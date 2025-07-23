"""SC62015 emulator with Perfetto tracing support."""

from typing import Optional, Union
from pathlib import Path
import struct

from sc62015.pysc62015.emulator import Emulator, Memory, RegisterName
from .trace_manager import g_tracer
from .tracing_config import TracingConfig


class TracingMemoryWrapper(Memory):
    """Memory wrapper that tracks reads and writes for tracing."""
    
    def __init__(self, wrapped_memory: Memory, tracer: 'TracingEmulator'):
        self._wrapped = wrapped_memory
        self._tracer = tracer
        
    def read_byte(self, address: int) -> int:
        self._tracer._memory_reads += 1
        # Set context if TracingEmulator has _current_pc
        if hasattr(self._wrapped, 'set_context') and hasattr(self._tracer, '_current_pc'):
            self._wrapped.set_context({'cpu_pc': self._tracer._current_pc})
        return self._wrapped.read_byte(address)
        
    def write_byte(self, address: int, value: int) -> None:
        self._tracer._memory_writes += 1
        # Set context if TracingEmulator has _current_pc
        if hasattr(self._wrapped, 'set_context') and hasattr(self._tracer, '_current_pc'):
            self._wrapped.set_context({'cpu_pc': self._tracer._current_pc})
        self._wrapped.write_byte(address, value)
        
    def read(self, address: int, size: int) -> bytes:
        self._tracer._memory_reads += size
        if hasattr(self._wrapped, 'read'):
            return self._wrapped.read(address, size)
        # Fallback for MemoryMapper - read_byte already increments counter
        return bytes([self._wrapped.read_byte(address + i) for i in range(size)])
        
    def write(self, address: int, data: bytes) -> None:
        self._tracer._memory_writes += len(data)
        if hasattr(self._wrapped, 'write'):
            self._wrapped.write(address, data)
        else:
            # Fallback for MemoryMapper - write_byte already increments counter
            for i, byte in enumerate(data):
                self._wrapped.write_byte(address + i, byte)
                
    def read_bytes(self, size: int, address: int) -> int:
        """For eval_llil compatibility - returns integer value."""
        # First try to delegate to wrapped memory's read_bytes if it has one
        if hasattr(self._wrapped, 'read_bytes'):
            return self._wrapped.read_bytes(size, address)
        
        # Otherwise, read bytes and convert to integer (little-endian)
        data = self.read(address, size)
        result = 0
        for i, byte in enumerate(data):
            result |= byte << (i * 8)
        return result
        
    def write_bytes(self, size: int, address: int, data) -> None:
        """For eval_llil compatibility."""
        if isinstance(data, int):
            # Handle both signed and unsigned integers
            # Mask to the appropriate size to avoid overflow
            max_val = (1 << (size * 8))
            data = data & (max_val - 1)
            data = data.to_bytes(size, byteorder='little', signed=False)
        self.write(address, data)
        
    def set_context(self, context: dict) -> None:
        """Pass through context setting."""
        if hasattr(self._wrapped, 'set_context'):
            self._wrapped.set_context(context)
            
    def __getattr__(self, name):
        """Forward any other attributes to wrapped memory."""
        return getattr(self._wrapped, name)


class TracingEmulator(Emulator):
    """SC62015 emulator with integrated Perfetto tracing.
    
    This emulator extends the base SC62015 emulator with comprehensive
    tracing support for CPU execution, jumps, and peripheral operations.
    """
    
    def __init__(self, memory: Memory, trace_output: Optional[Union[str, Path]] = None):
        """Initialize the tracing emulator.
        
        Args:
            memory: Memory interface for the emulator
            trace_output: Optional path to output .perfetto-trace file.
                         If provided, tracing starts automatically.
                         If not provided, checks TracingConfig for settings.
        """
        # Wrap memory to track accesses
        self._original_memory = memory
        wrapped_memory = TracingMemoryWrapper(memory, self)
        super().__init__(wrapped_memory)
        
        # Counter tracking
        self._instruction_count = 0
        self._memory_reads = 0
        self._memory_writes = 0
        self._last_counter_trace = 0
        self._counter_interval = 1000  # Trace counters every N instructions
        
        # PC tracking for memory context
        self._current_pc = 0
        self._last_pc = 0
        
        # Flow ID tracking for interrupts
        self._next_interrupt_id = 1
        self._interrupt_stack = []  # Stack of interrupt flow IDs
        
        # Check if tracing should be enabled
        if trace_output:
            g_tracer.start_tracing(trace_output)
        elif TracingConfig.is_enabled():
            g_tracer.start_tracing(TracingConfig.get_output_path())
    
    def execute_instruction(self, address: int) -> None:
        """Execute a single instruction with tracing support.
        
        This method adds Perfetto tracing for:
        - Function calls and returns (CALL/RET/CALLF/RETF/IR/RETI)
        - Jump instructions (JP/JPZ/JR etc)
        - Call stack depth tracking
        
        Args:
            address: Memory address of the instruction to execute
        """
        # Track PC history for tracing
        self._last_pc = self._current_pc
        self._current_pc = address
        
        self.regs.set(RegisterName.PC, address)
        instr = self.decode_instruction(address)
        assert instr is not None, f"Failed to decode instruction at {address:04X}"
        
        # Get opcode for tracing decisions
        opcode = self.memory.read_byte(address)
        # Read up to 4 bytes for full instruction
        data = bytes([self.memory.read_byte(address + i) for i in range(4)])
        
        # Track call stack depth and emit trace events
        # Monitor specific opcodes for call stack tracking and tracing
        if opcode == 0x04:  # CALL mn
            if g_tracer.is_tracing():
                new_pc = struct.unpack('<H', data[1:3])[0]
                g_tracer.begin_function("CPU", new_pc, address)
                
        elif opcode == 0x05:  # CALLF lmn
            if g_tracer.is_tracing():
                new_pc = data[1] | (data[2] << 8) | (data[3] << 16)
                g_tracer.begin_function("CPU", new_pc, address)
                
        elif opcode == 0xFE:  # IR - Interrupt entry
            if g_tracer.is_tracing():
                # Interrupt vector is at 0xFFFFA
                int_vector = bytes([self.memory.read_byte(0xFFFFA + i) for i in range(3)])
                new_pc = int_vector[0] | (int_vector[1] << 8) | (int_vector[2] << 16)
                
                # Start interrupt flow
                interrupt_id = self._next_interrupt_id
                self._next_interrupt_id += 1
                self._interrupt_stack.append(interrupt_id)
                
                g_tracer.begin_flow("Interrupt", interrupt_id)
                g_tracer.trace_interrupt("IR_Entry", vector=0xFFFFA, pc=address)
                g_tracer.begin_function("CPU", new_pc, address, name="IRQ_Handler")
                
        elif opcode == 0x06:  # RET
            if g_tracer.is_tracing():
                g_tracer.end_function("CPU", address)
            
        elif opcode == 0x07:  # RETF
            if g_tracer.is_tracing():
                g_tracer.end_function("CPU", address)
            
        elif opcode == 0x01:  # RETI - Return from interrupt
            if g_tracer.is_tracing():
                g_tracer.end_function("CPU", address)
                
                # End interrupt flow if we have one
                if self._interrupt_stack:
                    interrupt_id = self._interrupt_stack.pop()
                    g_tracer.trace_interrupt("RETI_Exit", pc=address)
                    g_tracer.end_flow("Interrupt", interrupt_id)
            
        # Trace jump instructions
        elif opcode == 0x02:  # JP mn
            if g_tracer.is_tracing():
                new_pc = struct.unpack('<H', data[1:3])[0]
                g_tracer.trace_jump("CPU", address, new_pc)
                
        elif opcode == 0x03:  # JPF lmn
            if g_tracer.is_tracing():
                new_pc = data[1] | (data[2] << 8) | (data[3] << 16)
                g_tracer.trace_jump("CPU", address, new_pc)
                
        elif opcode in [0x12, 0x13]:  # JR +n / JR -n
            if g_tracer.is_tracing():
                offset = data[1]
                if opcode == 0x13:  # Negative offset
                    offset = -offset
                new_pc = (address + 2 + offset) & 0xFFFFF  # 20-bit PC
                g_tracer.trace_jump("CPU", address, new_pc)
                
        elif opcode in range(0x14, 0x20):  # Conditional jumps
            # Store pre-execution state for conditional jump detection
            pc_before = address
            
        # Execute the instruction normally
        super().execute_instruction(address)
        
        # Increment instruction counter
        self._instruction_count += 1
        
        # Emit counter events periodically
        if self._instruction_count - self._last_counter_trace >= self._counter_interval:
            if g_tracer.is_tracing():
                g_tracer.trace_counter("CPU", "call_depth", self.regs.call_sub_level)
                g_tracer.trace_counter("CPU", "instructions", self._instruction_count)
                g_tracer.trace_counter("Memory", "reads", self._memory_reads)
                g_tracer.trace_counter("Memory", "writes", self._memory_writes)
                g_tracer.trace_counter("CPU", "stack_pointer", self.regs.get(RegisterName.S))
            self._last_counter_trace = self._instruction_count
        
        # Check if conditional jump was taken
        if opcode in range(0x14, 0x20) and g_tracer.is_tracing():
            pc_after = self.regs.get(RegisterName.PC)
            
            # Decode the expected jump target
            jump_target = None
            condition_name = self._get_condition_name(opcode)
            
            if opcode in [0x14, 0x15, 0x16, 0x17]:  # JP[Z/NZ/C/NC] mn
                jump_target = struct.unpack('<H', data[1:3])[0]
            elif opcode in range(0x18, 0x20):  # JR conditional
                offset = data[1]
                # Odd opcodes are negative offsets
                if opcode & 1:
                    offset = -offset
                jump_target = (pc_before + 2 + offset) & 0xFFFFF
            
            # If PC changed to expected target, jump was taken
            if jump_target and pc_after == jump_target:
                g_tracer.trace_instant(
                    "CPU",
                    f"Jump_{condition_name}_taken",
                    {
                        "from": f"0x{pc_before:06X}",
                        "to": f"0x{jump_target:06X}",
                        "opcode": f"0x{opcode:02X}"
                    }
                )
    
    def read_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory with optional tracing.
        
        Args:
            address: Memory address to read from
            size: Number of bytes to read
            
        Returns:
            Bytes read from memory
        """
        # Read bytes from memory (wrapper handles context and counting)
        if hasattr(self.memory, 'read'):
            data = self.memory.read(address, size)
        else:
            # MemoryMapper doesn't have read(), use read_byte
            data = bytes([self.memory.read_byte(address + i) for i in range(size)])
        
        # Add memory access tracing for significant reads
        if g_tracer.is_tracing() and size > 1:
            g_tracer.trace_instant(
                "Memory",
                f"Read_{size}B",
                {"addr": f"0x{address:06X}", "pc": f"0x{self._current_pc:06X}"}
            )
        
        return data
    
    def write_memory(self, address: int, data: bytes) -> None:
        """Write memory with optional tracing.
        
        Args:
            address: Memory address to write to
            data: Bytes to write
        """
        # Write bytes to memory (wrapper handles context and counting)
        if hasattr(self.memory, 'write'):
            self.memory.write(address, data)
        else:
            # MemoryMapper doesn't have write(), use write_byte
            for i, byte in enumerate(data):
                self.memory.write_byte(address + i, byte)
        
        # Add memory access tracing for writes
        if g_tracer.is_tracing():
            g_tracer.trace_instant(
                "Memory",
                f"Write_{len(data)}B",
                {"addr": f"0x{address:06X}", "pc": f"0x{self._current_pc:06X}"}
            )
    
    def trace_io_read(self, port: int, value: int) -> None:
        """Trace an I/O port read operation.
        
        Args:
            port: I/O port number
            value: Value read from the port
        """
        if g_tracer.is_tracing():
            g_tracer.trace_instant(
                "I/O",
                f"Read_Port_{port:02X}",
                {"value": f"0x{value:02X}", "pc": f"0x{self._current_pc:06X}"}
            )
    
    def trace_io_write(self, port: int, value: int) -> None:
        """Trace an I/O port write operation.
        
        Args:
            port: I/O port number
            value: Value written to the port
        """
        if g_tracer.is_tracing():
            g_tracer.trace_instant(
                "I/O",
                f"Write_Port_{port:02X}",
                {"value": f"0x{value:02X}", "pc": f"0x{self._current_pc:06X}"}
            )
    
    def get_current_pc(self) -> int:
        """Get the current program counter for tracing context.
        
        Returns:
            Current PC value
        """
        return self._current_pc
    
    def _get_condition_name(self, opcode: int) -> str:
        """Get human-readable condition name from opcode.
        
        Args:
            opcode: Conditional jump opcode (0x14-0x1F)
            
        Returns:
            Condition name like "Z", "NZ", "C", "NC"
        """
        # Mapping based on SC62015 opcode table
        conditions = {
            0x14: "Z",   # JPZ mn
            0x15: "NZ",  # JPNZ mn
            0x16: "C",   # JPC mn
            0x17: "NC",  # JPNC mn
            0x18: "Z",   # JRZ +n
            0x19: "Z",   # JRZ -n
            0x1A: "NZ",  # JRNZ +n
            0x1B: "NZ",  # JRNZ -n
            0x1C: "C",   # JRC +n
            0x1D: "C",   # JRC -n
            0x1E: "NC",  # JRNC +n
            0x1F: "NC",  # JRNC -n
        }
        return conditions.get(opcode, f"COND_{opcode:02X}")
    
    def stop_tracing(self) -> None:
        """Stop tracing and close the output file."""
        g_tracer.stop_tracing()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure tracing is stopped."""
        self.stop_tracing()
        return False