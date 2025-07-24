"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import time
from typing import Optional, Dict, Any, Set
from pathlib import Path

# Import the SC62015 emulator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.emulator import Emulator as SC62015Emulator, RegisterName
from sc62015.pysc62015.instr.instructions import (
    CALL, RetInstruction, JumpInstruction, IR
)

from .memory import PCE500Memory
from .display.hd61202 import HD61202Controller

from .trace_manager import g_tracer


class TrackedMemory(PCE500Memory):
    """Memory wrapper that tracks read/write counts for the emulator."""

    def __init__(self, emulator):
        super().__init__()
        self.emulator = emulator

    def read_byte(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read byte and increment counter."""
        self.emulator.memory_read_count += 1
        # Use current PC if not provided
        if cpu_pc is None and hasattr(self.emulator, '_current_pc'):
            cpu_pc = self.emulator._current_pc
        return super().read_byte(address, cpu_pc)

    def write_byte(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write byte and increment counter."""
        self.emulator.memory_write_count += 1
        # Use current PC if not provided
        if cpu_pc is None and hasattr(self.emulator, '_current_pc'):
            cpu_pc = self.emulator._current_pc
        super().write_byte(address, value, cpu_pc)

    def read_bytes(self, address: int, size: int) -> int:
        """Read bytes according to binja_test_mocks.eval_llil.Memory interface."""
        self.emulator.memory_read_count += size
        # Read bytes and convert to integer (little-endian)
        result = 0
        for i in range(size):
            cpu_pc = getattr(self.emulator, '_current_pc', None)
            byte_val = super().read_byte(address + i, cpu_pc)
            result |= byte_val << (i * 8)
        return result

    def write_bytes(self, size: int, address: int, value: int) -> None:
        """Write bytes according to binja_test_mocks.eval_llil.Memory interface."""
        # Convert value to bytes of specified size (little-endian)
        for i in range(size):
            byte_value = (value >> (i * 8)) & 0xFF
            self.write_byte(address + i, byte_value)

    def set_context(self, context: dict) -> None:
        """Set context for memory operations (compatibility method)."""
        # Context is handled automatically through _current_pc
        pass


class PCE500Emulator:
    """PC-E500 emulator with integrated machine configuration.

    This is the simplified implementation that replaces the original
    multi-layered architecture with a cleaner, more maintainable design.
    """

    # Memory constants
    INTERNAL_ROM_START = 0xC0000
    INTERNAL_ROM_SIZE = 0x40000    # 256KB
    INTERNAL_RAM_START = 0xB8000
    INTERNAL_RAM_SIZE = 0x8000     # 32KB

    def __init__(self, trace_enabled: bool = False, perfetto_trace: bool = False):
        """Initialize the PC-E500 emulator.

        Args:
            trace_enabled: Enable simple list-based tracing
            perfetto_trace: Enable Perfetto tracing (if available)
        """
        # Create memory and LCD controller
        self.memory = TrackedMemory(self)
        self.lcd = HD61202Controller()
        self.memory.set_lcd_controller(self.lcd)

        # Create CPU emulator with our memory
        self.cpu = SC62015Emulator(self.memory)

        # Emulation state
        self.breakpoints: Set[int] = set()
        self.cycle_count = 0
        self.start_time = time.time()

        # Simple tracing
        self.trace_enabled = trace_enabled
        self.trace: Optional[list] = [] if trace_enabled else None

        # Perfetto tracing
        self.perfetto_enabled = perfetto_trace
        if self.perfetto_enabled:
            g_tracer.start_tracing("pc-e500.trace")
            # Enable LCD and memory tracing
            self.lcd.set_perfetto_enabled(True)
            self.memory.set_perfetto_enabled(True)

        # Call stack tracking for tracing
        self.call_depth = 0
        self._interrupt_stack = []  # Track interrupt IDs
        self._next_interrupt_id = 1  # For generating unique interrupt flow IDs

        # PC tracking for memory context and jump analysis
        self._current_pc = 0
        self._last_pc = 0

        # Performance counters
        self.instruction_count = 0
        self.memory_read_count = 0
        self.memory_write_count = 0

    def load_rom(self, rom_data: bytes, start_address: Optional[int] = None) -> None:
        """Load ROM data."""
        if start_address is None:
            start_address = self.INTERNAL_ROM_START

        if start_address == self.INTERNAL_ROM_START:
            # Loading as internal ROM
            self.memory.load_rom(rom_data)
        else:
            # Loading at arbitrary address
            self.memory.add_rom(start_address, rom_data, "Loaded ROM")

    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        """Load a memory card (8KB, 16KB, 32KB, or 64KB)."""
        self.memory.load_memory_card(card_data, card_size)

    def expand_ram(self, size: int, start_address: int) -> None:
        """Add RAM expansion module."""
        self.memory.add_ram(start_address, size, f"RAM Expansion ({size//1024}KB)")

    def reset(self) -> None:
        """Reset the emulator to initial state."""
        # Reset memory
        self.memory.reset()

        # Reset LCD
        self.lcd.reset()

        # Reset CPU state by setting PC to reset vector
        # Get reset vector from ROM
        reset_vector = self.memory.read_long(0xFFFFD)
        self.cpu.regs.set(RegisterName.PC, reset_vector)

        # Reset emulation state
        self.cycle_count = 0
        self.start_time = time.time()
        if self.trace is not None:
            self.trace.clear()

    def step(self) -> bool:
        """Execute a single instruction.

        Returns:
            True if execution should continue, False if breakpoint hit.
        """
        pc = self.cpu.regs.get(RegisterName.PC)

        # Track PC for memory context
        self._last_pc = self._current_pc
        self._current_pc = pc

        # Check breakpoint
        if pc in self.breakpoints:
            return False

        # Trace if enabled
        if self.trace is not None:
            self.trace.append(('exec', pc, self.cycle_count))

        # Pre-execution - capture state for tracing
        opcode = None
        fc_before = None
        fz_before = None
        
        if self.perfetto_enabled:
            # Read the opcode and flag state before execution
            opcode = self.memory.read_byte(pc)
            fc_before = self.cpu.regs.get(RegisterName.FC)
            fz_before = self.cpu.regs.get(RegisterName.FZ)

        # Store PC before execution for conditional jump analysis
        pc_before = pc

        # Execute instruction
        try:
            eval_info = self.cpu.execute_instruction(pc)
            self.cycle_count += 1

            # Update counters
            self.instruction_count += 1

            # Post-execution analysis - detailed tracing after execution
            if self.perfetto_enabled:
                # Capture flag state after execution
                fc_after = self.cpu.regs.get(RegisterName.FC)
                fz_after = self.cpu.regs.get(RegisterName.FZ)
                
                # Trace execution with both before and after flag states
                g_tracer.trace_instant("Execution", f"Exec@0x{pc:06X}", {
                    "pc": f"0x{pc:06X}",
                    "opcode": f"0x{opcode:02X}",
                    "C_before": fc_before,
                    "Z_before": fz_before,
                    "C_after": fc_after,
                    "Z_after": fz_after
                })

                # Analyze control flow instructions
                self._analyze_control_flow(pc_before, eval_info)

                # Check if a conditional jump was taken
                pc_after = self.cpu.regs.get(RegisterName.PC)
                self._check_conditional_jump_taken(pc_before, pc_after, eval_info)

            # Update Perfetto counters
            if self.perfetto_enabled:
                g_tracer.trace_counter("CPU", "cycles", self.cycle_count)
                g_tracer.trace_counter("CPU", "call_depth", self.call_depth)
                g_tracer.trace_counter("CPU", "instructions", self.instruction_count)

                # Get and trace stack pointer
                sp = self.cpu.regs.get(RegisterName.S)
                g_tracer.trace_counter("CPU", "stack_pointer", sp)

                # Trace memory operation counters
                g_tracer.trace_counter("Memory", "read_ops", self.memory_read_count)
                g_tracer.trace_counter("Memory", "write_ops", self.memory_write_count)
        except Exception as e:
            if self.trace is not None:
                self.trace.append(('error', pc, str(e)))
            if self.perfetto_enabled:
                g_tracer.trace_instant("CPU", "Error", {"error": str(e), "pc": f"0x{pc:06X}"})
            raise

        return True

    def run(self, max_instructions: Optional[int] = None) -> int:
        """Run emulation until breakpoint or instruction limit.

        Args:
            max_instructions: Maximum instructions to execute (None for unlimited)

        Returns:
            Number of instructions executed.
        """
        count = 0
        while True:
            if max_instructions is not None and count >= max_instructions:
                break

            if not self.step():
                break  # Breakpoint hit

            count += 1

        return count

    def add_breakpoint(self, address: int) -> None:
        """Add a breakpoint at the specified address."""
        self.breakpoints.add(address & 0xFFFFFF)

    def remove_breakpoint(self, address: int) -> None:
        """Remove a breakpoint."""
        self.breakpoints.discard(address & 0xFFFFFF)

    def get_cpu_state(self) -> Dict[str, Any]:
        """Get current CPU state."""
        return {
            'pc': self.cpu.regs.get(RegisterName.PC),
            'a': self.cpu.regs.get(RegisterName.A),
            'b': self.cpu.regs.get(RegisterName.B),
            'ba': self.cpu.regs.get(RegisterName.BA),
            'i': self.cpu.regs.get(RegisterName.I),
            'x': self.cpu.regs.get(RegisterName.X),
            'y': self.cpu.regs.get(RegisterName.Y),
            'u': self.cpu.regs.get(RegisterName.U),
            's': self.cpu.regs.get(RegisterName.S),
            'flags': {
                'z': self.cpu.regs.get(RegisterName.FZ),
                'c': self.cpu.regs.get(RegisterName.FC)
            },
            'cycles': self.cycle_count
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get emulation performance statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        if elapsed > 0:
            instructions_per_second = self.cycle_count / elapsed
            # Assume 2MHz CPU clock for speed ratio
            speed_ratio = instructions_per_second / 2_000_000
        else:
            instructions_per_second = 0
            speed_ratio = 0

        return {
            'instructions_executed': self.cycle_count,
            'elapsed_time': elapsed,
            'instructions_per_second': instructions_per_second,
            'speed_ratio': speed_ratio
        }

    def get_memory_info(self) -> str:
        """Get information about memory configuration."""
        return self.memory.get_memory_info()

    def get_display_buffer(self):
        """Get the current display buffer."""
        return self.lcd.get_display_buffer()

    @property
    def display_on(self) -> bool:
        """Check if display is on."""
        return any(self.lcd.display_on)

    # Convenience properties for compatibility
    @property
    def main_lcd(self):
        return self.lcd

    @property
    def regs(self):
        return self.cpu.regs

    def stop_tracing(self) -> None:
        """Stop Perfetto tracing and save trace file."""
        if self.perfetto_enabled:
            print("Stopping Perfetto tracing...")
            g_tracer.stop_tracing()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure tracing is stopped."""
        self.stop_tracing()
        return False
    
    
    def _get_condition_code(self, instr) -> Optional[str]:
        """Get condition code from conditional instruction."""
        return getattr(instr, '_cond', None)
    
    
    def _trace_call(self, pc: int, dest_addr: int):
        """Trace call instruction."""
        g_tracer.begin_function("CPU", dest_addr, pc, f"func_0x{dest_addr:05X}")
        self.call_depth += 1
        g_tracer.trace_instant("CPU", "call", {
            "from": f"0x{pc:06X}",
            "to": f"0x{dest_addr:05X}",
            "depth": self.call_depth
        })
    
    def _trace_return(self, pc: int, instr_class_name: str):
        """Trace return instruction."""
        g_tracer.end_function("CPU", pc)
        self.call_depth = max(0, self.call_depth - 1)
        
        # Special handling for RETI
        if instr_class_name == "RETI" and self._interrupt_stack:
            flow_id = self._interrupt_stack.pop()
            g_tracer.end_flow("CPU", flow_id, f"RETI@0x{pc:06X}")
        
        g_tracer.trace_instant("CPU", "return", {
            "at": f"0x{pc:06X}",
            "type": instr_class_name.lower(),
            "depth": self.call_depth
        })
    
    def _trace_interrupt(self, pc: int, vector_addr: int):
        """Trace software interrupt."""
        interrupt_id = self._next_interrupt_id
        self._next_interrupt_id += 1
        self._interrupt_stack.append(interrupt_id)

        g_tracer.begin_flow("CPU", interrupt_id, f"IR@0x{pc:06X}")
        g_tracer.begin_function("CPU", vector_addr, pc, f"int_0x{vector_addr:05X}")
        self.call_depth += 1

        g_tracer.trace_instant("CPU", "interrupt", {
            "from": f"0x{pc:06X}",
            "vector": f"0x{vector_addr:05X}",
            "interrupt_id": interrupt_id
        })
    
    def _trace_jump(self, pc: int, dest_addr: int, condition: Optional[str] = None, taken: bool = True):
        """Trace jump instruction - only if taken."""
        # For conditional jumps, only trace if taken
        if condition and not taken:
            return
            
        trace_data = {
            "from": f"0x{pc:06X}",
            "to": f"0x{dest_addr:06X}"
        }
        
        if condition:
            trace_data["condition"] = condition
            trace_data["type"] = "conditional_taken"
        else:
            trace_data["type"] = "unconditional"
            
        g_tracer.trace_instant("CPU", "jump", trace_data)

    def _analyze_control_flow(self, pc: int, eval_info) -> None:
        """Analyze control flow instructions for Perfetto tracing.

        Args:
            pc: Current program counter
            eval_info: InstructionEvalInfo containing instruction and metadata
        """
        instr = eval_info.instruction
        
        # Handle CALL instructions
        if isinstance(instr, CALL):
            dest_addr = instr.dest_addr(pc)
            self._trace_call(pc, dest_addr)
        
        # Handle return instructions (RET, RETF, RETI)
        elif isinstance(instr, RetInstruction):
            self._trace_return(pc, type(instr).__name__)
        
        # Handle IR (software interrupt)
        elif isinstance(instr, IR):
            # Read interrupt vector from 0xFFFFA
            vector_addr = self.memory.read_long(0xFFFFA)
            self._trace_interrupt(pc, vector_addr)
        
        # Handle jump instructions - don't trace here, wait until after execution
        elif isinstance(instr, JumpInstruction):
            # We'll check if the jump was taken in _check_conditional_jump_taken
            pass


    def _check_conditional_jump_taken(self, pc_before: int, pc_after: int, eval_info) -> None:
        """Check if a jump was taken and trace it.

        Args:
            pc_before: PC before instruction execution
            pc_after: PC after instruction execution
            eval_info: InstructionEvalInfo containing instruction and metadata
        """
        instr = eval_info.instruction
        
        # Only process jump instructions
        if not isinstance(instr, JumpInstruction):
            return
            
        # Calculate expected PC if no jump was taken
        expected_pc = (pc_before + eval_info.instruction_info.length) & 0xFFFFFF
        
        # If PC changed from expected, jump was taken
        jump_taken = (pc_after != expected_pc)
        
        # Get condition code (None for unconditional jumps)
        condition = self._get_condition_code(instr)
        
        # For unconditional jumps, always trace
        # For conditional jumps, only trace if taken
        if not condition or jump_taken:
            self._trace_jump(pc_before, pc_after, condition, jump_taken)
