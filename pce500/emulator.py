"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import time
import struct
from typing import Optional, Dict, Any, Set
from pathlib import Path

# Import the SC62015 emulator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.emulator import Emulator as SC62015Emulator, RegisterName

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
        self.trace: list = [] if trace_enabled else None

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

        # Read instruction for analysis
        instr_bytes = bytes(self.memory.read_byte(pc + i) for i in range(4))
        opcode = instr_bytes[0] if instr_bytes else 0

        # Debug output
        # print(f"DEBUG: PC=0x{pc:06X}, opcode=0x{opcode:02X}, perfetto_enabled={self.perfetto_enabled}, call_depth={self.call_depth}")

        # Perfetto tracing with instruction analysis
        if self.perfetto_enabled:
            # Trace execution instant
            g_tracer.trace_instant("CPU", f"Exec@0x{pc:06X}", {
                "pc": f"0x{pc:06X}",
                "opcode": f"0x{opcode:02X}"
            })

            # Analyze control flow instructions
            self._analyze_control_flow(pc, opcode, instr_bytes)

        # Store PC before execution for conditional jump analysis
        pc_before = pc

        # Execute instruction
        try:
            self.cpu.execute_instruction(pc)
            self.cycle_count += 1

            # Update counters
            self.instruction_count += 1

            # Check if a conditional jump was taken
            if self.perfetto_enabled:
                pc_after = self.cpu.regs.get(RegisterName.PC)
                self._check_conditional_jump_taken(pc_before, pc_after, opcode)

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

    def _analyze_control_flow(self, pc: int, opcode: int, instr_bytes: bytes) -> None:
        """Analyze control flow instructions for Perfetto tracing.

        Args:
            pc: Current program counter
            opcode: Instruction opcode
            instr_bytes: Raw instruction bytes (at least 4 bytes)
        """
        # Get next PC after this instruction
        instr_len = self._get_instruction_length(opcode)
        new_pc = (pc + instr_len) & 0xFFFFFF

        # CALL - 16-bit absolute call
        if opcode == 0x04:
            dest_addr = struct.unpack('<H', instr_bytes[1:3])[0]
            g_tracer.begin_function("CPU", dest_addr, pc, f"func_0x{dest_addr:04X}")
            self.call_depth += 1
            g_tracer.trace_instant("CPU", "CALL", {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:04X}",
                "depth": self.call_depth
            })

        # CALLF - 20-bit far call
        elif opcode == 0x05:
            # 20-bit address in little-endian (3 bytes)
            dest_addr = (instr_bytes[1] | (instr_bytes[2] << 8) |
                        ((instr_bytes[3] & 0x0F) << 16))
            g_tracer.begin_function("CPU", dest_addr, pc, f"func_0x{dest_addr:05X}")
            self.call_depth += 1
            g_tracer.trace_instant("CPU", "CALLF", {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:05X}",
                "depth": self.call_depth
            })

        # RET - Return from subroutine
        elif opcode == 0x06:
            g_tracer.end_function("CPU", pc)
            self.call_depth = max(0, self.call_depth - 1)
            g_tracer.trace_instant("CPU", "RET", {
                "at": f"0x{pc:06X}",
                "depth": self.call_depth
            })

        # RETF - Far return
        elif opcode == 0x07:
            g_tracer.end_function("CPU", pc)
            self.call_depth = max(0, self.call_depth - 1)
            g_tracer.trace_instant("CPU", "RETF", {
                "at": f"0x{pc:06X}",
                "depth": self.call_depth
            })

        # IR - Software interrupt
        elif opcode == 0xFE:
            # Read interrupt vector from 0xFFFFA
            vector_addr = self.memory.read_long(0xFFFFA)
            # Generate unique interrupt ID
            interrupt_id = self._next_interrupt_id
            self._next_interrupt_id += 1
            self._interrupt_stack.append(interrupt_id)

            g_tracer.begin_flow("CPU", interrupt_id, f"IR@0x{pc:06X}")
            g_tracer.begin_function("CPU", vector_addr, pc, f"int_0x{vector_addr:05X}")
            self.call_depth += 1

            g_tracer.trace_instant("CPU", "IR", {
                "from": f"0x{pc:06X}",
                "vector": f"0x{vector_addr:05X}",
                "interrupt_id": interrupt_id
            })

        # RETI - Return from interrupt
        elif opcode == 0x01:
            g_tracer.end_function("CPU", pc)
            self.call_depth = max(0, self.call_depth - 1)

            if self._interrupt_stack:
                flow_id = self._interrupt_stack.pop()
                g_tracer.end_flow("CPU", flow_id, f"RETI@0x{pc:06X}")

            g_tracer.trace_instant("CPU", "RETI", {
                "at": f"0x{pc:06X}",
                "depth": self.call_depth
            })

        # JP - 16-bit absolute jump
        elif opcode == 0x0E:
            dest_addr = struct.unpack('<H', instr_bytes[1:3])[0]
            g_tracer.trace_instant("CPU", "JP", {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:04X}"
            })

        # JPF - 20-bit far jump
        elif opcode == 0x0F:
            dest_addr = (instr_bytes[1] | (instr_bytes[2] << 8) |
                        ((instr_bytes[3] & 0x0F) << 16))
            g_tracer.trace_instant("CPU", "JPF", {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:05X}"
            })

        # JR - 8-bit relative jump
        elif opcode == 0x0D:
            # Sign-extend 8-bit offset
            offset = instr_bytes[1]
            if offset & 0x80:
                offset |= 0xFFFFFF00
            dest_addr = (new_pc + offset) & 0xFFFFFF
            g_tracer.trace_instant("CPU", "JR", {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:06X}",
                "offset": offset
            })

        # Conditional jumps (JRZ, JRNZ, JRC, JRNC)
        elif opcode in [0x08, 0x09, 0x0A, 0x0B]:
            # These are 2-byte instructions with 8-bit relative offset
            offset = instr_bytes[1]
            if offset & 0x80:
                offset |= 0xFFFFFF00
            dest_addr = (pc + 2 + offset) & 0xFFFFFF

            # Determine jump type
            jump_types = {0x08: "JRZ", 0x09: "JRNZ", 0x0A: "JRC", 0x0B: "JRNC"}
            jump_type = jump_types.get(opcode, "JR??")

            # We can't know if the jump is taken until after execution
            # For now, just trace that it's a conditional jump
            g_tracer.trace_instant("CPU", jump_type, {
                "from": f"0x{pc:06X}",
                "to": f"0x{dest_addr:06X}",
                "offset": offset,
                "conditional": True
            })

    def _get_instruction_length(self, opcode: int) -> int:
        """Get instruction length based on opcode.

        This is a simplified version - full implementation would need
        complete opcode table.
        """
        # Control flow instructions
        if opcode in [0x04, 0x0E]:  # CALL, JP
            return 3
        elif opcode in [0x05, 0x0F]:  # CALLF, JPF
            return 4
        elif opcode in [0x08, 0x09, 0x0A, 0x0B, 0x0D]:  # JRZ, JRNZ, JRC, JRNC, JR
            return 2
        elif opcode in [0x00, 0x01, 0x06, 0x07, 0xFE]:  # NOP, RETI, RET, RETF, IR
            return 1
        else:
            return 1  # Default for simple instructions

    def _check_conditional_jump_taken(self, pc_before: int, pc_after: int, opcode: int) -> None:
        """Check if a conditional jump was taken and trace it.

        Args:
            pc_before: PC before instruction execution
            pc_after: PC after instruction execution
            opcode: The instruction opcode
        """
        # Conditional jumps: JRZ, JRNZ, JRC, JRNC
        if opcode in [0x08, 0x09, 0x0A, 0x0B]:
            instr_len = 2  # Conditional jumps are 2 bytes
            expected_pc = (pc_before + instr_len) & 0xFFFFFF

            # If PC changed from expected, jump was taken
            jump_taken = pc_after != expected_pc
            jump_types = {0x08: "JRZ", 0x09: "JRNZ", 0x0A: "JRC", 0x0B: "JRNC"}
            jump_type = jump_types.get(opcode, "JR??")

            g_tracer.trace_instant("CPU", f"{jump_type}_Result", {
                "from": f"0x{pc_before:06X}",
                "to": f"0x{pc_after:06X}",
                "taken": jump_taken
            })
