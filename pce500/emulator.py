"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Set
from collections import deque

# Import the SC62015 emulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.emulator import Emulator as SC62015Emulator, RegisterName
from sc62015.pysc62015.instr.instructions import (
    CALL, RetInstruction, JumpInstruction, IR
)

from .memory import PCE500Memory, MemoryOverlay
from .display import HD61202Controller
from .keyboard import PCE500KeyboardHandler
from .trace_manager import g_tracer

# Define constants locally to avoid heavy imports
INTERNAL_MEMORY_START = 0x100000  # SC62015 internal memory starts at this address
KOL = 0xF0  # Key Output Low register offset
KOH = 0xF1  # Key Output High register offset
KIL = 0xF2  # Key Input register offset


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
    INTERNAL_ROM_START = 0xC0000  # ROM location
    INTERNAL_ROM_SIZE = 0x40000    # 256KB
    INTERNAL_RAM_START = 0xB8000
    INTERNAL_RAM_SIZE = 0x8000     # 32KB
    
    # Debug/dump configuration
    MEMORY_DUMP_PC = 0x0F119C      # PC to trigger internal memory dump
    MEMORY_DUMP_DIR = "."          # Directory for dump files

    def __init__(self, trace_enabled: bool = False, perfetto_trace: bool = False, 
                 save_lcd_on_exit: bool = True):
        """Initialize the PC-E500 emulator.

        Args:
            trace_enabled: Enable simple list-based tracing
            perfetto_trace: Enable Perfetto tracing (if available)
            save_lcd_on_exit: Save LCD displays as PNG when exiting context manager
        """
        # Initialize performance counters first (needed by TrackedMemory)
        self.instruction_count = 0
        self.memory_read_count = 0
        self.memory_write_count = 0
        
        # Store save LCD option
        self.save_lcd_on_exit = save_lcd_on_exit
        
        # Create memory and LCD controller
        self.memory = TrackedMemory(self)
        self.lcd = HD61202Controller()
        self.memory.set_lcd_controller(self.lcd)
        
        # Create and integrate keyboard handler
        self.keyboard = PCE500KeyboardHandler()
        keyboard_overlay = MemoryOverlay(
            start=INTERNAL_MEMORY_START + KOL,
            end=INTERNAL_MEMORY_START + KIL,
            name="keyboard_io",
            read_only=False,
            read_handler=self._keyboard_read_handler,
            write_handler=self._keyboard_write_handler,
            perfetto_thread="I/O"
        )
        self.memory.add_overlay(keyboard_overlay)

        # Create CPU emulator with our memory (temporarily disable tracing during reset)
        old_perfetto_state = self.memory.perfetto_enabled
        self.memory.set_perfetto_enabled(False)
        self.cpu = SC62015Emulator(self.memory, reset_on_init=True)
        self.memory.set_perfetto_enabled(old_perfetto_state)
        
        # Give memory access to CPU for accessing internal registers
        self.memory.set_cpu(self.cpu)
        
        # After power-on reset, set PC to entry point (not reset vector)
        # The reset vector at 0xFFFFA is only used for RESET instruction
        # Normal startup uses entry point at 0xFFFFD
        # Check if ROM is loaded by looking for internal_rom overlay
        rom_loaded = any(overlay.name == 'internal_rom' for overlay in self.memory.overlays)
        if rom_loaded:
            entry_point = self.memory.read_long(0xFFFFD)
            self.cpu.regs.set(RegisterName.PC, entry_point)

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
        
        # Instruction history tracking
        self.instruction_history: deque = deque(maxlen=100)

    def load_rom(self, rom_data: bytes, start_address: Optional[int] = None) -> None:
        """Load ROM data."""
        if start_address is None:
            start_address = self.INTERNAL_ROM_START

        if start_address == self.INTERNAL_ROM_START or start_address == 0xC0000:
            # Loading as internal ROM (at 0xC0000)
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

        # Reset CPU state (temporarily disable tracing during reset)
        old_perfetto_state = self.memory.perfetto_enabled
        self.memory.set_perfetto_enabled(False)
        self.cpu.power_on_reset()
        self.memory.set_perfetto_enabled(old_perfetto_state)
        
        # After power-on reset, set PC to entry point (not reset vector)
        # The reset vector at 0xFFFFA is only used for RESET instruction
        # Normal startup uses entry point at 0xFFFFD
        # Check if ROM is loaded by looking for internal_rom overlay
        rom_loaded = any(overlay.name == 'internal_rom' for overlay in self.memory.overlays)
        if rom_loaded:
            entry_point = self.memory.read_long(0xFFFFD)
            self.cpu.regs.set(RegisterName.PC, entry_point)

        # Reset emulation state
        self.cycle_count = 0
        self.start_time = time.time()
        if self.trace is not None:
            self.trace.clear()
        
        # Clear instruction history
        self.instruction_history.clear()
        
        # Clear IMEM register access tracking
        self.memory.clear_imem_access_tracking()

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

        # Check for internal memory dump trigger
        if pc == self.MEMORY_DUMP_PC and self.perfetto_enabled:
            # Dump internal memory when reaching target PC
            internal_mem = self.memory.get_internal_memory_bytes()
            
            # Save to binary file with timestamp
            import time
            import os
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dump_filename = f"internal_memory_dump_{timestamp}_pc_{pc:06X}.bin"
            dump_path = os.path.join(self.MEMORY_DUMP_DIR, dump_filename)
            
            with open(dump_path, 'wb') as f:
                f.write(internal_mem)
            
            print(f"\nInternal memory dumped to: {dump_path}")
            print(f"  PC: 0x{pc:06X}")
            print(f"  Size: {len(internal_mem)} bytes")
            
            # Also trace it in Perfetto
            g_tracer.trace_instant("Debug", "InternalMemoryDump", {
                "pc": f"0x{pc:06X}",
                "filename": dump_filename,
                "size": str(len(internal_mem)),
                "trigger": f"PC match 0x{self.MEMORY_DUMP_PC:06X}"
            })
        
        # Pre-execution - capture state for tracing
        opcode = None
        
        if self.perfetto_enabled:
            # Read the opcode before execution
            opcode = self.memory.read_byte(pc)

        # Store PC before execution for conditional jump analysis
        pc_before = pc

        # Execute instruction
        try:
            eval_info = self.cpu.execute_instruction(pc)
            self.cycle_count += 1

            # Update counters
            self.instruction_count += 1
            
            # Add to instruction history
            from binja_test_mocks.tokens import asm_str
            disassembly = asm_str(eval_info.instruction.render())
            self.instruction_history.append({
                "pc": f"0x{pc:06X}",
                "disassembly": disassembly
            })

            # Post-execution analysis - detailed tracing after execution
            if self.perfetto_enabled:
                # Get full CPU state after execution
                state = self.get_cpu_state()
                
                # Trace execution with all registers (same naming as function calls)
                event = g_tracer.trace_instant("Execution", f"Exec@0x{pc:06X}", {
                    "pc": f"0x{pc:06X}",
                    "opcode": f"0x{opcode:02X}"
                })
                
                # Add all register values as annotations (same as function calls)
                if event:
                    event.add_annotations({
                        "reg_A": f"0x{state['a']:02X}",
                        "reg_B": f"0x{state['b']:02X}",
                        "reg_BA": f"0x{state['ba']:04X}",
                        "reg_I": f"0x{state['i']:04X}",
                        "reg_X": f"0x{state['x']:06X}",
                        "reg_Y": f"0x{state['y']:06X}",
                        "reg_U": f"0x{state['u']:06X}",
                        "reg_S": f"0x{state['s']:06X}",
                        "reg_PC": f"0x{state['pc']:06X}",
                        "flag_C": state['flags']['c'],
                        "flag_Z": state['flags']['z']
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
        
    def set_memory_dump_pc(self, address: int) -> None:
        """Set the PC address that triggers internal memory dump."""
        self.MEMORY_DUMP_PC = address & 0xFFFFFF
        print(f"Internal memory dump will trigger at PC=0x{self.MEMORY_DUMP_PC:06X}")

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
        
    def save_lcd_displays(self, combined_filename: str = "lcd_display.png",
                         save_individual: bool = False) -> None:
        """Save LCD display to PNG file.
        
        Args:
            combined_filename: Filename for combined display image
            save_individual: Also save individual chip displays for debugging
        """
        # Save combined display
        combined = self.lcd.get_combined_display(zoom=2)
        combined.save(combined_filename)
        print(f"LCD display saved to {combined_filename}")
        
        # Optionally save individual chips
        if save_individual:
            self.lcd.save_displays_to_png("lcd_left.png", "lcd_right.png")
            print("Individual chip displays saved to lcd_left.png and lcd_right.png")

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
        """Context manager exit - ensure tracing is stopped and optionally save LCD displays."""
        # Save LCD displays as PNG images if enabled
        if hasattr(self, 'save_lcd_on_exit') and self.save_lcd_on_exit:
            # Save combined display
            combined = self.lcd.get_combined_display(zoom=2)
            combined.save("lcd_display.png")
            print("LCD display saved to lcd_display.png")
            
            # Also save individual chips for debugging
            self.lcd.save_displays_to_png("lcd_left.png", "lcd_right.png")
            print("Individual chip displays saved to lcd_left.png and lcd_right.png")
        
        # Stop tracing
        self.stop_tracing()
        return False
    
    
    def _get_condition_code(self, instr) -> Optional[str]:
        """Get condition code from conditional instruction."""
        return getattr(instr, '_cond', None)
    
    
    def _trace_call(self, pc: int, dest_addr: int):
        """Trace call instruction with register state."""
        # Begin function duration event
        func_event = g_tracer.begin_function("CPU", dest_addr, pc, f"func_0x{dest_addr:05X}")
        self.call_depth += 1
        
        # Add register annotations to the function slice event
        if func_event:
            state = self.get_cpu_state()
            
            # Add all register values as annotations
            func_event.add_annotations({
                "reg_A": f"0x{state['a']:02X}",
                "reg_B": f"0x{state['b']:02X}",
                "reg_BA": f"0x{state['ba']:04X}",
                "reg_I": f"0x{state['i']:04X}",
                "reg_X": f"0x{state['x']:06X}",
                "reg_Y": f"0x{state['y']:06X}",
                "reg_U": f"0x{state['u']:06X}",
                "reg_S": f"0x{state['s']:06X}",
                "reg_PC": f"0x{state['pc']:06X}",
                "flag_C": state['flags']['c'],
                "flag_Z": state['flags']['z']
            })
        
        # Create call instant event with basic info only
        g_tracer.trace_instant("CPU", "call", {
            "from": f"0x{pc:06X}",
            "to": f"0x{dest_addr:05X}",
            "depth": self.call_depth
        })
    
    def _trace_return(self, pc: int, instr_class_name: str):
        """Trace return instruction with register state."""
        g_tracer.end_function("CPU", pc)
        self.call_depth = max(0, self.call_depth - 1)
        
        # Special handling for RETI
        if instr_class_name == "RETI" and self._interrupt_stack:
            flow_id = self._interrupt_stack.pop()
            g_tracer.end_flow("CPU", flow_id, f"RETI@0x{pc:06X}")
        
        # Create return instant event with basic info
        return_event = g_tracer.trace_instant("CPU", "return", {
            "at": f"0x{pc:06X}",
            "type": instr_class_name.lower(),
            "depth": self.call_depth
        })
        
        # Add register annotations to the return event
        if return_event:
            state = self.get_cpu_state()
            
            # Add all register values as annotations
            return_event.add_annotations({
                "reg_A": f"0x{state['a']:02X}",
                "reg_B": f"0x{state['b']:02X}",
                "reg_BA": f"0x{state['ba']:04X}",
                "reg_I": f"0x{state['i']:04X}",
                "reg_X": f"0x{state['x']:06X}",
                "reg_Y": f"0x{state['y']:06X}",
                "reg_U": f"0x{state['u']:06X}",
                "reg_S": f"0x{state['s']:06X}",
                "reg_PC": f"0x{state['pc']:06X}",
                "flag_C": state['flags']['c'],
                "flag_Z": state['flags']['z']
            })
    
    def _trace_interrupt(self, pc: int, vector_addr: int):
        """Trace software interrupt with register state."""
        interrupt_id = self._next_interrupt_id
        self._next_interrupt_id += 1
        self._interrupt_stack.append(interrupt_id)

        g_tracer.begin_flow("CPU", interrupt_id, f"IR@0x{pc:06X}")
        func_event = g_tracer.begin_function("CPU", vector_addr, pc, f"int_0x{vector_addr:05X}")
        self.call_depth += 1

        # Add register annotations to the interrupt function slice event
        if func_event:
            state = self.get_cpu_state()
            
            # Add all register values as annotations
            func_event.add_annotations({
                "reg_A": f"0x{state['a']:02X}",
                "reg_B": f"0x{state['b']:02X}",
                "reg_BA": f"0x{state['ba']:04X}",
                "reg_I": f"0x{state['i']:04X}",
                "reg_X": f"0x{state['x']:06X}",
                "reg_Y": f"0x{state['y']:06X}",
                "reg_U": f"0x{state['u']:06X}",
                "reg_S": f"0x{state['s']:06X}",
                "reg_PC": f"0x{state['pc']:06X}",
                "flag_C": state['flags']['c'],
                "flag_Z": state['flags']['z']
            })

        # Create interrupt instant event with basic info only
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
    
    def press_key(self, key_code: str):
        """Simulates pressing a key on the keyboard."""
        if self.keyboard:
            self.keyboard.press_key(key_code)

    def release_key(self, key_code: str):
        """Simulates releasing a key on the keyboard."""
        if self.keyboard:
            self.keyboard.release_key(key_code)
    
    def _keyboard_read_handler(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Handle keyboard register reads."""
        offset = address - INTERNAL_MEMORY_START
        
        # Track register access for internal register watch
        if cpu_pc is not None:
            reg_name = None
            if offset == 0xF0:
                reg_name = "KOL"
            elif offset == 0xF1:
                reg_name = "KOH"
            elif offset == 0xF2:
                reg_name = "KIL"
            
            if reg_name:
                if reg_name not in self.memory.imem_access_tracking:
                    self.memory.imem_access_tracking[reg_name] = {"reads": [], "writes": []}
                reads_list = self.memory.imem_access_tracking[reg_name]["reads"]
                if reads_list and reads_list[-1][0] == cpu_pc:
                    # Increment count for same PC
                    reads_list[-1] = (cpu_pc, reads_list[-1][1] + 1)
                else:
                    # Add new PC with count 1
                    reads_list.append((cpu_pc, 1))
                    if len(reads_list) > 10:
                        reads_list.pop(0)
        
        return self.keyboard.handle_register_read(offset)
    
    def _keyboard_write_handler(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Handle keyboard register writes."""
        offset = address - INTERNAL_MEMORY_START
        
        # Track register access for internal register watch
        if cpu_pc is not None:
            reg_name = None
            if offset == 0xF0:
                reg_name = "KOL"
            elif offset == 0xF1:
                reg_name = "KOH"
            elif offset == 0xF2:
                reg_name = "KIL"
            
            if reg_name:
                if reg_name not in self.memory.imem_access_tracking:
                    self.memory.imem_access_tracking[reg_name] = {"reads": [], "writes": []}
                writes_list = self.memory.imem_access_tracking[reg_name]["writes"]
                if writes_list and writes_list[-1][0] == cpu_pc:
                    # Increment count for same PC
                    writes_list[-1] = (cpu_pc, writes_list[-1][1] + 1)
                else:
                    # Add new PC with count 1
                    writes_list.append((cpu_pc, 1))
                    if len(writes_list) > 10:
                        writes_list.pop(0)
        
        self.keyboard.handle_register_write(offset, value)
