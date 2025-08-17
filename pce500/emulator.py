"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Set
from collections import deque

# Import the SC62015 emulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.emulator import Emulator as SC62015Emulator, RegisterName
from sc62015.pysc62015.instr.instructions import (
    CALL,
    RetInstruction,
    JumpInstruction,
    IR,
)
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from .memory import PCE500Memory, MemoryOverlay
from .display import HD61202Controller
from .keyboard_compat import PCE500KeyboardHandler as KeyboardCompat
from .keyboard_hardware import KeyboardHardware
from .trace_manager import g_tracer
from .tracing.perfetto_tracing import tracer as new_tracer, perf_trace

# Define constants locally to avoid heavy imports
INTERNAL_MEMORY_START = 0x100000
KOL, KOH, KIL = IMEMRegisters.KOL, IMEMRegisters.KOH, IMEMRegisters.KIL


def _trace_probe_pc_and_opcode(emu):
    """Extract PC and opcode for tracing without side effects."""
    pc = emu.cpu.regs.get(RegisterName.PC) if hasattr(emu, "cpu") else None
    opcode = None
    if pc is not None and hasattr(emu, "memory"):
        try:
            opcode = emu.memory.read_byte(pc) & 0xFF
        except Exception:
            pass
    return pc, opcode


class PCE500Emulator:
    """PC-E500 emulator with integrated machine configuration."""

    INTERNAL_ROM_START = 0xC0000
    INTERNAL_ROM_SIZE = 0x40000
    INTERNAL_RAM_START = 0xB8000
    INTERNAL_RAM_SIZE = 0x8000
    MEMORY_DUMP_PC = 0x0F119C
    MEMORY_DUMP_DIR = "."

    def __init__(
        self,
        trace_enabled: bool = False,
        perfetto_trace: bool = False,
        save_lcd_on_exit: bool = True,
        keyboard_impl: str = "compat",
        enable_new_tracing: bool = False,
        trace_path: str = "pc-e500.perfetto-trace",
    ):
        self.instruction_count = 0
        self.memory_read_count = 0
        self.memory_write_count = 0
        self.save_lcd_on_exit = save_lcd_on_exit

        self.memory = PCE500Memory()
        self.memory._emulator = self  # Set reference for tracking counters
        self.lcd = HD61202Controller()
        self.memory.set_lcd_controller(self.lcd)

        # Select keyboard implementation.
        # Default is 'compat' to preserve CLI/tests behavior and performance.
        if keyboard_impl == "hardware":
            self.keyboard = KeyboardHardware(self.memory.read_byte)
        else:
            self.keyboard = KeyboardCompat()
        self.memory.add_overlay(
            MemoryOverlay(
                start=INTERNAL_MEMORY_START + KOL,
                end=INTERNAL_MEMORY_START + KIL,
                name="keyboard_io",
                read_only=False,
                read_handler=self._keyboard_read_handler,
                write_handler=self._keyboard_write_handler,
                perfetto_thread="I/O",
            )
        )
        
        # Add LCC register overlay for hardware keyboard (if applicable)
        if keyboard_impl == "hardware":
            self.memory.add_overlay(
                MemoryOverlay(
                    start=INTERNAL_MEMORY_START + 0xFE,  # LCC register
                    end=INTERNAL_MEMORY_START + 0xFE,
                    name="lcc_register",
                    read_only=False,
                    write_handler=self._lcc_write_handler,
                    perfetto_thread="I/O",
                )
            )

        self.cpu = SC62015Emulator(self.memory, reset_on_init=True)
        self.memory.set_cpu(self.cpu)
        
        # Set performance tracer for SC62015 integration if available
        if new_tracer.enabled:
            self.memory.set_perf_tracer(new_tracer)

        if any(o.name == "internal_rom" for o in self.memory.overlays):
            self.cpu.regs.set(RegisterName.PC, self.memory.read_long(0xFFFFD))

        self.breakpoints: Set[int] = set()
        self.cycle_count = 0
        self.start_time = time.time()
        self.trace_enabled = trace_enabled
        self.trace: Optional[list] = [] if trace_enabled else None
        self.perfetto_enabled = perfetto_trace
        if self.perfetto_enabled:
            g_tracer.start_tracing("pc-e500.trace")
            self.lcd.set_perfetto_enabled(True)
            self.memory.set_perfetto_enabled(True)

        # New tracing system
        self._new_trace_enabled = enable_new_tracing
        self._trace_path = trace_path
        self._trace_instr_count = 0
        if self._new_trace_enabled:
            new_tracer.start(self._trace_path)

        self.call_depth = 0
        self._interrupt_stack = []
        self._next_interrupt_id = 1
        self._current_pc = 0
        self._last_pc = 0
        self.instruction_history: deque = deque(maxlen=100)
        # Keyboard read monitoring
        self._kil_read_count = 0
        self._last_kil_columns = []
        self._last_kol = 0
        self._last_koh = 0

    @perf_trace("System")
    def load_rom(self, rom_data: bytes, start_address: Optional[int] = None) -> None:
        if start_address is None:
            start_address = self.INTERNAL_ROM_START
        if start_address in (self.INTERNAL_ROM_START, 0xC0000):
            self.memory.load_rom(rom_data)
        else:
            self.memory.add_rom(start_address, rom_data, "Loaded ROM")

    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        self.memory.load_memory_card(card_data, card_size)

    def expand_ram(self, size: int, start_address: int) -> None:
        self.memory.add_ram(start_address, size, f"RAM Expansion ({size // 1024}KB)")

    @perf_trace("System")
    def reset(self) -> None:
        self.memory.reset()
        self.lcd.reset()
        self.cpu.power_on_reset()
        if any(o.name == "internal_rom" for o in self.memory.overlays):
            self.cpu.regs.set(RegisterName.PC, self.memory.read_long(0xFFFFD))
        self.cycle_count = 0
        self.start_time = time.time()
        if self.trace is not None:
            self.trace.clear()
        self.instruction_history.clear()
        self.memory.clear_imem_access_tracking()

    @perf_trace("Emulation", include_op_num=True)
    def step(self) -> bool:
        pc = self.cpu.regs.get(RegisterName.PC)
        self._last_pc, self._current_pc = self._current_pc, pc

        if pc in self.breakpoints:
            return False

        if self.trace is not None:
            self.trace.append(("exec", pc, self.cycle_count))

        if pc == self.MEMORY_DUMP_PC and self.perfetto_enabled:
            self._dump_internal_memory(pc)

        try:
            # Decode instruction first to get opcode name for tracing
            instr = self.cpu.decode_instruction(pc)
            opcode = (
                self.memory.read_byte(pc)
                if (self.perfetto_enabled or self._new_trace_enabled)
                else None
            )
            
            # Build opcode name for performance tracing
            opcode_name = instr.name()
            
            # Execute instruction with opcode-level tracing
            pc_before = pc
            with new_tracer.slice("Opcodes", opcode_name, {"pc": f"0x{pc:06X}", "opcode": f"0x{opcode:02X}" if opcode else None, "op_num": self.instruction_count}):
                eval_info = self.cpu.execute_instruction(pc)
            
            self.cycle_count += 1
            self.instruction_count += 1

            # Only compute disassembly when tracing is enabled to avoid overhead
            if self.trace is not None:
                from binja_test_mocks.tokens import asm_str

                self.instruction_history.append(
                    {
                        "pc": f"0x{pc:06X}",
                        "disassembly": asm_str(eval_info.instruction.render()),
                    }
                )

            if self.perfetto_enabled:
                self._trace_execution(pc, opcode)
                self._trace_control_flow(pc_before, eval_info)
                self._update_perfetto_counters()

            # New tracing system
            if new_tracer.enabled:
                self._trace_instr_count += 1
                pc, opcode = _trace_probe_pc_and_opcode(self)

                # Instruction instant event
                name = f"Exec@{pc:#06x}" if isinstance(pc, int) else "Exec@?"
                args = {}
                if isinstance(pc, int):
                    args["pc"] = pc
                if isinstance(opcode, int):
                    args["opcode"] = opcode

                # Add register values
                state = self.get_cpu_state()
                args["a"] = state["a"] & 0xFF
                args["b"] = state["b"] & 0xFF
                args["x"] = state["x"] & 0xFFFFFF
                args["y"] = state["y"] & 0xFFFFFF
                args["sp"] = state["s"] & 0xFFFFFF
                args["flags"] = (state["flags"]["c"] << 1) | state["flags"]["z"]

                new_tracer.instant("Execution", name, args)

                # Update counters
                new_tracer.counter(
                    "Counters", "instructions", float(self._trace_instr_count)
                )
        except Exception as e:
            if self.trace is not None:
                self.trace.append(("error", pc, str(e)))
            if self.perfetto_enabled:
                g_tracer.trace_instant(
                    "CPU", "Error", {"error": str(e), "pc": f"0x{pc:06X}"}
                )
            raise
        return True

    def run(self, max_instructions: Optional[int] = None) -> int:
        count = 0
        while max_instructions is None or count < max_instructions:
            if not self.step():
                break
            count += 1
        return count

    def add_breakpoint(self, address: int) -> None:
        self.breakpoints.add(address & 0xFFFFFF)

    def set_memory_dump_pc(self, address: int) -> None:
        self.MEMORY_DUMP_PC = address & 0xFFFFFF
        print(f"Internal memory dump will trigger at PC=0x{self.MEMORY_DUMP_PC:06X}")

    def remove_breakpoint(self, address: int) -> None:
        self.breakpoints.discard(address & 0xFFFFFF)

    def get_cpu_state(self) -> Dict[str, Any]:
        return {
            "pc": self.cpu.regs.get(RegisterName.PC),
            "a": self.cpu.regs.get(RegisterName.A),
            "b": self.cpu.regs.get(RegisterName.B),
            "ba": self.cpu.regs.get(RegisterName.BA),
            "i": self.cpu.regs.get(RegisterName.I),
            "x": self.cpu.regs.get(RegisterName.X),
            "y": self.cpu.regs.get(RegisterName.Y),
            "u": self.cpu.regs.get(RegisterName.U),
            "s": self.cpu.regs.get(RegisterName.S),
            "flags": {
                "z": self.cpu.regs.get(RegisterName.FZ),
                "c": self.cpu.regs.get(RegisterName.FC),
            },
            "cycles": self.cycle_count,
        }

    def get_performance_stats(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        ips = self.cycle_count / elapsed if elapsed > 0 else 0
        return {
            "instructions_executed": self.cycle_count,
            "elapsed_time": elapsed,
            "instructions_per_second": ips,
            "speed_ratio": ips / 2_000_000,
        }

    def get_memory_info(self) -> str:
        return self.memory.get_memory_info()

    def get_display_buffer(self):
        return self.lcd.get_display_buffer()

    def save_lcd_displays(
        self, combined_filename: str = "lcd_display.png", save_individual: bool = False
    ) -> None:
        self.lcd.get_combined_display(zoom=2).save(combined_filename)
        print(f"LCD display saved to {combined_filename}")
        if save_individual:
            self.lcd.save_displays_to_png("lcd_left.png", "lcd_right.png")
            print("Individual chip displays saved to lcd_left.png and lcd_right.png")

    def stop_tracing(self) -> None:
        if self.perfetto_enabled:
            print("Stopping Perfetto tracing...")
            g_tracer.stop_tracing()
        if self._new_trace_enabled and new_tracer.enabled:
            print(f"Stopping new tracing, saved to {self._trace_path}")
            new_tracer.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "save_lcd_on_exit") and self.save_lcd_on_exit:
            self.save_lcd_displays(save_individual=True)
        self.stop_tracing()
        return False

    def _add_register_annotations(self, event: Any):
        if not event:
            return
        state = self.get_cpu_state()
        event.add_annotations(
            {
                "reg_A": f"0x{state['a']:02X}",
                "reg_B": f"0x{state['b']:02X}",
                "reg_BA": f"0x{state['ba']:04X}",
                "reg_I": f"0x{state['i']:04X}",
                "reg_X": f"0x{state['x']:06X}",
                "reg_Y": f"0x{state['y']:06X}",
                "reg_U": f"0x{state['u']:06X}",
                "reg_S": f"0x{state['s']:06X}",
                "reg_PC": f"0x{state['pc']:06X}",
                "flag_C": state["flags"]["c"],
                "flag_Z": state["flags"]["z"],
            }
        )

    def _trace_execution(self, pc: int, opcode: Optional[int]):
        event = g_tracer.trace_instant(
            "Execution",
            f"Exec@0x{pc:06X}",
            {"pc": f"0x{pc:06X}", "opcode": f"0x{opcode:02X}"},
        )
        self._add_register_annotations(event)

    def _update_perfetto_counters(self):
        g_tracer.trace_counter("CPU", "cycles", self.cycle_count)
        g_tracer.trace_counter("CPU", "call_depth", self.call_depth)
        g_tracer.trace_counter("CPU", "instructions", self.instruction_count)
        g_tracer.trace_counter(
            "CPU", "stack_pointer", self.cpu.regs.get(RegisterName.S)
        )
        g_tracer.trace_counter("Memory", "read_ops", self.memory_read_count)
        g_tracer.trace_counter("Memory", "write_ops", self.memory_write_count)

    def _trace_control_flow(self, pc_before: int, eval_info):
        instr = eval_info.instruction
        pc_after = self.cpu.regs.get(RegisterName.PC)

        if isinstance(instr, CALL):
            dest_addr = instr.dest_addr(pc_before)
            event = g_tracer.begin_function(
                "CPU", dest_addr, pc_before, f"func_0x{dest_addr:05X}"
            )
            self.call_depth += 1
            self._add_register_annotations(event)
            g_tracer.trace_instant(
                "CPU",
                "call",
                {
                    "from": f"0x{pc_before:06X}",
                    "to": f"0x{dest_addr:05X}",
                    "depth": self.call_depth,
                },
            )

        elif isinstance(instr, RetInstruction):
            g_tracer.end_function("CPU", pc_before)
            self.call_depth = max(0, self.call_depth - 1)
            instr_name = type(instr).__name__
            if instr_name == "RETI" and self._interrupt_stack:
                flow_id = self._interrupt_stack.pop()
                g_tracer.end_flow("CPU", flow_id, f"RETI@0x{pc_before:06X}")
            event = g_tracer.trace_instant(
                "CPU",
                "return",
                {
                    "at": f"0x{pc_before:06X}",
                    "type": instr_name.lower(),
                    "depth": self.call_depth,
                },
            )
            self._add_register_annotations(event)

        elif isinstance(instr, IR):
            vector_addr = self.memory.read_long(0xFFFFA)
            interrupt_id = self._next_interrupt_id
            self._next_interrupt_id += 1
            self._interrupt_stack.append(interrupt_id)
            g_tracer.begin_flow("CPU", interrupt_id, f"IR@0x{pc_before:06X}")
            event = g_tracer.begin_function(
                "CPU", vector_addr, pc_before, f"int_0x{vector_addr:05X}"
            )
            self.call_depth += 1
            self._add_register_annotations(event)
            g_tracer.trace_instant(
                "CPU",
                "interrupt",
                {
                    "from": f"0x{pc_before:06X}",
                    "vector": f"0x{vector_addr:05X}",
                    "interrupt_id": interrupt_id,
                },
            )

        elif isinstance(instr, JumpInstruction):
            expected_pc = (pc_before + eval_info.instruction_info.length) & 0xFFFFFF
            if pc_after != expected_pc:
                condition = getattr(instr, "_cond", None)
                trace_data = {
                    "from": f"0x{pc_before:06X}",
                    "to": f"0x{pc_after:06X}",
                    "type": "unconditional" if not condition else "conditional_taken",
                }
                if condition:
                    trace_data["condition"] = condition
                g_tracer.trace_instant("CPU", "jump", trace_data)

    def press_key(self, key_code: str) -> bool:
        result = self.keyboard.press_key(key_code) if self.keyboard else False
        if new_tracer.enabled:
            new_tracer.instant("I/O", "KeyPress", {"key": key_code})
        return result

    def release_key(self, key_code: str):
        if self.keyboard:
            self.keyboard.release_key(key_code)
        if new_tracer.enabled:
            new_tracer.instant("I/O", "KeyRelease", {"key": key_code})

    def _track_imem_access(self, offset: int, access_type: str, cpu_pc: Optional[int]):
        if cpu_pc is None or not self.perfetto_enabled:
            return
        reg_name = {0xF0: "KOL", 0xF1: "KOH", 0xF2: "KIL"}.get(offset)
        if reg_name:
            tracking = self.memory.imem_access_tracking.setdefault(
                reg_name, {"reads": [], "writes": []}
            )
            access_list = tracking[access_type]
            if access_list and access_list[-1][0] == cpu_pc:
                access_list[-1] = (cpu_pc, access_list[-1][1] + 1)
            else:
                access_list.append((cpu_pc, 1))
                if len(access_list) > 10:
                    access_list.pop(0)

    def _keyboard_read_handler(self, address: int, cpu_pc: Optional[int] = None) -> int:
        offset = address - INTERNAL_MEMORY_START
        self._track_imem_access(offset, "reads", cpu_pc)
        # Support both compat and hardware keyboard APIs
        if hasattr(self.keyboard, "read_register"):
            result = self.keyboard.read_register(offset)
        else:
            result = self.keyboard.handle_register_read(offset)

        # Trace keyboard matrix I/O
        if new_tracer.enabled and offset == KIL:
            new_tracer.instant(
                "I/O", "KB_InputRead", {"addr": offset, "value": result & 0xFF}
            )

        # Monitor KIL reads and active columns (for test harness automation)
        if offset == KIL:
            self._kil_read_count += 1
            # Capture active columns if available
            cols = []
            try:
                if hasattr(self.keyboard, "get_active_columns"):
                    cols = list(self.keyboard.get_active_columns())
            except Exception:
                cols = []
            self._last_kil_columns = cols
            # Capture last KOL/KOH values for debugging
            try:
                if hasattr(self.keyboard, "kol_value"):
                    self._last_kol = int(self.keyboard.kol_value) & 0xFF
                if hasattr(self.keyboard, "koh_value"):
                    self._last_koh = int(self.keyboard.koh_value) & 0xFF
            except Exception:
                pass

        return result

    def _keyboard_write_handler(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        offset = address - INTERNAL_MEMORY_START
        self._track_imem_access(offset, "writes", cpu_pc)

        # Trace keyboard matrix I/O
        if new_tracer.enabled and offset in (KOL, KOH):
            new_tracer.instant(
                "I/O", "KB_ColumnStrobe", {"addr": offset, "value": value & 0xFF}
            )
        # Support both compat and hardware keyboard APIs
        if hasattr(self.keyboard, "write_register"):
            self.keyboard.write_register(offset, value)
        else:
            self.keyboard.handle_register_write(offset, value)
    
    def _lcc_write_handler(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        """Handle LCC register write for hardware keyboard."""
        offset = address - INTERNAL_MEMORY_START
        self._track_imem_access(offset, "writes", cpu_pc)
        
        # Invalidate KSD cache in hardware keyboard
        if hasattr(self.keyboard, "invalidate_ksd_cache"):
            self.keyboard.invalidate_ksd_cache()

    def _dump_internal_memory(self, pc: int):
        internal_mem = self.memory.get_internal_memory_bytes()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"internal_memory_dump_{timestamp}_pc_{pc:06X}.bin"
        path = os.path.join(self.MEMORY_DUMP_DIR, filename)
        with open(path, "wb") as f:
            f.write(internal_mem)
        print(f"\nInternal memory dumped to: {path}")
        g_tracer.trace_instant(
            "Debug",
            "InternalMemoryDump",
            {
                "pc": f"0x{pc:06X}",
                "filename": filename,
                "size": str(len(internal_mem)),
                "trigger": f"PC match 0x{self.MEMORY_DUMP_PC:06X}",
            },
        )
