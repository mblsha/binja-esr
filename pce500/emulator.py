"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Set, List
from enum import Enum
from collections import deque
from datetime import datetime

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
from .trace_manager import g_tracer
from .tracing.perfetto_tracing import tracer as new_tracer, perf_trace


class IRQSource(Enum):
    # Enum values store ISR bit index directly
    MTI = 0  # Main timer interrupt → ISR bit 0
    STI = 1  # Sub timer interrupt  → ISR bit 1
    KEY = 2  # Keyboard interrupt   → ISR bit 2


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
        # Single keyboard implementation (compat); parameter removed
        enable_new_tracing: bool = False,
        trace_path: str = "pc-e500.perfetto-trace",
        disasm_trace: bool = False,
    ):
        self.instruction_count = 0
        self.memory_read_count = 0
        self.memory_write_count = 0
        self.save_lcd_on_exit = save_lcd_on_exit

        # Disassembly trace data structures
        self.disasm_trace_enabled = disasm_trace
        self.executed_instructions: Dict[
            int, Dict[str, Any]
        ] = {}  # PC -> instruction info
        self.control_flow_edges: Dict[int, Set[int]] = {}  # dest_pc -> set(source_pcs)
        self.execution_order: list[int] = []  # PCs in execution order
        self.last_pc: Optional[int] = None
        self.register_accesses: Dict[
            int, List[Dict[str, Any]]
        ] = {}  # PC -> list of register accesses
        self.current_instruction_accesses: List[
            Dict[str, Any]
        ] = []  # Accumulate during instruction

        self.memory = PCE500Memory()
        self.memory._emulator = self  # Set reference for tracking counters

        # Set callback for IMEM register access tracking
        if self.disasm_trace_enabled:
            self.memory.set_imem_access_callback(self._on_imem_register_access)
        self.lcd = HD61202Controller()
        self.memory.set_lcd_controller(self.lcd)

        # Keyboard implementation: compat only (hardware impl removed)
        # Pass memory accessor so compat can honor KSD (LCC bit)
        self.keyboard = KeyboardCompat(self.memory.read_byte)
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

        # Note: LCC overlay not needed for compat keyboard

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

        # Interrupt accounting (counts since last reset)
        self.irq_counts: Dict[str, int] = {"total": 0, "KEY": 0, "MTI": 0, "STI": 0}
        self.last_irq: Dict[str, Any] = {"src": None, "pc": None, "vector": None}
        # Track IMR/ISR bit set/clear PCs (last few)
        self.irq_bit_watch: Dict[str, Dict[int, Dict[str, list[int]]]] = {
            "IMR": {i: {"set": [], "clear": []} for i in range(8)},
            "ISR": {i: {"set": [], "clear": []} for i in range(8)},
        }
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
        # Keyboard strobe monitoring (KOL/KOH writes)
        self._kb_strobe_count = 0
        self._kb_col_hist = [0 for _ in range(11)]
        # Synthetic keyboard interrupt wiring (enable for both compat and hardware)
        self._kb_irq_enabled = True
        self._irq_pending = False
        self._in_interrupt = False
        self._kb_irq_count = 0
        # Simple periodic timers (rough emulation)
        self._timer_enabled = True
        # Periods in instructions (tunable): main ~ msec, sub ~ 10x slower here
        self._timer_mti_period = 500  # MTI (bit 0)
        self._timer_sti_period = 5000  # STI (bit 1)
        self._timer_next_mti = self._timer_mti_period
        self._timer_next_sti = self._timer_sti_period
        self._irq_source: Optional["IRQSource"] = None
        # Fast mode: minimize step() overhead to run many instructions
        self.fast_mode = False

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
        # Optional debug: force display ON at reset
        try:
            if getattr(self, "force_display_on", False) and hasattr(self.lcd, "chips"):
                for chip in self.lcd.chips:
                    chip.state.on = True
        except Exception:
            pass
        self.cpu.power_on_reset()
        if any(o.name == "internal_rom" for o in self.memory.overlays):
            self.cpu.regs.set(RegisterName.PC, self.memory.read_long(0xFFFFD))
        self.cycle_count = 0
        self.start_time = time.time()
        if self.trace is not None:
            self.trace.clear()
        self.instruction_history.clear()
        self.memory.clear_imem_access_tracking()
        # Reset interrupt accounting
        try:
            self.irq_counts.update({"total": 0, "KEY": 0, "MTI": 0, "STI": 0})
            self.last_irq.update({"src": None, "pc": None, "vector": None})
            self.irq_bit_watch = {
                "IMR": {i: {"set": [], "clear": []} for i in range(8)},
                "ISR": {i: {"set": [], "clear": []} for i in range(8)},
            }
        except Exception:
            pass

    @perf_trace("Emulation", include_op_num=True)
    def step(self) -> bool:
        # Tick rough timers to set ISR bits and arm IRQ when due
        try:
            if self._timer_enabled:
                self._tick_timers()
        except Exception:
            pass
        # Check for pending synthetic interrupt before executing next instruction
        if getattr(self, "_irq_pending", False) and not getattr(
            self, "_in_interrupt", False
        ):
            try:
                # Respect IMR/ISR masks: deliver only if IRM=1 and (IMR & ISR)!=0
                imr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.IMR
                isr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.ISR
                imr_val_chk = self.memory.read_byte(imr_addr_chk) & 0xFF
                isr_val_chk = self.memory.read_byte(isr_addr_chk) & 0xFF
                if (imr_val_chk & 0x80) == 0 or (imr_val_chk & isr_val_chk) == 0:
                    # Keep pending; CPU continues executing normal flow
                    pass
                else:
                    # Push PC (3 bytes), then F (1), then IMR (1), clear IMR.IRM
                    cur_pc = self.cpu.regs.get(RegisterName.PC)
                    s = self.cpu.regs.get(RegisterName.S)
                    # Require a valid, initialized stack pointer; defer IRQ until firmware sets SP
                    if not isinstance(s, int) or s < 5:
                        raise RuntimeError(
                            "IRQ deferred: stack pointer not initialized"
                        )
                    # push PC (little-endian 3 bytes)
                    s_new = s - 3
                    self.memory.write_bytes(3, s_new, cur_pc)
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push F (1 byte)
                    f_val = self.cpu.regs.get(RegisterName.F)
                    s_new = self.cpu.regs.get(RegisterName.S) - 1
                    self.memory.write_bytes(1, s_new, f_val)
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push IMR (1 byte) and clear IRM bit 7
                    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
                    imr_val = self.memory.read_byte(imr_addr)
                    s_new = self.cpu.regs.get(RegisterName.S) - 1
                    self.memory.write_bytes(1, s_new, imr_val)
                    self.cpu.regs.set(RegisterName.S, s_new)
                    self.memory.write_byte(imr_addr, imr_val & 0x7F)
                    # ISR status was set by the triggering source (device/timer)
                    # Do not modify ISR here; only deliver the interrupt.
                    # Jump to interrupt vector (0xFFFFA little-endian 3 bytes)
                    vector_addr = self.memory.read_long(0xFFFFA)
                    self.cpu.regs.set(RegisterName.PC, vector_addr)
                    self._in_interrupt = True
                    self._irq_pending = False
                    # Interrupt accounting
                    try:
                        src = (
                            self._irq_source.name
                            if isinstance(self._irq_source, IRQSource)
                            else "KEY"
                        )
                        # Increment counts at delivery time
                        self.irq_counts["total"] = (
                            int(self.irq_counts.get("total", 0)) + 1
                        )
                        if src in ("KEY", "MTI", "STI"):
                            self.irq_counts[src] = int(self.irq_counts.get(src, 0)) + 1
                        self.last_irq = {
                            "src": src,
                            "pc": cur_pc,
                            "vector": vector_addr,
                        }
                    except Exception:
                        pass
                    # Debug/trace: note interrupt delivery
                    try:
                        assert isinstance(self._irq_source, IRQSource)
                        if self.trace is not None:
                            self.trace.append(
                                (
                                    "irq",
                                    cur_pc,
                                    vector_addr,
                                    self._irq_source.name,
                                )
                            )
                        if new_tracer.enabled:
                            new_tracer.instant(
                                "CPU",
                                "IRQ_Delivered",
                                {
                                    "from": cur_pc,
                                    "to": vector_addr,
                                    "src": self._irq_source.name,
                                },
                            )
                        self._kb_irq_count += 1
                    except Exception:
                        pass
            except Exception:
                pass

        pc = self.cpu.regs.get(RegisterName.PC)
        self._last_pc, self._current_pc = self._current_pc, pc

        if pc in self.breakpoints:
            return False

        if self.trace is not None:
            self.trace.append(("exec", pc, self.cycle_count))

        if pc == self.MEMORY_DUMP_PC and self.perfetto_enabled:
            self._dump_internal_memory(pc)

        try:
            # Pre-read opcode and I for WAIT simulation (so we can model time passing)
            # Only simulate when tracing is enabled to keep performance tests predictable
            wait_sim_count = 0
            if self.perfetto_enabled or getattr(self, "_new_trace_enabled", False):
                try:
                    opcode_peek = self.memory.read_byte(pc) & 0xFF
                    if opcode_peek == 0xEF:  # WAIT
                        i_before = self.cpu.regs.get(RegisterName.I) & 0xFFFF
                        if i_before > 0:
                            wait_sim_count = i_before
                except Exception:
                    pass

            if getattr(self, "fast_mode", False):
                # Minimal execution path for speed
                pc_before = pc

                # Clear current instruction accesses before execution
                if self.disasm_trace_enabled:
                    self.current_instruction_accesses = []

                eval_info = self.cpu.execute_instruction(pc)

                # Associate accumulated register accesses with this instruction
                if self.disasm_trace_enabled and self.current_instruction_accesses:
                    if pc_before not in self.register_accesses:
                        self.register_accesses[pc_before] = []
                    self.register_accesses[pc_before].extend(
                        self.current_instruction_accesses
                    )

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    for _ in range(int(wait_sim_count)):
                        # Advance instruction counter and tick timers
                        self.instruction_count += 1
                        if self._timer_enabled:
                            try:
                                self._tick_timers()
                            except Exception:
                                pass
                if self.perfetto_enabled:
                    # In fast mode, keep lightweight counters only
                    self._update_perfetto_counters()
                # Emit lightweight Exec@ events for new tracer even in fast mode
                if new_tracer.enabled:
                    pc2, opcode2 = _trace_probe_pc_and_opcode(self)
                    name = f"Exec@{pc2:#06x}" if isinstance(pc2, int) else "Exec@?"
                    args = {}
                    if isinstance(pc2, int):
                        args["pc"] = pc2
                    if isinstance(opcode2, int):
                        args["opcode"] = opcode2
                    # Minimal register state to keep overhead low
                    try:
                        state = self.get_cpu_state()
                        args["sp"] = state["s"] & 0xFFFFFF
                        args["flags"] = (state["flags"]["c"] << 1) | state["flags"]["z"]
                    except Exception:
                        pass
                    new_tracer.instant("Execution", name, args)
                    new_tracer.counter(
                        "Counters", "instructions", float(self.instruction_count)
                    )
            else:
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

                # Clear current instruction accesses before execution
                if self.disasm_trace_enabled:
                    self.current_instruction_accesses = []

                with new_tracer.slice(
                    "Opcodes",
                    opcode_name,
                    {
                        "pc": f"0x{pc:06X}",
                        "opcode": f"0x{opcode:02X}" if opcode else None,
                        "op_num": self.instruction_count,
                    },
                ):
                    eval_info = self.cpu.execute_instruction(pc)

                # Associate accumulated register accesses with this instruction
                if self.disasm_trace_enabled and self.current_instruction_accesses:
                    if pc_before not in self.register_accesses:
                        self.register_accesses[pc_before] = []
                    self.register_accesses[pc_before].extend(
                        self.current_instruction_accesses
                    )

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    for _ in range(int(wait_sim_count)):
                        # Advance instruction counter and tick timers
                        self.instruction_count += 1
                        if self._timer_enabled:
                            try:
                                self._tick_timers()
                            except Exception:
                                pass

                # Only compute disassembly when tracing is enabled to avoid overhead
                if self.trace is not None:
                    from binja_test_mocks.tokens import asm_str

                    self.instruction_history.append(
                        {
                            "pc": f"0x{pc:06X}",
                            "disassembly": asm_str(eval_info.instruction.render()),
                        }
                    )

                # Capture disassembly trace if enabled
                if self.disasm_trace_enabled:
                    self._capture_disasm_trace(pc_before, eval_info.instruction, instr)

                if self.perfetto_enabled:
                    self._trace_execution(pc, opcode)
                    self._trace_control_flow(pc_before, eval_info)
                    self._update_perfetto_counters()

                # Track control flow edges for disassembly trace
                if self.disasm_trace_enabled:
                    pc_after = self.cpu.regs.get(RegisterName.PC)
                    # Detect non-sequential control flow
                    expected_next = pc_before + eval_info.instruction.length()
                    if pc_after != expected_next:
                        # This was a taken branch/jump/call/return
                        if pc_after not in self.control_flow_edges:
                            self.control_flow_edges[pc_after] = set()
                        self.control_flow_edges[pc_after].add(pc_before)

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
        # Detect end of interrupt roughly by RETI opcode name
        try:
            instr_name = type(eval_info.instruction).__name__
            if instr_name == "RETI":
                self._in_interrupt = False
                # After returning from interrupt, clear IRQ source marker
                self._irq_source = None
        except Exception:
            pass
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
        img = self.lcd.get_combined_display(zoom=1)
        img.save(combined_filename)
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
        # Arm an interrupt eagerly in hardware mode to ensure delivery
        try:
            if self._kb_irq_enabled and result:
                # Arm pending and record source
                setattr(self, "_irq_pending", True)
                self._irq_source = IRQSource.KEY
                # Ensure ISR.KEY asserts for delivery if not already set, but avoid duplicates
                try:
                    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
                    isr_val = self.memory.read_byte(isr_addr) & 0xFF
                    if (isr_val & (1 << IRQSource.KEY.value)) == 0:
                        self._set_isr_bits(1 << IRQSource.KEY.value)
                except Exception:
                    pass
        except Exception:
            pass
        # Optional debug: make a visible mark on LCD to confirm key handling
        try:
            if result and getattr(self, "debug_draw_on_key", False):
                # Turn displays on and draw a simple pattern
                for chip in getattr(self.lcd, "chips", []):
                    chip.state.on = True
                # Draw a small block in the top-left corner
                # Left chip page 0, columns 0..7 with alternating bits
                if getattr(self.lcd, "chips", None):
                    for col in range(8):
                        # Write zeros to render as dark pixels on a white background
                        self.lcd.chips[0].vram[0][col] = (
                            0x00 if (col % 2 == 0) else 0x18
                        )
        except Exception:
            pass
        if new_tracer.enabled:
            new_tracer.instant("I/O", "KeyPress", {"key": key_code})
        return result

    def _set_isr_bits(self, mask: int) -> None:
        """OR mask into ISR register."""
        isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
        val = self.memory.read_byte(isr_addr)
        self.memory.write_byte(isr_addr, (val | (mask & 0xFF)) & 0xFF)

    def _tick_timers(self) -> None:
        """Rough timer emulation: set ISR bits periodically and arm IRQ."""
        ic = self.instruction_count
        fired = False
        # Main timer (MTI, bit 0)
        if ic >= self._timer_next_mti:
            self._set_isr_bits(1 << IRQSource.MTI.value)
            self._timer_next_mti = ic + self._timer_mti_period
            self._irq_pending = True
            self._irq_source = IRQSource.MTI
            fired = True
        # Sub timer (STI, bit 1) – less frequent
        if ic >= self._timer_next_sti:
            self._set_isr_bits(1 << IRQSource.STI.value)
            self._timer_next_sti = ic + self._timer_sti_period
            self._irq_pending = True
            self._irq_source = IRQSource.STI
            fired = True
        if fired and new_tracer.enabled:
            assert isinstance(self._irq_source, IRQSource)
            new_tracer.instant(
                "CPU",
                "TimerIRQ",
                {"ic": ic, "src": self._irq_source.name},
            )

    def get_interrupt_stats(self) -> Dict[str, Any]:
        """Return interrupt counts and last delivery info.

        Structure:
            {
              "total": int,
              "by_source": {"KEY": int, "MTI": int, "STI": int},
              "last": {"src": str|None, "pc": int|None, "vector": int|None},
            }
        """
        try:
            by_source = {
                "KEY": int(self.irq_counts.get("KEY", 0)),
                "MTI": int(self.irq_counts.get("MTI", 0)),
                "STI": int(self.irq_counts.get("STI", 0)),
            }
            return {
                "total": int(self.irq_counts.get("total", 0)),
                "by_source": by_source,
                "last": {
                    "src": self.last_irq.get("src"),
                    "pc": self.last_irq.get("pc"),
                    "vector": self.last_irq.get("vector"),
                },
                # Provide recent set/clear PCs for key bits of interest; full data remains on emulator
                "watch": {
                    "IMR": {
                        7: self.irq_bit_watch.get("IMR", {}).get(
                            7, {"set": [], "clear": []}
                        ),
                        2: self.irq_bit_watch.get("IMR", {}).get(
                            2, {"set": [], "clear": []}
                        ),
                    },
                    "ISR": {
                        2: self.irq_bit_watch.get("ISR", {}).get(
                            2, {"set": [], "clear": []}
                        ),
                    },
                },
            }
        except Exception:
            return {
                "total": 0,
                "by_source": {"KEY": 0, "MTI": 0, "STI": 0},
                "last": {"src": None, "pc": None, "vector": None},
            }

    def _record_irq_bit_watch(
        self, reg_name: str, prev_val: int, new_val: int, pc: int
    ) -> None:
        try:
            table = self.irq_bit_watch.get(reg_name)
            if not table:
                return
            for bit in range(8):
                prev_b = (prev_val >> bit) & 1
                new_b = (new_val >> bit) & 1
                if prev_b == new_b:
                    continue
                action = "set" if new_b == 1 else "clear"
                lst = table[bit][action]
                if lst and lst[-1] == pc:
                    # coalesce consecutive entries from same PC
                    continue
                lst.append(pc)
                if len(lst) > 10:
                    lst.pop(0)
        except Exception:
            pass

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
        # Honor KSD (keyboard strobe disable) bit: when set, firmware expects KIL=0x00
        if offset == KIL:
            try:
                lcc_addr = INTERNAL_MEMORY_START + IMEMRegisters.LCC
                lcc_val = self.memory.read_byte(lcc_addr)
                if (lcc_val & 0x04) != 0:
                    result = 0x00
                    # Trace keyboard matrix I/O
                    if new_tracer.enabled:
                        new_tracer.instant(
                            "I/O", "KB_InputRead", {"addr": offset, "value": result}
                        )
                    self._kil_read_count += 1
                    self._last_kil_columns = []
                    return result
            except Exception:
                pass
        # Single keyboard implementation: use compat handler
        result = self.keyboard.handle_register_read(offset)

        # Trace keyboard matrix I/O
        if new_tracer.enabled and offset == KIL:
            new_tracer.instant(
                "I/O", "KB_InputRead", {"addr": offset, "value": result & 0xFF}
            )

        # (Keyboard interrupt status is handled via explicit key events and
        # strobe detection. We do not mutate ISR here to avoid spurious
        # interrupts during early firmware boot scans.)

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
        # Monitor strobe count on changes and track active columns
        try:
            if offset == KOL:
                if getattr(self.keyboard, "kol_value", None) != (value & 0xFF):
                    self._kb_strobe_count += 1
                self._last_kol = value & 0xFF
            elif offset == KOH:
                if getattr(self.keyboard, "koh_value", None) != (value & 0xFF):
                    self._kb_strobe_count += 1
                self._last_koh = value & 0xFF

            # Update histogram using keyboard's active column calculation
            active_cols = []
            if hasattr(self.keyboard, "get_active_columns"):
                active_cols = list(self.keyboard.get_active_columns())
            for col in active_cols:
                if 0 <= col < len(self._kb_col_hist):
                    self._kb_col_hist[col] += 1
        except Exception:
            pass
        # If IRQ wiring enabled: set pending IRQ when a pressed key's column is active
        try:
            if self._kb_irq_enabled and hasattr(self.keyboard, "get_pressed_keys"):
                pressed = set(self.keyboard.get_pressed_keys())
                if pressed:
                    active_cols = []
                    if hasattr(self.keyboard, "get_active_columns"):
                        active_cols = list(self.keyboard.get_active_columns())
                    # Any pressed key in active columns?
                    for kc in pressed:
                        loc = getattr(self.keyboard, "key_locations", {}).get(kc)
                        if loc and loc.column in active_cols:
                            # Arm pending interrupt (delivered at next step)
                            self._set_isr_bits(1 << IRQSource.KEY.value)
                            setattr(self, "_irq_pending", True)
                            self._irq_source = IRQSource.KEY
                            break
        except Exception:
            pass
        # Single keyboard implementation: use compat handler
        self.keyboard.handle_register_write(offset, value)

    # Note: LCC write handler not required with single compat keyboard

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

    def _capture_disasm_trace(
        self, pc: int, instruction: Any, decoded_instr: Any
    ) -> None:
        """Capture disassembly trace information for an executed instruction."""
        if pc not in self.executed_instructions:
            # Read instruction bytes
            instr_length = instruction.length()
            instr_bytes = bytearray()
            for i in range(instr_length):
                instr_bytes.append(self.memory.read_byte(pc + i))

            # Get disassembly text
            from binja_test_mocks.tokens import asm_str

            disasm_text = asm_str(instruction.render())

            # Determine instruction type
            instr_type = "normal"
            if isinstance(instruction, JumpInstruction):
                instr_type = "jump"
            elif isinstance(instruction, CALL):
                instr_type = "call"
            elif isinstance(instruction, RetInstruction):
                instr_type = "return"
            elif isinstance(instruction, IR):
                instr_type = "interrupt"

            self.executed_instructions[pc] = {
                "bytes": bytes(instr_bytes),
                "disasm": disasm_text,
                "type": instr_type,
                "length": instr_length,
            }

        # Track execution order
        self.execution_order.append(pc)
        self.last_pc = pc

    def _on_imem_register_access(
        self, pc: int, reg_name: str, access_type: str, value: int
    ) -> None:
        """Callback for internal memory register accesses.

        Args:
            pc: Program counter where access occurred
            reg_name: Name of the register (e.g., 'KOL', 'ISR')
            access_type: 'read' or 'write'
            value: Value read or written
        """
        # Skip BP, PX, PY as they're too frequent
        if reg_name in ("BP", "PX", "PY"):
            return

        # Add to current instruction's accesses
        self.current_instruction_accesses.append(
            {"register": reg_name, "type": access_type, "value": value}
        )

    def save_disasm_trace(self, output_dir: str = "data") -> str:
        """Generate and save the disassembly trace to a file."""
        if not self.disasm_trace_enabled:
            return ""

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_trace_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        # Build reverse edge map for source annotations
        source_edges: Dict[int, Set[int]] = {}  # source_pc -> set(dest_pcs)
        for dest, sources in self.control_flow_edges.items():
            for source in sources:
                if source not in source_edges:
                    source_edges[source] = set()
                source_edges[source].add(dest)

        # Generate output
        with open(filepath, "w") as f:
            # Write header
            f.write("; PC-E500 Execution Trace\n")
            f.write(f"; Instructions executed: {len(self.execution_order)}\n")
            f.write(f"; Unique PCs: {len(self.executed_instructions)}\n")
            f.write(
                f"; Control flow edges: {sum(len(s) for s in self.control_flow_edges.values())}\n"
            )
            f.write("\n")

            # Get all executed PCs in address order
            sorted_pcs = sorted(self.executed_instructions.keys())

            # Group instructions into basic blocks
            blocks = []
            current_block = []
            for pc in sorted_pcs:
                if current_block:
                    last_pc = current_block[-1]
                    last_info = self.executed_instructions[last_pc]
                    expected_next = last_pc + last_info["length"]
                    # Check if this PC is the expected sequential next instruction
                    if pc != expected_next:
                        # End current block and start new one
                        blocks.append(current_block)
                        current_block = [pc]
                    else:
                        current_block.append(pc)
                else:
                    current_block.append(pc)
            if current_block:
                blocks.append(current_block)

            # Write each block
            for block_idx, block in enumerate(blocks):
                if block_idx > 0:
                    f.write("\n")  # Separator between blocks

                for pc in block:
                    info = self.executed_instructions[pc]

                    # Format instruction bytes
                    bytes_str = " ".join(f"{b:02X}" for b in info["bytes"])
                    bytes_str = bytes_str.ljust(12)  # Align to 12 chars (4 bytes max)

                    # Format base line
                    line = f"0x{pc:06X}: {bytes_str} {info['disasm']}"

                    # Add annotations
                    annotations = []

                    # Annotate control flow sources (where this instruction jumps/calls to)
                    if pc in source_edges:
                        dests = sorted(source_edges[pc])
                        if info["type"] == "call":
                            annotations.append(
                                f"Calls: {', '.join(f'0x{d:06X}' for d in dests)}"
                            )
                        elif info["type"] == "jump":
                            annotations.append(
                                f"Jumps to: {', '.join(f'0x{d:06X}' for d in dests)}"
                            )
                        elif info["type"] == "return":
                            annotations.append(
                                f"Returns to: {', '.join(f'0x{d:06X}' for d in dests)}"
                            )

                    # Annotate control flow destinations (where jumps to this instruction)
                    if pc in self.control_flow_edges:
                        sources = sorted(self.control_flow_edges[pc])
                        annotations.append(
                            f"From: {', '.join(f'0x{s:06X}' for s in sources)}"
                        )

                    # Annotate register accesses
                    if pc in self.register_accesses:
                        reads = {}
                        writes = {}
                        for access in self.register_accesses[pc]:
                            if access["type"] == "read":
                                # Store only unique values per register
                                if access["register"] not in reads:
                                    reads[access["register"]] = set()
                                reads[access["register"]].add(access["value"])
                            else:  # write
                                # Store only unique values per register
                                if access["register"] not in writes:
                                    writes[access["register"]] = set()
                                writes[access["register"]].add(access["value"])

                        # Format unique reads
                        if reads:
                            read_strs = []
                            for reg, values in sorted(reads.items()):
                                if len(values) == 1:
                                    read_strs.append(f"{reg}=0x{list(values)[0]:02X}")
                                else:
                                    # Multiple unique values - show them all
                                    vals = ",".join(
                                        f"0x{v:02X}" for v in sorted(values)
                                    )
                                    read_strs.append(f"{reg}=[{vals}]")
                            annotations.append(f"Reads: {', '.join(read_strs)}")

                        # Format unique writes
                        if writes:
                            write_strs = []
                            for reg, values in sorted(writes.items()):
                                if len(values) == 1:
                                    write_strs.append(f"{reg}=0x{list(values)[0]:02X}")
                                else:
                                    # Multiple unique values - show them all
                                    vals = ",".join(
                                        f"0x{v:02X}" for v in sorted(values)
                                    )
                                    write_strs.append(f"{reg}=[{vals}]")
                            annotations.append(f"Writes: {', '.join(write_strs)}")

                    # Special annotation for entry point
                    if pc == 0x0F10C2:  # Common PC-E500 entry point
                        annotations.append("Entry point")

                    # Write line with annotations
                    if annotations:
                        line += "    ; " + "; ".join(annotations)
                    f.write(line + "\n")

        print(f"\nDisassembly trace saved to: {filepath}")
        return filepath
