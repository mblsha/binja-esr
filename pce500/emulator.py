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
from sc62015.pysc62015.constants import IMRFlag, ISRFlag

from .memory import (
    PCE500Memory,
    MemoryOverlay,
    IMEM_ACCESS_HISTORY_LIMIT,
)
from .display import HD61202Controller
from .keyboard_compat import PCE500KeyboardHandler as KeyboardCompat
from .trace_manager import g_tracer
from .tracing.perfetto_tracing import tracer as new_tracer, perf_trace

# Default timer periods in cycles (rough emulation)
MTI_PERIOD_CYCLES_DEFAULT = 500
STI_PERIOD_CYCLES_DEFAULT = 5000


class IRQSource(Enum):
    # Enum values store ISR bit index directly
    MTI = 0  # Main timer interrupt → ISR bit 0
    STI = 1  # Sub timer interrupt  → ISR bit 1
    KEY = 2  # Keyboard interrupt   → ISR bit 2
    ONK = 3  # On-key interrupt     → ISR bit 3


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
    _symbol_cache: Optional[Dict[int, str]] = None
    _function_cache: Optional[List[int]] = None

    def __init__(
        self,
        trace_enabled: bool = False,
        perfetto_trace: bool = False,
        save_lcd_on_exit: bool = True,
        keyboard_columns_active_high: bool = True,
        enable_new_tracing: bool = False,
        trace_path: str = "pc-e500.perfetto-trace",
        disasm_trace: bool = False,
        enable_display_trace: bool = False,
        display_trace_functions: Optional[Dict[int, str]] = None,
        display_trace_event_limit: int = 2048,
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

        # Display tracing hooks (optional)
        self.display_trace_enabled = bool(enable_display_trace)
        self._display_trace_watch: Dict[int, str] = (
            dict(display_trace_functions)
            if display_trace_functions is not None
            else self._default_display_trace_functions()
        )
        self._display_trace_symbols: Dict[int, str] = {}
        self._display_trace_stack: list[Dict[str, Any]] = []
        self.display_trace_log: list[Dict[str, Any]] = []
        self._display_trace_events: list[Dict[str, Any]] = []
        self._display_trace_event_limit = max(1, int(display_trace_event_limit))
        self._display_trace_function_index: List[int] = []
        self._display_trace_summary: Dict[int, Dict[str, Any]] = {}
        if self.display_trace_enabled:
            self._display_trace_symbols = self._load_symbol_map()
            for addr, name in self._display_trace_symbols.items():
                if addr in self._display_trace_watch:
                    self._display_trace_watch[addr] = name
            self._display_trace_function_index = self._load_function_addresses()
            self.lcd.set_write_trace_callback(self._on_lcd_trace_event)

        self._keyboard_columns_active_high = keyboard_columns_active_high

        # Keyboard implementation: compat handler parameterised for column polarity
        # Pass memory accessor so compat can honor KSD (LCC bit)
        self.keyboard = KeyboardCompat(
            self.memory.read_byte, columns_active_high=keyboard_columns_active_high
        )
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
        # Periods in cycles (tunable): main ~ msec, sub ~ 10x slower here
        self._timer_mti_period = MTI_PERIOD_CYCLES_DEFAULT  # MTI (bit 0)
        self._timer_sti_period = STI_PERIOD_CYCLES_DEFAULT  # STI (bit 1)
        self._timer_next_mti = self._timer_mti_period
        self._timer_next_sti = self._timer_sti_period
        self._irq_source: Optional["IRQSource"] = None
        # Fast mode: minimize step() overhead to run many instructions
        self.fast_mode = False

    def _reset_instruction_access_log(self) -> None:
        """Clear the per-instruction register access accumulator when tracing."""

        if self.disasm_trace_enabled:
            self.current_instruction_accesses = []

    def _flush_instruction_access_log(self, pc: int) -> None:
        """Persist register accesses captured while executing the current opcode."""

        if not self.disasm_trace_enabled or not self.current_instruction_accesses:
            return

        self.register_accesses.setdefault(pc, []).extend(
            self.current_instruction_accesses
        )

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
        # Reset pending interrupt and timer state to match power-on behaviour
        self._irq_pending = False
        self._in_interrupt = False
        self._irq_source = None
        self._interrupt_stack.clear()
        self._next_interrupt_id = 1
        self._timer_next_mti = self._timer_mti_period
        self._timer_next_sti = self._timer_sti_period
        self._kb_irq_count = 0
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
        # Honor HALT: do not execute instructions while halted. Any ISR bit cancels HALT.
        try:
            if getattr(self.cpu.state, "halted", False):
                isr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.ISR
                isr_val_chk = self.memory.read_byte(isr_addr_chk) & 0xFF
                if isr_val_chk != 0:
                    # Cancel HALT and arm a pending interrupt; infer a plausible source
                    self.cpu.state.halted = False
                    for b in (
                        IRQSource.MTI.value,
                        IRQSource.STI.value,
                        IRQSource.KEY.value,
                        IRQSource.ONK.value,
                    ):
                        if isr_val_chk & (1 << b):
                            self._irq_source = IRQSource(b)
                            break
                    setattr(self, "_irq_pending", True)
                else:
                    # Remain halted; model passage of one cycle of time
                    try:
                        # Keep instruction_count aligned with steps for perf tests
                        self.instruction_count += 1
                        self.cycle_count += 1
                    except Exception:
                        pass
                    return True
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
                if (imr_val_chk & int(IMRFlag.IRM)) == 0 or (
                    imr_val_chk & isr_val_chk
                ) == 0:
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
                    self.memory.write_byte(
                        imr_addr, imr_val & (~int(IMRFlag.IRM) & 0xFF)
                    )
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
            # Always simulate WAIT loops to advance timers, regardless of tracing.
            wait_sim_count = 0
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
                self._reset_instruction_access_log()

                eval_info = self.cpu.execute_instruction(pc)

                # Associate accumulated register accesses with this instruction
                self._flush_instruction_access_log(pc_before)

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    self._simulate_wait(wait_sim_count)
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
                self._reset_instruction_access_log()

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
                self._flush_instruction_access_log(pc_before)

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    self._simulate_wait(wait_sim_count)

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

    def _simulate_wait(self, cycles: int) -> None:
        """Advance cycle count and timers for simulated WAIT loops."""
        for _ in range(int(cycles)):
            self.cycle_count += 1
            if self._timer_enabled:
                try:
                    self._tick_timers()
                except Exception:
                    pass

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
            if isinstance(dest_addr, int):
                self._push_display_trace(dest_addr, pc_before)

        elif isinstance(instr, RetInstruction):
            ret_depth = self.call_depth
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
            self._pop_display_trace(ret_depth)

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
        # Special-case the ON key: not part of the matrix; set ISR.ONKI and arm IRQ
        if key_code == "KEY_ON":
            try:
                self._set_isr_bits(int(ISRFlag.ONKI))
                setattr(self, "_irq_pending", True)
                self._irq_source = IRQSource.ONK
            except Exception:
                pass
            if new_tracer.enabled:
                new_tracer.instant("I/O", "KeyPress", {"key": key_code})
            return True

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
                    if (isr_val & int(IMRFlag.KEY)) == 0:
                        self._set_isr_bits(int(ISRFlag.KEYI))
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
        ic = self.cycle_count
        fired = False
        # Main timer (MTI, bit 0)
        if ic >= self._timer_next_mti:
            self._set_isr_bits(int(ISRFlag.MTI))
            self._timer_next_mti = ic + self._timer_mti_period
            self._irq_pending = True
            self._irq_source = IRQSource.MTI
            fired = True
        # Sub timer (STI, bit 1) – less frequent
        if ic >= self._timer_next_sti:
            self._set_isr_bits(int(ISRFlag.STI))
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
            # Build watch table for all 8 bits of IMR/ISR
            watch_imr: Dict[int, Dict[str, list[int]]] = {}
            watch_isr: Dict[int, Dict[str, list[int]]] = {}
            try:
                for bit in range(8):
                    watch_imr[bit] = self.irq_bit_watch.get("IMR", {}).get(
                        bit, {"set": [], "clear": []}
                    )
                    watch_isr[bit] = self.irq_bit_watch.get("ISR", {}).get(
                        bit, {"set": [], "clear": []}
                    )
            except Exception:
                # Fallback to empty structure if any issue
                watch_imr = {bit: {"set": [], "clear": []} for bit in range(8)}
                watch_isr = {bit: {"set": [], "clear": []} for bit in range(8)}

            return {
                "total": int(self.irq_counts.get("total", 0)),
                "by_source": by_source,
                "last": {
                    "src": self.last_irq.get("src"),
                    "pc": self.last_irq.get("pc"),
                    "vector": self.last_irq.get("vector"),
                },
                "watch": {"IMR": watch_imr, "ISR": watch_isr},
            }
        except Exception:
            return {
                "total": 0,
                "by_source": {"KEY": 0, "MTI": 0, "STI": 0},
                "last": {"src": None, "pc": None, "vector": None},
            }

    @staticmethod
    def _default_display_trace_functions() -> Dict[int, str]:
        return {
            0xE5A78: "sub_e5a78_slow_timer",
            0xE5B38: "sub_e5b38_draw_text",
            0xE51C3: "sub_e51c3_display_write",
            0xE0D0C: "sub_e0d0c_draw_string",
            0xAC60: "sub_ac60_keyboard_decode",
            0xF2E24: "sub_f2e24_screen_write",
            0xF2E50: "sub_f2e50_screen_blit",
            0xF2E9E: "sub_f2e9e_screen_fill",
            0xF2DA5: "sub_f2da5_buffer_copy",
            0xF29B8: "sub_f29b8_draw_menu",
            0x02E24: "sub_f2e24_screen_write",
            0x02E50: "sub_f2e50_screen_blit",
            0x02E9E: "sub_f2e9e_screen_fill",
            0x02DA5: "sub_f2da5_buffer_copy",
            0x029B8: "sub_f29b8_draw_menu",
        }

    def _load_symbol_map(self) -> Dict[int, str]:
        if PCE500Emulator._symbol_cache is not None:
            return PCE500Emulator._symbol_cache
        import json
        from pathlib import Path

        symbol_map: Dict[int, str] = {}
        try:
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir.parent / "rom-analysis" / "bnida.json"
            if not candidate.exists():
                candidate = base_dir / "rom-analysis" / "bnida.json"
            if candidate.exists():
                data = json.loads(candidate.read_text())
                for key, name in data.get("names", {}).items():
                    try:
                        addr = int(key)
                    except ValueError:
                        continue
                    symbol_map[addr] = name
        except Exception:
            symbol_map = {}
        PCE500Emulator._symbol_cache = symbol_map
        return symbol_map

    def _load_function_addresses(self) -> List[int]:
        if PCE500Emulator._function_cache is not None:
            return PCE500Emulator._function_cache
        import json
        from pathlib import Path

        addresses: List[int] = []
        try:
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir.parent / "rom-analysis" / "bnida.json"
            if not candidate.exists():
                candidate = base_dir / "rom-analysis" / "bnida.json"
            if candidate.exists():
                data = json.loads(candidate.read_text())
                addresses = sorted(int(addr) for addr in data.get("functions", []) if isinstance(addr, int))
        except Exception:
            addresses = []
        PCE500Emulator._function_cache = addresses
        return addresses

    def _resolve_symbol_name(self, address: int) -> str:
        if address in self._display_trace_watch:
            return self._display_trace_watch[address]
        if address in self._display_trace_symbols:
            return self._display_trace_symbols[address]
        return f"sub_{address:05X}"

    def _lookup_function(self, pc: Optional[int]) -> tuple[int, str]:
        if pc is None:
            return (0, "unknown")
        if not self._display_trace_function_index:
            return (pc, self._resolve_symbol_name(pc))
        # Binary search for greatest address <= pc
        funcs = self._display_trace_function_index
        lo, hi = 0, len(funcs) - 1
        best = funcs[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            addr = funcs[mid]
            if addr <= pc:
                best = addr
                lo = mid + 1
            else:
                hi = mid - 1
        return (best, self._resolve_symbol_name(best))

    def _on_lcd_trace_event(self, event: Dict[str, Any]) -> None:
        if not self.display_trace_enabled:
            return
        payload = dict(event)
        payload.setdefault("pc", self.cpu.regs.get(RegisterName.PC))
        payload["instruction_count"] = self.instruction_count
        payload["cycle_count"] = self.cycle_count
        func_addr, func_name = self._lookup_function(payload.get("pc"))
        payload["function_addr"] = func_addr
        payload["function_name"] = func_name
        self._display_trace_events.append(payload)
        if len(self._display_trace_events) > self._display_trace_event_limit:
            self._display_trace_events.pop(0)
        if self._display_trace_stack:
            self._display_trace_stack[-1]["writes"].append(payload)
        summary = self._display_trace_summary.setdefault(
            func_addr,
            {
                "name": func_name,
                "address": func_addr,
                "writes": 0,
                "data_writes": 0,
                "instruction_writes": 0,
                "samples": [],
            },
        )
        summary["writes"] += 1
        if payload.get("type") == "data":
            summary["data_writes"] += 1
        else:
            summary["instruction_writes"] += 1
        if len(summary["samples"]) < 5:
            summary["samples"].append(payload)

    def _push_display_trace(self, dest_addr: int, caller_pc: int) -> None:
        if not self.display_trace_enabled:
            return
        if dest_addr not in self._display_trace_watch:
            return
        entry = {
            "name": self._resolve_symbol_name(dest_addr),
            "address": dest_addr,
            "caller": caller_pc,
            "start_instr": self.instruction_count,
            "start_cycle": self.cycle_count,
            "frame_depth": self.call_depth,
            "writes": [],
        }
        self._display_trace_stack.append(entry)

    def _pop_display_trace(self, ret_depth: int) -> None:
        if not self.display_trace_enabled or not self._display_trace_stack:
            return
        if self._display_trace_stack[-1]["frame_depth"] != ret_depth:
            return
        entry = self._display_trace_stack.pop()
        entry["end_instr"] = self.instruction_count
        entry["end_cycle"] = self.cycle_count
        entry["duration_instr"] = entry["end_instr"] - entry["start_instr"]
        entry["duration_cycle"] = entry["end_cycle"] - entry["start_cycle"]
        self.display_trace_log.append(entry)
        if len(self.display_trace_log) > self._display_trace_event_limit:
            self.display_trace_log.pop(0)

    def get_display_trace_log(self) -> Dict[str, Any]:
        return {
            "spans": [dict(entry) for entry in self.display_trace_log],
            "events": [dict(ev) for ev in self._display_trace_events],
            "summary": [
                {
                    **{
                        "name": meta["name"],
                        "address": addr,
                        "writes": meta["writes"],
                        "data_writes": meta["data_writes"],
                        "instruction_writes": meta["instruction_writes"],
                    },
                    "samples": list(meta["samples"]),
                }
                for addr, meta in sorted(self._display_trace_summary.items())
            ],
        }

    def bootstrap_from_rom_image(
        self,
        rom_image: bytes,
        *,
        reset: bool = True,
        restore_internal_ram: bool = True,
        configure_interrupt_mask: bool = True,
        imr_value: int = 0x43,
        isr_value: int = 0x00,
    ) -> None:
        """Reapply ROM-provided runtime state after a test reset.

        The PC-E500 firmware expects the internal RAM window (0xB8000-0xBFFFF)
        and certain IMEM registers to be initialised before the fast-timer ISR
        can service keyboard interrupts (see ``docs/interrupt_rom_analysis.md``).
        The emulator's :meth:`reset` method clears RAM/IMEM for determinism, so
        tests that rely on ROM behaviour need to restore those bytes manually.

        Args:
            rom_image: Full 1MB dump matching ``data/pc-e500.bin``.
            reset: When true (default) perform a fresh :meth:`reset` before
                restoring the RAM snapshot so CPU state matches power-on.
            restore_internal_ram: Copy the 0x8000-byte RAM block from the ROM
                image back into ``external_memory`` (default True).
            configure_interrupt_mask: Seed IMR/ISR with the ROM defaults
                (IMR=0x43, ISR=0x00 by default) so timer/keyboard IRQs deliver.
            imr_value: Value written to IMR when ``configure_interrupt_mask`` is
                enabled.
            isr_value: Value written to ISR when ``configure_interrupt_mask`` is
                enabled.

        Raises:
            ValueError: if ``rom_image`` is not exactly 1MB.
        """

        expected_size = 0x100000
        if len(rom_image) != expected_size:
            raise ValueError(
                f"Expected 0x{expected_size:05X} bytes in ROM image, "
                f"got {len(rom_image)}"
            )

        if reset:
            self.reset()

        if restore_internal_ram:
            ram_start = self.INTERNAL_RAM_START
            ram_end = ram_start + self.INTERNAL_RAM_SIZE
            # Copy RAM window directly into the mutable backing store.
            self.memory.external_memory[ram_start:ram_end] = rom_image[
                ram_start:ram_end
            ]

        if configure_interrupt_mask:
            imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
            isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            self.memory.write_byte(imr_addr, imr_value & 0xFF)
            self.memory.write_byte(isr_addr, isr_value & 0xFF)

        # Ensure PC points to the ROM entry vector in case callers expect to
        # execute instructions immediately after bootstrapping.
        try:
            entry_vector = self.memory.read_long(0xFFFFD)
            self.cpu.regs.set(RegisterName.PC, entry_vector)
        except Exception:
            pass

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
                reg_name,
                {
                    "reads": deque(maxlen=IMEM_ACCESS_HISTORY_LIMIT),
                    "writes": deque(maxlen=IMEM_ACCESS_HISTORY_LIMIT),
                },
            )
            access_list = tracking[access_type]
            if access_list and access_list[-1][0] == cpu_pc:
                access_list[-1] = (cpu_pc, access_list[-1][1] + 1)
            else:
                access_list.append((cpu_pc, 1))

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
                            self._set_isr_bits(int(ISRFlag.KEYI))
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
