"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import json
import os
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

# Import the SC62015 emulator
from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.instr.instructions import (
    CALL,
    RetInstruction,
    JumpInstruction,
    IR,
)
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.pysc62015.constants import IMRFlag, ISRFlag
from sc62015.pysc62015.stepper import CPURegistersSnapshot

from .memory import (
    PCE500Memory,
    MemoryOverlay,
    IMEM_ACCESS_HISTORY_LIMIT,
)
from .display import HD61202Controller
from .keyboard_handler import PCE500KeyboardHandler as KeyboardHandler
from .keyboard_matrix import MatrixEvent
from .tracing import trace_dispatcher
from .tracing.perfetto_tracing import tracer as new_tracer, perf_trace
from .scheduler import TimerScheduler, TimerSource
from .peripherals import PeripheralManager

# Default timer periods in cycles (rough emulation)
MTI_PERIOD_CYCLES_DEFAULT = 500
STI_PERIOD_CYCLES_DEFAULT = 5000

SNAPSHOT_MAGIC = "pc-e500.snapshot"
SNAPSHOT_VERSION = 1
_SNAPSHOT_REGISTER_LAYOUT = (
    ("pc", 3),
    ("ba", 2),
    ("i", 2),
    ("x", 3),
    ("y", 3),
    ("u", 3),
    ("s", 3),
    ("f", 1),
)


def _pack_register_bytes(snapshot: CPURegistersSnapshot) -> bytes:
    """Pack the core registers into a deterministic little-endian blob."""

    chunks: list[bytes] = []
    for name, width in _SNAPSHOT_REGISTER_LAYOUT:
        value = getattr(snapshot, name)
        chunks.append(int(value).to_bytes(width, byteorder="little", signed=False))
    return b"".join(chunks)


def _unpack_register_bytes(payload: bytes) -> Dict[str, int]:
    """Unpack a register blob created by ``_pack_register_bytes``."""

    expected = sum(width for _, width in _SNAPSHOT_REGISTER_LAYOUT)
    if len(payload) != expected:
        raise ValueError(
            f"registers.bin length mismatch (expected {expected}, got {len(payload)})"
        )
    offset = 0
    values: Dict[str, int] = {}
    for name, width in _SNAPSHOT_REGISTER_LAYOUT:
        values[name] = int.from_bytes(
            payload[offset : offset + width], byteorder="little", signed=False
        )
        offset += width
    return values


class IRQSource(Enum):
    # Enum values store ISR bit index directly
    MTI = 0  # Main timer interrupt → ISR bit 0
    STI = 1  # Sub timer interrupt  → ISR bit 1
    KEY = 2  # Keyboard interrupt   → ISR bit 2
    ONK = 3  # On-key interrupt     → ISR bit 3


# Define constants locally to avoid heavy imports
INTERNAL_MEMORY_START = 0x100000
KOL, KOH, KIL = IMEMRegisters.KOL, IMEMRegisters.KOH, IMEMRegisters.KIL
IRQ_STACK_TRACE_ENABLED = os.getenv("IRQ_STACK_TRACE") == "1"
PYTHON_PC_TRACE_ENABLED = os.getenv("PYTHON_PC_TRACE") == "1"

_STACK_SNAPSHOT_RANGE: tuple[int, int] | None = None
_STACK_SNAPSHOT_LEN: int | None = None


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw not in {"0", "false", "False", "off", ""}


IRQ_DEBUG_ENABLED = _env_flag("IRQ_DEBUG")


def _log_irq_debug(message: str) -> None:
    if IRQ_DEBUG_ENABLED:
        print(f"[irq-debug] {message}")


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

    _TRACE_UNITS_PER_INSTRUCTION = 10_000
    _TRACE_REGISTERS = (
        RegisterName.PC,
        RegisterName.A,
        RegisterName.B,
        RegisterName.BA,
        RegisterName.I,
        RegisterName.IL,
        RegisterName.IH,
        RegisterName.X,
        RegisterName.Y,
        RegisterName.U,
        RegisterName.S,
        RegisterName.F,
        RegisterName.FC,
        RegisterName.FZ,
    )
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
        lcd_trace_file: Optional[str] = None,
        lcd_trace_event_limit: int = 50000,
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

        self.lcd = HD61202Controller()
        self._lcd_trace_events: List[Dict[str, Any]] = []
        self._lcd_trace_limit = max(0, int(lcd_trace_event_limit))
        self._lcd_trace_truncated = False
        self._lcd_trace_path = Path(lcd_trace_file) if lcd_trace_file else None
        if self._lcd_trace_path and self._lcd_trace_limit > 0:
            self.lcd.add_write_trace_callback(self._record_lcd_trace_event)

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
            self.lcd.add_write_trace_callback(self._on_lcd_trace_event)

        self._keyboard_columns_active_high = keyboard_columns_active_high

        # Keyboard implementation parameterised for column polarity
        self.keyboard = KeyboardHandler(
            self.memory, columns_active_high=keyboard_columns_active_high
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

        # Note: LCC overlay not needed for the keyboard handler

        backend = os.getenv("SC62015_CPU_BACKEND")
        try:
            self.cpu = CPU(self.memory, reset_on_init=True, backend=backend)
        except RuntimeError as exc:
            # Fall back to the legacy backend if the requested one is unavailable.
            print(f"[pce500] Falling back to python CPU backend: {exc}")
            self.cpu = CPU(self.memory, reset_on_init=True, backend="python")

        self.memory.set_cpu(self.cpu)
        backend_name = getattr(self.cpu, "backend", "python")

        pure_keyboard_env = os.getenv("RUST_PURE_KEYBOARD")
        if pure_keyboard_env is not None:
            disable_keyboard_overlay = pure_keyboard_env == "1"
        else:
            disable_keyboard_overlay = backend_name == "llama"
        self.memory.set_keyboard_handler(
            self._keyboard_read_handler,
            self._keyboard_write_handler,
            enable_overlay=not disable_keyboard_overlay,
        )
        try:
            self.keyboard.set_bridge_cpu(
                self.cpu if disable_keyboard_overlay else None, disable_keyboard_overlay
            )
        except Exception:
            pass

        pure_lcd_env = os.getenv("RUST_PURE_LCD")
        if pure_lcd_env is not None:
            disable_overlay = pure_lcd_env == "1"
        else:
            disable_overlay = backend_name == "llama"
        enable_overlay = not disable_overlay
        self.memory.set_lcd_controller(self.lcd, enable_overlay=enable_overlay)
        self._llama_pure_lcd = disable_overlay
        self._llama_lcd_write = (
            getattr(self.memory, "_llama_lcd_write", None) if disable_overlay else None
        )

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
            trace_dispatcher.start_trace(trace_path)
            self.lcd.set_perfetto_enabled(True)
            self.memory.set_perfetto_enabled(True)
            if getattr(self.cpu, "backend", None) == "llama":
                try:
                    self.cpu.set_perfetto_trace(trace_path)
                except Exception:
                    pass
        elif trace_dispatcher.has_observers():
            trace_dispatcher.start_trace(trace_path)

        # New tracing system
        self._new_trace_enabled = enable_new_tracing
        self._trace_path = trace_path
        self._trace_instr_count = 0
        self._trace_units_per_instruction = self._TRACE_UNITS_PER_INSTRUCTION
        self._trace_substep = 0
        self._active_trace_instruction: Optional[int] = None
        if self._new_trace_enabled:
            new_tracer.start(self._trace_path)
            new_tracer.set_manual_clock_mode(True, tick_ns=1)
            self.memory.set_perf_tracer(new_tracer)

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
        # Synthetic keyboard interrupt wiring (enable for both handler and hardware modes)
        self._kb_irq_enabled = True
        self._irq_pending = False
        self._in_interrupt = False
        self._kb_irq_count = 0
        self._scheduler = TimerScheduler(
            mti_period=MTI_PERIOD_CYCLES_DEFAULT,
            sti_period=STI_PERIOD_CYCLES_DEFAULT,
        )
        self._irq_source: Optional["IRQSource"] = None
        # Fast mode: minimize step() overhead to run many instructions
        self.fast_mode = False

        # Peripheral adapters and IMEM callbacks
        self.peripherals = PeripheralManager(self.memory, self._scheduler)
        self.memory.set_imem_access_callback(self._handle_imem_access)

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
        self._scheduler.reset()
        self._kb_irq_count = 0
        if getattr(self, "_llama_pure_lcd", False):
            self._seed_interrupt_mask()
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
        trace_snapshot: Optional[Dict[str, Any]] = None
        # Tick rough timers to set ISR bits and arm IRQ when due
        try:
            if self._timer_enabled and not getattr(self, "_in_interrupt", False):
                if not getattr(self.keyboard, "_bridge_enabled", False) or not getattr(
                    self.cpu, "_timer_in_llama", False
                ):
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
                    _log_irq_debug(
                        f"HALT wake IRQ pending (ISR=0x{isr_val_chk:02X}) source={self._irq_source}"
                    )
                else:
                    # Remain halted; model passage of one cycle of time
                    try:
                        # Model one cycle of idle time while halted; no instruction executed.
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
                _log_irq_debug(
                    f"pending IRQ check pc=0x{self.cpu.regs.get(RegisterName.PC):06X} "
                    f"imr=0x{imr_val_chk:02X} isr=0x{isr_val_chk:02X} in_interrupt={self._in_interrupt}"
                )
                if (imr_val_chk & int(IMRFlag.IRM)) == 0 or (
                    imr_val_chk & isr_val_chk
                ) == 0:
                    # Keep pending; CPU continues executing normal flow
                    _log_irq_debug(
                        f"IRQ masked; pending retained (IMR=0x{imr_val_chk:02X} ISR=0x{isr_val_chk:02X})"
                    )
                    pass
                else:
                    # Push PC (3 bytes), then F (1), then IMR (1), clear IMR.IRM
                    cur_pc = self.cpu.regs.get(RegisterName.PC)
                    s = self.cpu.regs.get(RegisterName.S)
                    _log_irq_debug(
                        f"Delivering IRQ src={self._irq_source} pc=0x{cur_pc:06X} s=0x{s:06X}"
                    )
                    # Require a valid, initialized stack pointer; defer IRQ until firmware sets SP
                    if not isinstance(s, int) or s < 5:
                        raise RuntimeError(
                            "IRQ deferred: stack pointer not initialized"
                        )
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] deliver start pc=0x{cur_pc:06X} s=0x{s:06X} f=0x{int(self.cpu.regs.get(RegisterName.F)) & 0xFF:02X} imr=0x{imr_val_chk:02X}"
                        )
                    # push PC (little-endian 3 bytes)
                    s_new = s - 3
                    self.memory.write_bytes(3, s_new, cur_pc)
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] push_pc from 0x{s:06X} to 0x{s_new:06X} value=0x{cur_pc & 0xFFFFFF:06X}"
                        )
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push F (1 byte)
                    f_val = self.cpu.regs.get(RegisterName.F)
                    s_new = self.cpu.regs.get(RegisterName.S) - 1
                    self.memory.write_bytes(1, s_new, f_val)
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] push_f value=0x{int(f_val) & 0xFF:02X} new_s=0x{s_new:06X}"
                        )
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push IMR (1 byte) and clear IRM bit 7
                    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
                    imr_val = self.memory.read_byte(imr_addr)
                    s_new = self.cpu.regs.get(RegisterName.S) - 1
                    self.memory.write_bytes(1, s_new, imr_val)
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] push_imr value=0x{imr_val & 0xFF:02X} new_s=0x{s_new:06X}"
                        )
                    self.cpu.regs.set(RegisterName.S, s_new)
                    self.memory.write_byte(
                        imr_addr, imr_val & (~int(IMRFlag.IRM) & 0xFF)
                    )
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] deliver done s=0x{int(self.cpu.regs.get(RegisterName.S)):06X}"
                        )
                    # ISR status was set by the triggering source (device/timer)
                    # Do not modify ISR here; only deliver the interrupt.
                    # Jump to interrupt vector (0xFFFFA little-endian 3 bytes)
                    vector_addr = self.memory.read_long(0xFFFFA)
                    self.cpu.regs.set(RegisterName.PC, vector_addr)
                    self._in_interrupt = True
                    self._irq_pending = False
                    _log_irq_debug(
                        f"IRQ delivered vector=0x{vector_addr:06X} new_s=0x{self.cpu.regs.get(RegisterName.S):06X}"
                    )
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
        if os.getenv("IRQ_DEBUG_CYCLES") == "1" and pc in (
            0x0F2051,
            0x0F2053,
            0x0F2055,
            0x0F2056,
            0x0F2059,
            0x0F205C,
        ):
            print(
                f"[irq-cycle] pc=0x{pc:06X} cycles={self.cycle_count} in_irq={self._in_interrupt}"
            )

        if self._new_trace_enabled and new_tracer.enabled:
            trace_snapshot = self._snapshot_instruction_trace(
                pc, self.instruction_count
            )
            if trace_snapshot is not None:
                self._active_trace_instruction = self.instruction_count
                self._trace_substep = 0

        if pc in self.breakpoints:
            return False

        if self.trace is not None:
            self.trace.append(("exec", pc, self.cycle_count))

        if pc == self.MEMORY_DUMP_PC and self.perfetto_enabled:
            self._dump_internal_memory(pc)

        opcode = None
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
                opcode = opcode_peek
            except Exception:
                pass

            if getattr(self, "fast_mode", False):
                # Minimal execution path for speed
                pc_before = pc

                # Clear current instruction accesses before execution
                self._reset_instruction_access_log()

                eval_info = self.cpu.execute_instruction(pc)
                if PYTHON_PC_TRACE_ENABLED:
                    print(f"[python-pc] PC=0x{pc:06X}")
                _log_stack_snapshot_emulator(self, pc)

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
                if new_tracer.enabled:
                    new_tracer.instant(
                        "Execution",
                        f"Exec@0x{pc_before:06X}",
                        {"pc": f"0x{pc_before:06X}"},
                    )
                elif self.perfetto_enabled or trace_dispatcher.has_observers():
                    trace_dispatcher.record_instant(
                        "Execution",
                        f"Exec@0x{pc_before:06X}",
                        {"pc": f"0x{pc_before:06X}"},
                    )
                # Emit lightweight Exec@ events for new tracer even in fast mode
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
                    if PYTHON_PC_TRACE_ENABLED:
                        print(f"[python-pc] PC=0x{pc:06X}")
                _log_stack_snapshot_emulator(self, pc)

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

                # Always emit an execution event so registered observers (and the
                # new tracer when enabled) see every instruction regardless of
                # perfetto wiring.
                self._trace_execution(pc, opcode)

                if self.perfetto_enabled:
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

        except Exception as e:
            if self.trace is not None:
                self.trace.append(("error", pc, str(e)))
            if self.perfetto_enabled:
                trace_dispatcher.record_instant(
                    "CPU", "Error", {"error": str(e), "pc": f"0x{pc:06X}"}
                )
            raise
        # Always emit an Execution instant when the new tracer is enabled so the
        # Perfetto stream has one Exec@ per instruction regardless of legacy
        # dispatcher state.
        if self._new_trace_enabled and new_tracer.enabled and opcode is not None:
            self._trace_execution(pc, opcode)
        self._emit_instruction_trace_event(trace_snapshot)
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
        self._sync_lcd_from_backend()
        img = self.lcd.get_combined_display(zoom=1)
        img.save(combined_filename)

    def close(self) -> None:
        """Release backend resources (flush traces, etc.)."""
        try:
            if hasattr(self.cpu, "flush_perfetto"):
                self.cpu.flush_perfetto()
            trace_dispatcher.stop_trace()
            if self._new_trace_enabled and new_tracer.enabled:
                new_tracer.stop()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _sync_lcd_from_backend(self) -> None:
        if getattr(self.cpu, "backend", None) != "llama" or not getattr(
            self, "_llama_pure_lcd", False
        ):
            return
        exporter = getattr(self.cpu, "export_lcd_snapshot", None)
        if not callable(exporter):
            return
        metadata, payload = exporter()
        if metadata and payload:
            try:
                self.lcd.load_snapshot(metadata, payload)
            except Exception as exc:  # pragma: no cover - diagnostic path
                print(
                    f"WARNING: failed to apply LCD snapshot from LLAMA backend: {exc}"
                )

    def _capture_lcd_snapshot(self) -> Tuple[Dict[str, object], bytes]:
        self._sync_lcd_from_backend()
        snapshot = self.lcd.get_snapshot()
        chips = snapshot.chips
        meta: Dict[str, object] = {
            "chip_count": len(chips),
            "pages": len(chips[0].vram) if chips else 0,
            "width": len(chips[0].vram[0]) if chips and chips[0].vram else 0,
            "chips": [],
            "cs_both_count": getattr(self.lcd, "cs_both_count", 0),
            "cs_left_count": getattr(self.lcd, "cs_left_count", 0),
            "cs_right_count": getattr(self.lcd, "cs_right_count", 0),
        }
        payload = bytearray()
        for chip_snap in chips:
            meta["chips"].append(
                {
                    "on": chip_snap.on,
                    "start_line": chip_snap.start_line,
                    "page": chip_snap.page,
                    "y_address": chip_snap.y_address,
                    "instruction_count": chip_snap.instruction_count,
                    "data_write_count": chip_snap.data_write_count,
                }
            )
            for page in chip_snap.vram:
                for value in page:
                    payload.append(int(value) & 0xFF)
        return meta, bytes(payload)

    def _restore_lcd_snapshot(
        self, metadata: Optional[Dict[str, object]], payload: Optional[bytes]
    ) -> None:
        if not metadata or payload is None:
            return
        self.lcd.load_snapshot(metadata, payload)

    def _patch_llama_snapshot_metadata(self, target: Path) -> None:
        """Add Python-only metadata fields to a LLAMA-authored snapshot."""

        try:
            with zipfile.ZipFile(target, "r") as zf:
                entries = {name: zf.read(name) for name in zf.namelist()}
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[snapshot] Unable to post-process LLAMA snapshot: {exc}")
            return

        try:
            raw_meta = entries.get("snapshot.json", b"{}")
            metadata = json.loads(raw_meta.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[snapshot] Failed to parse snapshot metadata: {exc}")
            return

        cpu_snapshot = self.cpu.snapshot_registers()
        temps = {str(k): int(v) for k, v in getattr(cpu_snapshot, "temps", {}).items()}
        metadata.update(
            {
                "backend": getattr(self.cpu, "backend", "llama"),
                "call_depth": int(getattr(self, "call_depth", 0)),
                "call_sub_level": int(getattr(cpu_snapshot, "call_sub_level", 0)),
                "temps": temps,
                "memory_reads": int(getattr(self, "memory_read_count", 0)),
                "memory_writes": int(getattr(self, "memory_write_count", 0)),
                "memory_dump_pc": int(getattr(self, "MEMORY_DUMP_PC", 0)),
                "fast_mode": bool(getattr(self, "fast_mode", False)),
            }
        )

        keyboard_state = (
            self.keyboard.snapshot_state() if hasattr(self, "keyboard") else None
        )
        if keyboard_state is not None:
            metadata["keyboard"] = keyboard_state

        kb_metrics = {
            "irq_count": int(getattr(self, "_kb_irq_count", 0)),
            "strobe_count": int(getattr(self, "_kb_strobe_count", 0)),
            "column_hist": list(getattr(self, "_kb_col_hist", [])),
            "last_cols": list(getattr(self, "_last_kil_columns", [])),
            "last_kol": int(getattr(self, "_last_kol", 0)),
            "last_koh": int(getattr(self, "_last_koh", 0)),
            "kil_reads": int(getattr(self, "_kil_read_count", 0)),
            "kb_irq_enabled": bool(getattr(self, "_kb_irq_enabled", True)),
        }
        metadata["kb_metrics"] = kb_metrics

        entries["snapshot.json"] = json.dumps(
            metadata, indent=2, sort_keys=True
        ).encode("utf-8")

        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, data in entries.items():
                    zf.writestr(name, data)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[snapshot] Failed to rewrite Rust snapshot metadata: {exc}")

    def save_snapshot(self, path: str | Path) -> Path:
        """Persist CPU/memory/peripheral state to a .pcsnap bundle."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if getattr(self.cpu, "backend", None) == "llama":
            llama_impl = self.cpu.unwrap()
            saver = getattr(llama_impl, "save_snapshot", None)
            if callable(saver):
                try:
                    is_synced = getattr(llama_impl, "is_memory_synced", None)
                    if callable(is_synced) and not is_synced():
                        reinit = getattr(llama_impl, "_initialise_rust_memory", None)
                        if callable(reinit):
                            reinit()
                    saver(str(target))
                    self._patch_llama_snapshot_metadata(target)
                    return target
                except Exception as exc:  # pragma: no cover - fallback path
                    print(
                        f"[snapshot] LLAMA save failed, falling back to python path: {exc}"
                    )

        cpu_snapshot = self.cpu.snapshot_registers()
        registers_blob = _pack_register_bytes(cpu_snapshot)
        flat_memory, fallback_ranges, readonly_ranges = self.memory.export_flat_memory()

        internal_slice = self.memory.external_memory[
            self.INTERNAL_RAM_START : self.INTERNAL_RAM_START + self.INTERNAL_RAM_SIZE
        ]
        imem_bytes = self.memory.get_internal_memory_bytes()
        lcd_meta, lcd_payload = self._capture_lcd_snapshot()
        keyboard_state = (
            self.keyboard.snapshot_state() if hasattr(self, "keyboard") else None
        )

        timer_info = {
            "enabled": bool(self._timer_enabled),
            "mti_period": int(self._timer_mti_period),
            "sti_period": int(self._timer_sti_period),
            "next_mti": int(self._timer_next_mti),
            "next_sti": int(self._timer_next_sti),
        }
        irq_source_name = self._irq_source.name if self._irq_source else None
        interrupts = {
            "pending": bool(getattr(self, "_irq_pending", False)),
            "in_interrupt": bool(getattr(self, "_in_interrupt", False)),
            "source": irq_source_name,
            "stack": list(self._interrupt_stack),
            "next_id": int(self._next_interrupt_id),
            "irq_counts": dict(self.irq_counts),
            "last_irq": dict(self.last_irq),
            "irq_bit_watch": self.irq_bit_watch,
        }

        kb_metrics = {
            "irq_count": int(getattr(self, "_kb_irq_count", 0)),
            "strobe_count": int(getattr(self, "_kb_strobe_count", 0)),
            "column_hist": list(getattr(self, "_kb_col_hist", [])),
            "last_cols": list(getattr(self, "_last_kil_columns", [])),
            "last_kol": int(getattr(self, "_last_kol", 0)),
            "last_koh": int(getattr(self, "_last_koh", 0)),
            "kil_reads": int(getattr(self, "_kil_read_count", 0)),
            "kb_irq_enabled": bool(getattr(self, "_kb_irq_enabled", True)),
        }

        metadata = {
            "magic": SNAPSHOT_MAGIC,
            "version": SNAPSHOT_VERSION,
            "backend": self.cpu.backend,
            "created": datetime.utcnow().isoformat() + "Z",
            "instruction_count": int(self.instruction_count),
            "cycle_count": int(self.cycle_count),
            "memory_reads": int(self.memory_read_count),
            "memory_writes": int(self.memory_write_count),
            "pc": int(cpu_snapshot.pc),
            "call_depth": int(self.call_depth),
            "call_sub_level": int(cpu_snapshot.call_sub_level),
            "temps": {str(k): int(v) for k, v in cpu_snapshot.temps.items()},
            "timer": timer_info,
            "interrupts": interrupts,
            "keyboard": keyboard_state,
            "lcd": lcd_meta,
            "fallback_ranges": [
                [int(start), int(end)] for start, end in fallback_ranges
            ],
            "readonly_ranges": [
                [int(start), int(end)] for start, end in readonly_ranges
            ],
            "internal_ram": {
                "start": self.INTERNAL_RAM_START,
                "size": self.INTERNAL_RAM_SIZE,
            },
            "imem": {"start": INTERNAL_MEMORY_START, "size": 0x100},
            "kb_metrics": kb_metrics,
            "memory_dump_pc": int(self.MEMORY_DUMP_PC),
            "fast_mode": bool(getattr(self, "fast_mode", False)),
            "memory_image_size": len(flat_memory),
            "lcd_payload_size": len(lcd_payload),
        }

        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("snapshot.json", json.dumps(metadata, indent=2, sort_keys=True))
            zf.writestr("registers.bin", registers_blob)
            zf.writestr("external_ram.bin", bytes(flat_memory))
            zf.writestr("internal_ram.bin", bytes(internal_slice))
            zf.writestr("imem.bin", imem_bytes)
            zf.writestr("lcd_vram.bin", lcd_payload)

        return target

    def load_snapshot(self, path: str | Path, *, backend: Optional[str] = None) -> None:
        """Load a snapshot created by ``save_snapshot``."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        with zipfile.ZipFile(source, "r") as zf:
            metadata = json.loads(zf.read("snapshot.json"))
            if metadata.get("magic") != SNAPSHOT_MAGIC:
                raise ValueError("Snapshot magic mismatch")
            if int(metadata.get("version", -1)) != SNAPSHOT_VERSION:
                raise ValueError("Unsupported snapshot version")
            registers_blob = zf.read("registers.bin")
            flat_memory = zf.read("external_ram.bin")
            imem_bytes = zf.read("imem.bin") if "imem.bin" in zf.namelist() else b""
            lcd_payload = (
                zf.read("lcd_vram.bin") if "lcd_vram.bin" in zf.namelist() else None
            )

        if len(flat_memory) != len(self.memory.external_memory):
            raise ValueError("external_ram.bin size mismatch")
        self.memory.external_memory[:] = flat_memory
        for overlay in self.memory.overlays:
            if overlay.data is None:
                continue
            start = max(overlay.start, 0)
            end = min(overlay.end + 1, len(flat_memory))
            span = max(0, end - start)
            if span and span <= len(overlay.data):
                overlay.data[:span] = flat_memory[start : start + span]

        if imem_bytes:
            self.memory.external_memory[-len(imem_bytes) :] = imem_bytes

        # Hardware invariant: USR bits 3/4 (TXR/TXE) come up set after reset.
        try:
            usr_addr = INTERNAL_MEMORY_START + IMEMRegisters.USR.value
            usr_val = self.memory.read_byte(usr_addr) & 0xFF
            self.memory.write_byte(usr_addr, usr_val | 0x18)
            # IMR often defaults to IRM|EXM|STM|MTM in the Python backend; seed
            # a sane value so Rust and Python start aligned when snapshots lack it.
            imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR.value
            imr_val = self.memory.read_byte(imr_addr) & 0xFF
            if imr_val == 0:
                self.memory.write_byte(imr_addr, 0xC3)
        except Exception:
            pass

        if getattr(self.cpu, "backend", None) == "llama":
            llama_impl = getattr(self.cpu, "_impl", None)
            marker = getattr(llama_impl, "mark_memory_dirty", None)
            if callable(marker):
                try:
                    marker()
                except Exception:
                    pass

        self._restore_lcd_snapshot(metadata.get("lcd"), lcd_payload)

        keyboard_state = metadata.get("keyboard")
        if isinstance(keyboard_state, dict) and hasattr(self, "keyboard"):
            self.keyboard.load_state(keyboard_state)

        reg_values = _unpack_register_bytes(registers_blob)
        temps = {
            int(key): int(value) for key, value in (metadata.get("temps") or {}).items()
        }
        snapshot = CPURegistersSnapshot(
            pc=reg_values["pc"],
            ba=reg_values["ba"],
            i=reg_values["i"],
            x=reg_values["x"],
            y=reg_values["y"],
            u=reg_values["u"],
            s=reg_values["s"],
            f=reg_values["f"],
            temps=temps,
            call_sub_level=int(metadata.get("call_sub_level", 0)),
        )
        self.cpu.apply_snapshot(snapshot)

        snapshot_backend = metadata.get("backend")
        if backend and snapshot_backend and backend != snapshot_backend:
            print(
                f"[snapshot] Warning: snapshot backend {snapshot_backend} "
                f"!= requested {backend}"
            )
        elif snapshot_backend and snapshot_backend != self.cpu.backend:
            print(
                f"[snapshot] Warning: emulator backend {self.cpu.backend} "
                f"!= snapshot backend {snapshot_backend}"
            )

        self.instruction_count = int(metadata.get("instruction_count", 0))
        self.cycle_count = int(metadata.get("cycle_count", 0))
        self.memory_read_count = int(metadata.get("memory_reads", 0))
        self.memory_write_count = int(metadata.get("memory_writes", 0))
        self.call_depth = int(metadata.get("call_depth", 0))
        self._current_pc = int(metadata.get("pc", snapshot.pc)) & 0xFFFFFF
        self._last_pc = self._current_pc
        self._trace_instr_count = self.instruction_count
        self._active_trace_instruction = None
        self._trace_substep = 0
        self.start_time = time.time()

        timer_info = metadata.get("timer", {})
        self._timer_enabled = bool(timer_info.get("enabled", True))
        self._timer_mti_period = int(
            timer_info.get("mti_period", self._timer_mti_period)
        )
        self._timer_sti_period = int(
            timer_info.get("sti_period", self._timer_sti_period)
        )
        self._timer_next_mti = int(timer_info.get("next_mti", self._timer_next_mti))
        self._timer_next_sti = int(timer_info.get("next_sti", self._timer_next_sti))

        interrupts = metadata.get("interrupts", {})
        self._irq_pending = bool(interrupts.get("pending", False))
        self._in_interrupt = bool(interrupts.get("in_interrupt", False))
        source_name = interrupts.get("source")
        self._irq_source = IRQSource[source_name] if source_name else None
        self._interrupt_stack = list(interrupts.get("stack", []))
        self._next_interrupt_id = int(interrupts.get("next_id", 1))
        irq_counts = interrupts.get("irq_counts")
        if isinstance(irq_counts, dict):
            self.irq_counts = {key: int(val) for key, val in irq_counts.items()}
        last_irq = interrupts.get("last_irq")
        if isinstance(last_irq, dict):
            self.last_irq = last_irq
        irq_watch = interrupts.get("irq_bit_watch")
        if isinstance(irq_watch, dict):
            self.irq_bit_watch = irq_watch

        kb_metrics = metadata.get("kb_metrics", {})
        self._kb_irq_count = int(kb_metrics.get("irq_count", 0))
        self._kb_strobe_count = int(kb_metrics.get("strobe_count", 0))
        hist = kb_metrics.get("column_hist")
        if isinstance(hist, list) and hist:
            self._kb_col_hist = [int(val) for val in hist]
        last_cols = kb_metrics.get("last_cols")
        if isinstance(last_cols, list):
            self._last_kil_columns = [int(val) for val in last_cols]
        self._last_kol = int(kb_metrics.get("last_kol", self._last_kol)) & 0xFF
        self._last_koh = int(kb_metrics.get("last_koh", self._last_koh)) & 0xFF
        self._kil_read_count = int(kb_metrics.get("kil_reads", self._kil_read_count))
        self._kb_irq_enabled = bool(kb_metrics.get("kb_irq_enabled", True))

        self.MEMORY_DUMP_PC = int(metadata.get("memory_dump_pc", self.MEMORY_DUMP_PC))
        self.fast_mode = bool(
            metadata.get("fast_mode", getattr(self, "fast_mode", False))
        )

        self.instruction_history.clear()
        self.memory.clear_imem_access_tracking()

        if self.cpu.backend == "llama":
            llama_impl = self.cpu.unwrap()
            reinit = getattr(llama_impl, "_initialise_rust_memory", None)
            if callable(reinit):
                try:
                    setattr(llama_impl, "_memory_synced", False)
                except Exception:
                    pass
                reinit()
            sync_irq = getattr(llama_impl, "set_interrupt_state", None)
            if callable(sync_irq):
                try:
                    sync_irq(
                        bool(self._irq_pending),
                        int(
                            self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.IMR
                            )
                        )
                        & 0xFF,
                        int(
                            self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.ISR
                            )
                        )
                        & 0xFF,
                        int(self._scheduler.next_mti),
                        int(self._scheduler.next_sti),
                        self._irq_source.name if self._irq_source else None,
                        bool(self._in_interrupt),
                        list(self._interrupt_stack),
                        int(self._next_interrupt_id),
                    )
                except Exception:
                    pass

        self.memory.set_cpu(self.cpu)

    def stop_tracing(self) -> None:
        if self.perfetto_enabled:
            print("Stopping Perfetto tracing...")
            trace_dispatcher.stop_trace()
        if self._new_trace_enabled and new_tracer.enabled:
            print(f"Stopping new tracing, saved to {self._trace_path}")
            new_tracer.set_manual_clock_mode(False)
            new_tracer.stop()
        self._new_trace_enabled = False

    def start_tracing(self, path: Optional[str] = None) -> None:
        """Enable instruction tracing to the provided path."""
        if path:
            self._trace_path = path
        self._new_trace_enabled = True
        if not new_tracer.enabled:
            new_tracer.start(self._trace_path)

    @property
    def tracing_enabled(self) -> bool:
        """Return ``True`` when instruction tracing is active."""
        return self._new_trace_enabled and new_tracer.enabled

    def get_keyboard_register_state(self) -> Dict[str, int]:
        """Expose current keyboard matrix register values."""
        if not hasattr(self, "keyboard"):
            return {"kol": 0, "koh": 0, "kil": 0}
        keyboard = self.keyboard
        kol = keyboard.kol_value if hasattr(keyboard, "kol_value") else 0
        koh = keyboard.koh_value if hasattr(keyboard, "koh_value") else 0
        kil = keyboard.peek_keyboard_input()
        return {"kol": kol & 0xFF, "koh": koh & 0xFF, "kil": kil & 0xFF}

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
            if self._timer_enabled and not getattr(self, "_in_interrupt", False):
                try:
                    self._tick_timers()
                except Exception:
                    pass

    def _build_register_annotations(self) -> Dict[str, Any]:
        state = self.get_cpu_state()
        return {
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

    def _snapshot_instruction_trace(
        self, pc: Optional[int], instr_index: int
    ) -> Optional[Dict[str, Any]]:
        """Capture register + opcode state prior to executing an instruction."""

        if not (self._new_trace_enabled and new_tracer.enabled):
            return None
        if pc is None:
            return None
        try:
            opcode = self.memory.read_byte(pc) & 0xFF
        except Exception:
            opcode = None

        registers = self._collect_trace_registers()

        units = instr_index * self._trace_units_per_instruction
        return {
            "pc": pc & 0xFFFFFF,
            "opcode": opcode,
            "op_index": instr_index,
            "registers": registers,
            "units": units,
        }

    def _emit_instruction_trace_event(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Emit the captured instruction snapshot as a Perfetto instant event."""

        if not snapshot or not new_tracer.enabled:
            return
        units = snapshot.get("units")
        if isinstance(units, int):
            new_tracer.set_manual_clock_units(units)
        pc = snapshot.get("pc")
        name = f"Exec@0x{pc:06X}" if isinstance(pc, int) else "Exec@?"
        payload: Dict[str, Any] = {
            "backend": getattr(self.cpu, "backend", "python"),
        }
        if isinstance(pc, int):
            payload["pc"] = pc & 0xFFFFFF
        opcode = snapshot.get("opcode")
        if isinstance(opcode, int):
            payload["opcode"] = opcode & 0xFF
        op_index = snapshot.get("op_index")
        if isinstance(op_index, int):
            payload["op_index"] = op_index
        registers = snapshot.get("registers", {})
        for reg_name, value in registers.items():
            payload[f"reg_{reg_name.lower()}"] = int(value)
        new_tracer.instant("InstructionTrace", name, payload)
        self._trace_instr_count = int(snapshot.get("op_index", 0)) + 1
        new_tracer.counter(
            "InstructionClock", "instructions", float(self._trace_instr_count)
        )
        self._active_trace_instruction = None

    def _collect_trace_registers(self) -> Dict[str, int]:
        """Collect the register snapshot for Perfetto tracing with minimal overhead."""

        snapshot_func = getattr(self.cpu, "snapshot_registers", None)
        if callable(snapshot_func):
            try:
                snapshot = snapshot_func()
            except Exception:
                snapshot = None
            else:
                registers = self._collect_trace_registers_from_snapshot(snapshot)
                if registers:
                    return registers
        return self._collect_trace_registers_legacy()

    def _collect_trace_registers_from_snapshot(self, snapshot: Any) -> Dict[str, int]:
        """Extract trace registers from a CPURegistersSnapshot-like object."""

        registers: Dict[str, int] = {}
        try:
            pc_val = int(getattr(snapshot, "pc", 0)) & 0xFFFFFF
            registers["PC"] = pc_val
        except Exception:
            pass

        def _mask(attr: str, mask: int) -> int:
            try:
                return int(getattr(snapshot, attr, 0)) & mask
            except Exception:
                return 0

        ba_val = _mask("ba", 0xFFFF)
        registers["BA"] = ba_val
        registers["A"] = ba_val & 0xFF
        registers["B"] = (ba_val >> 8) & 0xFF

        i_val = _mask("i", 0xFFFF)
        registers["I"] = i_val
        registers["IL"] = i_val & 0xFF
        registers["IH"] = (i_val >> 8) & 0xFF

        registers["X"] = _mask("x", 0xFFFFFF)
        registers["Y"] = _mask("y", 0xFFFFFF)
        registers["U"] = _mask("u", 0xFFFFFF)
        registers["S"] = _mask("s", 0xFFFFFF)

        f_val = _mask("f", 0xFF)
        registers["F"] = f_val
        registers["FC"] = f_val & 0x01
        registers["FZ"] = (f_val >> 1) & 0x01

        return registers

    def _collect_trace_registers_legacy(self) -> Dict[str, int]:
        """Fallback register collector using per-register accessor calls."""

        registers: Dict[str, int] = {}
        for reg in self._TRACE_REGISTERS:
            try:
                registers[reg.name] = int(self.cpu.regs.get(reg))
            except Exception:
                continue
        return registers

    def _next_memory_trace_units(self) -> Optional[int]:
        """Reserve Perfetto clock units for a memory write within the current opcode."""

        if not (self._new_trace_enabled and new_tracer.enabled):
            return None
        instr_index = self._active_trace_instruction
        if instr_index is None:
            return None
        # Space out memory writes within the same instruction.
        self._trace_substep += 1
        return instr_index * self._trace_units_per_instruction + self._trace_substep

    def _trace_execution(self, pc: int, opcode: Optional[int]):
        payload: Dict[str, Any] = {"pc": f"0x{pc:06X}"}
        if opcode is not None:
            payload["opcode"] = f"0x{opcode:02X}"
        payload.update(self._build_register_annotations())
        if new_tracer.enabled:
            new_tracer.instant("Execution", f"Exec@0x{pc:06X}", payload)
        else:
            trace_dispatcher.record_instant("Execution", f"Exec@0x{pc:06X}", payload)

    def _update_perfetto_counters(self):
        trace_dispatcher.record_counter("cycles", self.cycle_count)
        trace_dispatcher.record_counter("call_depth", self.call_depth)
        trace_dispatcher.record_counter("instructions", self.instruction_count)
        trace_dispatcher.record_counter(
            "stack_pointer", self.cpu.regs.get(RegisterName.S)
        )
        trace_dispatcher.record_counter(
            "read_ops", self.memory_read_count, thread="Memory"
        )
        trace_dispatcher.record_counter(
            "write_ops", self.memory_write_count, thread="Memory"
        )

    def _trace_control_flow(self, pc_before: int, eval_info):
        instr = eval_info.instruction
        pc_after = self.cpu.regs.get(RegisterName.PC)

        if isinstance(instr, CALL):
            dest_addr = instr.dest_addr(pc_before)
            annotations = self._build_register_annotations()
            trace_dispatcher.begin_function(
                "CPU",
                dest_addr,
                pc_before,
                f"func_0x{dest_addr:05X}",
                annotations=annotations,
            )
            self.call_depth += 1
            trace_dispatcher.record_instant(
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
            trace_dispatcher.end_function("CPU", pc_before)
            self.call_depth = max(0, self.call_depth - 1)
            instr_name = type(instr).__name__
            if instr_name == "RETI" and self._interrupt_stack:
                flow_id = self._interrupt_stack.pop()
                trace_dispatcher.end_flow("CPU", flow_id, f"RETI@0x{pc_before:06X}")
            payload = {
                "at": f"0x{pc_before:06X}",
                "type": instr_name.lower(),
                "depth": self.call_depth,
            }
            payload.update(self._build_register_annotations())
            trace_dispatcher.record_instant("CPU", "return", payload)
            self._pop_display_trace(ret_depth)

        elif isinstance(instr, IR):
            vector_addr = self.memory.read_long(0xFFFFA)
            interrupt_id = self._next_interrupt_id
            self._next_interrupt_id += 1
            self._interrupt_stack.append(interrupt_id)
            trace_dispatcher.begin_flow("CPU", interrupt_id, f"IR@0x{pc_before:06X}")
            annotations = self._build_register_annotations()
            trace_dispatcher.begin_function(
                "CPU",
                vector_addr,
                pc_before,
                f"int_0x{vector_addr:05X}",
                annotations=annotations,
            )
            self.call_depth += 1
            trace_dispatcher.record_instant(
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
                trace_dispatcher.record_instant("CPU", "jump", trace_data)

    # ------------------------------------------------------------------ #
    # LCD write tracing helpers

    def _record_lcd_trace_event(self, event: Dict[str, Any]) -> None:
        if self._lcd_trace_limit <= 0:
            return
        if len(self._lcd_trace_events) >= self._lcd_trace_limit:
            self._lcd_trace_truncated = True
            return
        record = dict(event)
        record.setdefault("pc", event.get("pc"))
        record["instruction_index"] = self.instruction_count
        self._lcd_trace_events.append(record)

    def get_lcd_trace_events(self) -> List[Dict[str, Any]]:
        return list(self._lcd_trace_events)

    def save_lcd_trace(self, path: Optional[str] = None) -> Optional[Path]:
        target = Path(path) if path else self._lcd_trace_path
        if target is None or not self._lcd_trace_events:
            return None
        payload = {
            "backend": getattr(self.cpu, "backend", "python"),
            "event_count": len(self._lcd_trace_events),
            "truncated": self._lcd_trace_truncated,
            "limit": self._lcd_trace_limit,
            "events": self._lcd_trace_events,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2))
        return target

    def notify_lcd_interrupt(
        self, address: int, value: int, pc: Optional[int] = None
    ) -> None:
        """Handle a pure-Rust LCD write that should nudge the KEY interrupt."""

        if not getattr(self, "_llama_pure_lcd", False):
            return
        if not getattr(self, "_kb_irq_enabled", True):
            return
        self._ensure_key_irq_mask()
        self._set_isr_bits(int(ISRFlag.KEYI))
        self._irq_pending = True
        self._irq_source = IRQSource.KEY
        try:
            self.irq_counts["KEY"] += 1
            self.irq_counts["total"] += 1
        except Exception:
            pass
        if IRQ_DEBUG_ENABLED:
            pc_str = f"0x{pc:06X}" if pc is not None else "N/A"
            _log_irq_debug(
                f"lcd-notify irq addr=0x{address:06X} value=0x{value:02X} pc={pc_str}"
            )

    def _seed_interrupt_mask(
        self, imr_value: int = 0x43, isr_value: int = 0x00
    ) -> None:
        imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
        isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
        try:
            self.memory.write_byte(imr_addr, imr_value & 0xFF)
            self.memory.write_byte(isr_addr, isr_value & 0xFF)
        except Exception:
            pass

    def _ensure_key_irq_mask(self) -> None:
        if not getattr(self, "_llama_pure_lcd", False):
            return
        try:
            imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
            current = self.memory.read_byte(imr_addr) & 0xFF
        except Exception:
            return
        required = int(IMRFlag.IRM | IMRFlag.KEY)
        if (current & required) == required:
            return
        try:
            self.memory.write_byte(imr_addr, (current | required) & 0xFF)
        except Exception:
            pass

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
        if result and self._kb_irq_enabled:
            events = self.keyboard.scan_tick()
            if events:
                self._kb_irq_count += len(events)
            self._ensure_key_irq_mask()
            self._set_isr_bits(int(ISRFlag.KEYI))
            self._irq_pending = True
            self._irq_source = IRQSource.KEY
            try:
                self._last_kol = getattr(self.keyboard, "kol_value", self._last_kol)
                self._last_koh = getattr(self.keyboard, "koh_value", self._last_koh)
                self._kb_strobe_count = getattr(
                    self.keyboard, "strobe_count", self._kb_strobe_count
                )
                hist = getattr(self.keyboard, "column_histogram", None)
                if hist:
                    for idx, val in enumerate(hist):
                        if idx < len(self._kb_col_hist):
                            self._kb_col_hist[idx] = val
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
        if mask & int(ISRFlag.KEYI):
            self._ensure_key_irq_mask()
        if IRQ_DEBUG_ENABLED:
            pc = self.cpu.regs.get(RegisterName.PC)
            _log_irq_debug(
                f"set_isr_bits mask=0x{mask:02X} prev=0x{val & 0xFF:02X} new=0x{(val | (mask & 0xFF)) & 0xFF:02X} pc=0x{pc:06X}"
            )

    def _tick_timers(self) -> None:
        """Rough timer emulation: set ISR bits periodically and arm IRQ."""
        fired_sources = tuple(self._scheduler.advance(self.cycle_count))
        if not fired_sources:
            return

        key_events: List[MatrixEvent] = []
        for source in fired_sources:
            if source is TimerSource.MTI:
                # When the Rust bridge is active, the keyboard scan runs inside the
                # Rust instruction loop; skip the Python-side scan to avoid double
                # counting or flag changes.
                if not getattr(self.keyboard, "_bridge_enabled", False):
                    key_events = self.keyboard.scan_tick()
                self._set_isr_bits(int(ISRFlag.MTI))
                self._irq_pending = True
                self._irq_source = IRQSource.MTI
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"timer fired source=MTI cycle={self.cycle_count} next_mti={self._scheduler.next_mti}"
                    )
            elif source is TimerSource.STI:
                self._set_isr_bits(int(ISRFlag.STI))
                self._irq_pending = True
                self._irq_source = IRQSource.STI
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"timer fired source=STI cycle={self.cycle_count} next_sti={self._scheduler.next_sti}"
                    )

        if key_events:
            if self._kb_irq_enabled:
                self._set_isr_bits(int(ISRFlag.KEYI))
                self._irq_pending = True
                self._irq_source = IRQSource.KEY
                self._kb_irq_count += len(key_events)
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"key_events_irq count={len(key_events)} cycle={self.cycle_count}"
                    )

        if new_tracer.enabled and self._irq_source is not None:
            new_tracer.instant(
                "CPU",
                "TimerIRQ",
                {"ic": self.cycle_count, "src": self._irq_source.name},
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

    @property
    def _timer_enabled(self) -> bool:
        return self._scheduler.enabled

    @_timer_enabled.setter
    def _timer_enabled(self, value: bool) -> None:
        self._scheduler.enabled = bool(value)

    @property
    def _timer_mti_period(self) -> int:
        return self._scheduler.mti_period

    @_timer_mti_period.setter
    def _timer_mti_period(self, value: int) -> None:
        self._scheduler.mti_period = int(value)

    @property
    def _timer_sti_period(self) -> int:
        return self._scheduler.sti_period

    @_timer_sti_period.setter
    def _timer_sti_period(self, value: int) -> None:
        self._scheduler.sti_period = int(value)

    @property
    def _timer_next_mti(self) -> int:
        return self._scheduler.next_mti

    @_timer_next_mti.setter
    def _timer_next_mti(self, value: int) -> None:
        self._scheduler.next_mti = int(value)

    @property
    def _timer_next_sti(self) -> int:
        return self._scheduler.next_sti

    @_timer_next_sti.setter
    def _timer_next_sti(self, value: int) -> None:
        self._scheduler.next_sti = int(value)

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

        addresses: List[int] = []
        try:
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir.parent / "rom-analysis" / "bnida.json"
            if not candidate.exists():
                candidate = base_dir / "rom-analysis" / "bnida.json"
            if candidate.exists():
                data = json.loads(candidate.read_text())
                addresses = sorted(
                    int(addr)
                    for addr in data.get("functions", [])
                    if isinstance(addr, int)
                )
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
        try:
            payload["registers"] = {
                "X": self.cpu.regs.get(RegisterName.X),
                "Y": self.cpu.regs.get(RegisterName.Y),
                "BA": self.cpu.regs.get(RegisterName.BA),
                "S": self.cpu.regs.get(RegisterName.S),
            }
        except Exception:
            payload["registers"] = {}
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
        # Single keyboard implementation: use keyboard handler
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
                self._kb_strobe_count = getattr(
                    self.keyboard, "strobe_count", self._kb_strobe_count
                )
                hist = getattr(self.keyboard, "column_histogram", None)
                if hist:
                    for idx, val in enumerate(hist):
                        if idx < len(self._kb_col_hist):
                            self._kb_col_hist[idx] = val
            except Exception:
                pass

        return result

    def _keyboard_write_handler(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        offset = address - INTERNAL_MEMORY_START
        self._track_imem_access(offset, "writes", cpu_pc)

        if new_tracer.enabled and offset in (KOL, KOH):
            new_tracer.instant(
                "I/O", "KB_ColumnStrobe", {"addr": offset, "value": value & 0xFF}
            )

        # Update keyboard state via keyboard handler
        self.keyboard.handle_register_write(offset, value)

        # Cache register values and metrics for diagnostics
        try:
            self._last_kol = getattr(self.keyboard, "kol_value", self._last_kol)
            self._last_koh = getattr(self.keyboard, "koh_value", self._last_koh)
            self._kb_strobe_count = getattr(
                self.keyboard, "strobe_count", self._kb_strobe_count
            )
            hist = getattr(self.keyboard, "column_histogram", None)
            if hist:
                for idx, val in enumerate(hist):
                    if idx < len(self._kb_col_hist):
                        self._kb_col_hist[idx] = val
            self._last_kil_columns = list(self.keyboard.get_active_columns())
        except Exception:
            pass

    # Note: LCC write handler not required with the keyboard handler

    def _dump_internal_memory(self, pc: int):
        internal_mem = self.memory.get_internal_memory_bytes()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"internal_memory_dump_{timestamp}_pc_{pc:06X}.bin"
        path = os.path.join(self.MEMORY_DUMP_DIR, filename)
        with open(path, "wb") as f:
            f.write(internal_mem)
        print(f"\nInternal memory dumped to: {path}")
        trace_dispatcher.record_instant(
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

    def _handle_imem_access(
        self, pc: int, reg_name: Optional[str], access_type: str, value: int
    ) -> None:
        """Forward IMEM register notifications to peripherals and tracing."""

        if reg_name:
            self.peripherals.handle_imem_access(pc, reg_name, access_type, value)

        if self.disasm_trace_enabled and reg_name:
            self._on_imem_register_access(pc, reg_name, access_type, value)

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


def _stack_snapshot_range() -> tuple[int, int] | None:
    global _STACK_SNAPSHOT_RANGE
    if _STACK_SNAPSHOT_RANGE is not None:
        return _STACK_SNAPSHOT_RANGE
    raw = os.getenv("STACK_SNAPSHOT_RANGE")
    if not raw:
        return None
    try:
        start_str, end_str = raw.split("-", 1)
        start = int(start_str, 0)
        end = int(end_str, 0)
    except ValueError:
        try:
            start = end = int(raw, 0)
        except ValueError:
            return None
    if start > end:
        start, end = end, start
    _STACK_SNAPSHOT_RANGE = (start & 0xFFFFFF, end & 0xFFFFFF)
    return _STACK_SNAPSHOT_RANGE


def _stack_snapshot_len() -> int:
    global _STACK_SNAPSHOT_LEN
    if _STACK_SNAPSHOT_LEN is not None:
        return _STACK_SNAPSHOT_LEN
    raw = os.getenv("STACK_SNAPSHOT_LEN")
    length = 10
    if raw:
        try:
            candidate = int(raw, 0)
            if candidate > 0:
                length = candidate
        except ValueError:
            pass
    _STACK_SNAPSHOT_LEN = length
    return length


def _log_stack_snapshot_emulator(emu: "PCE500Emulator", pc: int) -> None:
    rng = _stack_snapshot_range()
    if not rng:
        return
    start, end = rng
    if not (start <= (pc & 0xFFFFFF) <= end):
        return
    try:
        s = int(emu.cpu.regs.get(RegisterName.S)) & 0xFFFFFF
    except Exception:
        return
    length = _stack_snapshot_len()
    bytes_: list[int] = []
    for offset in range(length):
        try:
            bytes_.append(emu.memory.read_byte((s + offset) & 0xFFFFFF))
        except Exception:
            bytes_.append(0)
    byte_str = " ".join(f"{b:02X}" for b in bytes_)
    print(
        f"[stack-snapshot] backend={emu.cpu.backend} pc=0x{pc:06X} S=0x{s:06X} bytes={byte_str}"
    )
