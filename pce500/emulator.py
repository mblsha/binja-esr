"""Simplified PC-E500 emulator combining machine and emulator functionality."""

import json
import operator
import os
import tempfile
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

# Import the SC62015 emulator
from sc62015.pysc62015 import CPU, RegisterName, Registers
from sc62015.pysc62015.emulator import fetch_validated_vector_transfer
from sc62015.pysc62015.instr.instructions import (
    CALL,
    RetInstruction,
    JumpInstruction,
    IR,
)
from sc62015.pysc62015.instr.opcodes import (
    ENTRY_POINT_ADDR,
    IMEMRegisters,
    INTERRUPT_VECTOR_ADDR,
)
from sc62015.pysc62015.constants import IMRFlag, ISRFlag
from sc62015.pysc62015.stepper import CPURegistersSnapshot

from .memory import (
    PCE500Memory,
    MemoryOverlay,
    IMEM_ACCESS_HISTORY_LIMIT,
    SUPPORTED_MEMORY_CARD_SIZES,
)
from .display import HD61202Controller
from .keyboard_handler import PCE500KeyboardHandler as KeyboardHandler
from .keyboard_matrix import MatrixEvent
from .tracing import trace_dispatcher
from .tracing.perfetto_tracing import tracer as new_tracer, perf_trace
from .scheduler import TimerScheduler, TimerSource
from .peripherals import PeripheralManager

# Default timer periods in cycles (match Rust core timing defaults).
DEFAULT_CPU_HZ = 1_024_000
MTI_PERIOD_CYCLES_DEFAULT = DEFAULT_CPU_HZ // 1000 * 2  # 2 ms tick
STI_PERIOD_CYCLES_DEFAULT = DEFAULT_CPU_HZ // 2  # 0.5 s tick

SNAPSHOT_MAGIC = "pc-e500.snapshot"
SNAPSHOT_VERSION = 4
_SNAPSHOT_JSON_MAX_BYTES = 4 * 1024 * 1024
_SNAPSHOT_LCD_MAX_BYTES = 0x100000
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


def _snapshot_json_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Reject duplicate JSON members instead of accepting the last value."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"snapshot JSON contains duplicate member {key!r}")
        result[key] = value
    return result


def _snapshot_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"snapshot {label} must be an object")
    return value


def _snapshot_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"snapshot {label} must be an array")
    return value


def _snapshot_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"snapshot {label} must be a boolean")
    return value


def _snapshot_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 64) - 1,
) -> int:
    """Extract an exact JSON integer without accepting booleans or floats."""

    if isinstance(value, bool):
        raise TypeError(f"snapshot {label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"snapshot {label} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"snapshot {label} must be between {minimum} and {maximum}")
    return result


def _snapshot_range(value: object, label: str) -> tuple[int, int]:
    if isinstance(value, dict):
        mapping = _snapshot_mapping(value, label)
        if set(mapping) != {"start", "size"}:
            raise ValueError(f"snapshot {label} must contain exactly start and size")
        start = _snapshot_int(mapping["start"], f"{label}.start", maximum=0xFFFFFF)
        size = _snapshot_int(mapping["size"], f"{label}.size", maximum=0x1000000)
        return start, size
    items = _snapshot_list(value, label)
    if len(items) != 2:
        raise ValueError(f"snapshot {label} must contain exactly two integers")
    return (
        _snapshot_int(items[0], f"{label}[0]", maximum=0xFFFFFF),
        _snapshot_int(items[1], f"{label}[1]", maximum=0x1000000),
    )


def _snapshot_ranges(value: object, label: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for index, item in enumerate(_snapshot_list(value, label)):
        pair = _snapshot_list(item, f"{label}[{index}]")
        if len(pair) != 2:
            raise ValueError(
                f"snapshot {label}[{index}] must contain exactly two addresses"
            )
        start = _snapshot_int(pair[0], f"{label}[{index}][0]", maximum=0xFFFFFF)
        end = _snapshot_int(pair[1], f"{label}[{index}][1]", maximum=0xFFFFFF)
        if end < start:
            raise ValueError(f"snapshot {label}[{index}] has an inverted range")
        result.append((start, end))
    return tuple(result)


_SNAPSHOT_BASE_ENTRY_SIZES = {
    "registers.bin": sum(width for _, width in _SNAPSHOT_REGISTER_LAYOUT),
    "external_ram.bin": 0x100000,
    "internal_ram.bin": 0x8000,
    "imem.bin": 0x100,
}
_SNAPSHOT_NATIVE_BASE_ENTRIES = frozenset(
    {"snapshot.json", *_SNAPSHOT_BASE_ENTRY_SIZES}
)
_SNAPSHOT_V4_ENTRIES = frozenset(
    {*_SNAPSHOT_NATIVE_BASE_ENTRIES, "lcd_vram.bin", "memory_card.bin"}
)


def _snapshot_archive_infos(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    """Index archive members without quadratic duplicate-name checks."""

    infos: dict[str, zipfile.ZipInfo] = {}
    duplicates: list[str] = []
    duplicate_names: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if name in infos:
            if name not in duplicate_names:
                duplicate_names.add(name)
                duplicates.append(name)
        else:
            infos[name] = info
    if duplicates:
        raise ValueError(f"snapshot archive contains duplicate entries: {duplicates}")
    return infos


def _read_snapshot_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    exact_size: int | None = None,
    maximum_size: int | None = None,
) -> bytes:
    """Read one member with both declared and actual decompressed-size bounds."""

    if exact_size is not None:
        if info.file_size != exact_size:
            raise ValueError(
                f"snapshot entry {info.filename!r} must contain exactly "
                f"{exact_size} bytes (declared {info.file_size})"
            )
        limit = exact_size
    elif maximum_size is not None:
        if info.file_size > maximum_size:
            raise ValueError(
                f"snapshot entry {info.filename!r} exceeds the "
                f"{maximum_size}-byte limit"
            )
        limit = maximum_size
    else:
        raise AssertionError("snapshot member read requires a size contract")

    with archive.open(info, "r") as member:
        payload = member.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(
            f"snapshot entry {info.filename!r} exceeds its decompressed-size limit"
        )
    if exact_size is not None and len(payload) != exact_size:
        raise ValueError(
            f"snapshot entry {info.filename!r} must contain exactly "
            f"{exact_size} bytes (read {len(payload)})"
        )
    return payload


def _read_snapshot_archive(
    path: str | Path,
    *,
    native_for_patch: bool = False,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Strictly read a PCE v4 archive or a native v4 patch input.

    Native LLAMA output uses v4 with the same fixed-size base
    entries and an LCD member exactly when its metadata carries LCD state.  The
    caller patches that already-strict native image into the PCE v4 envelope.
    """

    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        infos = _snapshot_archive_infos(archive)
        names = set(infos)

        allowed = (
            _SNAPSHOT_NATIVE_BASE_ENTRIES | {"lcd_vram.bin"}
            if native_for_patch
            else _SNAPSHOT_V4_ENTRIES
        )
        unexpected = sorted(names - allowed)
        if unexpected:
            raise ValueError(
                f"snapshot archive contains unexpected entries: {unexpected}"
            )
        if "snapshot.json" not in infos:
            raise ValueError("snapshot archive is missing entries: ['snapshot.json']")

        metadata_payload = _read_snapshot_member(
            archive,
            infos["snapshot.json"],
            maximum_size=_SNAPSHOT_JSON_MAX_BYTES,
        )
        metadata = _snapshot_mapping(
            json.loads(
                metadata_payload,
                object_pairs_hook=_snapshot_json_object_pairs,
            ),
            "metadata",
        )

        if not native_for_patch:
            # Inspect the bounded, duplicate-free metadata before applying the
            # v4 entry-set contract.  This preserves a clear compatibility
            # error for genuine v3 archives, which legitimately lack the v4
            # memory-card member and payload.
            version = _snapshot_int(
                metadata.get("version"),
                "version",
                maximum=(1 << 32) - 1,
            )
            if version != SNAPSHOT_VERSION:
                raise ValueError("Unsupported snapshot version")
            missing = sorted(_SNAPSHOT_V4_ENTRIES - names)
            if missing:
                raise ValueError(f"snapshot archive is missing entries: {missing}")
        else:
            expected = set(_SNAPSHOT_NATIVE_BASE_ENTRIES)
            if metadata.get("lcd") is not None:
                expected.add("lcd_vram.bin")
            missing = sorted(expected - names)
            if missing:
                raise ValueError(f"snapshot archive is missing entries: {missing}")
            unexpected_for_shape = sorted(names - expected)
            if unexpected_for_shape:
                raise ValueError(
                    "snapshot archive contains unexpected entries: "
                    f"{unexpected_for_shape}"
                )

        for name, exact_size in _SNAPSHOT_BASE_ENTRY_SIZES.items():
            info = infos.get(name)
            if info is None:
                raise ValueError(f"snapshot archive is missing entries: [{name!r}]")
            if info.file_size != exact_size:
                raise ValueError(
                    f"snapshot entry {name!r} must contain exactly {exact_size} "
                    f"bytes (declared {info.file_size})"
                )

        lcd_payload_size = _snapshot_int(
            metadata.get("lcd_payload_size", 0),
            "lcd_payload_size",
            maximum=_SNAPSHOT_LCD_MAX_BYTES,
        )
        if "lcd_vram.bin" in infos:
            if infos["lcd_vram.bin"].file_size != lcd_payload_size:
                raise ValueError(
                    "snapshot entry 'lcd_vram.bin' size disagrees with lcd_payload_size"
                )
        elif lcd_payload_size != 0:
            raise ValueError("snapshot LCD metadata requires lcd_vram.bin")

        card_capacity: int | None = None
        if not native_for_patch:
            card_metadata = _snapshot_mapping(
                metadata.get("memory_card"), "memory_card"
            )
            card_capacity = _snapshot_int(
                card_metadata.get("capacity"),
                "memory_card.capacity",
                maximum=max(SUPPORTED_MEMORY_CARD_SIZES),
            )
            if card_capacity not in SUPPORTED_MEMORY_CARD_SIZES:
                raise ValueError("snapshot memory_card.capacity is unsupported")
            if infos["memory_card.bin"].file_size != card_capacity:
                raise ValueError(
                    "snapshot memory_card.bin size disagrees with memory_card.capacity"
                )

        entries = {"snapshot.json": metadata_payload}
        for name, exact_size in _SNAPSHOT_BASE_ENTRY_SIZES.items():
            entries[name] = _read_snapshot_member(
                archive, infos[name], exact_size=exact_size
            )
        if "lcd_vram.bin" in infos:
            entries["lcd_vram.bin"] = _read_snapshot_member(
                archive,
                infos["lcd_vram.bin"],
                exact_size=lcd_payload_size,
            )
        if card_capacity is not None:
            entries["memory_card.bin"] = _read_snapshot_member(
                archive,
                infos["memory_card.bin"],
                exact_size=card_capacity,
            )
    return metadata, entries


class IRQSource(Enum):
    # Enum values store ISR bit index directly
    MTI = 0  # Main timer interrupt → ISR bit 0
    STI = 1  # Sub timer interrupt  → ISR bit 1
    KEY = 2  # Keyboard interrupt   → ISR bit 2
    ONK = 3  # On-key interrupt     → ISR bit 3


_IRQ_PRIORITY = (
    IRQSource.ONK,
    IRQSource.KEY,
    IRQSource.STI,
    IRQSource.MTI,
)


def _highest_pending_irq_source(isr: int) -> Optional[IRQSource]:
    """Return the highest-priority low interrupt source selected by the ROM."""
    for source in _IRQ_PRIORITY:
        if isr & (1 << source.value):
            return source
    return None


# Define constants locally to avoid heavy imports
INTERNAL_MEMORY_START = 0x100000
KOL, KOH, KIL = IMEMRegisters.KOL, IMEMRegisters.KOH, IMEMRegisters.KIL
IRQ_STACK_TRACE_ENABLED = False
PYTHON_PC_TRACE_ENABLED = False

_STACK_SNAPSHOT_RANGE: tuple[int, int] | None = None
_STACK_SNAPSHOT_LEN: int | None = None


def _env_flag(name: str) -> bool:
    _ = name
    return False


IRQ_DEBUG_ENABLED = False


def _log_irq_debug(message: str) -> None:
    if IRQ_DEBUG_ENABLED:
        print(f"[irq-debug] {message}")


def _trace_probe_pc_and_opcode(emu):
    """Extract PC and opcode for tracing without side effects."""
    pc = emu.cpu.regs.get(RegisterName.PC) if hasattr(emu, "cpu") else None
    opcode = None
    if pc is not None and hasattr(emu, "memory"):
        try:
            opcode = emu.memory.peek_byte_for_preflight(pc, pc) & 0xFF
        except Exception:
            pass
    return pc, opcode


class PCE500Emulator:
    """PC-E500 emulator with integrated machine configuration."""

    _TRACE_UNITS_PER_INSTRUCTION = 1
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
        memory_card_present: bool = True,
        keyboard_columns_active_high: bool = True,
        enable_new_tracing: bool = False,
        trace_path: str = "pc-e500.perfetto-trace",
        disasm_trace: bool = False,
        enable_display_trace: bool = False,
        display_trace_functions: Optional[Dict[int, str]] = None,
        display_trace_event_limit: int = 2048,
        lcd_trace_file: Optional[str] = None,
        lcd_trace_event_limit: int = 50000,
        timer_scale: float = 1.0,
    ):
        # Avoid leaking a previously-enabled perfetto tracer into runs that do not
        # request tracing.
        if not (perfetto_trace or enable_new_tracing) and new_tracer.enabled:
            try:
                new_tracer.safe_stop()
            except Exception:
                pass

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
        self.memory.set_memory_card_present(bool(memory_card_present))

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
        self._timer_scale = float(timer_scale) if timer_scale else 1.0
        if self._timer_scale <= 0:
            self._timer_scale = 1.0

        # Keyboard implementation parameterised for column polarity
        self.keyboard = KeyboardHandler(
            self.memory, columns_active_high=keyboard_columns_active_high
        )
        # Match Rust PC-E500: assert KEYI on first visible scan.
        try:
            if hasattr(self.keyboard, "_matrix"):
                self.keyboard._matrix.press_threshold = 1
        except Exception:
            pass
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
        # An explicit backend request is part of the validation contract.  In
        # particular, a missing LLAMA extension must not turn a Python-vs-Rust
        # parity run into Python-vs-Python while still reporting success.
        self.cpu = CPU(
            self.memory,
            reset_on_init=True,
            backend=backend,
            timer_scale=self._timer_scale,
        )

        self.memory.set_cpu(self.cpu)
        cpu_backend = getattr(self.cpu, "backend", None)
        if cpu_backend == "llama":
            setter = getattr(self.cpu, "set_timer_scale", None)
            if callable(setter):
                try:
                    setter(self._timer_scale)
                except Exception:
                    pass

        # Keep keyboard overlays active even on LLAMA so Python handlers can trace KIO writes.
        disable_keyboard_overlay = False
        self.memory.set_keyboard_handler(
            self._keyboard_read_handler,
            self._keyboard_write_handler,
            enable_overlay=not disable_keyboard_overlay,
        )
        # Ensure strobing is active in LLAMA path even when overlays are disabled.
        self.keyboard.force_strobe_enabled = True
        # Even with Python KIO overlays enabled, ON-key mutations must go
        # through LLAMA's transactional native keyboard path.  Matrix-key
        # forwarding remains controlled independently by ``enable_overlay``.
        self.keyboard.set_bridge_cpu(
            self.cpu if cpu_backend == "llama" else None, disable_keyboard_overlay
        )

        # Keep LCD overlays enabled so Python snapshots and traces stay in sync with LLAMA.
        disable_overlay = False
        enable_overlay = not disable_overlay
        self.memory.set_lcd_controller(self.lcd, enable_overlay=enable_overlay)
        self._llama_pure_lcd = disable_overlay
        self._llama_lcd_write = (
            getattr(self.memory, "_llama_lcd_write", None) if disable_overlay else None
        )

        # Set performance tracer for SC62015 integration if available
        if new_tracer.enabled:
            self.memory.set_perf_tracer(new_tracer)
            # LLAMA core understands set_perfetto_trace(path); hand it the file path
            # so Rust-side IMEM tagging can emit into the same trace file.
            try:
                if getattr(self.cpu, "backend", None) == "llama":
                    self.cpu.set_perfetto_trace(trace_path)
                    # Also mirror into the Rust static tracer for KIO logging.
                    impl = self.cpu.unwrap()
                    setter = getattr(impl, "set_perfetto_trace", None)
                    if callable(setter):
                        setter(trace_path)
            except Exception:
                pass
            # Seed a Rust-side IRQ event to mark tracing start for cadence parity.
            try:
                from _sc62015_rustcore import record_irq_event as rust_irq_event

                rust_irq_event(
                    "Trace_Start",
                    {"pc": int(self.cpu.regs.get(RegisterName.PC)) & 0xFFFFFF},
                )
            except Exception:
                pass

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
        # Track last-observed IMR/ISR values for trace diffs.
        self._last_imem_values: Dict[str, int] = {}
        rust_trace_path = str(trace_path)
        if enable_new_tracing:
            rust_trace_path = f"{rust_trace_path}.rust"
        self.perfetto_enabled = perfetto_trace
        self._new_trace_enabled = perfetto_trace and enable_new_tracing
        if self.perfetto_enabled:
            trace_dispatcher.start_trace(trace_path)
            self.lcd.set_perfetto_enabled(True)
            self.memory.set_perfetto_enabled(True)
            if getattr(self.cpu, "backend", None) == "llama":
                try:
                    self.cpu.set_perfetto_trace(rust_trace_path)
                except Exception:
                    pass
        elif len(list(trace_dispatcher.observers())) > 1:
            trace_dispatcher.start_trace(trace_path)

        # New tracing system
        self._trace_path = trace_path
        self._rust_trace_path = rust_trace_path
        self._trace_instr_count = 0
        self._trace_units_per_instruction = self._TRACE_UNITS_PER_INSTRUCTION
        self._trace_substep = 0
        self._active_trace_instruction: Optional[int] = None
        if self._new_trace_enabled:
            if not new_tracer.enabled:
                new_tracer.start(self._trace_path)
            # Keep Perfetto timestamps aligned to instruction indices:
            # 1 instruction tick == 1us in the final Perfetto UI.
            new_tracer.set_manual_clock_mode(True, tick_ns=1_000)
            self.memory.set_perf_tracer(new_tracer)
            # Ensure the Rust LLAMA core writes into the same Perfetto trace file.
            try:
                if getattr(self.cpu, "backend", None) == "llama":
                    self.cpu.set_perfetto_trace(self._rust_trace_path)
            except Exception:
                pass

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
        # PC-E500 scans the key matrix each instruction (not just on MTI).
        self._scan_on_timer = False
        self._irq_pending = False
        self._in_interrupt = False
        self._kb_irq_count = 0
        self._key_irq_latched = False
        self._timer_scale = float(timer_scale) if timer_scale else 1.0
        if self._timer_scale <= 0:
            self._timer_scale = 1.0
        effective_timer_scale = self._timer_scale if cpu_backend == "llama" else 1.0
        self._scheduler = TimerScheduler(
            mti_period=max(1, int(MTI_PERIOD_CYCLES_DEFAULT * effective_timer_scale)),
            sti_period=max(1, int(STI_PERIOD_CYCLES_DEFAULT * effective_timer_scale)),
        )
        self._irq_source: Optional["IRQSource"] = None
        # Pending-IRQ delivery mutates host memory before all later reads and
        # register writes can be proven to succeed.  If one of those operations
        # fails, only a full reset is safe; retrying step() could otherwise push
        # a second, overlapping interrupt frame.
        self._poisoned: Optional[str] = None
        # Fast mode: minimize step() overhead to run many instructions
        self.fast_mode = False

        # Peripheral adapters and IMEM callbacks
        self.peripherals = PeripheralManager(self.memory, self._scheduler)
        self._default_imem_access_callback = self._handle_imem_access
        self.memory.set_imem_access_callback(self._default_imem_access_callback)
        self.memory._mark_snapshot_baseline()
        try:
            # Tap into keyboard scan events to surface KEYI progression in logs/perfetto.
            if hasattr(self.keyboard, "_matrix"):
                matrix = self.keyboard._matrix  # type: ignore[attr-defined]
                matrix._trace_hook = self._trace_key_event  # type: ignore[attr-defined]
                # Dedicated hook for KIO reads/writes so we can capture PC/op index.
                matrix._kio_trace_hook = self._trace_kio_access  # type: ignore[attr-defined]
                # Pass perfetto tracer down to matrix so trace_kio can emit directly.
                try:
                    if new_tracer.enabled:
                        matrix._perf_tracer = new_tracer  # type: ignore[attr-defined]
                    elif self.perfetto_enabled or trace_dispatcher.has_observers():
                        # Use the dispatcher as a tracer proxy for legacy perfetto mode.
                        matrix._perf_tracer = trace_dispatcher  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    def _irq_perfetto_track(self, source: Optional["IRQSource"]) -> str:
        if source in (IRQSource.MTI, IRQSource.STI):
            return "irq.timer"
        if source in (IRQSource.KEY, IRQSource.ONK):
            return "irq.key"
        return "irq.misc"

    def _trace_irq_instant(
        self,
        name: str,
        source: Optional["IRQSource"],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a lightweight IRQ instant to Perfetto/new tracer."""

        if not (new_tracer.enabled or self.perfetto_enabled):
            return
        data = dict(payload or {})
        if source is not None:
            data.setdefault("src", source.name)
        if new_tracer.enabled:
            new_tracer.instant(self._irq_perfetto_track(source), name, data)
        elif self.perfetto_enabled or trace_dispatcher.has_observers():
            trace_dispatcher.record_instant(
                self._irq_perfetto_track(source), name, data
            )
        try:
            from _sc62015_rustcore import record_irq_event as rust_irq_event

            rust_payload = {}
            for key, value in data.items():
                try:
                    if value is None:
                        rust_payload[key] = 0
                    elif isinstance(value, bool):
                        rust_payload[key] = int(value)
                    elif isinstance(value, (int, float)):
                        rust_payload[key] = int(value)
                    elif isinstance(value, str):
                        rust_payload[key] = int(value, 0)
                except (TypeError, ValueError):
                    continue
            rust_irq_event(name, rust_payload)
        except Exception:
            pass

    def _trace_kio_access(
        self, name: str, kol: int, koh: int, kil: int, *, pc: Optional[int] = None
    ) -> bool:
        """Log KIO accesses (KOL/KOH/KIL) with best-effort PC/op index."""

        if not (
            new_tracer.enabled
            or self.perfetto_enabled
            or trace_dispatcher.has_observers()
        ):
            return False

        eff_pc = pc
        if eff_pc is None:
            try:
                eff_pc = self.cpu.regs.get(RegisterName.PC)
            except Exception:
                eff_pc = None

        payload: Dict[str, Any] = {
            "kol": kol & 0xFF,
            "koh": koh & 0x0F,
            "kil": kil & 0xFF,
        }
        if eff_pc is not None:
            payload["pc"] = eff_pc & 0xFFFFFF
        op_index = getattr(self, "_active_trace_instruction", None)
        if op_index is not None:
            payload["op_index"] = op_index

        try:
            if new_tracer.enabled:
                if getattr(self, "_new_trace_enabled", False):
                    units = self._next_memory_trace_units()
                    setter = getattr(new_tracer, "set_manual_clock_units", None)
                    if units is not None and callable(setter):
                        setter(units)
                new_tracer.instant("KIO", name, payload)
            else:
                trace_dispatcher.record_instant("KIO", name, payload)
        except Exception:
            return False
        return True

    def _trace_key_event(self, col: int, row: int, pressed: bool) -> None:
        """Optional hook from keyboard scan to log KEY events into perfetto."""
        if not (new_tracer.enabled or self.perfetto_enabled):
            return
        payload = {
            "col": col,
            "row": row,
            "pressed": bool(pressed),
            "pc": self.cpu.regs.get(RegisterName.PC),
            "imr": self.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
            & 0xFF,
            "isr": self.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
            & 0xFF,
        }
        if new_tracer.enabled:
            if getattr(self, "_new_trace_enabled", False):
                op_index = getattr(self, "_active_trace_instruction", None)
                if not isinstance(op_index, int):
                    op_index = int(getattr(self, "instruction_count", 0))
                new_tracer.set_manual_clock_units(op_index)
            new_tracer.instant("irq.key", "KeyScanEvent", payload)
        elif self.perfetto_enabled or trace_dispatcher.has_observers():
            trace_dispatcher.record_instant("irq.key", "KeyScanEvent", payload)

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
        if start_address is None or start_address in (
            self.INTERNAL_ROM_START,
            0xC0000,
            0xE0000,
        ):
            self.memory.load_rom(rom_data, start_address=start_address)
        else:
            self.memory.add_rom(start_address, rom_data, "Loaded ROM")

    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        self.memory.load_memory_card(card_data, card_size)

    def expand_ram(self, size: int, start_address: int) -> None:
        self.memory.add_ram(start_address, size, f"RAM Expansion ({size // 1024}KB)")

    def _fetch_validated_vector(
        self, vector_address: int, *, source_pc: int | None = None
    ) -> int:
        """Perform one architectural vector fetch matching a silent preflight."""

        vector_pc = (
            self.cpu.regs.get(RegisterName.PC) if source_pc is None else int(source_pc)
        )
        # Keep every predictable rejection in the silent phase.  Once the
        # factory below starts, any failure is conservatively treated as a
        # failed architectural read because a custom host reader may already
        # have consumed external state.
        self.cpu.preflight_vector_transfer_for_scheduling(
            vector_address,
            source_pc=vector_pc,
        )
        try:
            transfer = fetch_validated_vector_transfer(
                self.memory,
                self.cpu.regs,
                vector_address,
                source_pc=vector_pc,
                require_stability_metadata=True,
            )
            return transfer.consume(self.memory, vector_address, vector_pc)
        except Exception as exc:
            if self._poisoned is None:
                self._poisoned = (
                    "vector fetch may have consumed external state: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise

    @perf_trace("System")
    def reset(self) -> None:
        # Prove RESET can complete before clearing RAM, LCD state, scheduler
        # counters, or wrapper poison.  A noncanonical vector or quarantined
        # target must leave the entire machine available for inspection.
        try:
            self.cpu.prepare_power_on_reset()
        except Exception as exc:
            contract_poison = getattr(self.cpu, "_contract_poisoned", None)
            if contract_poison is not None and self._poisoned is None:
                self._poisoned = (
                    "machine reset vector preparation may have consumed "
                    f"external state: {type(exc).__name__}: {exc}"
                )
            raise
        previous_poison = self._poisoned
        try:
            self._commit_prepared_reset()
        except Exception as exc:
            self.cpu.cancel_prepared_vector_transfer()
            if previous_poison is None:
                self._poisoned = (
                    "machine reset failed after mutation began: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                self._poisoned = previous_poison
            raise

    def _commit_prepared_reset(self) -> None:
        """Commit a machine reset after its immutable vector was prepared."""

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
        self._key_irq_latched = False
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
        # A failed native callback can leave host writes queued after they have
        # already changed Python memory.  Only discard them once every reset
        # component above has succeeded; until then the original poison and its
        # recovery evidence remain authoritative.
        self.memory.discard_deferred_llama_host_writes()
        # Clear wrapper poison only after the complete machine reset succeeds.
        # If any operation above raises, the prior poison remains authoritative.
        self._poisoned = None

    def _execute_instruction_and_flush(self, pc: int):
        """Execute once, then reconcile deferred LLAMA callback writes."""

        try:
            eval_info = self.cpu.execute_instruction(pc)
        except Exception as exc:
            if getattr(self.cpu, "backend", None) == "llama" and self._poisoned is None:
                self._poisoned = (
                    f"native instruction execution failed: {type(exc).__name__}: {exc}"
                )
            raise

        try:
            self.memory.flush_deferred_llama_host_writes()
        except Exception as exc:
            if getattr(self.cpu, "backend", None) == "llama" and self._poisoned is None:
                self._poisoned = (
                    "native deferred host-write flush failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise
        return eval_info

    @perf_trace("Emulation", include_op_num=True)
    def step(self) -> bool:
        if self._poisoned is not None:
            raise RuntimeError(
                "PCE500 emulator is poisoned after a failed pre-instruction "
                f"side effect; reset required: {self._poisoned}"
            )

        trace_snapshot: Optional[Dict[str, Any]] = None
        pc = self.cpu.regs.get(RegisterName.PC)
        # A debugger breakpoint is an observational boundary. Do not sample
        # IRQ state, relatch keys, advance low-power wake logic, preflight a
        # vector, or touch timers until the caller elects to continue.
        if pc in self.breakpoints:
            return False
        was_halted = bool(getattr(self.cpu.state, "halted", False))
        in_interrupt = bool(getattr(self, "_in_interrupt", False))
        key_will_reassert = bool(self._key_irq_latched) and not in_interrupt
        onk_will_reassert = (
            bool(getattr(self, "_pending_onk", False)) and not in_interrupt
        )
        potential_pending = (
            bool(self._irq_pending) or key_will_reassert or onk_will_reassert
        )
        pending_irq_replaces_pc = False
        if not was_halted and not in_interrupt and potential_pending:
            # An interrupt is taken between instructions: the saved PC does
            # not need to decode when an already-unmasked IRQ will replace it.
            # Include status bits that the host will reassert below, and decide
            # entirely through the side-effect-free IMEM view.
            imr_silent = (
                self.memory.peek_byte_for_preflight(
                    INTERNAL_MEMORY_START + IMEMRegisters.IMR, pc
                )
                & 0xFF
            )
            isr_silent = (
                self.memory.peek_byte_for_preflight(
                    INTERNAL_MEMORY_START + IMEMRegisters.ISR, pc
                )
                & 0xFF
            )
            if key_will_reassert:
                isr_silent |= int(ISRFlag.KEYI)
            if onk_will_reassert:
                isr_silent |= int(ISRFlag.ONKI)
            pending_irq_replaces_pc = (imr_silent & int(IMRFlag.IRM)) != 0 and (
                imr_silent & isr_silent
            ) != 0
            if pending_irq_replaces_pc:
                # This pass will actually attempt delivery. Prove the fixed
                # vector and callback-free destination before relatching any
                # host status or changing interrupt metadata. HALT/OFF wake is
                # an idle boundary, an active handler bars nested delivery, and
                # a timer tick below can only arm an IRQ for the next pass.
                self.cpu.preflight_vector_transfer_for_scheduling(
                    INTERRUPT_VECTOR_ADDR,
                    source_pc=pc,
                )
        if not was_halted and not pending_irq_replaces_pc:
            # Validate the current PC before keyboard latches, IRQ sampling, or
            # timer state can change. Interrupt delivery may replace this PC,
            # so the selected handler is validated again below.
            self.cpu.validate_before_scheduling(pc)
        # Reassert latched KEY/ONK interrupts even when timers are disabled so
        # firmware ISR clearing does not drop pending keyboard events.
        if self._key_irq_latched and not getattr(self, "_in_interrupt", False):
            isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            isr_val = self.memory.read_byte(isr_addr) & 0xFF
            if (isr_val & int(ISRFlag.KEYI)) == 0:
                self._set_isr_bits(int(ISRFlag.KEYI))
                self._irq_pending = True
                if getattr(self, "_irq_source", None) not in (
                    IRQSource.KEY,
                    IRQSource.ONK,
                ):
                    self._irq_source = IRQSource.KEY
        # Honor low-power state without collapsing OFF into HALT.  The current
        # model allows any asserted ISR source to wake HALT, but only ONKI to
        # wake OFF.  Exact silicon wake policy remains hardware-trace work.
        if getattr(self.cpu.state, "halted", False):
            power_state = getattr(self.cpu.state, "power_state", "halted")
            is_off = power_state == "off"
            # Mirror Rust: tick timers while halted to allow ISR bits to wake the CPU.
            if (
                not is_off
                and self._timer_enabled
                and not getattr(self, "_in_interrupt", False)
            ):
                try:
                    self._tick_timers()
                except Exception as exc:
                    self._poisoned = (
                        f"low-power timer tick failed: {type(exc).__name__}: {exc}"
                    )
                    raise
            # Scan keyboard while halted so key presses can wake the CPU.  Any
            # peripheral failure propagates while the CPU remains stopped; it
            # must never become implicit permission to execute the next opcode.
            self._scan_keyboard_per_instruction()
            isr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            isr_val_chk = self.memory.read_byte(isr_addr_chk) & 0xFF
            wake_isr = isr_val_chk & int(ISRFlag.ONKI) if is_off else isr_val_chk
            if wake_isr != 0:
                # Cancel the low-power state and arm a pending interrupt. Wake
                # is an idle-step boundary: the dormant PC is saved as-is and
                # is not fetched or decoded until the next scheduling pass.
                self.cpu.state.halted = False
                self.cpu.state.power_state = "running"
                self._irq_source = _highest_pending_irq_source(wake_isr)
                setattr(self, "_irq_pending", True)
                _log_irq_debug(
                    f"{power_state.upper()} wake IRQ pending "
                    f"(ISR=0x{isr_val_chk:02X}) source={self._irq_source}"
                )
                self.cycle_count += 1
                return True
            else:
                # Model one idle cycle while remaining stopped; no instruction
                # was executed.
                self.cycle_count += 1
                return True
        # Emit a focused diagnostic instant around the IRQ stub split to understand
        # branch/return flow differences (e.g., PC≈0xF205C → 0xF1769 vs 0xF1FB5).
        if (
            new_tracer.enabled
            and pc not in self.breakpoints
            and (
                0x0F205C <= pc <= 0x0F2064
                or 0x0F1760 <= pc <= 0x0F2070
                or 0x0F1FB0 <= pc <= 0x0F1FC0
            )
        ):
            try:
                imr_probe_diag = (
                    self.memory.peek_byte_for_preflight(
                        INTERNAL_MEMORY_START + IMEMRegisters.IMR, cpu_pc=pc
                    )
                    & 0xFF
                )
                isr_probe_diag = (
                    self.memory.peek_byte_for_preflight(
                        INTERNAL_MEMORY_START + IMEMRegisters.ISR, cpu_pc=pc
                    )
                    & 0xFF
                )
            except Exception:
                imr_probe_diag = None
                isr_probe_diag = None
            try:
                opcode_diag = (
                    self.memory.peek_byte_for_preflight(pc & 0xFFFFF, cpu_pc=pc) & 0xFF
                )
            except Exception:
                opcode_diag = None
            try:
                s_reg = self.cpu.regs.get(RegisterName.S) & 0xFFFFFF
                u_reg = self.cpu.regs.get(RegisterName.U) & 0xFFFFFF
                f_reg = self.cpu.regs.get(RegisterName.F) & 0xFF
                fc_reg = getattr(self.cpu.regs, "get_fc", lambda: None)()
                fz_reg = getattr(self.cpu.regs, "get_fz", lambda: None)()
            except Exception:
                s_reg = u_reg = f_reg = fc_reg = fz_reg = None
            stack_bytes = None
            if s_reg is not None:
                try:
                    base = s_reg & 0xFFFFF
                    stack_bytes = [
                        self.memory.peek_byte_for_preflight(
                            (base + idx) & 0xFFFFF, cpu_pc=pc
                        )
                        & 0xFF
                        for idx in range(5)
                    ]
                except Exception:
                    stack_bytes = None
            new_tracer.instant(
                "diag.stub",
                "StubWindow",
                {
                    "pc": pc,
                    "opcode": opcode_diag,
                    "s": s_reg,
                    "u": u_reg,
                    "f": f_reg,
                    "fc": fc_reg,
                    "fz": fz_reg,
                    "imr": imr_probe_diag,
                    "isr": isr_probe_diag,
                    "stack0": stack_bytes,
                },
            )

        # Always probe IMR/ISR so tests and diagnostics can observe dispatcher state, but only
        # arm synthetic pending IRQs from the IMR/ISR snapshot while we're already in an
        # interrupt. This prevents repeatedly re-arming the same latched ISR bit after RETI.
        try:
            imr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.IMR
            isr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            pc_for_irq = pc

            trace_imem = False
            try:
                active_cb = getattr(self.memory, "_imem_access_callback", None)
                trace_imem = (
                    active_cb is not None
                    and getattr(self, "_default_imem_access_callback", None) is not None
                    and active_cb is not self._default_imem_access_callback
                )
            except Exception:
                trace_imem = False

            if self.perfetto_enabled or new_tracer.enabled or trace_imem:
                imr_probe = (
                    self.memory.read_byte(imr_addr_chk, cpu_pc=pc_for_irq) & 0xFF
                )
                isr_probe = (
                    self.memory.read_byte(isr_addr_chk, cpu_pc=pc_for_irq) & 0xFF
                )
            else:
                imr_probe = self.memory.read_byte(imr_addr_chk) & 0xFF
                isr_probe = self.memory.read_byte(isr_addr_chk) & 0xFF

            pending_src = _highest_pending_irq_source(isr_probe & imr_probe)

            if (
                getattr(self, "_in_interrupt", False)
                and not getattr(self, "_irq_pending", False)
                and (imr_probe & int(IMRFlag.IRM)) != 0
                and (imr_probe & isr_probe) != 0
            ):
                self._irq_pending = True
                if pending_src is not None:
                    self._irq_source = pending_src
                if self.perfetto_enabled or new_tracer.enabled:
                    try:
                        self._trace_irq_instant(
                            "IRQ_PendingArm",
                            pending_src,
                            {
                                "pc": pc_for_irq,
                                "imr": imr_probe,
                                "isr": isr_probe,
                                "pending_src": pending_src.name
                                if pending_src
                                else None,
                            },
                        )
                    except Exception:
                        pass

            if self.perfetto_enabled or new_tracer.enabled:
                try:
                    self._trace_irq_instant(
                        "IRQ_Check",
                        pending_src,
                        {
                            "pc": pc_for_irq,
                            "imr": imr_probe,
                            "isr": isr_probe,
                            "pending_flag": bool(getattr(self, "_irq_pending", False)),
                            "in_interrupt": bool(getattr(self, "_in_interrupt", False)),
                            "pending_src": pending_src.name if pending_src else None,
                        },
                    )
                except Exception:
                    pass
        except Exception as exc:
            # These reads feed functional pending-IRQ state, not just tracing.
            # A failed sample cannot become permission to execute an opcode.
            raise RuntimeError("failed to sample pending IRQ state") from exc
        # Check for pending synthetic interrupt before executing next instruction
        if getattr(self, "_irq_pending", False) and not getattr(
            self, "_in_interrupt", False
        ):
            irq_delivery_started = False
            try:
                # Respect IMR/ISR masks: deliver only if IRM=1 and (IMR & ISR)!=0
                imr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.IMR
                isr_addr_chk = INTERNAL_MEMORY_START + IMEMRegisters.ISR
                imr_val_chk = self.memory.read_byte(imr_addr_chk, cpu_pc=pc) & 0xFF
                isr_val_chk = self.memory.read_byte(isr_addr_chk, cpu_pc=pc) & 0xFF
                try:
                    kil_val_chk = (
                        self.memory.read_byte(
                            INTERNAL_MEMORY_START + IMEMRegisters.KIL, cpu_pc=pc
                        )
                        & 0xFF
                    )
                except Exception:
                    # KIL is included only as diagnostic context; it is not part
                    # of the interrupt-delivery decision.
                    kil_val_chk = None
                # Capture a second IMR read via CPU regs (LLAMA) to spot divergence.
                imr_reg_val = None
                try:
                    imr_reg_val = self.cpu.regs.get(RegisterName.IMR) & 0xFF
                except Exception:
                    pass
                pending_src = _highest_pending_irq_source(isr_val_chk & imr_val_chk)
                # Trace the pending check to see if KEYI is masked/cleared.
                try:
                    self._trace_irq_instant(
                        "IRQ_PendingCheck",
                        pending_src,
                        {
                            "pc": self.cpu.regs.get(RegisterName.PC),
                            "imr": imr_val_chk,
                            "isr": isr_val_chk,
                            "kil": kil_val_chk,
                            "imr_reg": imr_reg_val,
                            "irq_source": getattr(self, "_irq_source", None)
                            and getattr(self._irq_source, "name", None),
                            "pending_src": pending_src.name if pending_src else None,
                        },
                    )
                except Exception:
                    pass
                # Trace unexpected IMR=0 reads to spot masking.
                if imr_val_chk == 0:
                    try:
                        self._trace_irq_instant(
                            "IMR_ReadZero",
                            pending_src,
                            {
                                "pc": self.cpu.regs.get(RegisterName.PC),
                                "isr": isr_val_chk,
                            },
                        )
                    except Exception:
                        pass
                _log_irq_debug(
                    f"pending IRQ check pc=0x{self.cpu.regs.get(RegisterName.PC):06X} "
                    f"imr=0x{imr_val_chk:02X} isr=0x{isr_val_chk:02X} in_interrupt={self._in_interrupt}"
                )
                irm_enabled = (imr_val_chk & int(IMRFlag.IRM)) != 0
                if not irm_enabled or (imr_val_chk & isr_val_chk) == 0:
                    # Keep pending; CPU continues executing normal flow
                    _log_irq_debug(
                        f"IRQ masked; pending retained (IMR=0x{imr_val_chk:02X} ISR=0x{isr_val_chk:02X})"
                    )
                    pass
                else:
                    # Push PC (3 bytes), then F (1), then IMR (1), clear IMR.IRM
                    cur_pc = self.cpu.regs.get(RegisterName.PC)
                    s = self.cpu.regs.get(RegisterName.S)
                    # Resolve and statically preflight the handler silently,
                    # then perform exactly one architectural vector fetch and
                    # require it to match before the frame changes.  This
                    # catches a failing/volatile bus without leaving a partial
                    # stack frame or cleared IMR.
                    vector_addr = self._fetch_validated_vector(
                        INTERRUPT_VECTOR_ADDR, source_pc=cur_pc
                    )
                    # Commit wrapper metadata only after the vector read has
                    # proved that delivery can begin atomically.
                    if pending_src is not None:
                        self._irq_source = pending_src
                    _log_irq_debug(
                        f"Delivering IRQ src={self._irq_source} pc=0x{cur_pc:06X} s=0x{s:06X}"
                    )
                    if self._irq_source == IRQSource.KEY:
                        try:
                            self._trace_irq_instant(
                                "KeyDeliver",
                                self._irq_source,
                                {
                                    "from": cur_pc,
                                    "imr": imr_val_chk,
                                    "isr": isr_val_chk,
                                    "s": s,
                                },
                            )
                        except Exception:
                            pass
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] deliver start pc=0x{cur_pc:06X} s=0x{s:06X} f=0x{int(self.cpu.regs.get(RegisterName.F)) & 0xFF:02X} imr=0x{imr_val_chk:02X}"
                        )
                    # push PC (little-endian 3 bytes)
                    s_new = (s - 3) & 0xFFFFF
                    irq_delivery_started = True
                    self.memory.write_bytes(3, s_new, cur_pc)
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] push_pc from 0x{s:06X} to 0x{s_new:06X} value=0x{cur_pc & 0xFFFFFF:06X}"
                        )
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push F (1 byte)
                    f_val = self.cpu.regs.get(RegisterName.F)
                    s_new = (self.cpu.regs.get(RegisterName.S) - 1) & 0xFFFFF
                    self.memory.write_bytes(1, s_new, f_val)
                    if IRQ_STACK_TRACE_ENABLED:
                        print(
                            f"[irq-stack] push_f value=0x{int(f_val) & 0xFF:02X} new_s=0x{s_new:06X}"
                        )
                    self.cpu.regs.set(RegisterName.S, s_new)
                    # push IMR (1 byte) and clear IRM bit 7
                    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
                    imr_val = self.memory.read_byte(imr_addr)
                    s_new = (self.cpu.regs.get(RegisterName.S) - 1) & 0xFFFFF
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
                    # Jump to the already validated interrupt vector candidate.
                    if self._irq_source == IRQSource.KEY:
                        # Mark keyboard IRQ delivery explicitly on irq.key track.
                        try:
                            self._trace_irq_instant(
                                "KeyDeliver",
                                self._irq_source,
                                {
                                    "from": cur_pc,
                                    "vector": vector_addr,
                                    "imr": imr_val_chk,
                                    "isr": isr_val_chk,
                                    "s": s_new,
                                },
                            )
                        except Exception:
                            pass
                    try:
                        self._trace_irq_instant(
                            "IRQ_Enter",
                            self._irq_source,
                            {
                                "from": cur_pc,
                                "vector": vector_addr,
                                "imr_before": imr_val_chk,
                                "imr_after": imr_val & 0xFF,
                                "isr": isr_val_chk,
                                "y": self.cpu.regs.get(RegisterName.Y),
                                "s": s,
                            },
                        )
                    except Exception:
                        pass
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
            except Exception as exc:
                if irq_delivery_started:
                    self._poisoned = f"IRQ delivery failed: {type(exc).__name__}: {exc}"
                raise

        # Interrupt delivery above may replace the opcode which was about to
        # execute, so preflight the final PC here.  Quarantined behavior (for
        # example counted instructions with I=0 or TCL) must fail before the
        # per-instruction timer tick mutates scheduler/ISR state.
        scheduled_pc = self.cpu.regs.get(RegisterName.PC)
        if scheduled_pc in self.breakpoints:
            return False
        try:
            self.cpu.prepare_instruction_before_scheduling(scheduled_pc)
        except Exception as exc:
            if getattr(self.cpu, "_contract_poisoned", None) is not None:
                self._poisoned = (
                    "instruction preparation may have performed an architectural "
                    f"read: {type(exc).__name__}: {exc}"
                )
            raise

        # Tick rough timers after pending IRQ delivery check to match Rust ordering.
        if self._timer_enabled:
            # Hardware timers continue advancing while an interrupt handler
            # runs; delivery remains deferred by the interrupt-state checks.
            try:
                self._tick_timers()
            except Exception as exc:
                # Timer advancement may already have updated the scheduler or
                # ISR before a later host operation fails.  Require reset rather
                # than retrying a partially completed tick and then executing.
                self._poisoned = f"timer tick failed: {type(exc).__name__}: {exc}"
                self.cpu.cancel_prepared_vector_transfer()
                raise

        pc = self.cpu.regs.get(RegisterName.PC)
        self._last_pc, self._current_pc = self._current_pc, pc
        if False and pc in (
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
            try:
                trace_snapshot = self._snapshot_instruction_trace(
                    pc, self.instruction_count
                )
                if trace_snapshot is not None:
                    self._active_trace_instruction = self.instruction_count
                    self._trace_substep = 0
            except Exception as exc:
                self.cpu.cancel_prepared_vector_transfer()
                self._poisoned = (
                    f"post-prepare trace setup failed: {type(exc).__name__}: {exc}"
                )
                raise

        if self.trace is not None:
            self.trace.append(("exec", pc, self.cycle_count))

        if pc == self.MEMORY_DUMP_PC and self.perfetto_enabled:
            try:
                self._dump_internal_memory(pc)
            except Exception as exc:
                self.cpu.cancel_prepared_vector_transfer()
                self._poisoned = (
                    f"post-prepare memory dump failed: {type(exc).__name__}: {exc}"
                )
                raise

        opcode = None
        try:
            # Pre-read opcode and I for WAIT simulation (so we can model time passing)
            # Always simulate WAIT loops to advance timers, regardless of tracing.
            wait_sim_count = 0
            try:
                opcode_peek = self.memory.peek_byte_for_preflight(pc, pc) & 0xFF
                if opcode_peek == 0xEF:  # WAIT
                    i_before = self.cpu.regs.get(RegisterName.I) & 0xFFFF
                    if i_before > 0:
                        wait_sim_count = i_before
                opcode = opcode_peek
            except Exception:
                pass

            fast_mode = getattr(self, "fast_mode", False)
            if fast_mode:
                # Minimal execution path for speed
                pc_before = pc

                # Clear current instruction accesses before execution
                self._reset_instruction_access_log()

                eval_info = self._execute_instruction_and_flush(pc)
                if PYTHON_PC_TRACE_ENABLED:
                    print(f"[python-pc] PC=0x{pc:06X}")
                _log_stack_snapshot_emulator(self, pc)

                # Associate accumulated register accesses with this instruction
                self._flush_instruction_access_log(pc_before)

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    if hasattr(self.memory, "wait_cycles"):
                        # Core backends with WAIT fast-path delegate timer progress via
                        # memory.wait_cycles(); do not apply WAIT cycles twice.
                        pass
                    else:
                        self._simulate_wait(wait_sim_count)
                if self.perfetto_enabled:
                    # In fast mode, keep lightweight counters only
                    self._update_perfetto_counters()
                if self._new_trace_enabled:
                    # In fast mode, still emit full execution instants when new tracing is enabled
                    # so Perfetto/trace consumers stay aligned.
                    self._trace_execution(pc_before, opcode)
                elif new_tracer.enabled:
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
            else:
                # Decode instruction first to get opcode name for tracing
                instr = self.cpu.decode_instruction(
                    pc,
                    read_fn=lambda address: self.memory.peek_byte_for_preflight(
                        address, pc
                    ),
                )
                opcode = (
                    int(instr.opcode) & 0xFF
                    if (self.perfetto_enabled or self._new_trace_enabled)
                    else None
                )
                # Build opcode name for performance tracing
                opcode_name = instr.name()
                # Execute instruction with opcode-level tracing
                pc_before = pc

                # Clear current instruction accesses before execution
                self._reset_instruction_access_log()

                if new_tracer.enabled and not self._new_trace_enabled:
                    with new_tracer.slice(
                        "Opcodes",
                        opcode_name,
                        {
                            "pc": f"0x{pc:06X}",
                            "opcode": f"0x{opcode:02X}" if opcode else None,
                            "op_num": self.instruction_count,
                        },
                    ):
                        eval_info = self._execute_instruction_and_flush(pc)
                        if PYTHON_PC_TRACE_ENABLED:
                            print(f"[python-pc] PC=0x{pc:06X}")
                else:
                    eval_info = self._execute_instruction_and_flush(pc)
                    if PYTHON_PC_TRACE_ENABLED:
                        print(f"[python-pc] PC=0x{pc:06X}")
                _log_stack_snapshot_emulator(self, pc)

                # Associate accumulated register accesses with this instruction
                self._flush_instruction_access_log(pc_before)

                self.cycle_count += 1
                self.instruction_count += 1
                # If this was WAIT, simulate the skipped loop to keep timers aligned
                if wait_sim_count:
                    if hasattr(self.memory, "wait_cycles"):
                        # Core backends with WAIT fast-path delegate timer progress via
                        # memory.wait_cycles(); do not apply WAIT cycles twice.
                        pass
                    else:
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
            self.cpu.cancel_prepared_vector_transfer()
            if self._poisoned is None:
                self._poisoned = (
                    f"post-prepare instruction path failed: {type(e).__name__}: {e}"
                )
            if self.trace is not None:
                self.trace.append(("error", pc, str(e)))
            if self.perfetto_enabled:
                trace_dispatcher.record_instant(
                    "CPU", "Error", {"error": str(e), "pc": f"0x{pc:06X}"}
                )
            raise
        self._emit_instruction_trace_event(trace_snapshot)
        self._scan_keyboard_per_instruction()
        # Detect end of interrupt roughly by RETI opcode name
        try:
            instr_name = type(eval_info.instruction).__name__
            if instr_name == "RETI":
                self._trace_irq_instant(
                    "IRQ_Return",
                    self._irq_source,
                    {
                        "pc": pc,
                        "y": self.cpu.regs.get(RegisterName.Y),
                        "imr": self.memory.read_byte(
                            INTERNAL_MEMORY_START + IMEMRegisters.IMR
                        )
                        & 0xFF,
                        "isr": self.memory.read_byte(
                            INTERNAL_MEMORY_START + IMEMRegisters.ISR
                        )
                        & 0xFF,
                    },
                )
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
        if metadata is None and payload is None:
            return
        if metadata is None or payload is None:
            raise RuntimeError(
                "LLAMA LCD snapshot returned incomplete metadata/payload"
            )
        self.lcd.load_snapshot(metadata, payload)

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
                    "data_read_count": chip_snap.data_read_count,
                    "on_off_count": chip_snap.on_off_count,
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

    def _capture_keyboard_snapshot_metadata(self) -> dict[str, object]:
        state = self.keyboard.snapshot_state()
        matrix = state.get("matrix")
        if not isinstance(matrix, dict):
            raise RuntimeError("keyboard snapshot is missing matrix state")
        matrix["kil_read_count"] = int(getattr(self, "_kil_read_count", 0))
        return state

    def _capture_scheduler_snapshot_metadata(
        self,
        imem_bytes: bytes,
        *,
        delivered_masks: object = None,
        native_timer: object = None,
        native_interrupts: object = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Capture the host-authoritative timer and IRQ state exactly."""

        if len(imem_bytes) != 0x100:
            raise ValueError("internal memory snapshot must contain exactly 256 bytes")
        timer_info: dict[str, object] = {
            "enabled": bool(self._timer_enabled),
            "mti_period": int(self._timer_mti_period),
            "sti_period": int(self._timer_sti_period),
            "next_mti": int(self._timer_next_mti),
            "next_sti": int(self._timer_next_sti),
            "kb_irq_enabled": bool(getattr(self, "_kb_irq_enabled", True)),
            # Python snapshots are taken at an instruction boundary. Native
            # snapshots overwrite these with TimerContext's exact phase state.
            "instruction_start_cycle": int(self.cycle_count),
            "last_mti_fire_cycle": None,
            "last_sti_fire_cycle": None,
            "fired_mti_since_boundary": False,
            "fired_sti_since_boundary": False,
            "preserve_phase": True,
        }
        if isinstance(native_timer, dict):
            for key in (
                "enabled",
                "mti_period",
                "sti_period",
                "next_mti",
                "next_sti",
                "kb_irq_enabled",
            ):
                if key not in native_timer:
                    raise RuntimeError(
                        f"LLAMA snapshot timer metadata is missing {key!r}"
                    )
                if native_timer[key] != timer_info[key]:
                    raise RuntimeError(
                        f"LLAMA native and PCE host timer fields disagree: {key}"
                    )
            for key in (
                "instruction_start_cycle",
                "last_mti_fire_cycle",
                "last_sti_fire_cycle",
                "fired_mti_since_boundary",
                "fired_sti_since_boundary",
                "preserve_phase",
            ):
                if key not in native_timer:
                    raise RuntimeError(
                        f"LLAMA snapshot timer metadata is missing {key!r}"
                    )
                timer_info[key] = native_timer[key]
        irq_source_name = self._irq_source.name if self._irq_source else None
        last_fired = None
        if isinstance(native_interrupts, dict):
            if "last_fired" not in native_interrupts:
                raise RuntimeError(
                    "LLAMA snapshot interrupt metadata is missing 'last_fired'"
                )
            last_fired = native_interrupts["last_fired"]
        interrupts: dict[str, object] = {
            "pending": bool(getattr(self, "_irq_pending", False)),
            "in_interrupt": bool(getattr(self, "_in_interrupt", False)),
            "key_irq_latched": bool(getattr(self, "_key_irq_latched", False)),
            "source": irq_source_name,
            "last_fired": last_fired,
            "stack": list(self._interrupt_stack),
            "next_id": int(self._next_interrupt_id),
            "imr": int(imem_bytes[IMEMRegisters.IMR.value]),
            "isr": int(imem_bytes[IMEMRegisters.ISR.value]),
            "irq_counts": dict(self.irq_counts),
            "last_irq": dict(self.last_irq),
            "irq_bit_watch": self.irq_bit_watch,
            "delivered_masks": (
                list(delivered_masks) if isinstance(delivered_masks, list) else []
            ),
        }
        for source in ("mti", "sti"):
            fired = bool(timer_info[f"fired_{source}_since_boundary"])
            fire_cycle = timer_info[f"last_{source}_fire_cycle"]
            if fired != (fire_cycle is not None):
                raise ValueError(
                    f"snapshot {source.upper()} phase flag/fire cycle disagree"
                )
        if self.cpu.backend != "llama" and (
            timer_info["instruction_start_cycle"] != self.cycle_count
            or timer_info["last_mti_fire_cycle"] is not None
            or timer_info["last_sti_fire_cycle"] is not None
            or timer_info["fired_mti_since_boundary"]
            or timer_info["fired_sti_since_boundary"]
            or not timer_info["preserve_phase"]
        ):
            raise ValueError(
                "Python backend cannot exactly restore transient native timer phase"
            )
        return timer_info, interrupts

    def _synchronize_llama_snapshot_shadow(self, llama_impl: object) -> None:
        """Copy host-owned peripheral state into LLAMA before serialization.

        The PCE facade advances timers, scans the keyboard, and delivers IRQs.
        LLAMA's duplicate TimerContext/KeyboardMatrix fields are therefore a
        bridge shadow in this integration, not a second source of truth.  The
        native helper validates all candidates before committing any of them.
        """

        synchronizer = getattr(llama_impl, "synchronize_host_snapshot_state", None)
        if not callable(synchronizer):
            raise RuntimeError(
                "LLAMA backend lacks atomic host snapshot-state synchronization"
            )
        imem_bytes = self.memory.get_internal_memory_bytes()
        timer_info, interrupts = self._capture_scheduler_snapshot_metadata(imem_bytes)
        keyboard_state = self._capture_keyboard_snapshot_metadata()
        synchronizer(
            timer_info,
            interrupts,
            keyboard_state,
            int(self.instruction_count),
            int(self.cycle_count),
            int(self.memory_read_count),
            int(self.memory_write_count),
        )

    def _patch_llama_snapshot_metadata(self, target: Path) -> None:
        """Add Python-only metadata fields to a LLAMA-authored snapshot."""

        try:
            metadata, entries = _read_snapshot_archive(target, native_for_patch=True)
        except json.JSONDecodeError as exc:
            raise RuntimeError("failed to parse LLAMA snapshot metadata") from exc
        except ValueError as exc:
            if "snapshot JSON contains duplicate member" in str(exc):
                raise RuntimeError("failed to parse LLAMA snapshot metadata") from exc
            raise RuntimeError(
                f"unable to read LLAMA snapshot for metadata patch: {target}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"unable to read LLAMA snapshot for metadata patch: {target}"
            ) from exc

        imem_bytes = entries.get("imem.bin")
        if imem_bytes is None:
            raise RuntimeError("LLAMA snapshot is missing imem.bin")
        previous_timer = metadata.get("timer")
        previous_interrupts = metadata.get("interrupts")
        delivered_masks = (
            previous_interrupts.get("delivered_masks")
            if isinstance(previous_interrupts, dict)
            else None
        )
        timer_info, interrupts = self._capture_scheduler_snapshot_metadata(
            imem_bytes,
            delivered_masks=delivered_masks,
            native_timer=previous_timer,
            native_interrupts=previous_interrupts,
        )
        if isinstance(previous_interrupts, dict):
            native_latch = previous_interrupts.get("key_irq_latched")
            if not isinstance(native_latch, bool):
                raise RuntimeError(
                    "LLAMA snapshot interrupt metadata is missing boolean "
                    "key_irq_latched"
                )
            if native_latch != interrupts["key_irq_latched"]:
                raise RuntimeError(
                    "LLAMA native and PCE host keyboard IRQ latches disagree"
                )
            for field in (
                "pending",
                "in_interrupt",
                "source",
                "stack",
                "next_id",
                "imr",
                "isr",
                "irq_counts",
                "last_irq",
                "irq_bit_watch",
            ):
                if field not in previous_interrupts:
                    raise RuntimeError(
                        f"LLAMA snapshot interrupt metadata is missing {field!r}"
                    )
                host_value = interrupts[field]
                if field == "irq_bit_watch":
                    # Python keeps bit indices as integers in memory; JSON object
                    # keys are necessarily strings in the native archive.
                    host_value = json.loads(json.dumps(host_value))
                if previous_interrupts[field] != host_value:
                    raise RuntimeError(
                        f"LLAMA native and PCE host interrupt fields disagree: {field}"
                    )

        cpu_snapshot = self.cpu.snapshot_registers()
        temps = {str(k): int(v) for k, v in getattr(cpu_snapshot, "temps", {}).items()}
        for field, host_value in (
            ("instruction_count", int(getattr(self, "instruction_count", 0))),
            ("cycle_count", int(getattr(self, "cycle_count", 0))),
            ("memory_reads", int(getattr(self, "memory_read_count", 0))),
            ("memory_writes", int(getattr(self, "memory_write_count", 0))),
            ("pc", int(getattr(cpu_snapshot, "pc", 0)) & 0xFFFFF),
            ("temps", temps),
            ("power_state", str(getattr(self.cpu.state, "power_state", "running"))),
        ):
            if metadata.get(field) != host_value:
                raise RuntimeError(
                    f"LLAMA native and PCE host runtime fields disagree: {field}"
                )
        native_call_depth = metadata.get("call_depth")
        native_call_sub_level = metadata.get("call_sub_level")
        host_call_depth = int(getattr(self, "call_depth", 0))
        host_call_sub_level = int(getattr(cpu_snapshot, "call_sub_level", 0))
        if native_call_depth != host_call_depth:
            raise RuntimeError("LLAMA native and PCE host call depths disagree")
        if native_call_sub_level != host_call_sub_level:
            raise RuntimeError("LLAMA native and PCE host call sub-levels disagree")
        for field in ("call_stack", "call_page_stack", "call_return_widths"):
            if not isinstance(metadata.get(field), list):
                raise RuntimeError(f"LLAMA snapshot metadata is missing {field!r}")
        for field, label in (
            ("external_interrupt_level", "external IRQ"),
            ("onk_level", "native ON-key"),
        ):
            level = metadata.get(field)
            if not isinstance(level, bool):
                raise RuntimeError(
                    f"LLAMA snapshot metadata is missing boolean {field!r}"
                )
            if level:
                raise RuntimeError(
                    f"PCE host cannot exactly represent asserted {label} level"
                )
        metadata.update(
            {
                "backend": getattr(self.cpu, "backend", "llama"),
                "instruction_count": int(getattr(self, "instruction_count", 0)),
                "cycle_count": int(getattr(self, "cycle_count", 0)),
                "pc": int(getattr(cpu_snapshot, "pc", 0)) & 0xFFFFF,
                "call_depth": host_call_depth,
                "call_sub_level": host_call_sub_level,
                "temps": temps,
                "memory_reads": int(getattr(self, "memory_read_count", 0)),
                "memory_writes": int(getattr(self, "memory_write_count", 0)),
                "memory_dump_pc": int(getattr(self, "MEMORY_DUMP_PC", 0)),
                "fast_mode": bool(getattr(self, "fast_mode", False)),
                "power_state": str(getattr(self.cpu.state, "power_state", "running")),
                "external_interrupt_level": False,
                "onk_level": False,
                "timer": timer_info,
                "interrupts": interrupts,
            }
        )

        keyboard_state = self._capture_keyboard_snapshot_metadata()
        previous_keyboard = metadata.get("keyboard")
        if not isinstance(previous_keyboard, dict):
            raise RuntimeError("LLAMA snapshot is missing native keyboard metadata")
        host_matrix = keyboard_state.get("matrix")
        if not isinstance(host_matrix, dict):
            raise RuntimeError("PCE host keyboard snapshot is missing matrix metadata")
        for field in (
            "keyi_on_any_press",
            "raw_kil",
            "emit_events",
            "repeat_enabled",
            "keyi_latch",
            "kil_read_count",
        ):
            if previous_keyboard.get(field) != host_matrix.get(field):
                raise RuntimeError(
                    f"LLAMA native and PCE host keyboard fields disagree: {field}"
                )
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

        card_metadata, card_payload = self.memory.export_memory_card_snapshot()
        metadata["version"] = SNAPSHOT_VERSION
        metadata["memory_card"] = card_metadata
        entries["memory_card.bin"] = card_payload

        entries["snapshot.json"] = json.dumps(
            metadata, indent=2, sort_keys=True
        ).encode("utf-8")

        temp_fd, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(
                temp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for name, data in entries.items():
                    zf.writestr(name, data)
            os.replace(temp_path, target)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"failed to atomically rewrite LLAMA snapshot metadata: {target}"
            ) from exc

    def _reject_poisoned_snapshot_save(self) -> None:
        """Refuse to checkpoint state which cannot safely resume execution."""

        poison_sources = (
            ("PCE500 wrapper", getattr(self, "_poisoned", None)),
            ("SC62015 CPU facade", getattr(self.cpu, "_contract_poisoned", None)),
            ("SC62015 CPU backend", getattr(self.cpu, "_poisoned", None)),
        )
        for source, reason in poison_sources:
            if reason is not None:
                raise RuntimeError(
                    f"cannot save snapshot while {source} is poisoned; "
                    f"reset required: {reason}"
                )

    def _reject_unrepresented_snapshot_state(self, action: str) -> None:
        """Refuse a v4 operation that would silently lose host device state."""

        active: list[str] = []
        memory = getattr(self, "memory", None)
        memory_check = getattr(memory, "unrepresented_snapshot_state", None)
        if callable(memory_check):
            active.extend(str(item) for item in memory_check())
        peripherals = getattr(self, "peripherals", None)
        peripheral_check = getattr(peripherals, "unrepresented_snapshot_state", None)
        if callable(peripheral_check):
            active.extend(str(item) for item in peripheral_check())
        if active:
            raise RuntimeError(
                f"cannot {action} snapshot v4 while unrepresented state is active: "
                + ", ".join(active)
            )

    def save_snapshot(self, path: str | Path) -> Path:
        """Persist exactly represented CPU, memory, display, and input state."""

        self._reject_poisoned_snapshot_save()
        self._reject_unrepresented_snapshot_state("save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if getattr(self.cpu, "backend", None) == "llama":
            llama_impl = self.cpu.unwrap()
            saver = getattr(llama_impl, "save_snapshot", None)
            if callable(saver):
                is_synced = getattr(llama_impl, "is_memory_synced", None)
                if callable(is_synced) and not is_synced():
                    reinit = getattr(llama_impl, "_initialise_rust_memory", None)
                    if callable(reinit):
                        reinit()
                self._synchronize_llama_snapshot_shadow(llama_impl)
                # An explicitly active LLAMA backend owns its snapshot
                # contract. Shape, callback, and write errors must reach the
                # caller; falling through to a second serializer can turn a
                # rejected partial image into an apparently valid snapshot.
                temp_fd, temp_name = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".native.tmp",
                )
                os.close(temp_fd)
                temp_path = Path(temp_name)
                try:
                    saver(str(temp_path))
                    self._patch_llama_snapshot_metadata(temp_path)
                    self._validate_snapshot_archive(temp_path)
                    self._reject_poisoned_snapshot_save()
                    os.replace(temp_path, target)
                except Exception:
                    temp_path.unlink(missing_ok=True)
                    raise
                return target

        cpu_snapshot = self.cpu.snapshot_registers()
        registers_blob = _pack_register_bytes(cpu_snapshot)
        flat_memory, fallback_ranges, readonly_ranges = self.memory.export_flat_memory()

        internal_slice = flat_memory[
            self.INTERNAL_RAM_START : self.INTERNAL_RAM_START + self.INTERNAL_RAM_SIZE
        ]
        imem_bytes = self.memory.get_internal_memory_bytes()
        lcd_meta, lcd_payload = self._capture_lcd_snapshot()
        keyboard_state = self._capture_keyboard_snapshot_metadata()
        card_metadata, card_payload = self.memory.export_memory_card_snapshot()

        timer_info, interrupts = self._capture_scheduler_snapshot_metadata(imem_bytes)

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
            # Use timezone-aware UTC timestamp to avoid deprecated utcnow().
            "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "instruction_count": int(self.instruction_count),
            "cycle_count": int(self.cycle_count),
            "memory_reads": int(self.memory_read_count),
            "memory_writes": int(self.memory_write_count),
            "pc": int(cpu_snapshot.pc),
            "power_state": str(getattr(self.cpu.state, "power_state", "running")),
            "external_interrupt_level": False,
            "onk_level": False,
            "call_depth": int(self.call_depth),
            "call_sub_level": int(cpu_snapshot.call_sub_level),
            "call_stack": [],
            "call_page_stack": [],
            "call_return_widths": [],
            "temps": {
                str(index): int(cpu_snapshot.temps.get(index, 0)) for index in range(16)
            },
            "timer": timer_info,
            "interrupts": interrupts,
            "keyboard": keyboard_state,
            "memory_card": card_metadata,
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

        temp_fd, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(
                temp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr(
                    "snapshot.json", json.dumps(metadata, indent=2, sort_keys=True)
                )
                zf.writestr("registers.bin", registers_blob)
                zf.writestr("external_ram.bin", bytes(flat_memory))
                zf.writestr("internal_ram.bin", bytes(internal_slice))
                zf.writestr("imem.bin", imem_bytes)
                zf.writestr("lcd_vram.bin", lcd_payload)
                zf.writestr("memory_card.bin", card_payload)
            self._validate_snapshot_archive(temp_path)
            self._reject_poisoned_snapshot_save()
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return target

    def _validate_snapshot_archive(self, path: str | Path) -> None:
        """Build every load candidate without committing it to the emulator."""

        self._load_snapshot(path, backend=None, commit=False)

    def load_snapshot(self, path: str | Path, *, backend: Optional[str] = None) -> None:
        """Validate a complete snapshot off-side, then commit it once."""

        self._load_snapshot(path, backend=backend, commit=True)

    def _load_snapshot(
        self,
        path: str | Path,
        *,
        backend: Optional[str],
        commit: bool,
    ) -> None:
        """Parse and validate a snapshot, optionally committing its candidates."""

        self._reject_unrepresented_snapshot_state("load")

        def required(mapping: dict[str, object], key: str, label: str) -> object:
            if key not in mapping:
                raise ValueError(f"snapshot {label} is missing required member {key!r}")
            return mapping[key]

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        metadata, entries = _read_snapshot_archive(source)
        registers_blob = entries["registers.bin"]
        flat_memory = entries["external_ram.bin"]
        internal_ram = entries["internal_ram.bin"]
        imem_bytes = entries["imem.bin"]
        lcd_payload = entries["lcd_vram.bin"]
        card_payload = entries["memory_card.bin"]

        metadata = _snapshot_mapping(metadata, "metadata")
        required_metadata_fields = {
            "magic",
            "version",
            "backend",
            "created",
            "instruction_count",
            "cycle_count",
            "memory_reads",
            "memory_writes",
            "pc",
            "power_state",
            "external_interrupt_level",
            "onk_level",
            "call_depth",
            "call_sub_level",
            "call_stack",
            "call_page_stack",
            "call_return_widths",
            "temps",
            "timer",
            "interrupts",
            "keyboard",
            "memory_card",
            "kb_metrics",
            "fallback_ranges",
            "readonly_ranges",
            "internal_ram",
            "imem",
            "memory_dump_pc",
            "fast_mode",
            "memory_image_size",
            "lcd_payload_size",
            "lcd",
        }
        if required_metadata_fields - set(metadata) or set(metadata) - (
            required_metadata_fields | {"device_model"}
        ):
            raise ValueError("snapshot metadata has an unexpected top-level shape")
        if "device_model" in metadata and metadata["device_model"] not in (
            None,
            "pc-e500",
            "pc-e500-jp",
        ):
            raise ValueError("snapshot device model is not a PC-E500 variant")
        if required(metadata, "magic", "metadata") != SNAPSHOT_MAGIC:
            raise ValueError("Snapshot magic mismatch")
        version = _snapshot_int(
            required(metadata, "version", "metadata"),
            "version",
            maximum=(1 << 32) - 1,
        )
        if version != SNAPSHOT_VERSION:
            raise ValueError("Unsupported snapshot version")
        snapshot_backend = required(metadata, "backend", "metadata")
        if snapshot_backend not in ("python", "llama"):
            raise ValueError(f"snapshot backend is unsupported: {snapshot_backend!r}")
        if not isinstance(required(metadata, "created", "metadata"), str):
            raise TypeError("snapshot created must be a string")

        card_metadata = _snapshot_mapping(
            required(metadata, "memory_card", "metadata"), "memory_card"
        )
        if set(card_metadata) != {"mode", "capacity", "writable", "payload_size"}:
            raise ValueError(
                "snapshot memory_card must contain exactly mode, capacity, "
                "writable, and payload_size"
            )
        card_mode = required(card_metadata, "mode", "memory_card")
        if card_mode not in ("present", "absent"):
            raise ValueError("snapshot memory_card.mode must be 'present' or 'absent'")
        card_capacity = _snapshot_int(
            required(card_metadata, "capacity", "memory_card"),
            "memory_card.capacity",
            maximum=max(SUPPORTED_MEMORY_CARD_SIZES),
        )
        if card_capacity not in SUPPORTED_MEMORY_CARD_SIZES:
            raise ValueError("snapshot memory_card.capacity is unsupported")
        card_writable = _snapshot_bool(
            required(card_metadata, "writable", "memory_card"),
            "memory_card.writable",
        )
        card_payload_size = _snapshot_int(
            required(card_metadata, "payload_size", "memory_card"),
            "memory_card.payload_size",
            maximum=max(SUPPORTED_MEMORY_CARD_SIZES),
        )
        if card_payload_size != card_capacity:
            raise ValueError(
                "snapshot memory_card.payload_size must equal its capacity"
            )
        if len(card_payload) != card_capacity:
            raise ValueError("memory_card.bin size does not match card capacity")
        card_candidate = (
            card_mode == "present",
            card_writable,
            card_capacity,
            bytearray(card_payload),
        )

        memory_size = len(self.memory.external_memory)
        if len(flat_memory) != memory_size:
            raise ValueError("external_ram.bin size mismatch")
        if (
            _snapshot_int(
                required(metadata, "memory_image_size", "metadata"),
                "memory_image_size",
                maximum=0x1000000,
            )
            != memory_size
        ):
            raise ValueError("snapshot memory_image_size mismatch")
        if len(internal_ram) != self.INTERNAL_RAM_SIZE:
            raise ValueError("internal_ram.bin size mismatch")
        expected_internal = flat_memory[
            self.INTERNAL_RAM_START : self.INTERNAL_RAM_START + self.INTERNAL_RAM_SIZE
        ]
        if internal_ram != expected_internal:
            raise ValueError("internal_ram.bin disagrees with external_ram.bin")
        if len(imem_bytes) != 0x100:
            raise ValueError("imem.bin size mismatch")
        if _snapshot_range(
            required(metadata, "internal_ram", "metadata"), "internal_ram"
        ) != (self.INTERNAL_RAM_START, self.INTERNAL_RAM_SIZE):
            raise ValueError("snapshot internal RAM layout mismatch")
        if _snapshot_range(required(metadata, "imem", "metadata"), "imem") != (
            INTERNAL_MEMORY_START,
            0x100,
        ):
            raise ValueError("snapshot internal-memory layout mismatch")

        _, current_fallback, current_readonly = self.memory.export_flat_memory()
        if _snapshot_ranges(
            required(metadata, "fallback_ranges", "metadata"), "fallback_ranges"
        ) != tuple(current_fallback):
            raise ValueError(
                "snapshot Python fallback ranges do not match this machine"
            )
        if _snapshot_ranges(
            required(metadata, "readonly_ranges", "metadata"), "readonly_ranges"
        ) != tuple(current_readonly):
            raise ValueError("snapshot read-only ranges do not match this machine")

        overlay_updates: list[tuple[bytearray, bytes]] = []
        for overlay in self.memory.overlays:
            if overlay.data is None or overlay.start >= memory_size:
                continue
            start = max(overlay.start, 0)
            end = min(overlay.end + 1, memory_size)
            span = max(0, end - start)
            if span == 0:
                continue
            if span > len(overlay.data):
                raise ValueError(
                    f"configured overlay {overlay.name!r} has an incomplete payload"
                )
            payload = flat_memory[start:end]
            if overlay.read_only:
                if bytes(overlay.data[:span]) != payload:
                    raise ValueError(
                        f"snapshot attempts to replace read-only overlay {overlay.name!r}"
                    )
            else:
                overlay_updates.append((overlay.data, payload))

        reg_values = _unpack_register_bytes(registers_blob)
        for name in ("pc", "x", "y", "u", "s"):
            if reg_values[name] > 0xFFFFF:
                raise ValueError(f"snapshot register {name.upper()} exceeds 20 bits")
        temps_raw = _snapshot_mapping(required(metadata, "temps", "metadata"), "temps")
        temps: dict[int, int] = {}
        for key, value in temps_raw.items():
            if not isinstance(key, str) or not key.isdecimal():
                raise ValueError(f"snapshot temporary register key is invalid: {key!r}")
            index = int(key)
            if index in temps:
                raise ValueError(f"snapshot repeats temporary register TEMP{index}")
            if not 0 <= index < 16:
                raise ValueError(
                    f"snapshot contains unknown temporary register TEMP{index}"
                )
            temps[index] = _snapshot_int(value, f"TEMP{index}", maximum=0xFFFFFF)
        if set(temps) != set(range(16)):
            raise ValueError("snapshot must contain exactly TEMP0 through TEMP15")
        call_sub_level = _snapshot_int(
            required(metadata, "call_sub_level", "metadata"),
            "call_sub_level",
            maximum=(1 << 31) - 1,
        )
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
            call_sub_level=call_sub_level,
        )
        snapshot.apply_to(Registers())

        instruction_count = _snapshot_int(
            required(metadata, "instruction_count", "metadata"), "instruction_count"
        )
        cycle_count = _snapshot_int(
            required(metadata, "cycle_count", "metadata"), "cycle_count"
        )
        memory_reads = _snapshot_int(
            required(metadata, "memory_reads", "metadata"), "memory_reads"
        )
        memory_writes = _snapshot_int(
            required(metadata, "memory_writes", "metadata"), "memory_writes"
        )
        call_depth = _snapshot_int(
            required(metadata, "call_depth", "metadata"),
            "call_depth",
            maximum=(1 << 32) - 1,
        )
        call_stack = [
            _snapshot_int(value, f"call_stack[{index}]", maximum=0xFFFFF)
            for index, value in enumerate(
                _snapshot_list(
                    required(metadata, "call_stack", "metadata"), "call_stack"
                )
            )
        ]
        call_page_stack = [
            _snapshot_int(value, f"call_page_stack[{index}]", maximum=0xFFFFF)
            for index, value in enumerate(
                _snapshot_list(
                    required(metadata, "call_page_stack", "metadata"),
                    "call_page_stack",
                )
            )
        ]
        if any(page & 0xFFFF for page in call_page_stack):
            raise ValueError("snapshot call_page_stack contains a noncanonical page")
        call_return_widths = [
            _snapshot_int(value, f"call_return_widths[{index}]", maximum=24)
            for index, value in enumerate(
                _snapshot_list(
                    required(metadata, "call_return_widths", "metadata"),
                    "call_return_widths",
                )
            )
        ]
        if len(call_stack) != len(call_return_widths):
            raise ValueError(
                "snapshot call stack and return-width stack have different lengths"
            )
        if any(width not in (0, 16, 24) for width in call_return_widths):
            raise ValueError(
                "snapshot call_return_widths contains an unsupported width"
            )
        native_call_metrics = bool(call_stack or call_page_stack or call_return_widths)
        if snapshot_backend == "python" and native_call_metrics:
            raise ValueError(
                "Python snapshot cannot restore native call-metrics stacks"
            )
        if self.cpu.backend == "python" and native_call_metrics:
            raise ValueError(
                "Python destination cannot restore native call-metrics stacks"
            )
        current_pc = _snapshot_int(
            required(metadata, "pc", "metadata"), "pc", maximum=0xFFFFF
        )
        if current_pc != snapshot.pc:
            raise ValueError("snapshot metadata PC disagrees with registers.bin")
        power_state = required(metadata, "power_state", "metadata")
        if power_state not in ("running", "halted", "off"):
            raise ValueError(f"snapshot power_state is invalid: {power_state!r}")
        if _snapshot_bool(
            required(metadata, "external_interrupt_level", "metadata"),
            "external_interrupt_level",
        ):
            raise ValueError(
                "PCE500Emulator cannot exactly restore an asserted external IRQ level"
            )
        if _snapshot_bool(
            required(metadata, "onk_level", "metadata"),
            "onk_level",
        ):
            raise ValueError(
                "PCE500Emulator cannot exactly restore an asserted native ON-key level"
            )

        timer_raw = _snapshot_mapping(required(metadata, "timer", "metadata"), "timer")
        expected_timer_fields = {
            "enabled",
            "mti_period",
            "sti_period",
            "next_mti",
            "next_sti",
            "kb_irq_enabled",
            "instruction_start_cycle",
            "last_mti_fire_cycle",
            "last_sti_fire_cycle",
            "fired_mti_since_boundary",
            "fired_sti_since_boundary",
            "preserve_phase",
        }
        if set(timer_raw) != expected_timer_fields:
            raise ValueError("snapshot timer metadata has an unexpected shape")

        def optional_timer_cycle(name: str) -> int | None:
            value = required(timer_raw, name, "timer")
            if value is None:
                return None
            return _snapshot_int(
                value,
                f"timer.{name}",
                maximum=(1 << 64) - 1,
            )

        timer_info: dict[str, object] = {
            "enabled": _snapshot_bool(
                required(timer_raw, "enabled", "timer"), "timer.enabled"
            ),
            "mti_period": _snapshot_int(
                required(timer_raw, "mti_period", "timer"),
                "timer.mti_period",
                minimum=1,
                maximum=(1 << 63) - 1,
            ),
            "sti_period": _snapshot_int(
                required(timer_raw, "sti_period", "timer"),
                "timer.sti_period",
                minimum=1,
                maximum=(1 << 63) - 1,
            ),
            "next_mti": _snapshot_int(
                required(timer_raw, "next_mti", "timer"),
                "timer.next_mti",
                maximum=(1 << 64) - 1,
            ),
            "next_sti": _snapshot_int(
                required(timer_raw, "next_sti", "timer"),
                "timer.next_sti",
                maximum=(1 << 64) - 1,
            ),
            "kb_irq_enabled": _snapshot_bool(
                required(timer_raw, "kb_irq_enabled", "timer"),
                "timer.kb_irq_enabled",
            ),
            "instruction_start_cycle": _snapshot_int(
                required(timer_raw, "instruction_start_cycle", "timer"),
                "timer.instruction_start_cycle",
                maximum=(1 << 64) - 1,
            ),
            "last_mti_fire_cycle": optional_timer_cycle("last_mti_fire_cycle"),
            "last_sti_fire_cycle": optional_timer_cycle("last_sti_fire_cycle"),
            "fired_mti_since_boundary": _snapshot_bool(
                required(timer_raw, "fired_mti_since_boundary", "timer"),
                "timer.fired_mti_since_boundary",
            ),
            "fired_sti_since_boundary": _snapshot_bool(
                required(timer_raw, "fired_sti_since_boundary", "timer"),
                "timer.fired_sti_since_boundary",
            ),
            "preserve_phase": _snapshot_bool(
                required(timer_raw, "preserve_phase", "timer"),
                "timer.preserve_phase",
            ),
        }

        interrupts_raw = _snapshot_mapping(
            required(metadata, "interrupts", "metadata"), "interrupts"
        )
        if set(interrupts_raw) != {
            "pending",
            "in_interrupt",
            "key_irq_latched",
            "source",
            "last_fired",
            "stack",
            "next_id",
            "imr",
            "isr",
            "irq_counts",
            "last_irq",
            "irq_bit_watch",
            "delivered_masks",
        }:
            raise ValueError("snapshot interrupt metadata has an unexpected shape")
        source_name = required(interrupts_raw, "source", "interrupts")
        if source_name is not None and source_name not in IRQSource.__members__:
            raise ValueError(
                f"snapshot interrupt source is not representable: {source_name!r}"
            )
        last_fired_name = required(interrupts_raw, "last_fired", "interrupts")
        if last_fired_name is not None and last_fired_name not in {
            "RX",
            "EX",
            "TX",
            "ONK",
            "KEY",
            "STI",
            "MTI",
            "IR",
            "IRQ",
        }:
            raise ValueError(
                "snapshot last-fired interrupt source is not representable: "
                f"{last_fired_name!r}"
            )
        irq_pending = _snapshot_bool(
            required(interrupts_raw, "pending", "interrupts"),
            "interrupts.pending",
        )
        if irq_pending and source_name is None:
            raise ValueError("snapshot pending interrupt has no source")
        in_interrupt = _snapshot_bool(
            required(interrupts_raw, "in_interrupt", "interrupts"),
            "interrupts.in_interrupt",
        )
        key_irq_latched = _snapshot_bool(
            required(interrupts_raw, "key_irq_latched", "interrupts"),
            "interrupts.key_irq_latched",
        )
        interrupt_stack = [
            _snapshot_int(
                value,
                f"interrupts.stack[{index}]",
                maximum=(1 << 32) - 1,
            )
            for index, value in enumerate(
                _snapshot_list(
                    required(interrupts_raw, "stack", "interrupts"),
                    "interrupts.stack",
                )
            )
        ]
        next_interrupt_id = _snapshot_int(
            required(interrupts_raw, "next_id", "interrupts"),
            "interrupts.next_id",
            maximum=(1 << 32) - 1,
        )
        if interrupt_stack and next_interrupt_id <= max(interrupt_stack):
            raise ValueError(
                "snapshot next interrupt id must exceed every active flow id"
            )
        imr = _snapshot_int(
            required(interrupts_raw, "imr", "interrupts"),
            "interrupts.imr",
            maximum=0xFF,
        )
        isr = _snapshot_int(
            required(interrupts_raw, "isr", "interrupts"),
            "interrupts.isr",
            maximum=0xFF,
        )
        if imr != imem_bytes[IMEMRegisters.IMR.value]:
            raise ValueError("snapshot interrupt IMR disagrees with imem.bin")
        if isr != imem_bytes[IMEMRegisters.ISR.value]:
            raise ValueError("snapshot interrupt ISR disagrees with imem.bin")
        if irq_pending and isr == 0:
            raise ValueError("snapshot cannot be IRQ-pending with ISR == 0")
        if irq_pending and not in_interrupt and source_name is not None:
            source_mask = 1 << IRQSource[source_name].value
            if isr & source_mask == 0:
                raise ValueError("snapshot pending interrupt source disagrees with ISR")

        counts_raw = _snapshot_mapping(
            required(interrupts_raw, "irq_counts", "interrupts"),
            "interrupts.irq_counts",
        )
        if set(counts_raw) != {"total", "KEY", "MTI", "STI"}:
            raise ValueError("snapshot irq_counts has an unexpected shape")
        irq_counts = {
            key: _snapshot_int(
                counts_raw[key], f"interrupts.irq_counts.{key}", maximum=(1 << 32) - 1
            )
            for key in ("total", "KEY", "MTI", "STI")
        }
        last_irq_raw = _snapshot_mapping(
            required(interrupts_raw, "last_irq", "interrupts"),
            "interrupts.last_irq",
        )
        if set(last_irq_raw) != {"src", "pc", "vector"}:
            raise ValueError("snapshot last_irq has an unexpected shape")
        last_irq_src = last_irq_raw["src"]
        if last_irq_src is not None and not isinstance(last_irq_src, str):
            raise TypeError("snapshot interrupts.last_irq.src must be a string or null")
        last_irq = {
            "src": last_irq_src,
            "pc": (
                None
                if last_irq_raw["pc"] is None
                else _snapshot_int(
                    last_irq_raw["pc"], "interrupts.last_irq.pc", maximum=0xFFFFF
                )
            ),
            "vector": (
                None
                if last_irq_raw["vector"] is None
                else _snapshot_int(
                    last_irq_raw["vector"],
                    "interrupts.last_irq.vector",
                    maximum=0xFFFFF,
                )
            ),
        }
        watch_raw = _snapshot_mapping(
            required(interrupts_raw, "irq_bit_watch", "interrupts"),
            "interrupts.irq_bit_watch",
        )
        if set(watch_raw) != {"IMR", "ISR"}:
            raise ValueError("snapshot irq_bit_watch has an unexpected shape")
        irq_bit_watch: dict[str, dict[int, dict[str, list[int]]]] = {}
        for register_name in ("IMR", "ISR"):
            bits_raw = _snapshot_mapping(
                watch_raw[register_name], f"interrupts.irq_bit_watch.{register_name}"
            )
            normalized_bits: dict[int, dict[str, list[int]]] = {}
            for bit in range(8):
                raw_key = str(bit)
                if raw_key not in bits_raw:
                    raise ValueError(
                        f"snapshot irq_bit_watch.{register_name} is missing bit {bit}"
                    )
                actions = _snapshot_mapping(
                    bits_raw[raw_key],
                    f"interrupts.irq_bit_watch.{register_name}.{bit}",
                )
                if set(actions) != {"set", "clear"}:
                    raise ValueError("snapshot IRQ bit history has an unexpected shape")
                normalized_bits[bit] = {
                    action: [
                        _snapshot_int(
                            pc,
                            f"interrupts.irq_bit_watch.{register_name}.{bit}.{action}[{index}]",
                            maximum=0xFFFFF,
                        )
                        for index, pc in enumerate(
                            _snapshot_list(
                                actions[action],
                                f"interrupts.irq_bit_watch.{register_name}.{bit}.{action}",
                            )
                        )
                    ]
                    for action in ("set", "clear")
                }
            if set(bits_raw) != {str(bit) for bit in range(8)}:
                raise ValueError(
                    f"snapshot irq_bit_watch.{register_name} contains unknown bits"
                )
            irq_bit_watch[register_name] = normalized_bits
        delivered_masks = [
            _snapshot_int(value, f"interrupts.delivered_masks[{index}]", maximum=0xFF)
            for index, value in enumerate(
                _snapshot_list(
                    required(interrupts_raw, "delivered_masks", "interrupts"),
                    "interrupts.delivered_masks",
                )
            )
        ]
        native_timer_phase = bool(
            timer_info["instruction_start_cycle"] != cycle_count
            or timer_info["last_mti_fire_cycle"] is not None
            or timer_info["last_sti_fire_cycle"] is not None
            or timer_info["fired_mti_since_boundary"]
            or timer_info["fired_sti_since_boundary"]
            or not timer_info["preserve_phase"]
        )
        native_scheduler_state = bool(
            native_timer_phase or last_fired_name is not None or delivered_masks
        )
        if snapshot_backend == "python" and native_scheduler_state:
            raise ValueError("Python snapshot contains native-only scheduler state")
        if self.cpu.backend == "python" and native_scheduler_state:
            raise ValueError(
                "Python destination cannot restore native-only scheduler state"
            )
        interrupts: dict[str, object] = {
            "pending": irq_pending,
            "in_interrupt": in_interrupt,
            "key_irq_latched": key_irq_latched,
            "source": source_name,
            "last_fired": last_fired_name,
            "stack": interrupt_stack,
            "next_id": next_interrupt_id,
            "imr": imr,
            "isr": isr,
            "irq_counts": irq_counts,
            "last_irq": last_irq,
            "irq_bit_watch": irq_bit_watch,
            "delivered_masks": delivered_masks,
        }

        keyboard_state = _snapshot_mapping(
            required(metadata, "keyboard", "metadata"), "keyboard"
        )
        expected_keyboard_fields = {
            "matrix",
            "last_kol",
            "last_koh",
            "last_kil",
            "scan_enabled",
            "on_key_pressed",
        }
        if set(keyboard_state) != expected_keyboard_fields:
            raise ValueError("snapshot keyboard metadata has an unexpected shape")
        keyboard_last_kol = _snapshot_int(
            keyboard_state["last_kol"], "keyboard.last_kol", maximum=0xFF
        )
        keyboard_last_koh = _snapshot_int(
            keyboard_state["last_koh"], "keyboard.last_koh", maximum=0x0F
        )
        _snapshot_int(keyboard_state["last_kil"], "keyboard.last_kil", maximum=0xFF)
        keyboard_scan_enabled = _snapshot_bool(
            keyboard_state["scan_enabled"], "keyboard.scan_enabled"
        )
        on_key_pressed = _snapshot_bool(
            keyboard_state["on_key_pressed"], "keyboard.on_key_pressed"
        )
        if on_key_pressed != bool(imem_bytes[IMEMRegisters.SSR.value] & 0x08):
            raise ValueError("snapshot ON-key state disagrees with SSR.ONK")
        matrix_state = _snapshot_mapping(keyboard_state["matrix"], "keyboard.matrix")
        expected_matrix_fields = {
            "kol",
            "koh",
            "kil_latch",
            "scan_enabled",
            "pressed_keys",
            "key_states",
            "fifo",
            "head",
            "tail",
            "strobe_count",
            "column_histogram",
            "irq_count",
            "press_threshold",
            "release_threshold",
            "repeat_delay",
            "repeat_interval",
            "columns_active_high",
            "keyi_on_any_press",
            "raw_kil",
            "emit_events",
            "repeat_enabled",
            "keyi_latch",
            "kil_read_count",
        }
        if set(matrix_state) != expected_matrix_fields:
            raise ValueError("snapshot keyboard matrix has an unexpected shape")
        for key, maximum in (
            ("kol", 0xFF),
            ("koh", 0x0F),
            ("kil_latch", 0xFF),
            ("head", 7),
            ("tail", 7),
            ("irq_count", (1 << 32) - 1),
            ("strobe_count", (1 << 32) - 1),
            ("press_threshold", 0xFF),
            ("release_threshold", 0xFF),
            ("repeat_delay", 0xFF),
            ("repeat_interval", 0xFF),
        ):
            minimum = 1 if key in ("press_threshold", "release_threshold") else 0
            _snapshot_int(
                required(matrix_state, key, "keyboard.matrix"),
                f"keyboard.matrix.{key}",
                minimum=minimum,
                maximum=maximum,
            )
        for key in (
            "columns_active_high",
            "scan_enabled",
            "keyi_on_any_press",
            "raw_kil",
            "emit_events",
            "repeat_enabled",
            "keyi_latch",
        ):
            _snapshot_bool(
                required(matrix_state, key, "keyboard.matrix"),
                f"keyboard.matrix.{key}",
            )
        if keyboard_last_kol != matrix_state["kol"]:
            raise ValueError("snapshot keyboard KOL mirrors disagree")
        if keyboard_last_koh != matrix_state["koh"]:
            raise ValueError("snapshot keyboard KOH mirrors disagree")
        if keyboard_scan_enabled != matrix_state["scan_enabled"]:
            raise ValueError("snapshot keyboard scan-enable mirrors disagree")
        if (
            matrix_state["keyi_on_any_press"]
            or matrix_state["raw_kil"]
            or not matrix_state["emit_events"]
            or not matrix_state["repeat_enabled"]
        ):
            raise ValueError(
                "snapshot keyboard policy is not representable by the PCE host matrix"
            )
        matrix_kil_read_count = _snapshot_int(
            required(matrix_state, "kil_read_count", "keyboard.matrix"),
            "keyboard.matrix.kil_read_count",
            maximum=(1 << 32) - 1,
        )
        fifo = _snapshot_list(
            required(matrix_state, "fifo", "keyboard.matrix"), "keyboard.matrix.fifo"
        )
        if len(fifo) != 8:
            raise ValueError("snapshot keyboard FIFO must contain exactly eight bytes")
        for index, value in enumerate(fifo):
            _snapshot_int(value, f"keyboard.matrix.fifo[{index}]", maximum=0xFF)
        fifo_head = int(matrix_state["head"])
        fifo_tail = int(matrix_state["tail"])
        if bool(matrix_state["keyi_latch"]) != (fifo_head != fifo_tail):
            raise ValueError(
                "snapshot keyboard KEYI latch disagrees with FIFO occupancy"
            )
        histogram = _snapshot_list(
            required(matrix_state, "column_histogram", "keyboard.matrix"),
            "keyboard.matrix.column_histogram",
        )
        if len(histogram) != 16:
            raise ValueError("snapshot keyboard column histogram must have 16 entries")
        for index, value in enumerate(histogram):
            _snapshot_int(
                value,
                f"keyboard.matrix.column_histogram[{index}]",
                maximum=(1 << 32) - 1,
            )
        pressed_keys = _snapshot_list(
            required(matrix_state, "pressed_keys", "keyboard.matrix"),
            "keyboard.matrix.pressed_keys",
        )
        if any(not isinstance(key, str) for key in pressed_keys):
            raise TypeError("snapshot keyboard pressed_keys must contain only strings")
        if len(set(pressed_keys)) != len(pressed_keys):
            raise ValueError("snapshot keyboard pressed_keys contains duplicates")
        key_states = _snapshot_mapping(
            required(matrix_state, "key_states", "keyboard.matrix"),
            "keyboard.matrix.key_states",
        )
        expected_key_names = set(self.keyboard._matrix._key_states)
        if (
            set(key_states) != expected_key_names
            or not set(pressed_keys) <= expected_key_names
        ):
            raise ValueError("snapshot keyboard key map does not match this machine")
        physically_pressed: set[str] = set()
        for key_name, raw_state in key_states.items():
            key_state = _snapshot_mapping(
                raw_state, f"keyboard.matrix.key_states.{key_name}"
            )
            if set(key_state) != {
                "pressed",
                "debounced",
                "press_ticks",
                "release_ticks",
                "repeat_ticks",
            }:
                raise ValueError(
                    f"snapshot keyboard state for {key_name} is incomplete"
                )
            if _snapshot_bool(
                key_state["pressed"], f"keyboard.matrix.key_states.{key_name}.pressed"
            ):
                physically_pressed.add(key_name)
            _snapshot_bool(
                key_state["debounced"],
                f"keyboard.matrix.key_states.{key_name}.debounced",
            )
            for counter in ("press_ticks", "release_ticks", "repeat_ticks"):
                _snapshot_int(
                    key_state[counter],
                    f"keyboard.matrix.key_states.{key_name}.{counter}",
                    maximum=0xFF,
                )
        if set(pressed_keys) != physically_pressed:
            raise ValueError("snapshot pressed_keys disagrees with per-key state")
        keyboard_candidate = KeyboardHandler(
            None,
            columns_active_high=bool(matrix_state["columns_active_high"]),
        )
        keyboard_candidate.load_state(keyboard_state)
        candidate_matrix = keyboard_candidate.snapshot_state()["matrix"]
        if not isinstance(candidate_matrix, dict):
            raise ValueError("restored keyboard candidate is missing matrix state")
        if candidate_matrix["kil_latch"] != matrix_state["kil_latch"]:
            raise ValueError("snapshot keyboard KIL latch is not exactly representable")

        lcd_metadata = _snapshot_mapping(required(metadata, "lcd", "metadata"), "lcd")
        if set(lcd_metadata) != {
            "chip_count",
            "pages",
            "width",
            "chips",
            "cs_both_count",
            "cs_left_count",
            "cs_right_count",
        }:
            raise ValueError("snapshot LCD metadata has an unexpected shape")
        chips = _snapshot_list(required(lcd_metadata, "chips", "lcd"), "lcd.chips")
        if len(chips) != len(self.lcd.chips):
            raise ValueError("snapshot LCD chip count does not match this machine")
        chip_count = _snapshot_int(
            required(lcd_metadata, "chip_count", "lcd"), "lcd.chip_count", maximum=16
        )
        if chip_count != len(chips):
            raise ValueError("snapshot LCD chip_count disagrees with chips")
        current_lcd = self.lcd.get_snapshot()
        expected_pages = len(current_lcd.chips[0].vram)
        expected_width = len(current_lcd.chips[0].vram[0])
        pages = _snapshot_int(
            required(lcd_metadata, "pages", "lcd"), "lcd.pages", minimum=1
        )
        width = _snapshot_int(
            required(lcd_metadata, "width", "lcd"), "lcd.width", minimum=1
        )
        if (pages, width) != (expected_pages, expected_width):
            raise ValueError("snapshot LCD geometry does not match this machine")
        if len(lcd_payload) != len(chips) * pages * width:
            raise ValueError("lcd_vram.bin size mismatch")
        if _snapshot_int(
            required(metadata, "lcd_payload_size", "metadata"),
            "lcd_payload_size",
            maximum=0x1000000,
        ) != len(lcd_payload):
            raise ValueError("snapshot lcd_payload_size mismatch")
        for index, raw_chip in enumerate(chips):
            chip = _snapshot_mapping(raw_chip, f"lcd.chips[{index}]")
            if set(chip) != {
                "on",
                "start_line",
                "page",
                "y_address",
                "instruction_count",
                "data_write_count",
                "data_read_count",
                "on_off_count",
            }:
                raise ValueError(
                    f"snapshot LCD chip {index} metadata has an unexpected shape"
                )
            _snapshot_bool(required(chip, "on", "LCD chip"), f"lcd.chips[{index}].on")
            for key, maximum in (
                ("start_line", 0x3F),
                ("page", pages - 1),
                ("y_address", width - 1),
                ("instruction_count", (1 << 32) - 1),
                ("data_write_count", (1 << 32) - 1),
                ("data_read_count", (1 << 32) - 1),
                ("on_off_count", (1 << 32) - 1),
            ):
                _snapshot_int(
                    required(chip, key, f"lcd.chips[{index}]"),
                    f"lcd.chips[{index}].{key}",
                    maximum=maximum,
                )
        for key in ("cs_both_count", "cs_left_count", "cs_right_count"):
            _snapshot_int(
                required(lcd_metadata, key, "lcd"),
                f"lcd.{key}",
                maximum=(1 << 32) - 1,
            )
        lcd_candidate = HD61202Controller()
        lcd_candidate.load_snapshot(lcd_metadata, lcd_payload)

        kb_metrics_raw = _snapshot_mapping(
            required(metadata, "kb_metrics", "metadata"), "kb_metrics"
        )
        if set(kb_metrics_raw) != {
            "irq_count",
            "strobe_count",
            "column_hist",
            "last_cols",
            "last_kol",
            "last_koh",
            "kil_reads",
            "kb_irq_enabled",
        }:
            raise ValueError("snapshot kb_metrics has an unexpected shape")
        kb_irq_count = _snapshot_int(
            required(kb_metrics_raw, "irq_count", "kb_metrics"),
            "kb_metrics.irq_count",
        )
        kb_strobe_count = _snapshot_int(
            required(kb_metrics_raw, "strobe_count", "kb_metrics"),
            "kb_metrics.strobe_count",
        )
        kb_hist_raw = _snapshot_list(
            required(kb_metrics_raw, "column_hist", "kb_metrics"),
            "kb_metrics.column_hist",
        )
        if len(kb_hist_raw) != len(self._kb_col_hist):
            raise ValueError("snapshot keyboard metric histogram has wrong length")
        kb_hist = [
            _snapshot_int(value, f"kb_metrics.column_hist[{index}]")
            for index, value in enumerate(kb_hist_raw)
        ]
        last_columns = [
            _snapshot_int(value, f"kb_metrics.last_cols[{index}]", maximum=15)
            for index, value in enumerate(
                _snapshot_list(
                    required(kb_metrics_raw, "last_cols", "kb_metrics"),
                    "kb_metrics.last_cols",
                )
            )
        ]
        if len(set(last_columns)) != len(last_columns):
            raise ValueError("snapshot kb_metrics.last_cols contains duplicates")
        last_kol = _snapshot_int(
            required(kb_metrics_raw, "last_kol", "kb_metrics"),
            "kb_metrics.last_kol",
            maximum=0xFF,
        )
        last_koh = _snapshot_int(
            required(kb_metrics_raw, "last_koh", "kb_metrics"),
            "kb_metrics.last_koh",
            maximum=0x0F,
        )
        kil_reads = _snapshot_int(
            required(kb_metrics_raw, "kil_reads", "kb_metrics"),
            "kb_metrics.kil_reads",
        )
        if kil_reads != matrix_kil_read_count:
            raise ValueError("snapshot keyboard KIL-read counters disagree")
        kb_irq_enabled = _snapshot_bool(
            required(kb_metrics_raw, "kb_irq_enabled", "kb_metrics"),
            "kb_metrics.kb_irq_enabled",
        )
        if kb_irq_enabled != timer_info["kb_irq_enabled"]:
            raise ValueError("snapshot keyboard IRQ-enable fields disagree")
        memory_dump_pc = _snapshot_int(
            required(metadata, "memory_dump_pc", "metadata"),
            "memory_dump_pc",
            maximum=0xFFFFF,
        )
        fast_mode = _snapshot_bool(
            required(metadata, "fast_mode", "metadata"), "fast_mode"
        )

        llama_impl = None
        native_hooks: dict[str, Any] = {}
        if self.cpu.backend == "llama":
            llama_impl = self.cpu.unwrap()
            for name in (
                "validate_keyboard_snapshot",
                "validate_scheduler_snapshot",
                "validate_runtime_snapshot",
                "restore_keyboard_snapshot",
                "restore_scheduler_snapshot",
                "restore_runtime_snapshot",
                "_initialise_rust_memory",
                "set_perf_instr_counter",
            ):
                hook = getattr(llama_impl, name, None)
                if not callable(hook):
                    raise RuntimeError(
                        f"LLAMA backend is missing required snapshot hook {name}"
                    )
                native_hooks[name] = hook
            native_hooks["validate_keyboard_snapshot"](keyboard_state)
            native_hooks["validate_scheduler_snapshot"](
                timer_info, interrupts, cycle_count
            )
            native_hooks["validate_runtime_snapshot"](
                cycle_count,
                memory_reads,
                memory_writes,
                call_depth,
                call_sub_level,
                call_stack,
                call_page_stack,
                call_return_widths,
                power_state,
            )

        if not commit:
            return

        if backend and backend != snapshot_backend:
            print(
                f"[snapshot] Warning: snapshot backend {snapshot_backend} "
                f"!= requested {backend}"
            )
        elif snapshot_backend != self.cpu.backend:
            print(
                f"[snapshot] Warning: emulator backend {self.cpu.backend} "
                f"!= snapshot backend {snapshot_backend}"
            )

        # No validation below this line: every fallible parser and native
        # candidate builder has already succeeded. Unexpected integration
        # failures poison the wrapper so a partial commit cannot be executed.
        commit_started = False
        try:
            commit_started = True
            self.memory.external_memory[:] = flat_memory
            self.memory.external_memory[-len(imem_bytes) :] = imem_bytes
            for overlay_data, payload in overlay_updates:
                overlay_data[: len(payload)] = payload
            (
                self.memory._card_present,
                self.memory._card_writable,
                self.memory._card_len,
                self.memory._card_data,
            ) = card_candidate
            self.lcd.load_snapshot(lcd_metadata, lcd_payload)
            self.keyboard.load_state(keyboard_state)
            self.cpu.apply_snapshot(snapshot)

            self.instruction_count = instruction_count
            self.cycle_count = cycle_count
            self.memory_read_count = memory_reads
            self.memory_write_count = memory_writes
            self.call_depth = call_depth
            self._current_pc = current_pc
            self._last_pc = current_pc
            self._trace_instr_count = instruction_count
            self._active_trace_instruction = None
            self._trace_substep = 0
            self.start_time = time.time()

            self._scheduler.mti_period = int(timer_info["mti_period"])
            self._scheduler.sti_period = int(timer_info["sti_period"])
            self._scheduler.next_mti = int(timer_info["next_mti"])
            self._scheduler.next_sti = int(timer_info["next_sti"])
            self._scheduler.enabled = bool(timer_info["enabled"])
            self._irq_pending = irq_pending
            self._in_interrupt = in_interrupt
            self._key_irq_latched = key_irq_latched
            self._irq_source = IRQSource[source_name] if source_name else None
            self._interrupt_stack = interrupt_stack
            self._next_interrupt_id = next_interrupt_id
            self.irq_counts = irq_counts
            self.last_irq = last_irq
            self.irq_bit_watch = irq_bit_watch

            self._kb_irq_count = kb_irq_count
            self._kb_strobe_count = kb_strobe_count
            self._kb_col_hist = kb_hist
            self._last_kil_columns = last_columns
            self._last_kol = last_kol
            self._last_koh = last_koh
            self._kil_read_count = kil_reads
            self._kb_irq_enabled = kb_irq_enabled
            self.MEMORY_DUMP_PC = memory_dump_pc
            self.fast_mode = fast_mode
            # Python stores the compatibility ``halted`` flag separately.
            # Set it first so HALT/OFF snapshots cannot resume by executing an
            # opcode; setting power_state last preserves OFF in the native
            # proxy, where ``halted = True`` selects HALT.
            self.cpu.state.halted = power_state != "running"
            self.cpu.state.power_state = power_state

            self.instruction_history.clear()
            self.memory.clear_imem_access_tracking()

            if llama_impl is not None:
                native_hooks["_initialise_rust_memory"]()
                native_hooks["restore_keyboard_snapshot"](keyboard_state)
                native_hooks["restore_scheduler_snapshot"](
                    timer_info, interrupts, cycle_count
                )
                native_hooks["restore_runtime_snapshot"](
                    cycle_count,
                    memory_reads,
                    memory_writes,
                    call_depth,
                    call_sub_level,
                    call_stack,
                    call_page_stack,
                    call_return_widths,
                    power_state,
                )
                native_hooks["set_perf_instr_counter"](instruction_count)

            self.memory.set_cpu(self.cpu)
        except Exception as exc:
            if commit_started and self._poisoned is None:
                self._poisoned = (
                    "snapshot commit failed after validation: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise

    def stop_tracing(self) -> None:
        if self.perfetto_enabled:
            print("Stopping Perfetto tracing...")
            trace_dispatcher.stop_trace()
        if self._new_trace_enabled and new_tracer.enabled:
            print(f"Stopping new tracing, saved to {self._trace_path}")
            new_tracer.set_manual_clock_mode(False)
            new_tracer.stop()
        try:
            if hasattr(self.cpu, "flush_perfetto"):
                self.cpu.flush_perfetto()
        except Exception:
            pass
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
                self._tick_timers()

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
            opcode = self.memory.peek_byte_for_preflight(pc, pc) & 0xFF
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
        """Emit the captured instruction snapshot as a Perfetto slice event."""

        if not snapshot or not new_tracer.enabled:
            return
        op_index = snapshot.get("op_index")
        if not isinstance(op_index, int):
            return
        new_tracer.set_manual_clock_units(op_index)
        pc = snapshot.get("pc")
        opcode = snapshot.get("opcode")
        mnemonic: Optional[str] = None
        if isinstance(pc, int):
            try:
                mnemonic = self.cpu.decode_instruction(
                    pc,
                    read_fn=lambda address: self.memory.peek_byte_for_preflight(
                        address, pc
                    ),
                ).name()
            except Exception:
                mnemonic = None
        if isinstance(pc, int) and isinstance(mnemonic, str) and mnemonic:
            name = f"{mnemonic} @0x{pc:06X}"
        elif isinstance(pc, int) and isinstance(opcode, int):
            name = f"op=0x{opcode & 0xFF:02X} @0x{pc:06X}"
        elif isinstance(pc, int):
            name = f"@0x{pc:06X}"
        else:
            name = "instruction"
        payload: Dict[str, Any] = {
            "backend": "python",
        }
        if isinstance(pc, int):
            payload["pc"] = pc & 0xFFFFFF
        if isinstance(opcode, int):
            payload["opcode"] = opcode & 0xFF
        payload["op_index"] = op_index
        # Include IMR/ISR from internal memory so perfetto comparisons can catch
        # interrupt masking regressions across backends.
        try:
            imr = self.memory.internal_memory.read_byte(0xFB)
            isr = self.memory.internal_memory.read_byte(0xFC)
        except Exception:
            # Fall back to routed reads so InstructionTrace always carries IMR/ISR.
            try:
                imr = self.memory.read_byte(
                    INTERNAL_MEMORY_START + IMEMRegisters.IMR, cpu_pc=pc
                )
            except Exception:
                imr = None
            try:
                isr = self.memory.read_byte(
                    INTERNAL_MEMORY_START + IMEMRegisters.ISR, cpu_pc=pc
                )
            except Exception:
                isr = None
        if isinstance(imr, int):
            payload["mem_imr"] = imr & 0xFF
        if isinstance(isr, int):
            payload["mem_isr"] = isr & 0xFF
        registers = snapshot.get("registers", {})
        for reg_name, value in registers.items():
            payload[f"reg_{reg_name.lower()}"] = int(value)
        new_tracer.begin_slice("Instructions", name, payload)
        new_tracer.set_manual_clock_units(op_index + 1)
        new_tracer.end_slice("Instructions")

        self._trace_instr_count = op_index + 1
        new_tracer.set_manual_clock_units(op_index)
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
        return int(instr_index)

    def _trace_execution(self, pc: int, opcode: Optional[int]):
        payload: Dict[str, Any] = {"pc": f"0x{pc:06X}"}
        if opcode is not None:
            payload["opcode"] = f"0x{opcode:02X}"
        payload.update(self._build_register_annotations())
        # Always dispatch to legacy observers for register snapshots, even when
        # the perfetto tracer is active.
        trace_dispatcher.record_instant("Execution", f"Exec@0x{pc:06X}", payload)
        if new_tracer.enabled and not getattr(self, "_new_trace_enabled", False):
            new_tracer.instant("Execution", f"Exec@0x{pc:06X}", payload)

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
            if (
                self._new_trace_enabled
                and new_tracer.enabled
                and isinstance(dest_addr, int)
            ):
                op_index = max(0, int(self.instruction_count) - 1)
                new_tracer.set_manual_clock_units(op_index)
                new_tracer.begin_slice(
                    "Functions",
                    f"fn@0x{dest_addr:06X}",
                    {
                        "from": pc_before & 0xFFFFFF,
                        "to": dest_addr & 0xFFFFFF,
                        "depth": int(self.call_depth),
                        "op_index": op_index,
                    },
                )

        elif isinstance(instr, RetInstruction):
            ret_depth = self.call_depth
            trace_dispatcher.end_function("CPU", pc_before)
            self.call_depth = max(0, self.call_depth - 1)
            instr_name = type(instr).__name__
            if (
                self._new_trace_enabled
                and new_tracer.enabled
                and instr_name in {"RET", "RETF"}
            ):
                op_index = max(0, int(self.instruction_count) - 1)
                new_tracer.set_manual_clock_units(op_index + 1)
                new_tracer.end_slice("Functions")
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
            # Software IR execution already installed the one architectural
            # vector fetch as PC. Tracing must not read a callback-backed bus a
            # second time.
            vector_addr = pc_after
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
        self._set_isr_bits(int(ISRFlag.KEYI))
        self._key_irq_latched = True
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

    def press_key(self, key_code: str) -> bool:
        result = self.keyboard.press_key(key_code) if self.keyboard else False
        # The handler commits SSR.ONK and ISR.ONKI first (transactionally in
        # LLAMA).  Do not advertise a pending host IRQ if that mutation fails.
        if result and key_code == "KEY_ON":
            self._irq_pending = True
            self._irq_source = IRQSource.ONK
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
        val = self.memory.read_byte(isr_addr) & 0xFF
        new_val = (val | (mask & 0xFF)) & 0xFF
        self.memory.write_byte(isr_addr, new_val)
        if mask & int(ISRFlag.KEYI):
            # Latch a plausible IRQ source so pending checks can deliver.
            try:
                if getattr(self, "_irq_source", None) is None and not getattr(
                    self, "_in_interrupt", False
                ):
                    self._irq_source = IRQSource.KEY
            except Exception:
                pass
            # Emit a dedicated marker when KEYI is asserted so trace diffing can spot it.
            try:
                self._trace_irq_instant(
                    "KEYI_Set",
                    IRQSource.KEY,
                    {
                        "pc": self.cpu.regs.get(RegisterName.PC),
                        "prev": val,
                        "value": new_val,
                        "imr": self.memory.read_byte(
                            INTERNAL_MEMORY_START + IMEMRegisters.IMR
                        )
                        & 0xFF,
                        "still_set": bool(
                            self.memory.read_byte(isr_addr) & int(ISRFlag.KEYI)
                        ),
                    },
                )
            except Exception:
                pass
            if False:
                print(
                    f"[keyi-debug] pc=0x{int(self.cpu.regs.get(RegisterName.PC)) & 0xFFFFFF:06X} prev=0x{val:02X} new=0x{new_val:02X} imr=0x{self.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR) & 0xFF:02X}"
                )
        else:
            # Log when KEYI would not be set but was previously set.
            if (val & int(ISRFlag.KEYI)) and not (new_val & int(ISRFlag.KEYI)):
                try:
                    self._trace_irq_instant(
                        "KEYI_Clear",
                        IRQSource.KEY,
                        {
                            "pc": self.cpu.regs.get(RegisterName.PC),
                            "prev": val,
                            "value": new_val,
                        },
                    )
                except Exception:
                    pass
        if IRQ_DEBUG_ENABLED:
            pc = self.cpu.regs.get(RegisterName.PC)
            _log_irq_debug(
                f"set_isr_bits mask=0x{mask:02X} prev=0x{val:02X} new=0x{new_val:02X} pc=0x{pc:06X}"
            )

    def _tick_timers(self) -> None:
        """Rough timer emulation: set ISR bits periodically and arm IRQ."""
        fired_sources = tuple(self._scheduler.advance(self.cycle_count))
        if not fired_sources:
            return

        key_events: List[MatrixEvent] = []
        for source in fired_sources:
            if source is TimerSource.MTI:
                if self._scan_on_timer:
                    key_events = self.keyboard.scan_tick()
                self._set_isr_bits(int(ISRFlag.MTI))
                self._irq_pending = True
                self._irq_source = IRQSource.MTI
                try:
                    self._trace_irq_instant(
                        "TimerFired",
                        self._irq_source,
                        {
                            "cycle": self.cycle_count,
                            "imr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.IMR
                            )
                            & 0xFF,
                            "isr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.ISR
                            )
                            & 0xFF,
                        },
                    )
                except Exception:
                    pass
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"timer fired source=MTI cycle={self.cycle_count} next_mti={self._scheduler.next_mti}"
                    )
            elif source is TimerSource.STI:
                self._set_isr_bits(int(ISRFlag.STI))
                self._irq_pending = True
                self._irq_source = IRQSource.STI
                try:
                    self._trace_irq_instant(
                        "TimerFired",
                        self._irq_source,
                        {
                            "cycle": self.cycle_count,
                            "imr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.IMR
                            )
                            & 0xFF,
                            "isr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.ISR
                            )
                            & 0xFF,
                        },
                    )
                except Exception:
                    pass
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"timer fired source=STI cycle={self.cycle_count} next_sti={self._scheduler.next_sti}"
                    )

        # If we have pressed keys but no events surfaced, emit a diagnostic marker.
        if (
            not key_events
            and hasattr(self.keyboard, "_pressed_keys")
            and getattr(self.keyboard, "_pressed_keys")
        ):
            try:
                self._trace_irq_instant(
                    "KeyScanEmpty",
                    self._irq_source,
                    {
                        "pressed": len(getattr(self.keyboard, "_pressed_keys", [])),
                        "strobe_count": getattr(self.keyboard, "strobe_count", 0),
                        "kol": getattr(self.keyboard, "kol_value", 0),
                        "koh": getattr(self.keyboard, "koh_value", 0),
                        "pc": self.cpu.regs.get(RegisterName.PC),
                    },
                )
            except Exception:
                pass

        if key_events:
            if self._kb_irq_enabled:
                self._key_irq_latched = True
                self._set_isr_bits(int(ISRFlag.KEYI))
                self._irq_pending = True
                self._irq_source = IRQSource.KEY
                self._kb_irq_count += len(key_events)
                if IRQ_DEBUG_ENABLED:
                    _log_irq_debug(
                        f"key_events_irq count={len(key_events)} cycle={self.cycle_count}"
                    )
                # Emit a delivery marker for perfetto parity when scan raises KEYI.
                try:
                    self._trace_irq_instant(
                        "KeyIRQ",
                        self._irq_source,
                        {
                            "pc": self.cpu.regs.get(RegisterName.PC),
                            "y": self.cpu.regs.get(RegisterName.Y),
                            "events": len(key_events),
                            "imr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.IMR
                            )
                            & 0xFF,
                            "isr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.ISR
                            )
                            & 0xFF,
                        },
                    )
                except Exception:
                    pass
        # Trace scan outcome to perfetto to understand KEYI assertion cadence.
        try:
            pressed = len(getattr(self.keyboard, "_pressed_keys", []))
            active_cols = []
            try:
                if hasattr(self.keyboard, "get_active_columns"):
                    active_cols = list(self.keyboard.get_active_columns())
            except Exception:
                active_cols = []
            self._trace_irq_instant(
                "KeyScanEvent",
                IRQSource.KEY,
                {
                    "events": len(key_events),
                    "pressed": pressed,
                    "strobe_count": getattr(self.keyboard, "strobe_count", 0),
                    "kol": getattr(self.keyboard, "kol_value", 0),
                    "koh": getattr(self.keyboard, "koh_value", 0),
                    "active_cols": active_cols,
                    "pc": self.cpu.regs.get(RegisterName.PC),
                    "imr": self.memory.read_byte(
                        INTERNAL_MEMORY_START + IMEMRegisters.IMR
                    )
                    & 0xFF,
                    "isr": self.memory.read_byte(
                        INTERNAL_MEMORY_START + IMEMRegisters.ISR
                    )
                    & 0xFF,
                },
            )
        except Exception:
            pass

    def _scan_keyboard_per_instruction(self) -> None:
        """Scan the key matrix once per instruction when timer-driven scans are disabled."""
        if getattr(self, "keyboard", None) is None:
            return
        if self._scan_on_timer:
            return
        try:
            events = self.keyboard.scan_tick()
        except Exception:
            return
        fifo_pending = False
        try:
            fifo_pending = bool(self.keyboard.fifo_snapshot())
        except Exception:
            fifo_pending = False
        if events or fifo_pending:
            try:
                self.keyboard.drain_fifo_to_pce500_iocs_workspace(self._kb_irq_enabled)
            except Exception:
                pass
        if events:
            self._kb_irq_count += len(events)
        if self._kb_irq_enabled and (events or fifo_pending):
            self._key_irq_latched = True
            self._set_isr_bits(int(ISRFlag.KEYI))
            self._irq_pending = True
            if not getattr(self, "_in_interrupt", False):
                self._irq_source = IRQSource.KEY
            if events:
                try:
                    self._trace_irq_instant(
                        "KeyIRQ",
                        self._irq_source,
                        {
                            "pc": self.cpu.regs.get(RegisterName.PC),
                            "y": self.cpu.regs.get(RegisterName.Y),
                            "events": len(events),
                            "imr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.IMR
                            )
                            & 0xFF,
                            "isr": self.memory.read_byte(
                                INTERNAL_MEMORY_START + IMEMRegisters.ISR
                            )
                            & 0xFF,
                        },
                    )
                except Exception:
                    pass

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
            0xF299C: "lcd_stream_write_data_byte_f299c",
            0xF2E24: "lcd_write_clear_sequences_f2e24",
            0xF2E50: "sub_f2e50_screen_blit",
            0xF2E9E: "sub_f2e9e_screen_fill",
            0xF2DA5: "sub_f2da5_buffer_copy",
            0xF29B8: "sub_f29b8_draw_menu",
            0x0299C: "lcd_stream_write_data_byte_f299c",
            0x02E24: "lcd_write_clear_sequences_f2e24",
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
            rom_image: Full 1MB dump matching ``data/pc-e500-en.bin``.
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
            # reset() already validated and installed the current vector.
            entry_target = self.cpu.regs.get(RegisterName.PC)
        else:
            # This method used to mask and silently swallow a bad reset vector
            # after changing RAM/IMEM. Resolve it first through the same
            # safe-peek contract used by RESET so failure is atomic.
            entry_target = self._fetch_validated_vector(ENTRY_POINT_ADDR)

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

        # Ensure PC points to the already-validated ROM entry vector in case
        # callers expect to execute instructions immediately after bootstrapping.
        self.cpu.regs.set(RegisterName.PC, entry_target)

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
            if reg_name in ("IMR", "ISR"):
                self._trace_irq_instant(
                    f"{reg_name}_Write",
                    self._irq_source,
                    {"pc": pc, "prev": prev_val & 0xFF, "value": new_val & 0xFF},
                )
        except Exception:
            pass

    def release_key(self, key_code: str):
        pending_source: Optional[IRQSource] = None
        if key_code == "KEY_ON":
            # Validate the post-release bookkeeping before asking the handler
            # to commit.  The fields below are changed only after the Python or
            # native keyboard mutation succeeds. Physical release clears only
            # SSR.ONK; ISR.ONKI stays latched until firmware acknowledges it.
            isr = (
                self.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) & 0xFF
            )
            pending_source = _highest_pending_irq_source(isr)
        if self.keyboard:
            self.keyboard.release_key(key_code)
        if key_code == "KEY_ON":
            self._irq_pending = pending_source is not None
            self._irq_source = pending_source
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

        if reg_name == "KIL" and access_type == "read":
            try:
                if self.keyboard:
                    self.keyboard.consume_pending_events()
                self._key_irq_latched = False
            except Exception:
                pass

        # Perfetto logging for IMR/ISR writes to spot masking/clearing.
        try:
            if reg_name in ("IMR", "ISR") and access_type == "write":
                prev = self._last_imem_values.get(reg_name, value & 0xFF)
                self._last_imem_values[reg_name] = value & 0xFF
                src = None
                if reg_name == "ISR" and (value & int(ISRFlag.KEYI)):
                    src = IRQSource.KEY
                if reg_name == "IMR" and value == 0:
                    self._trace_irq_instant(
                        "IMR_Clear",
                        None,
                        {"pc": pc, "prev": prev & 0xFF},
                    )
                # Detect KEYI clear transitions for visibility.
                if (
                    reg_name == "ISR"
                    and (prev & int(ISRFlag.KEYI))
                    and not (value & int(ISRFlag.KEYI))
                ):
                    try:
                        imr_val = self.memory.read_byte(
                            INTERNAL_MEMORY_START + IMEMRegisters.IMR
                        )
                        key_unmasked = (
                            imr_val & (int(IMRFlag.IRM) | int(IMRFlag.KEY))
                        ) == (int(IMRFlag.IRM) | int(IMRFlag.KEY))
                        if key_unmasked:
                            if self.keyboard:
                                self.keyboard.consume_pending_events()
                            self._key_irq_latched = False
                    except Exception:
                        pass
                    self._trace_irq_instant(
                        "KEYI_Clear",
                        IRQSource.KEY,
                        {"pc": pc, "prev": prev & 0xFF, "value": value & 0xFF},
                    )
                self._trace_irq_instant(
                    "IMEM_Write",
                    src,
                    {
                        "reg": reg_name,
                        "prev": prev & 0xFF,
                        "value": value & 0xFF,
                        "pc": pc,
                    },
                )
        except Exception:
            pass

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
    return None


def _stack_snapshot_len() -> int:
    return 10


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
