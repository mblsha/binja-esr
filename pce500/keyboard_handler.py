"""Keyboard handler wrapping the deterministic matrix implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import PCE500Memory

from .keyboard_matrix import (
    DEFAULT_PRESS_TICKS,
    DEFAULT_RELEASE_TICKS,
    KEY_LOCATIONS,
    KeyboardMatrix,
    KeyLocation,
    MatrixEvent,
)
from .memory import INTERNAL_MEMORY_START
from .tracing import trace_dispatcher

# Keyboard IMEM register offsets (relative to internal RAM base).
KOL = 0xF0
KOH = 0xF1
KIL = 0xF2

# Re-export historical debounce constants for callers that still import them.
DEFAULT_DEBOUNCE_READS = DEFAULT_PRESS_TICKS
DEFAULT_RELEASE_READS = DEFAULT_RELEASE_TICKS


class PCE500KeyboardHandler:
    """High-level keyboard handler used by the emulator."""

    def __init__(
        self,
        memory: "PCE500Memory" | None = None,
        *,
        columns_active_high: bool = True,
    ):
        self._memory = memory
        self._matrix = KeyboardMatrix(
            columns_active_high=columns_active_high, memory=memory
        )
        self._last_kol = self._matrix.kol
        self._last_koh = self._matrix.koh
        self._last_kil = 0
        self._scan_enabled = True
        self._bridge_cpu = None
        self._bridge_enabled = False
        self._bridge_keys: set[str] = set()
        self._bridge_cpu = None
        self._bridge_enabled = False

    # ------------------------------------------------------------------ #
    # Public API used by emulator/tests
    # ------------------------------------------------------------------ #

    def press_key(self, key_code: str) -> bool:
        if self._bridge_enabled:
            if not self._forward_bridge_event(key_code, release=False):
                return False
            self._bridge_keys.add(key_code)
            self._last_kil = self._read_kil_from_memory()
            return True
        if not self._matrix.press_key(key_code):
            return False
        self._last_kil = self._matrix.peek_kil()
        self._forward_bridge_event(key_code, release=False)
        return True

    def release_key(self, key_code: str) -> None:
        if self._bridge_enabled:
            if self._forward_bridge_event(key_code, release=True):
                self._bridge_keys.discard(key_code)
                self._last_kil = self._read_kil_from_memory()
            return
        self._matrix.release_key(key_code)
        self._last_kil = self._matrix.peek_kil()
        self._forward_bridge_event(key_code, release=True)

    def release_all_keys(self) -> None:
        if self._bridge_enabled:
            for key_code in list(self._bridge_keys):
                self._forward_bridge_event(key_code, release=True)
            self._bridge_keys.clear()
            self._last_kil = self._read_kil_from_memory()
            return
        self._matrix.release_all_keys()
        self._last_kil = 0

    def handle_register_read(self, register: int) -> int | None:
        reg = register & 0xFF
        if reg == KOL:
            val = self._last_kol
            self._matrix.trace_kio("read_kol")
            return val
        if reg == KOH:
            val = self._last_koh
            self._matrix.trace_kio("read_koh")
            return val
        if reg == KIL:
            if not self._scan_enabled or self._ksd_masked():
                return 0x00
            self.scan_tick()
            self._last_kil = self._matrix.peek_kil()
            # Emit KIL read with a best-effort PC from CPU regs.
            pc = None
            try:
                from sc62015.pysc62015.emulator import RegisterName

                pc = (
                    self._memory.cpu.regs.get(RegisterName.PC)
                    if self._memory and getattr(self._memory, "cpu", None)
                    else None
                )
            except Exception:
                pc = None
            self._matrix.trace_kio("read_kil", pc=pc)
            # Perfetto/dispatcher logging of KIL read (ensure visibility even if hooks bypassed)
            try:
                tracer = getattr(self._memory, "_perf_tracer", None)
                if tracer is not None and hasattr(tracer, "instant"):
                    tracer.instant(
                        "KIO",
                        "read@KIL",
                        {
                            "pc": pc & 0xFFFFFF if pc is not None else None,
                            "value": self._last_kil & 0xFF,
                            "offset": reg,
                        },
                    )
                # Legacy dispatcher for completeness
                trace_dispatcher.record_instant(
                    "KIO",
                    "read@KIL",
                    {
                        "pc": f"0x{pc & 0xFFFFFF:06X}" if pc is not None else "N/A",
                        "value": f"0x{self._last_kil & 0xFF:02X}",
                        "offset": f"0x{reg:02X}",
                    },
                )
            except Exception:
                pass
            return self._last_kil
        return None

    def handle_register_write(self, register: int, value: int) -> bool:
        reg = register & 0xFF
        if reg == KOL:
            self._matrix.write_kol(value & 0xFF)
            self._last_kol = self._matrix.kol
            self._matrix.trace_kio("write_kol")
            return True
        if reg == KOH:
            self._matrix.write_koh(value & 0x0F)
            self._last_koh = self._matrix.koh
            self._matrix.trace_kio("write_koh")
            return True
        if reg == KIL:
            # KIL is read-only in hardware; ignore writes.
            return True
        return False

    def scan_tick(self) -> list[MatrixEvent]:
        events = self._matrix.scan_tick()
        if events:
            # Ensure KIL latch reflects the latest active rows after the tick.
            self._last_kil = self._matrix.peek_kil()
        return events

    def set_scan_enabled(self, enabled: bool) -> None:
        self._scan_enabled = bool(enabled)
        self._matrix.scan_enabled = bool(enabled)

    def set_bridge_cpu(self, bridge_cpu, enabled: bool) -> None:
        """Attach/detach the pure-Rust bridge CPU."""

        self._bridge_cpu = bridge_cpu
        self._bridge_enabled = bool(enabled and bridge_cpu)
        if not self._bridge_enabled:
            self._bridge_keys.clear()

    def _forward_bridge_event(self, key_code: str, *, release: bool) -> bool:
        if not self._bridge_enabled or not self._bridge_cpu:
            return False
        loc = KEY_LOCATIONS.get(key_code)
        if loc is None:
            return False
        matrix_code = ((loc.column & 0x0F) << 3) | (loc.row & 0x07)
        try:
            if release:
                return bool(self._bridge_cpu.keyboard_release_matrix_code(matrix_code))
            return bool(self._bridge_cpu.keyboard_press_matrix_code(matrix_code))
        except Exception:
            return False

    def _read_kil_from_memory(self) -> int:
        if not self._memory:
            return 0
        try:
            return self._memory.read_byte(INTERNAL_MEMORY_START + KIL) & 0xFF
        except Exception:
            return 0

    def get_active_columns(self) -> list[int]:
        return self._matrix.get_active_columns()

    def get_pressed_keys(self) -> list[str]:
        return list(self._matrix.get_pressed_keys())

    def peek_keyboard_input(self) -> int:
        return self._matrix.peek_kil()

    def get_debug_info(self) -> dict[str, object]:
        return {
            "pressed_keys": self.get_pressed_keys(),
            "kol": f"0x{self._last_kol:02X}",
            "koh": f"0x{self._last_koh:02X}",
            "kil": f"0x{self._last_kil:02X}",
            "selected_columns": self.get_active_columns(),
            "fifo": [f"0x{entry:02X}" for entry in self._matrix.fifo_snapshot()],
            "strobe_count": self._matrix.strobe_count,
            "irq_count": self._matrix.irq_count,
        }

    def get_queue_info(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        snapshot = self._matrix.fifo_snapshot()
        for raw in snapshot:
            release = bool(raw & 0x80)
            matrix_code = raw & 0x7F
            col = matrix_code >> 3
            row = matrix_code & 0x07
            entries.append(
                {
                    "matrix_code": matrix_code,
                    "release": release,
                    "column": col,
                    "row": row,
                    "raw": f"0x{raw:02X}",
                }
            )
        return entries

    def fifo_snapshot(self) -> list[int]:
        return self._matrix.fifo_snapshot()

    def snapshot_state(self) -> dict[str, object]:
        """Capture current keyboard handler + matrix state."""

        return {
            "matrix": self._matrix.snapshot_state(),
            "last_kol": self._last_kol,
            "last_koh": self._last_koh,
            "last_kil": self._last_kil,
            "scan_enabled": self._scan_enabled,
        }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore the keyboard handler from ``snapshot_state`` output."""

        matrix_state = state.get("matrix")
        if isinstance(matrix_state, dict):
            self._matrix.load_state(matrix_state)

        self._last_kol = int(state.get("last_kol", self._last_kol)) & 0xFF
        self._last_koh = int(state.get("last_koh", self._last_koh)) & 0xFF
        self._last_kil = int(state.get("last_kil", self._last_kil)) & 0xFF
        self._scan_enabled = bool(state.get("scan_enabled", self._scan_enabled))
        self._matrix.scan_enabled = self._scan_enabled

    # Metrics used by emulator instrumentation --------------------------------

    @property
    def strobe_count(self) -> int:
        return self._matrix.strobe_count

    @property
    def column_histogram(self) -> list[int]:
        return list(self._matrix.column_histogram)

    @property
    def irq_count(self) -> int:
        return self._matrix.irq_count

    @property
    def kol_value(self) -> int:
        return self._last_kol

    @property
    def koh_value(self) -> int:
        return self._last_koh

    @property
    def key_locations(self) -> dict[str, KeyLocation]:
        return KEY_LOCATIONS

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ksd_masked(self) -> bool:
        """Return True if the LCD/keyboard strobe disable bit is active."""
        if self._memory is None:
            return False
        try:
            from sc62015.pysc62015.instr.opcodes import IMEMRegisters

            addr = INTERNAL_MEMORY_START + IMEMRegisters.LCC
            value = self._memory.read_byte(addr)
            return (value & 0x04) != 0
        except Exception:
            return False


__all__ = [
    "PCE500KeyboardHandler",
    "DEFAULT_DEBOUNCE_READS",
    "DEFAULT_RELEASE_READS",
    "KOL",
    "KOH",
    "KIL",
]
