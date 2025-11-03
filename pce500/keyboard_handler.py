"""Keyboard handler wrapping the deterministic matrix implementation."""

from __future__ import annotations

from typing import Dict, List, Optional
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
        memory: Optional["PCE500Memory"] = None,
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

    # ------------------------------------------------------------------ #
    # Public API used by emulator/tests
    # ------------------------------------------------------------------ #

    def press_key(self, key_code: str) -> bool:
        if not self._matrix.press_key(key_code):
            return False
        self._last_kil = self._matrix.peek_kil()
        return True

    def release_key(self, key_code: str) -> None:
        self._matrix.release_key(key_code)
        self._last_kil = self._matrix.peek_kil()

    def release_all_keys(self) -> None:
        self._matrix.release_all_keys()
        self._last_kil = 0

    def handle_register_read(self, register: int) -> Optional[int]:
        reg = register & 0xFF
        if reg == KOL:
            return self._last_kol
        if reg == KOH:
            return self._last_koh
        if reg == KIL:
            if not self._scan_enabled or self._ksd_masked():
                return 0x00
            self.scan_tick()
            self._last_kil = self._matrix.peek_kil()
            return self._last_kil
        return None

    def handle_register_write(self, register: int, value: int) -> bool:
        reg = register & 0xFF
        if reg == KOL:
            self._matrix.write_kol(value & 0xFF)
            self._last_kol = self._matrix.kol
            return True
        if reg == KOH:
            self._matrix.write_koh(value & 0x0F)
            self._last_koh = self._matrix.koh
            return True
        if reg == KIL:
            # KIL is read-only in hardware; ignore writes.
            return True
        return False

    def scan_tick(self) -> List[MatrixEvent]:
        events = self._matrix.scan_tick()
        if events:
            # Ensure KIL latch reflects the latest active rows after the tick.
            self._last_kil = self._matrix.peek_kil()
        return events

    def set_scan_enabled(self, enabled: bool) -> None:
        self._scan_enabled = bool(enabled)
        self._matrix.scan_enabled = bool(enabled)

    def get_active_columns(self) -> List[int]:
        return self._matrix.get_active_columns()

    def get_pressed_keys(self) -> List[str]:
        return list(self._matrix.get_pressed_keys())

    def peek_keyboard_input(self) -> int:
        return self._matrix.peek_kil()

    def get_debug_info(self) -> Dict[str, object]:
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

    def get_queue_info(self) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
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

    def fifo_snapshot(self) -> List[int]:
        return self._matrix.fifo_snapshot()

    # Metrics used by emulator instrumentation --------------------------------

    @property
    def strobe_count(self) -> int:
        return self._matrix.strobe_count

    @property
    def column_histogram(self) -> List[int]:
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
    def key_locations(self) -> Dict[str, KeyLocation]:
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
