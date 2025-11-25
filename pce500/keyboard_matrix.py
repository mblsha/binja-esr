"""Deterministic keyboard matrix model with FIFO buffering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

# Internal RAM locations used by the ROM keyboard driver.
FIFO_BASE = 0x00BFC96
if TYPE_CHECKING:
    from .memory import PCE500Memory
FIFO_SIZE = 8  # 8 events mirrored in RAM
FIFO_HEAD_ADDR = 0x00BFC9D
FIFO_TAIL_ADDR = 0x00BFC9E

# Debounce / repeat defaults derived from ROM behaviour (16 ms fast-timer cadence).
DEFAULT_PRESS_TICKS = 6
DEFAULT_RELEASE_TICKS = 6
DEFAULT_REPEAT_DELAY_TICKS = 24  # ≈ 384 ms
DEFAULT_REPEAT_INTERVAL_TICKS = 6  # ≈ 96 ms


@dataclass(frozen=True)
class KeyLocation:
    column: int
    row: int


@dataclass
class KeyState:
    location: KeyLocation
    pressed: bool = False  # Physical state
    debounced: bool = False  # Logical state visible to FIFO/KIL
    press_ticks: int = 0  # Consecutive strobed ticks while pressed
    release_ticks: int = 0  # Consecutive ticks without strobe or after release
    repeat_ticks: int = 0  # Countdown for repeat events

    @property
    def matrix_code(self) -> int:
        return (self.location.column << 3) | self.location.row


@dataclass(frozen=True)
class MatrixEvent:
    """FIFO event describing a key press/release (bit 7 marks release)."""

    code: int
    release: bool = False
    repeat: bool = False

    def to_byte(self) -> int:
        value = self.code & 0x7F
        if self.release:
            value |= 0x80
        return value


def _build_key_locations() -> Tuple[Dict[str, KeyLocation], Dict[str, str]]:
    """Return keyboard mapping tables (key-code layout)."""

    # Layout matches KI rows (0-7) × KO columns (0-10).
    layout: List[List[Optional[str]]] = [
        ["▲▼", "W", "R", "Y", "I", "RCL", "STO", "C.CE", "↕", ")", "P"],
        ["Q", "E", "T", "U", "O", "hyp", "sin", "cos", "tan", "FSE", "2ndF"],
        ["MENU", "S", "F", "H", "K", "→HEX", "→DEG", "ln", "log", "1/x", "PF5"],
        ["A", "D", "G", "J", "L", "EXP", "Y^x", "√", "x²", "(", "PF4"],
        ["BASIC", "X", "V", "N", ",", "7", "8", "9", "÷", "DEL", "PF3"],
        ["Z", "C", "B", "M", ";", "4", "5", "6", "×", "BS", "PF2"],
        ["SHIFT", "CAPS", "SPACE", "↑", "▶", "1", "2", "3", "-", "INS", "PF1"],
        ["CTRL", "ANS", "↓", "▼", "◀", "0", "+/-", ".", "+", "=", None],
    ]

    names: Dict[str, str] = {
        # Letters
        "A": "KEY_A",
        "B": "KEY_B",
        "C": "KEY_C",
        "D": "KEY_D",
        "E": "KEY_E",
        "F": "KEY_F",
        "G": "KEY_G",
        "H": "KEY_H",
        "I": "KEY_I",
        "J": "KEY_J",
        "K": "KEY_K",
        "L": "KEY_L",
        "M": "KEY_M",
        "N": "KEY_N",
        "O": "KEY_O",
        "P": "KEY_P",
        "Q": "KEY_Q",
        "R": "KEY_R",
        "S": "KEY_S",
        "T": "KEY_T",
        "U": "KEY_U",
        "V": "KEY_V",
        "W": "KEY_W",
        "X": "KEY_X",
        "Y": "KEY_Y",
        "Z": "KEY_Z",
        # Digits
        "0": "KEY_0",
        "1": "KEY_1",
        "2": "KEY_2",
        "3": "KEY_3",
        "4": "KEY_4",
        "5": "KEY_5",
        "6": "KEY_6",
        "7": "KEY_7",
        "8": "KEY_8",
        "9": "KEY_9",
        # Navigation / modifiers
        "SPACE": "KEY_SPACE",
        "SHIFT": "KEY_SHIFT",
        "CTRL": "KEY_CTRL",
        "CAPS": "KEY_CAPS",
        "ANS": "KEY_ANS",
        "↑": "KEY_UP",
        "↓": "KEY_DOWN",
        "◀": "KEY_LEFT",
        "▶": "KEY_RIGHT",
        "↕": "KEY_UP_DOWN",
        "▲▼": "KEY_TRIANGLE_UP_DOWN",
        "▼": "KEY_DOWN_TRIANGLE",
        # Function keys
        "PF1": "KEY_F1",
        "PF2": "KEY_F2",
        "PF3": "KEY_F3",
        "PF4": "KEY_F4",
        "PF5": "KEY_F5",
        "2ndF": "KEY_2NDF",
        # Editing / control
        "C.CE": "KEY_C_CE",
        "DEL": "KEY_DELETE",
        "BS": "KEY_BACKSPACE",
        "INS": "KEY_INSERT",
        "BASIC": "KEY_BASIC",
        "MENU": "KEY_MENU",
        "FSE": "KEY_FSE",
        "STO": "KEY_STO",
        "RCL": "KEY_RCL",
        "CA": "KEY_CALC",
        "CALC": "KEY_CALC",
        # Math / operators
        "+": "KEY_PLUS",
        "-": "KEY_MINUS",
        "×": "KEY_MULTIPLY",
        "÷": "KEY_DIVIDE",
        "=": "KEY_EQUALS",
        ".": "KEY_PERIOD",
        ",": "KEY_COMMA",
        ";": "KEY_SEMICOLON",
        "(": "KEY_LPAREN",
        ")": "KEY_RPAREN",
        "+/-": "KEY_PLUSMINUS",
        "hyp": "KEY_HYP",
        "sin": "KEY_SIN",
        "cos": "KEY_COS",
        "tan": "KEY_TAN",
        "log": "KEY_LOG",
        "ln": "KEY_LN",
        "EXP": "KEY_EXP",
        "1/x": "KEY_1_X",
        "x²": "KEY_X2",
        "√": "KEY_SQRT",
        "Y^x": "KEY_Y_X",
        "→DEG": "KEY_TO_DEG",
        "→HEX": "KEY_TO_HEX",
    }

    locations: Dict[str, KeyLocation] = {}
    for row_idx, row in enumerate(layout):
        for col_idx, label in enumerate(row):
            if label is None:
                continue
            key_code = names.get(label)
            if key_code:
                locations[key_code] = KeyLocation(column=col_idx, row=row_idx)

    return locations, names


KEY_LOCATIONS, KEY_NAMES = _build_key_locations()


class KeyboardMatrix:
    """Pure keyboard matrix model backing the emulator and tests."""

    def __init__(
        self,
        *,
        columns_active_high: bool = True,
        press_threshold: int = DEFAULT_PRESS_TICKS,
        release_threshold: int = DEFAULT_RELEASE_TICKS,
        repeat_delay: int = DEFAULT_REPEAT_DELAY_TICKS,
        repeat_interval: int = DEFAULT_REPEAT_INTERVAL_TICKS,
        memory: Optional["PCE500Memory"] = None,
    ):
        self.columns_active_high = columns_active_high
        self.press_threshold = max(1, press_threshold)
        self.release_threshold = max(1, release_threshold)
        self.repeat_delay = max(0, repeat_delay)
        self.repeat_interval = max(0, repeat_interval)
        self._memory = memory
        self._reader = getattr(memory, "read_byte", None)
        self._writer = getattr(memory, "write_byte", None)

        self.kol = 0x00 if columns_active_high else 0xFF
        self.koh = 0x00 if columns_active_high else 0x0F
        self._kil_latch = 0x00
        self.scan_enabled = True

        self._key_states: Dict[str, KeyState] = {
            code: KeyState(location=loc) for code, loc in KEY_LOCATIONS.items()
        }
        self._pressed_keys: set[str] = set()

        self.strobe_count = 0
        self.column_histogram: List[int] = [0] * 11
        self.irq_count = 0
        # Optional trace hook for scan events (col, row, pressed)
        self._trace_hook = None

        self._fifo: List[int] = [0x00] * FIFO_SIZE
        self._head = 0
        self._tail = 0
        self._initialise_fifo_memory()

    # ------------------------------------------------------------------ #
    # Public API (used by emulator/tests)
    # ------------------------------------------------------------------ #

    def attach_memory(self, memory: "PCE500Memory") -> None:
        """Attach backing memory overlay at runtime."""

        self._memory = memory
        self._reader = getattr(memory, "read_byte", None)
        self._writer = getattr(memory, "write_byte", None)
        self._initialise_fifo_memory()

    def press_key(self, key_code: str) -> bool:
        state = self._key_states.get(key_code)
        if not state or state.pressed:
            return False
        state.pressed = True
        state.press_ticks = 0
        state.release_ticks = 0
        state.repeat_ticks = self.repeat_delay
        self._pressed_keys.add(key_code)
        return True

    def release_key(self, key_code: str) -> None:
        state = self._key_states.get(key_code)
        if not state:
            return
        state.pressed = False
        state.release_ticks = 0
        self._pressed_keys.discard(key_code)

    def release_all_keys(self) -> None:
        for state in self._key_states.values():
            state.pressed = False
            state.debounced = False
            state.press_ticks = 0
            state.release_ticks = 0
            state.repeat_ticks = 0
        self._pressed_keys.clear()

    def write_kol(self, value: int) -> None:
        value &= 0xFF
        if value != self.kol:
            self.kol = value
            self.strobe_count += 1
            self._update_column_histogram()
        self._kil_latch = self._compute_kil()

    def write_koh(self, value: int) -> None:
        value &= 0x0F
        if value != self.koh:
            self.koh = value
            self.strobe_count += 1
            self._update_column_histogram()
        self._kil_latch = self._compute_kil()

    def read_kil(self) -> int:
        self._kil_latch = self._compute_kil()
        return self._kil_latch

    def peek_kil(self) -> int:
        return self._compute_kil(allow_pending=True)

    def get_active_columns(self) -> List[int]:
        return list(self._active_columns())

    def get_pressed_keys(self) -> Iterable[str]:
        return tuple(sorted(self._pressed_keys))

    def get_key_locations(self) -> Dict[str, KeyLocation]:
        return KEY_LOCATIONS

    def scan_tick(self) -> List[MatrixEvent]:
        """Advance debounce/repeat state for a fast-timer tick."""

        if not self.scan_enabled:
            return []

        events: List[MatrixEvent] = []
        active_cols = set(self._active_columns())
        for key_code, state in self._key_states.items():
            generated = self._update_key_state(state, active_cols)
            if generated:
                events.extend(generated)

        if events:
            self.irq_count += len(events)
            for event in events:
                self._enqueue_event(event)

        self._kil_latch = self._compute_kil()
        return events

    def pop_fifo(self) -> Optional[int]:
        """Pop the next FIFO entry (mirrors ROM consumer behaviour)."""

        self._refresh_head_from_memory()
        if self._head == self._tail:
            return None

        value = self._fifo[self._head]
        self._head = (self._head + 1) % FIFO_SIZE
        self._write_head(self._head)
        return value

    def fifo_snapshot(self) -> List[int]:
        self._refresh_head_from_memory()
        snapshot: List[int] = []
        idx = self._head
        while idx != self._tail:
            snapshot.append(self._fifo[idx])
            idx = (idx + 1) % FIFO_SIZE
        return snapshot

    def inject_event(self, key_code: str, *, release: bool = False) -> bool:
        """Inject a debounced keyboard event into the FIFO (for scripted tests)."""

        state = self._key_states.get(key_code)
        if state is None:
            return False

        if release:
            state.pressed = False
            state.debounced = False
            state.press_ticks = 0
            state.release_ticks = 0
            state.repeat_ticks = 0
            self._pressed_keys.discard(key_code)
        else:
            state.pressed = True
            state.debounced = True
            state.press_ticks = self.press_threshold
            state.release_ticks = 0
            state.repeat_ticks = self.repeat_delay
            self._pressed_keys.add(key_code)

        event = MatrixEvent(code=state.matrix_code, release=release, repeat=False)
        self._enqueue_event(event)
        self.irq_count += 1
        self._kil_latch = self._compute_kil(allow_pending=True)
        return True

    # ------------------------------------------------------------------ #
    # Snapshot helpers
    # ------------------------------------------------------------------ #

    def snapshot_state(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the matrix state."""

        key_states: Dict[str, Dict[str, int | bool]] = {}
        for key_code, state in self._key_states.items():
            key_states[key_code] = {
                "pressed": state.pressed,
                "debounced": state.debounced,
                "press_ticks": state.press_ticks,
                "release_ticks": state.release_ticks,
                "repeat_ticks": state.repeat_ticks,
            }

        snapshot_fifo = list(self._fifo)

        return {
            "kol": self.kol,
            "koh": self.koh,
            "kil_latch": self._kil_latch,
            "scan_enabled": self.scan_enabled,
            "pressed_keys": list(self._pressed_keys),
            "key_states": key_states,
            "fifo": snapshot_fifo,
            "head": self._head,
            "tail": self._tail,
            "strobe_count": self.strobe_count,
            "column_histogram": list(self.column_histogram),
            "irq_count": self.irq_count,
            "press_threshold": self.press_threshold,
            "release_threshold": self.release_threshold,
            "repeat_delay": self.repeat_delay,
            "repeat_interval": self.repeat_interval,
            "columns_active_high": self.columns_active_high,
        }

    def load_state(self, state: Dict[str, object]) -> None:
        """Restore the matrix from a snapshot created by ``snapshot_state``."""

        def _get_int(key: str, default: int = 0) -> int:
            value = state.get(key, default)
            return _to_int(value, default)

        def _to_int(value: object, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        self.columns_active_high = bool(state.get("columns_active_high", True))
        self.press_threshold = max(1, _get_int("press_threshold", self.press_threshold))
        self.release_threshold = max(
            1, _get_int("release_threshold", self.release_threshold)
        )
        self.repeat_delay = max(0, _get_int("repeat_delay", self.repeat_delay))
        self.repeat_interval = max(0, _get_int("repeat_interval", self.repeat_interval))

        self.kol = _get_int("kol", self.kol) & 0xFF
        self.koh = _get_int("koh", self.koh) & 0x0F
        self._kil_latch = _get_int("kil_latch", self._kil_latch) & 0xFF
        self.scan_enabled = bool(state.get("scan_enabled", self.scan_enabled))

        pressed_keys = state.get("pressed_keys", [])
        if isinstance(pressed_keys, list):
            self._pressed_keys = {str(key) for key in pressed_keys}
        else:
            self._pressed_keys = set()

        key_states_snapshot = state.get("key_states", {})
        if isinstance(key_states_snapshot, dict):
            for key_code, saved in key_states_snapshot.items():
                state_obj = self._key_states.get(key_code)
                if state_obj is None or not isinstance(saved, dict):
                    continue
                state_obj.pressed = bool(saved.get("pressed", state_obj.pressed))
                state_obj.debounced = bool(saved.get("debounced", state_obj.debounced))
                state_obj.press_ticks = _to_int(saved.get("press_ticks", 0))
                state_obj.release_ticks = _to_int(saved.get("release_ticks", 0))
                state_obj.repeat_ticks = _to_int(saved.get("repeat_ticks", 0))

        fifo_data = state.get("fifo", [])
        if not isinstance(fifo_data, list):
            fifo_data = []
        for idx in range(FIFO_SIZE):
            value = _to_int(fifo_data[idx]) if idx < len(fifo_data) else 0
            self._fifo[idx] = value & 0xFF
            self._write_fifo_slot(idx, self._fifo[idx])

        self._head = _get_int("head", self._head) % FIFO_SIZE
        self._tail = _get_int("tail", self._tail) % FIFO_SIZE
        self._write_head(self._head)
        self._write_tail(self._tail)

        self.strobe_count = _get_int("strobe_count", self.strobe_count)
        hist = state.get("column_histogram")
        if isinstance(hist, list) and len(hist) == len(self.column_histogram):
            self.column_histogram = [int(val) for val in hist]
        self.irq_count = _get_int("irq_count", self.irq_count)

        # Recompute derived latch to keep KIL consistent with restored state.
        self._kil_latch = self._compute_kil(allow_pending=True)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _initialise_fifo_memory(self) -> None:
        if self._writer is None:
            return

        for offset in range(FIFO_SIZE):
            self._writer(FIFO_BASE + offset, 0x00)
        self._writer(FIFO_HEAD_ADDR, self._head)
        self._writer(FIFO_TAIL_ADDR, self._tail)

    def _active_columns(self) -> Iterable[int]:
        active: List[int] = []
        for col in range(8):
            bit = (self.kol >> col) & 1
            active_flag = bit == 1 if self.columns_active_high else bit == 0
            if active_flag:
                active.append(col)
        for col in range(3):
            bit = (self.koh >> col) & 1
            active_flag = bit == 1 if self.columns_active_high else bit == 0
            if active_flag:
                active.append(col + 8)
        return active

    def _update_column_histogram(self) -> None:
        active = self._active_columns()
        for col in active:
            if 0 <= col < len(self.column_histogram):
                self.column_histogram[col] += 1

    def _compute_kil(self, *, allow_pending: bool = False) -> int:
        value = 0
        active_cols = set(self._active_columns())
        for state in self._key_states.values():
            if state.location.column not in active_cols:
                continue
            if state.debounced or (
                allow_pending
                and state.pressed
                and state.press_ticks + 1 >= self.press_threshold
            ):
                value |= 1 << state.location.row
        return value & 0xFF

    def _update_key_state(
        self, state: KeyState, active_cols: set[int]
    ) -> List[MatrixEvent]:
        events: List[MatrixEvent] = []
        strobed = state.location.column in active_cols

        if state.pressed and strobed:
            if not state.debounced:
                state.press_ticks += 1
                if state.press_ticks >= self.press_threshold:
                    state.debounced = True
                    state.press_ticks = self.press_threshold
                    state.release_ticks = 0
                    state.repeat_ticks = self.repeat_delay
                    evt = MatrixEvent(code=state.matrix_code, release=False)
                    events.append(evt)
                    if callable(self._trace_hook):
                        try:
                            self._trace_hook(
                                state.location.column, state.location.row, True
                            )
                        except Exception:
                            pass
            else:
                state.release_ticks = 0
                if self.repeat_interval > 0 and self.repeat_delay >= 0:
                    if state.repeat_ticks > 0:
                        state.repeat_ticks -= 1
                    if state.repeat_ticks <= 0:
                        events.append(
                            MatrixEvent(
                                code=state.matrix_code, release=False, repeat=True
                            )
                        )
                        state.repeat_ticks = (
                            self.repeat_interval if self.repeat_interval > 0 else 0
                        )
        else:
            state.press_ticks = 0
            if state.debounced:
                state.release_ticks += 1
                if state.release_ticks >= self.release_threshold:
                    state.debounced = False
                    state.release_ticks = 0
                    state.repeat_ticks = 0
                    evt = MatrixEvent(code=state.matrix_code, release=True)
                    events.append(evt)
                    if callable(self._trace_hook):
                        try:
                            self._trace_hook(
                                state.location.column, state.location.row, False
                            )
                        except Exception:
                            pass

        if not state.pressed and not state.debounced:
            state.repeat_ticks = 0

        return events

    def _enqueue_event(self, event: MatrixEvent) -> None:
        self._refresh_head_from_memory()
        next_tail = (self._tail + 1) % FIFO_SIZE
        if next_tail == self._head:
            # FIFO full: drop oldest entry
            self._head = (self._head + 1) % FIFO_SIZE
            self._write_head(self._head)

        self._fifo[self._tail] = event.to_byte()
        self._write_fifo_slot(self._tail, self._fifo[self._tail])
        self._tail = next_tail
        self._write_tail(self._tail)

    def _write_fifo_slot(self, index: int, value: int) -> None:
        if self._writer is None:
            return
        self._writer(FIFO_BASE + index, value & 0xFF)

    def _write_head(self, value: int) -> None:
        if self._writer is not None:
            self._writer(FIFO_HEAD_ADDR, value & 0xFF)

    def _write_tail(self, value: int) -> None:
        if self._writer is not None:
            self._writer(FIFO_TAIL_ADDR, value & 0xFF)

    def _refresh_head_from_memory(self) -> None:
        if self._reader is None:
            return
        try:
            self._head = self._reader(FIFO_HEAD_ADDR) % FIFO_SIZE
        except Exception:
            pass


__all__ = [
    "KeyboardMatrix",
    "MatrixEvent",
    "KeyLocation",
    "KEY_LOCATIONS",
    "KEY_NAMES",
    "DEFAULT_PRESS_TICKS",
    "DEFAULT_RELEASE_TICKS",
    "DEFAULT_REPEAT_DELAY_TICKS",
    "DEFAULT_REPEAT_INTERVAL_TICKS",
    "FIFO_BASE",
    "FIFO_HEAD_ADDR",
    "FIFO_TAIL_ADDR",
    "FIFO_SIZE",
    "MatrixEvent",
]
