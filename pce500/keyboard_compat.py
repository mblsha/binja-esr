"""PC-E500 Keyboard Matrix Handler.

The PC-E500 uses a keyboard matrix scanning system:
- KOL (0xF0) and KOH (0xF1) are output registers that select which columns to scan
- KIL (0xF2) is the input register that reads which rows have keys pressed

The keyboard is organized as a matrix where:
- Columns are selected by setting bits in KOL (KO0-KO7) and KOH (KO8-KO10)
- Rows are read from KIL (KI0-KI7)
- A pressed key connects its column to its row
"""

from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass, field
import time

# Keyboard register addresses (offsets within internal memory)
# These are at internal memory offsets 0xF0-0xF2
KOL = 0xF0  # Key Output Low (controls KO0-KO7)
KOH = 0xF1  # Key Output High (controls KO8-KO10)
KIL = 0xF2  # Key Input (reads KI0-KI7)

# Default number of reads required for debouncing
DEFAULT_DEBOUNCE_READS = 6  # reads needed while strobed to accept a press
DEFAULT_RELEASE_READS = 6  # reads needed while strobed to accept a release


@dataclass
class QueuedKey:
    """Represents a key in the queue waiting to be processed."""

    key_code: str  # Human-readable key code (e.g., 'KEY_A')
    column: int  # Column index (0-10)
    row: int  # Row index (0-7)
    required_kol: int  # Required KOL value to activate this key
    required_koh: int  # Required KOH value to activate this key
    target_kil: int  # KIL value to return when active
    # Debounce state
    read_count: int = 0  # Number of times this key has been read while strobed
    target_reads: int = DEFAULT_DEBOUNCE_READS  # Reads required to accept press
    active: bool = False  # Becomes True after stable-press threshold
    release_reads: int = 0  # Number of times seen as not strobed for release
    release_target_reads: int = DEFAULT_RELEASE_READS  # Reads to accept release
    queued_time: float = field(default_factory=time.time)  # When key was queued
    released: bool = False  # Whether the key has been physically released
    columns_active_high: bool = True  # Whether KO lines are active-high

    def matches_output(self, kol: int, koh: int) -> bool:
        """Check if current KOL/KOH values match this key's column (compat, active-high).

        - KO0..KO7 are active when the corresponding KOL bit is 1
        - KO8..KO11 are active when the corresponding KOH bit (0..3) is 1
        """
        if self.column < 8:
            mask = 1 << self.column
            if self.columns_active_high:
                return (kol & mask) != 0
            return (kol & mask) == 0
        mask = 1 << (self.column - 8)
        if self.columns_active_high:
            return (koh & mask) != 0
        return (koh & mask) == 0

    def is_complete(self) -> bool:
        """Check if this key has completed its full press-release debounce."""
        return self.released and not self.active

    def increment_read(self) -> None:
        """Increment the read counter and update active state after threshold."""
        if not self.active:
            self.read_count += 1
            if self.read_count >= self.target_reads:
                self.active = True


class PCE500KeyboardHandler:
    """Handles PC-E500 keyboard matrix emulation."""

    # ### FIXED: Corrected KEYBOARD_LAYOUT to match the image ###
    # Visual keyboard layout matching the hardware matrix.
    # Rows are KI0-KI7, Columns are KO0-KO10, indexed from left to right.
    # This layout now correctly matches the provided key matrix diagram.
    KEYBOARD_LAYOUT: List[List[Optional[str]]] = [
        # KO0      KO1     KO2      KO3      KO4      KO5      KO6      KO7      KO8      KO9     KO10
        ["▲▼", "W", "R", "Y", "I", "RCL", "STO", "C.CE", "↕", ")", "P"],  # KI0
        ["Q", "E", "T", "U", "O", "hyp", "sin", "cos", "tan", "FSE", "2ndF"],  # KI1
        ["MENU", "S", "F", "H", "K", "→HEX", "→DEG", "ln", "log", "1/x", "PF5"],  # KI2
        ["A", "D", "G", "J", "L", "EXP", "Y^x", "√", "x²", "(", "PF4"],  # KI3
        ["BASIC", "X", "V", "N", ",", "7", "8", "9", "÷", "DEL", "PF3"],  # KI4
        ["Z", "C", "B", "M", ";", "4", "5", "6", "×", "BS", "PF2"],  # KI5
        ["SHIFT", "CAPS", "SPACE", "↑", "▶", "1", "2", "3", "-", "INS", "PF1"],  # KI6
        ["CTRL", "ANS", "↓", "▼", "◀", "0", "+/-", ".", "+", "=", None],  # KI7
    ]

    # ### FIXED: Updated KEY_NAMES to match the corrected layout ###
    # Mapping from key labels to key codes
    KEY_NAMES: Dict[str, str] = {
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
        # Numbers
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
        # Special keys
        "SPACE": "KEY_SPACE",
        "SHIFT": "KEY_SHIFT",
        "CTRL": "KEY_CTRL",
        "CAPS": "KEY_CAPS",
        "ANS": "KEY_ANS",
        "↑": "KEY_UP",
        "↓": "KEY_DOWN",
        "◀": "KEY_LEFT",
        "▶": "KEY_RIGHT",
        "BS": "KEY_BACKSPACE",
        "DEL": "KEY_DELETE",
        "INS": "KEY_INSERT",
        "KANA": "KEY_KANA",
        # Note: OFF/ON are hardware power buttons, not part of the keyboard matrix
        "OFF": "KEY_OFF",
        "ON": "KEY_ON",
        "C.CE": "KEY_C_CE",
        # Function keys
        "PF1": "KEY_F1",
        "PF2": "KEY_F2",
        "PF3": "KEY_F3",
        "PF4": "KEY_F4",
        "PF5": "KEY_F5",
        # Operators
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
        # PC-E500 specific keys
        "BASIC": "KEY_BASIC",
        "CA": "KEY_CALC",  # CA is on the C.CE key, likely a shifted function
        "MENU": "KEY_MENU",
        "2ndF": "KEY_2NDF",
        "FSE": "KEY_FSE",
        "STO": "KEY_STO",
        "RCL": "KEY_RCL",
        "CALC": "KEY_CALC",  # Alternative name for CA
        # Math functions
        "sin": "KEY_SIN",
        "cos": "KEY_COS",
        "tan": "KEY_TAN",
        "log": "KEY_LOG",
        "ln": "KEY_LN",
        "EXP": "KEY_EXP",
        "hyp": "KEY_HYP",
        "1/x": "KEY_1_X",
        "x²": "KEY_X2",
        "√": "KEY_SQRT",
        "Y^x": "KEY_Y_X",
        # Mode keys
        "→DEG": "KEY_TO_DEG",
        "→HEX": "KEY_TO_HEX",
        "+/-": "KEY_PLUSMINUS",
        # Special function keys
        "↕": "KEY_UP_DOWN",
        "▲▼": "KEY_TRIANGLE_UP_DOWN",
        "▼": "KEY_DOWN_TRIANGLE",  # Added missing key
    }

    def __init__(self, memory_accessor=None, columns_active_high: bool = True):
        """Initialize keyboard handler."""
        self._memory = memory_accessor
        self.pressed_keys: Set[str] = set()  # Keep for compatibility
        self.key_queue: List[QueuedKey] = []  # Queue of keys to be processed
        self._last_kol = 0
        self._last_koh = 0
        # Track last computed KIL for optional smoothing
        self._last_kil_value = 0x00
        self._columns_active_high = columns_active_high

        # Build the keyboard matrix from the layout
        self.KEYBOARD_MATRIX: Dict[str, Tuple[int, int]] = {}
        for row_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            for col_idx, key_label in enumerate(row):
                if key_label and key_label in self.KEY_NAMES:
                    key_code = self.KEY_NAMES[key_label]
                    # Store as (column, row) since KO selects columns and KI reads rows
                    self.KEYBOARD_MATRIX[key_code] = (col_idx, row_idx)

    def press_key(
        self, key_code: str, target_reads: int = DEFAULT_DEBOUNCE_READS
    ) -> bool:
        """Press a key.

        Args:
            key_code: Key identifier (e.g., 'KEY_A')
            target_reads: Number of reads required for debouncing

        Returns:
            True if key was queued, False if key is not mapped or already queued
        """
        if key_code not in self.KEYBOARD_MATRIX:
            # Key not in matrix - log for debugging
            print(f"Warning: Key '{key_code}' not in keyboard matrix, ignoring press")
            return False

        # Check if key is already queued
        for queued_key in self.key_queue:
            if queued_key.key_code == key_code and not queued_key.is_complete():
                # Key already queued and not complete, ignore
                return False

        # Add key to queue
        column, row = self.KEYBOARD_MATRIX[key_code]

        # Calculate required KOL/KOH values for this key
        required_kol = 0
        required_koh = 0
        if column < 8:
            if self._columns_active_high:
                required_kol = 1 << column
            else:
                required_kol = (~(1 << column)) & 0xFF
        else:
            bit = 1 << (column - 8)
            if self._columns_active_high:
                required_koh = bit
            else:
                required_koh = (~bit) & 0x0F

        # Calculate target KIL value (set the row bit - active high logic)
        target_kil = 1 << row

        # Create queued key
        queued_key = QueuedKey(
            key_code=key_code,
            column=column,
            row=row,
            required_kol=required_kol,
            required_koh=required_koh,
            target_kil=target_kil,
            target_reads=target_reads,
            columns_active_high=self._columns_active_high,
        )

        self.key_queue.append(queued_key)
        self.pressed_keys.add(key_code)  # Keep for compatibility
        self._update_keyboard_state()
        return True

    def release_key(self, key_code: str) -> None:
        """Release a key.

        Args:
            key_code: Key identifier (e.g., 'KEY_A')
        """
        # Mark key as released but keep it in queue until fully processed
        for queued_key in self.key_queue:
            if queued_key.key_code == key_code:
                queued_key.released = True
                # Start release debounce counting anew
                queued_key.release_reads = 0

        # Remove from pressed_keys set
        self.pressed_keys.discard(key_code)
        self._update_keyboard_state()

    def release_all_keys(self) -> None:
        """Release all pressed keys."""
        self.key_queue.clear()
        self.pressed_keys.clear()
        self._update_keyboard_state()

    def handle_register_read(self, register: int) -> Optional[int]:
        """Handle read from keyboard registers.

        Args:
            register: Register address (0xF0-0xF2)

        Returns:
            Register value or None if not a keyboard register
        """
        if register == KIL:
            # Honor KSD (keyboard strobe disable) bit if memory is available
            try:
                if self._memory is not None:
                    from sc62015.pysc62015.instr.opcodes import IMEMRegisters as _IMR

                    INTERNAL_MEMORY_START = 0x100000
                    lcc = self._memory(INTERNAL_MEMORY_START + _IMR.LCC)
                    if (lcc & 0x04) != 0:
                        return 0x00
            except Exception:
                pass
            # Read keyboard input based on current KOL/KOH values
            return self._read_keyboard_input()
        elif register == KOL:
            return self._last_kol
        elif register == KOH:
            return self._last_koh
        return None

    def handle_register_write(self, register: int, value: int) -> bool:
        """Handle write to keyboard registers.

        Args:
            register: Register address (0xF0-0xF2)
            value: Value to write

        Returns:
            True if handled, False otherwise
        """
        if register == KOL:
            self._last_kol = value & 0xFF
            return True
        elif register == KOH:
            self._last_koh = value & 0xFF
            return True
        elif register == KIL:
            # KIL is read-only
            return True
        return False

    def _read_keyboard_input(self) -> int:
        """Read keyboard input based on current column selection.

        Returns:
            Byte value representing pressed keys in selected columns
        """
        # Default: no keys pressed (active-high: 0)
        result = 0x00

        # Process key queue
        completed_keys = []

        for queued_key in self.key_queue:
            strobed = queued_key.matches_output(self._last_kol, self._last_koh)
            if strobed:
                # Count towards press debounce if not yet active
                queued_key.increment_read()
                # If active (after debounce), expose the row bit
                if queued_key.active:
                    result |= (1 << queued_key.row) & 0xFF
                # While strobed, do not count release
                queued_key.release_reads = 0
            else:
                # Not strobed this read; if released, count towards release debounce
                if queued_key.released and queued_key.active:
                    queued_key.release_reads += 1
                    if queued_key.release_reads >= queued_key.release_target_reads:
                        queued_key.active = False
                # If fully complete (released and inactive), schedule removal
                if queued_key.is_complete():
                    completed_keys.append(queued_key)

        # Remove completed keys (released and inactive)
        for completed_key in completed_keys:
            self.key_queue.remove(completed_key)
            self.pressed_keys.discard(completed_key.key_code)

        self._last_kil_value = result & 0xFF
        return self._last_kil_value

    def peek_keyboard_input(self) -> int:
        """Preview current KIL value without mutating debounce state."""

        result = 0x00

        for queued_key in self.key_queue:
            if not queued_key.matches_output(self._last_kol, self._last_koh):
                continue

            if queued_key.active or (
                not queued_key.active
                and queued_key.read_count + 1 >= queued_key.target_reads
            ):
                result |= (1 << queued_key.row) & 0xFF

        return result & 0xFF

    def _update_keyboard_state(self) -> None:
        """Update keyboard state in CPU memory."""
        # The keyboard state is automatically reflected when KIL is read
        # This method can be used for any additional state updates if needed
        pass

    # Expose last KOL/KOH values for instrumentation compatibility
    @property
    def kol_value(self) -> int:
        return int(self._last_kol) & 0xFF

    @property
    def koh_value(self) -> int:
        return int(self._last_koh) & 0xFF

    def get_active_columns(self) -> List[int]:
        """Return list of currently active (strobed) columns in compat mode.

        When columns are active-high (compat mode):
        - KO0..KO7 active when KOL bit 0..7 is 1
        - KO8..KO11 active when KOH bit 0..3 is 1
        For active-low strobes (hardware mode) the bits are inverted.
        """
        active: List[int] = []
        kol = self._last_kol & 0xFF
        koh = self._last_koh & 0xFF
        # KO0..KO7 from KOL bits 0..7
        for col in range(8):
            mask = 1 << col
            bit_set = kol & mask
            if self._columns_active_high:
                active_flag = bit_set
            else:
                active_flag = not bit_set
            if active_flag:
                active.append(col)
        # KO8..KO11 from KOH bits 0..3
        for col in range(4):
            mask = 1 << col
            bit_set = koh & mask
            if self._columns_active_high:
                active_flag = bit_set
            else:
                active_flag = not bit_set
            if active_flag:
                active.append(col + 8)
        return active

    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about keyboard state.

        Returns:
            Dictionary with debug information
        """
        return {
            "pressed_keys": list(self.pressed_keys),
            "kol": f"0x{self._last_kol:02X}",
            "koh": f"0x{self._last_koh:02X}",
            "kil": f"0x{self.peek_keyboard_input():02X}",
            "selected_columns": self._get_selected_columns(),
            "key_queue": self.get_queue_info(),
        }

    def get_queue_info(self) -> List[Dict[str, any]]:
        """Get information about queued keys.

        Returns:
            List of dictionaries with queue information
        """
        queue_info = []
        current_time = time.time()

        for queued_key in self.key_queue:
            # Check if this key is stuck (not being read for > 1 second)
            is_stuck = (
                current_time - queued_key.queued_time > 1.0
                and queued_key.read_count == 0
            )

            queue_info.append(
                {
                    "key_code": queued_key.key_code,
                    "column": queued_key.column,
                    "row": queued_key.row,
                    "ko_label": f"KO{queued_key.column}",
                    "ki_label": f"KI{queued_key.row}",
                    "kol": f"0x{queued_key.required_kol:02X}",
                    "koh": f"0x{queued_key.required_koh:02X}",
                    "kil": f"0x{queued_key.target_kil:02X}",
                    "read_count": queued_key.read_count,
                    "target_reads": queued_key.target_reads,
                    "progress": f"{queued_key.read_count}/{queued_key.target_reads}",
                    "is_stuck": is_stuck,
                    "released": queued_key.released,
                    "queued_time": queued_key.queued_time,
                    "age_seconds": current_time - queued_key.queued_time,
                }
            )

        return queue_info

    def _get_selected_columns(self) -> list:
        """Get list of currently selected columns."""
        columns = []
        for i in range(8):
            if self._last_kol & (1 << i):
                columns.append(f"KO{i}")
        for i in range(4):  # KO8-KO11 in compat mapping
            if self._last_koh & (1 << i):
                columns.append(f"KO{i + 8}")
        return columns
