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
DEFAULT_DEBOUNCE_READS = 10


@dataclass
class QueuedKey:
    """Represents a key in the queue waiting to be processed."""
    key_code: str               # Human-readable key code (e.g., 'KEY_A')
    column: int                 # Column index (0-10)
    row: int                    # Row index (0-7)
    required_kol: int           # Required KOL value to activate this key
    required_koh: int           # Required KOH value to activate this key
    target_kil: int             # KIL value to return when active
    read_count: int = 0         # Number of times this key has been read
    target_reads: int = DEFAULT_DEBOUNCE_READS  # Number of reads required
    queued_time: float = field(default_factory=time.time)  # When key was queued
    
    def matches_output(self, kol: int, koh: int) -> bool:
        """Check if current KOL/KOH values match this key's requirements."""
        # Check if the column bit is set
        if self.column < 8:
            # Column is in KOL (KO0-KO7)
            return bool(kol & (1 << self.column))
        else:
            # Column is in KOH (KO8-KO10)
            return bool(koh & (1 << (self.column - 8)))
    
    def is_complete(self) -> bool:
        """Check if this key has been read enough times."""
        return self.read_count >= self.target_reads
    
    def increment_read(self) -> None:
        """Increment the read counter."""
        self.read_count += 1


class PCE500KeyboardHandler:
    """Handles PC-E500 keyboard matrix emulation."""
    
    # Visual keyboard layout matching the hardware matrix
    # Rows are KI0-KI7, Columns are KO0-KO10
    KEYBOARD_LAYOUT: List[List[Optional[str]]] = [
        # KO0    KO1    KO2    KO3     KO4    KO5    KO6  KO7  KO8     KO9   KO10
        ['P',    ')',   '↕',   'CA',   'STO', 'RCL', 'I', 'Y', 'R',    'W',  '▲▼'],     # KI0
        ['2ndF', 'FSE', 'tan', 'cos',  'sin', 'hyp', '0', 'U', 'T',    'E',  'Q'],      # KI1
        ['PF5',  '1/x', 'log', 'ln',   '→DEG','→HEX','K', 'H', 'F',    'S',  'MENU'],   # KI2
        ['PF4',  '(',   'x²',  '√',    'Y^x', 'EXP', 'L', 'J', 'G',    'D',  'A'],      # KI3
        ['PF3',  'DEL', '÷',   '9',    '8',   '7',   ',', 'N', 'V',    'X',  'BASIC'],  # KI4
        ['PF2',  'BS',  '×',   '6',    '5',   '4',   ';', 'M', 'B',    'C',  'Z'],      # KI5
        ['PF1',  'INS', '-',   '3',    '2',   '1',   '▶', '↑', 'SPACE','CAPS','SHIFT'], # KI6
        [None,   '=',   '+',   '.',    '+/-', '0',   '↵', '◀', '↓',    'ANS', 'CTRL']   # KI7
    ]
    
    # Mapping from key labels to key codes
    KEY_NAMES: Dict[str, str] = {
        # Letters
        'A': 'KEY_A', 'B': 'KEY_B', 'C': 'KEY_C', 'D': 'KEY_D',
        'E': 'KEY_E', 'F': 'KEY_F', 'G': 'KEY_G', 'H': 'KEY_H',
        'I': 'KEY_I', 'J': 'KEY_J', 'K': 'KEY_K', 'L': 'KEY_L',
        'M': 'KEY_M', 'N': 'KEY_N', 'O': 'KEY_O', 'P': 'KEY_P',
        'Q': 'KEY_Q', 'R': 'KEY_R', 'S': 'KEY_S', 'T': 'KEY_T',
        'U': 'KEY_U', 'V': 'KEY_V', 'W': 'KEY_W', 'X': 'KEY_X',
        'Y': 'KEY_Y', 'Z': 'KEY_Z',
        
        # Numbers
        '0': 'KEY_0', '1': 'KEY_1', '2': 'KEY_2', '3': 'KEY_3',
        '4': 'KEY_4', '5': 'KEY_5', '6': 'KEY_6', '7': 'KEY_7',
        '8': 'KEY_8', '9': 'KEY_9',
        
        # Special keys
        'SPACE': 'KEY_SPACE',
        'SHIFT': 'KEY_SHIFT',
        'CTRL': 'KEY_CTRL',
        'CAPS': 'KEY_CAPS',
        'ANS': 'KEY_ANS',
        '↑': 'KEY_UP',
        '↓': 'KEY_DOWN',
        '◀': 'KEY_LEFT',
        '▶': 'KEY_RIGHT',
        '↵': 'KEY_ENTER',
        'BS': 'KEY_BACKSPACE',
        'DEL': 'KEY_DELETE',  
        'INS': 'KEY_INSERT',
        'KANA': 'KEY_KANA',
        'OFF': 'KEY_OFF',
        'ON': 'KEY_ON',
        'C.CE': 'KEY_C_CE',
        
        # Function keys
        'PF1': 'KEY_F1',
        'PF2': 'KEY_F2',
        'PF3': 'KEY_F3',
        'PF4': 'KEY_F4',
        'PF5': 'KEY_F5',
        
        # Operators
        '+': 'KEY_PLUS',
        '-': 'KEY_MINUS',
        '×': 'KEY_MULTIPLY',
        '÷': 'KEY_DIVIDE',
        '=': 'KEY_EQUALS',
        '.': 'KEY_PERIOD',
        ',': 'KEY_COMMA',
        ';': 'KEY_SEMICOLON',
        '(': 'KEY_LPAREN',
        ')': 'KEY_RPAREN',
        
        # PC-E500 specific keys
        'BASIC': 'KEY_BASIC',
        'CA': 'KEY_CALC',
        'MENU': 'KEY_MENU',
        '2ndF': 'KEY_2NDF',
        'FSE': 'KEY_FSE',
        'STO': 'KEY_STO',
        'RCL': 'KEY_RCL',
        'CALC': 'KEY_CALC',  # Alternative name for CA
        
        # Math functions
        'sin': 'KEY_SIN',
        'cos': 'KEY_COS',
        'tan': 'KEY_TAN',
        'log': 'KEY_LOG',
        'ln': 'KEY_LN',
        'EXP': 'KEY_EXP',
        'hyp': 'KEY_HYP',
        '1/x': 'KEY_1_X',
        'x²': 'KEY_X2',
        '√': 'KEY_SQRT',
        'Y^x': 'KEY_Y_X',
        
        # Mode keys
        '→DEG': 'KEY_TO_DEG',
        '→HEX': 'KEY_TO_HEX',
        '+/-': 'KEY_PLUSMINUS',
        
        # Special function keys
        '↕': 'KEY_UP_DOWN',
        '▲▼': 'KEY_TRIANGLE_UP_DOWN',
    }
    
    def __init__(self):
        """Initialize keyboard handler."""
        self.pressed_keys: Set[str] = set()  # Keep for compatibility
        self.key_queue: List[QueuedKey] = []  # Queue of keys to be processed
        self._last_kol = 0
        self._last_koh = 0
        
        # Build the keyboard matrix from the layout
        self.KEYBOARD_MATRIX: Dict[str, Tuple[int, int]] = {}
        for row_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            for col_idx, key_label in enumerate(row):
                if key_label and key_label in self.KEY_NAMES:
                    key_code = self.KEY_NAMES[key_label]
                    # Store as (column, row) since KO selects columns and KI reads rows
                    self.KEYBOARD_MATRIX[key_code] = (col_idx, row_idx)
        
    def press_key(self, key_code: str, target_reads: int = DEFAULT_DEBOUNCE_READS) -> None:
        """Press a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
            target_reads: Number of reads required for debouncing
        """
        if key_code in self.KEYBOARD_MATRIX:
            # Check if key is already queued
            for queued_key in self.key_queue:
                if queued_key.key_code == key_code and not queued_key.is_complete():
                    # Key already queued and not complete, ignore
                    return
            
            # Add key to queue
            column, row = self.KEYBOARD_MATRIX[key_code]
            
            # Calculate required KOL/KOH values for this key
            required_kol = 0
            required_koh = 0
            if column < 8:
                required_kol = 1 << column
            else:
                required_koh = 1 << (column - 8)
            
            # Calculate target KIL value (clear the row bit)
            target_kil = 0xFF & ~(1 << row)
            
            # Create queued key
            queued_key = QueuedKey(
                key_code=key_code,
                column=column,
                row=row,
                required_kol=required_kol,
                required_koh=required_koh,
                target_kil=target_kil,
                target_reads=target_reads
            )
            
            self.key_queue.append(queued_key)
            self.pressed_keys.add(key_code)  # Keep for compatibility
            self._update_keyboard_state()
            
    def release_key(self, key_code: str) -> None:
        """Release a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
        """
        # Remove from queue
        self.key_queue = [k for k in self.key_queue if k.key_code != key_code]
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
        result = 0xFF  # Default: no keys pressed (all bits high)
        
        # Process key queue
        completed_keys = []
        
        for queued_key in self.key_queue:
            # Check if this key matches current KOL/KOH
            if queued_key.matches_output(self._last_kol, self._last_koh):
                # This key is active, apply its KIL contribution
                result &= queued_key.target_kil
                
                # Increment read count
                queued_key.increment_read()
                
                # Check if key is complete
                if queued_key.is_complete():
                    completed_keys.append(queued_key)
        
        # Remove completed keys from queue
        for completed_key in completed_keys:
            self.key_queue.remove(completed_key)
            self.pressed_keys.discard(completed_key.key_code)
        
        # WORKAROUND: Avoid returning 0xFF which causes performance issues
        # in the LLIL evaluation when at address 0x1000F2.
        # If no keys are pressed (result is still 0xFF), set bit 0 to 0.
        # This returns 0xFE instead which doesn't trigger the performance issue.
        if result == 0xFF:
            result = 0xFE
                
        return result
        
    def _update_keyboard_state(self) -> None:
        """Update keyboard state in CPU memory."""
        # The keyboard state is automatically reflected when KIL is read
        # This method can be used for any additional state updates if needed
        pass
        
    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about keyboard state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'pressed_keys': list(self.pressed_keys),
            'kol': f'0x{self._last_kol:02X}',
            'koh': f'0x{self._last_koh:02X}',
            'kil': f'0x{self._read_keyboard_input():02X}',
            'selected_columns': self._get_selected_columns(),
            'key_queue': self.get_queue_info()
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
            is_stuck = (current_time - queued_key.queued_time > 1.0 and 
                       queued_key.read_count == 0)
            
            queue_info.append({
                'key_code': queued_key.key_code,
                'column': queued_key.column,
                'row': queued_key.row,
                'kol': f'0x{queued_key.required_kol:02X}',
                'koh': f'0x{queued_key.required_koh:02X}',
                'kil': f'0x{queued_key.target_kil:02X}',
                'read_count': queued_key.read_count,
                'target_reads': queued_key.target_reads,
                'progress': f'{queued_key.read_count}/{queued_key.target_reads}',
                'is_stuck': is_stuck,
                'queued_time': queued_key.queued_time,
                'age_seconds': current_time - queued_key.queued_time
            })
        
        return queue_info
        
    def _get_selected_columns(self) -> list:
        """Get list of currently selected columns."""
        columns = []
        for i in range(8):
            if self._last_kol & (1 << i):
                columns.append(f'KO{i}')
        for i in range(3):  # Only KO8-KO10 exist
            if self._last_koh & (1 << i):
                columns.append(f'KO{i+8}')
        return columns