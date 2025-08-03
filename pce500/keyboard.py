"""PC-E500 Keyboard Matrix Handler.

The PC-E500 uses a keyboard matrix scanning system:
- KOL (0xF0) and KOH (0xF1) are output registers that select which rows to scan
- KIL (0xF2) is the input register that reads which keys in selected rows are pressed

The keyboard is organized as a matrix where:
- Rows are selected by setting bits in KOL/KOH
- Columns are read from KIL
- A pressed key connects its row to its column
"""

from typing import Dict, Set, Tuple, Optional

# Keyboard register addresses (offsets within internal memory)
# These are at internal memory offsets 0xF0-0xF2
KOL = 0xF0  # Key Output Low
KOH = 0xF1  # Key Output High  
KIL = 0xF2  # Key Input


class PCE500KeyboardHandler:
    """Handles PC-E500 keyboard matrix emulation."""
    
    # Keyboard matrix mapping
    # Format: key_code -> (row_bit, column_bit)
    # Row bits are across KOL (bits 0-7) and KOH (bits 0-7) 
    # Column bits are read from KIL (bits 0-7)
    KEYBOARD_MATRIX: Dict[str, Tuple[int, int]] = {
        # This is a simplified example - the actual PC-E500 matrix needs to be reverse-engineered
        # Row 0 (KOL bit 0)
        'KEY_Q': (0, 0),
        'KEY_W': (0, 1), 
        'KEY_E': (0, 2),
        'KEY_R': (0, 3),
        'KEY_T': (0, 4),
        'KEY_Y': (0, 5),
        'KEY_U': (0, 6),
        'KEY_I': (0, 7),
        
        # Row 1 (KOL bit 1)
        'KEY_A': (1, 0),
        'KEY_S': (1, 1),
        'KEY_D': (1, 2),
        'KEY_F': (1, 3),
        'KEY_G': (1, 4),
        'KEY_H': (1, 5),
        'KEY_J': (1, 6),
        'KEY_K': (1, 7),
        
        # Row 2 (KOL bit 2)
        'KEY_Z': (2, 0),
        'KEY_X': (2, 1),
        'KEY_C': (2, 2),
        'KEY_V': (2, 3),
        'KEY_B': (2, 4),
        'KEY_N': (2, 5),
        'KEY_M': (2, 6),
        'KEY_L': (2, 7),
        
        # Row 3 (KOL bit 3)
        'KEY_1': (3, 0),
        'KEY_2': (3, 1),
        'KEY_3': (3, 2),
        'KEY_4': (3, 3),
        'KEY_5': (3, 4),
        'KEY_6': (3, 5),
        'KEY_7': (3, 6),
        'KEY_8': (3, 7),
        
        # Row 4 (KOL bit 4)
        'KEY_9': (4, 0),
        'KEY_0': (4, 1),
        'KEY_MINUS': (4, 2),
        'KEY_EQUALS': (4, 3),
        'KEY_BACKSPACE': (4, 4),
        'KEY_TAB': (4, 5),
        'KEY_O': (4, 6),
        'KEY_P': (4, 7),
        
        # Row 5 (KOL bit 5)
        'KEY_LBRACKET': (5, 0),
        'KEY_RBRACKET': (5, 1),
        'KEY_ENTER': (5, 2),
        'KEY_CAPS': (5, 3),
        'KEY_SEMICOLON': (5, 4),
        'KEY_QUOTE': (5, 5),
        'KEY_COMMA': (5, 6),
        'KEY_PERIOD': (5, 7),
        
        # Row 6 (KOL bit 6)
        'KEY_SLASH': (6, 0),
        'KEY_SPACE': (6, 1),
        'KEY_UP': (6, 2),
        'KEY_DOWN': (6, 3),
        'KEY_LEFT': (6, 4),
        'KEY_RIGHT': (6, 5),
        'KEY_SHIFT': (6, 6),
        'KEY_CTRL': (6, 7),
        
        # Row 7 (KOL bit 7)
        'KEY_F1': (7, 0),
        'KEY_F2': (7, 1),
        'KEY_F3': (7, 2),
        'KEY_F4': (7, 3),
        'KEY_F5': (7, 4),
        'KEY_F6': (7, 5),
        'KEY_INS': (7, 6),
        'KEY_DEL': (7, 7),
        
        # Row 8 (KOH bit 0) - Extended rows
        'KEY_CALC': (8, 0),
        'KEY_BASIC': (8, 1),
        'KEY_ON': (8, 2),
        # ... more keys can be added here
    }
    
    def __init__(self):
        """Initialize keyboard handler."""
        self.pressed_keys: Set[str] = set()
        self._last_kol = 0
        self._last_koh = 0
        
    def press_key(self, key_code: str) -> None:
        """Press a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
        """
        if key_code in self.KEYBOARD_MATRIX:
            self.pressed_keys.add(key_code)
            self._update_keyboard_state()
            
    def release_key(self, key_code: str) -> None:
        """Release a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
        """
        self.pressed_keys.discard(key_code)
        self._update_keyboard_state()
        
    def release_all_keys(self) -> None:
        """Release all pressed keys."""
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
        """Read keyboard input based on current row selection.
        
        Returns:
            Byte value representing pressed keys in selected rows
        """
        # WORKAROUND: Returning 0xFF causes severe performance issues
        # in the LLIL evaluation when at address 0x1000F2.
        # Return 0xFE instead (bit 0 clear) which doesn't trigger the issue.
        result = 0xFE  # Default: no keys pressed (almost all bits high)
        
        # Check each pressed key
        for key_code in self.pressed_keys:
            if key_code not in self.KEYBOARD_MATRIX:
                continue
                
            row, column = self.KEYBOARD_MATRIX[key_code]
            
            # Check if this row is selected
            row_selected = False
            if row < 8:
                # Row is in KOL
                if self._last_kol & (1 << row):
                    row_selected = True
            else:
                # Row is in KOH
                if self._last_koh & (1 << (row - 8)):
                    row_selected = True
                    
            if row_selected:
                # Clear the corresponding column bit (active low)
                result &= ~(1 << column)
        
        # WORKAROUND: Avoid returning 0xFF which causes performance issues
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
            'selected_rows': self._get_selected_rows()
        }
        
    def _get_selected_rows(self) -> list:
        """Get list of currently selected rows."""
        rows = []
        for i in range(8):
            if self._last_kol & (1 << i):
                rows.append(f'Row {i} (KOL bit {i})')
        for i in range(8):
            if self._last_koh & (1 << i):
                rows.append(f'Row {i+8} (KOH bit {i})')
        return rows