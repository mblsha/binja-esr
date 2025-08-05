"""PC-E500 Keyboard Hardware Simulation.

This module provides a hardware-accurate simulation of the PC-E500 keyboard matrix
scanning system with proper strobing support.
"""

from typing import Dict, Optional, Callable, List
from dataclasses import dataclass

from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START

# Keyboard register addresses (offsets within internal memory)
KOL = IMEMRegisters.KOL  # 0xF0 - Key Output Low (controls KO0-KO7)
KOH = IMEMRegisters.KOH  # 0xF1 - Key Output High (controls KO8-KO10)  
KIL = IMEMRegisters.KIL  # 0xF2 - Key Input (reads KI0-KI7)
LCC = IMEMRegisters.LCC  # 0xFE - LCD Contrast Control (contains KSD bit)


@dataclass
class KeyLocation:
    """Represents a key's position in the matrix."""
    column: int  # 0-10 (KO0-KO10)
    row: int     # 0-7 (KI0-KI7)


class KeyboardHardware:
    """Hardware-accurate simulation of PC-E500 keyboard matrix.
    
    This class simulates the electrical behavior of the keyboard matrix,
    including proper column strobing and row reading.
    """
    
    def __init__(self, memory_accessor: Callable[[int], int]):
        """Initialize keyboard hardware.
        
        Args:
            memory_accessor: Function to read from memory (for accessing LCC register)
        """
        self.memory = memory_accessor
        
        # State of output registers
        self.kol_value = 0x00  # KO0-KO7 output state
        self.koh_value = 0x00  # KO8-KO10 output state
        
        # 8x11 matrix state: matrix[row][column] = True if key pressed
        # Rows are KI0-KI7, Columns are KO0-KO10
        self.matrix_state = [[False for _ in range(11)] for _ in range(8)]
        
        # Keyboard layout matching hardware matrix
        self.KEYBOARD_LAYOUT: List[List[Optional[str]]] = [
            # KO0      KO1     KO2      KO3      KO4      KO5      KO6      KO7      KO8      KO9     KO10
            ['▲▼',    'W',    'R',     'Y',     'I',     'RCL',   'STO',   'C.CE',  '↕',     ')',    'P'   ],  # KI0
            ['Q',     'E',    'T',     'U',     'O',     'hyp',   'sin',   'cos',   'tan',   'FSE',  '2ndF'],  # KI1
            ['MENU',  'S',    'F',     'H',     'K',     '→HEX',  '→DEG',  'ln',    'log',   '1/x',  'PF5' ],  # KI2
            ['A',     'D',    'G',     'J',     'L',     'EXP',   'Y^x',   '√',     'x²',    '(',    'PF4' ],  # KI3
            ['BASIC', 'X',    'V',     'N',     ',',     '7',     '8',     '9',     '÷',     'DEL',  'PF3' ],  # KI4
            ['Z',     'C',    'B',     'M',     ';',     '4',     '5',     '6',     '×',     'BS',   'PF2' ],  # KI5
            ['SHIFT', 'CAPS', 'SPACE', '↑',     '▶',     '1',     '2',     '3',     '-',     'INS',  'PF1' ],  # KI6
            ['CTRL',  'ANS',  '↓',     '▼',     '◀',     '0',     '+/-',   '.',     '+',     '=',    None  ]   # KI7
        ]
        
        # Build key name to location mapping
        self._build_key_mappings()
    
    def _build_key_mappings(self):
        """Build mappings from key labels to matrix locations."""
        self.key_locations: Dict[str, KeyLocation] = {}
        
        # Map from standard key codes to labels in the layout
        label_to_keycode = {
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
            'SPACE': 'KEY_SPACE', 'SHIFT': 'KEY_SHIFT', 'CTRL': 'KEY_CTRL',
            'CAPS': 'KEY_CAPS', 'ANS': 'KEY_ANS',
            '↑': 'KEY_UP', '↓': 'KEY_DOWN', '◀': 'KEY_LEFT', '▶': 'KEY_RIGHT',
            'BS': 'KEY_BACKSPACE', 'DEL': 'KEY_DELETE', 'INS': 'KEY_INSERT',
            'C.CE': 'KEY_C_CE',
            
            # Function keys
            'PF1': 'KEY_F1', 'PF2': 'KEY_F2', 'PF3': 'KEY_F3',
            'PF4': 'KEY_F4', 'PF5': 'KEY_F5',
            
            # Operators
            '+': 'KEY_PLUS', '-': 'KEY_MINUS', '×': 'KEY_MULTIPLY',
            '÷': 'KEY_DIVIDE', '=': 'KEY_EQUALS', '.': 'KEY_PERIOD',
            ',': 'KEY_COMMA', ';': 'KEY_SEMICOLON',
            '(': 'KEY_LPAREN', ')': 'KEY_RPAREN',
            
            # PC-E500 specific
            'BASIC': 'KEY_BASIC', 'MENU': 'KEY_MENU', '2ndF': 'KEY_2NDF',
            'FSE': 'KEY_FSE', 'STO': 'KEY_STO', 'RCL': 'KEY_RCL',
            
            # Math functions
            'sin': 'KEY_SIN', 'cos': 'KEY_COS', 'tan': 'KEY_TAN',
            'log': 'KEY_LOG', 'ln': 'KEY_LN', 'EXP': 'KEY_EXP',
            'hyp': 'KEY_HYP', '1/x': 'KEY_1_X', 'x²': 'KEY_X2',
            '√': 'KEY_SQRT', 'Y^x': 'KEY_Y_X',
            
            # Mode keys
            '→DEG': 'KEY_TO_DEG', '→HEX': 'KEY_TO_HEX', '+/-': 'KEY_PLUSMINUS',
            
            # Special function keys
            '↕': 'KEY_UP_DOWN', '▲▼': 'KEY_TRIANGLE_UP_DOWN', '▼': 'KEY_DOWN_TRIANGLE',
        }
        
        # Build location mapping
        for row_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            for col_idx, label in enumerate(row):
                if label and label in label_to_keycode:
                    keycode = label_to_keycode[label]
                    self.key_locations[keycode] = KeyLocation(column=col_idx, row=row_idx)
    
    def press_key(self, key_code: str) -> bool:
        """Simulate pressing a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
            
        Returns:
            True if key was pressed, False if key not found
        """
        if key_code in self.key_locations:
            loc = self.key_locations[key_code]
            self.matrix_state[loc.row][loc.column] = True
            return True
        return False
    
    def release_key(self, key_code: str) -> bool:
        """Simulate releasing a key.
        
        Args:
            key_code: Key identifier (e.g., 'KEY_A')
            
        Returns:
            True if key was released, False if key not found
        """
        if key_code in self.key_locations:
            loc = self.key_locations[key_code]
            self.matrix_state[loc.row][loc.column] = False
            return True
        return False
    
    def release_all_keys(self):
        """Release all pressed keys."""
        for row in range(8):
            for col in range(11):
                self.matrix_state[row][col] = False
    
    def write_register(self, offset: int, value: int) -> None:
        """Handle write to keyboard output register.
        
        Args:
            offset: Register offset (KOL or KOH)
            value: Byte value to write
        """
        if offset == KOL:
            self.kol_value = value & 0xFF
        elif offset == KOH:
            self.koh_value = value & 0xFF
    
    def read_register(self, offset: int) -> int:
        """Handle read from keyboard register.
        
        Args:
            offset: Register offset (KOL, KOH, or KIL)
            
        Returns:
            Register value
        """
        if offset == KOL:
            return self.kol_value
        elif offset == KOH:
            return self.koh_value
        elif offset == KIL:
            return self._simulate_key_scan()
        return 0xFF
    
    def _simulate_key_scan(self) -> int:
        """Simulate hardware key scanning based on current KOL/KOH state.
        
        Returns:
            KIL value representing which rows have pressed keys in active columns
        """
        # Check if keyboard strobing is disabled
        lcc = self.memory(INTERNAL_MEMORY_START + LCC)
        ksd_bit = (lcc >> 2) & 1  # KSD is bit 2 of LCC
        if ksd_bit:
            # Keyboard strobing disabled - all KO lines forced low
            # This means no columns are active, so no keys can be read
            return 0xFF  # All bits high (no keys pressed)
        
        # Start with all input lines high (no keys pressed)
        kil_value = 0xFF
        
        # Check each column to see if it's being strobed (active low)
        for col in range(11):
            is_active = False
            
            if col < 8:  # KO0-KO7 controlled by KOL
                # Column is active if its bit is LOW (active-low logic)
                if not (self.kol_value & (1 << col)):
                    is_active = True
            else:  # KO8-KO10 controlled by KOH
                bit_index = col - 8
                if not (self.koh_value & (1 << bit_index)):
                    is_active = True
            
            if is_active:
                # This column is being strobed - check for pressed keys
                for row in range(8):
                    if self.matrix_state[row][col]:
                        # Key is pressed - pull the corresponding KI line low
                        kil_value &= ~(1 << row)
        
        return kil_value
    
    def get_pressed_keys(self) -> List[str]:
        """Get list of currently pressed key codes.
        
        Returns:
            List of key codes for pressed keys
        """
        pressed = []
        for key_code, loc in self.key_locations.items():
            if self.matrix_state[loc.row][loc.column]:
                pressed.append(key_code)
        return pressed
    
    def get_active_columns(self) -> List[int]:
        """Get list of currently active (strobed) columns.
        
        Returns:
            List of column indices (0-10) that are being strobed
        """
        active = []
        
        # Check KSD bit
        lcc = self.memory(INTERNAL_MEMORY_START + LCC)
        ksd_bit = (lcc >> 2) & 1
        if ksd_bit:
            # Keyboard strobing disabled
            return active
        
        # Check KOL columns (active-low)
        for col in range(8):
            if not (self.kol_value & (1 << col)):
                active.append(col)
        
        # Check KOH columns (active-low)
        for col in range(3):  # Only 3 bits used in KOH
            if not (self.koh_value & (1 << col)):
                active.append(col + 8)
        
        return active
    
    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about keyboard state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'kol': f'0x{self.kol_value:02X}',
            'koh': f'0x{self.koh_value:02X}',
            'kil': f'0x{self._simulate_key_scan():02X}',
            'active_columns': self.get_active_columns(),
            'pressed_keys': self.get_pressed_keys(),
            'ksd_enabled': bool((self.memory(INTERNAL_MEMORY_START + LCC) >> 2) & 1),
            'selected_columns': self._get_selected_columns()  # For compatibility
        }
    
    def get_queue_info(self) -> List[Dict[str, any]]:
        """Get information about pressed keys (for web UI compatibility).
        
        Returns:
            List of dictionaries with key information
        """
        # Since we don't have a queue in the hardware simulation,
        # return info about currently pressed keys
        queue_info = []
        
        for key_code in self.get_pressed_keys():
            loc = self.key_locations[key_code]
            # Calculate KOL/KOH values that would strobe this key
            required_kol = 0xFF
            required_koh = 0xFF
            
            if loc.column < 8:
                required_kol &= ~(1 << loc.column)  # Clear bit for active-low
            else:
                required_koh &= ~(1 << (loc.column - 8))  # Clear bit for active-low
            
            # Calculate KIL value when this key is pressed
            target_kil = ~(1 << loc.row) & 0xFF  # Clear bit for active-low
            
            queue_info.append({
                'key_code': key_code,
                'column': loc.column,
                'row': loc.row,
                'kol': f'0x{required_kol:02X}',
                'koh': f'0x{required_koh:02X}',
                'kil': f'0x{target_kil:02X}',
                'read_count': 10,  # Simulate as fully debounced
                'target_reads': 10,
                'progress': '10/10',
                'is_stuck': False,
                'released': False,
                'queued_time': 0,
                'age_seconds': 0
            })
        
        return queue_info
    
    def _get_selected_columns(self) -> List[str]:
        """Get list of currently selected columns (for compatibility).
        
        Returns:
            List of column names (e.g., ['KO0', 'KO1'])
        """
        columns = []
        for col in self.get_active_columns():
            columns.append(f'KO{col}')
        return columns
    
    # Properties for web UI compatibility
    @property
    def _last_kol(self) -> int:
        """Get last written KOL value (for web UI compatibility)."""
        return self.kol_value
    
    @property
    def _last_koh(self) -> int:
        """Get last written KOH value (for web UI compatibility)."""
        return self.koh_value
    
    def _read_keyboard_input(self) -> int:
        """Read keyboard input (for web UI compatibility)."""
        return self._simulate_key_scan()