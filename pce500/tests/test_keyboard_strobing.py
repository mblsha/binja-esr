"""Test keyboard hardware strobing functionality."""

from pce500.keyboard_hardware import KeyboardHardware, KOL, KOH, KIL, LCC
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START


class MockMemory:
    """Mock memory for testing keyboard hardware."""
    def __init__(self):
        self.data = {}
        # Initialize LCC register with KSD bit clear (keyboard enabled)
        self.data[INTERNAL_MEMORY_START + LCC] = 0x00
    
    def read_byte(self, address: int) -> int:
        return self.data.get(address, 0x00)
    
    def write_byte(self, address: int, value: int) -> None:
        self.data[address] = value & 0xFF


class TestKeyboardStrobing:
    """Test keyboard matrix strobing behavior."""
    
    def test_basic_key_press_with_strobing(self):
        """Test basic key press detection with proper column strobing."""
        # Create keyboard with mock memory
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press 'Q' key (KI1, KO0)
        keyboard.press_key('KEY_Q')
        
        # Strobe wrong column (KO1) - should not detect key
        keyboard.write_register(KOL, 0b11111101)  # KO1 low, others high
        keyboard.write_register(KOH, 0xFF)        # All high
        kil = keyboard.read_register(KIL)
        assert kil == 0xFF, "No key should be detected when wrong column is strobed"
        
        # Strobe correct column (KO0) - should detect key
        keyboard.write_register(KOL, 0b11111110)  # KO0 low, others high
        kil = keyboard.read_register(KIL)
        assert kil == 0b11111101, "KI1 should be low when Q is pressed and KO0 is strobed"
        
        # Release key
        keyboard.release_key('KEY_Q')
        kil = keyboard.read_register(KIL)
        assert kil == 0xFF, "No key should be detected after release"
    
    def test_multiple_columns_strobed(self):
        """Test multiple columns being strobed simultaneously."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press keys in different columns
        keyboard.press_key('KEY_Q')  # KI1, KO0
        keyboard.press_key('KEY_E')  # KI1, KO1
        
        # Strobe both KO0 and KO1
        keyboard.write_register(KOL, 0b11111100)  # KO0 and KO1 low
        kil = keyboard.read_register(KIL)
        # Both keys are on KI1, so KI1 should be low
        assert kil == 0b11111101, "KI1 should be low when either Q or E is pressed"
    
    def test_keys_in_different_rows(self):
        """Test keys in different rows of same column."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press keys in same column but different rows
        keyboard.press_key('KEY_Q')  # KI1, KO0
        keyboard.press_key('KEY_A')  # KI3, KO0
        
        # Strobe KO0
        keyboard.write_register(KOL, 0b11111110)  # KO0 low
        kil = keyboard.read_register(KIL)
        # Both KI1 and KI3 should be low
        assert kil == 0b11110101, "KI1 and KI3 should be low"
    
    def test_koh_column_strobing(self):
        """Test strobing columns controlled by KOH register."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press 'P' key (KI0, KO10)
        keyboard.press_key('KEY_P')
        
        # Strobe wrong KOH column
        keyboard.write_register(KOL, 0xFF)        # All high
        keyboard.write_register(KOH, 0b11111110)  # KO8 low (bit 0)
        kil = keyboard.read_register(KIL)
        assert kil == 0xFF, "No key should be detected when wrong column is strobed"
        
        # Strobe correct KOH column (KO10 = bit 2)
        keyboard.write_register(KOH, 0b11111011)  # KO10 low (bit 2)
        kil = keyboard.read_register(KIL)
        assert kil == 0b11111110, "KI0 should be low when P is pressed and KO10 is strobed"
    
    def test_keyboard_strobe_disable(self):
        """Test KSD (Key Strobe Disable) bit functionality."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press a key
        keyboard.press_key('KEY_Q')  # KI1, KO0
        
        # Strobe column with KSD disabled
        memory.write_byte(INTERNAL_MEMORY_START + LCC, 0b00000100)  # Set KSD bit (bit 2)
        keyboard.write_register(KOL, 0b11111110)  # Try to strobe KO0
        kil = keyboard.read_register(KIL)
        assert kil == 0xFF, "No keys should be detected when KSD is set"
        
        # Re-enable keyboard strobing
        memory.write_byte(INTERNAL_MEMORY_START + LCC, 0b00000000)  # Clear KSD bit
        kil = keyboard.read_register(KIL)
        assert kil == 0b11111101, "Key should be detected when KSD is cleared"
    
    def test_no_keys_pressed(self):
        """Test that KIL returns 0xFF when no keys are pressed."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Strobe all columns
        keyboard.write_register(KOL, 0x00)  # All KO0-KO7 low
        keyboard.write_register(KOH, 0x00)  # All KO8-KO10 low
        kil = keyboard.read_register(KIL)
        assert kil == 0xFF, "KIL should be 0xFF when no keys are pressed"
    
    def test_active_columns_list(self):
        """Test getting list of active (strobed) columns."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Strobe KO0 and KO2
        keyboard.write_register(KOL, 0b11111010)  # KO0 and KO2 low
        keyboard.write_register(KOH, 0xFF)
        active = keyboard.get_active_columns()
        assert active == [0, 2], "Should detect KO0 and KO2 as active"
        
        # Strobe KO9 (KOH bit 1)
        keyboard.write_register(KOL, 0xFF)
        keyboard.write_register(KOH, 0b11111101)  # KO9 low
        active = keyboard.get_active_columns()
        assert active == [9], "Should detect KO9 as active"
        
        # Test with KSD enabled
        memory.write_byte(INTERNAL_MEMORY_START + LCC, 0b00000100)  # Set KSD
        active = keyboard.get_active_columns()
        assert active == [], "No columns should be active when KSD is set"
    
    def test_debug_info(self):
        """Test debug information output."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Set up some state
        keyboard.write_register(KOL, 0xFE)
        keyboard.write_register(KOH, 0xFB)
        keyboard.press_key('KEY_A')
        
        debug = keyboard.get_debug_info()
        assert debug['kol'] == '0xFE'
        assert debug['koh'] == '0xFB'
        assert 'KEY_A' in debug['pressed_keys']
        assert debug['ksd_enabled'] is False
        assert 0 in debug['active_columns']  # KO0 is active
        assert 10 in debug['active_columns']  # KO10 is active
    
    def test_all_register_operations(self):
        """Test all register read/write operations."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Test KOL write/read
        keyboard.write_register(KOL, 0x55)
        assert keyboard.read_register(KOL) == 0x55
        
        # Test KOH write/read
        keyboard.write_register(KOH, 0xAA)
        assert keyboard.read_register(KOH) == 0xAA
        
        # Test that values are masked to 8 bits
        keyboard.write_register(KOL, 0x1FF)
        assert keyboard.read_register(KOL) == 0xFF
    
    def test_realistic_scanning_sequence(self):
        """Test a realistic keyboard scanning sequence."""
        memory = MockMemory()
        keyboard = KeyboardHardware(memory.read_byte)
        
        # Press '5' key (KI5, KO6)
        keyboard.press_key('KEY_5')
        
        # Scan all columns sequentially (typical scanning pattern)
        for col in range(11):
            if col < 8:
                # Scan KOL columns
                keyboard.write_register(KOL, ~(1 << col) & 0xFF)
                keyboard.write_register(KOH, 0xFF)
            else:
                # Scan KOH columns
                keyboard.write_register(KOL, 0xFF)
                keyboard.write_register(KOH, ~(1 << (col - 8)) & 0xFF)
            
            kil = keyboard.read_register(KIL)
            
            if col == 6:  # Column for '5' key
                assert kil == 0b11011111, f"KI5 should be low when scanning column {col}"
            else:
                assert kil == 0xFF, f"No key should be detected when scanning column {col}"