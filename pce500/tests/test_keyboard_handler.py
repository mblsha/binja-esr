"""Unit tests for PC-E500 keyboard handler."""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.keyboard import PCE500KeyboardHandler
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


class TestPCE500KeyboardHandler(unittest.TestCase):
    """Test keyboard handler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create keyboard handler directly (no CPU needed)
        self.handler = PCE500KeyboardHandler()
    
    def test_initial_state(self):
        """Test initial keyboard state."""
        # No keys should be pressed initially
        self.assertEqual(len(self.handler.pressed_keys), 0)
        
        # KIL should read 0xFF (no keys pressed)
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFF)
        
        # KOL/KOH should be 0
        self.assertEqual(self.handler._last_kol, 0)
        self.assertEqual(self.handler._last_koh, 0)
    
    def test_key_press_release(self):
        """Test basic key press and release."""
        # Press key Q
        self.handler.press_key('KEY_Q')
        self.assertIn('KEY_Q', self.handler.pressed_keys)
        
        # Release key Q
        self.handler.release_key('KEY_Q')
        self.assertNotIn('KEY_Q', self.handler.pressed_keys)
        
        # Release non-pressed key should not error
        self.handler.release_key('KEY_A')
        self.assertNotIn('KEY_A', self.handler.pressed_keys)
    
    def test_release_all_keys(self):
        """Test releasing all keys at once."""
        # Press multiple keys
        self.handler.press_key('KEY_A')
        self.handler.press_key('KEY_B')
        self.handler.press_key('KEY_C')
        self.assertEqual(len(self.handler.pressed_keys), 3)
        
        # Release all
        self.handler.release_all_keys()
        self.assertEqual(len(self.handler.pressed_keys), 0)
    
    def test_keyboard_matrix_scanning(self):
        """Test keyboard matrix row/column scanning."""
        # Press KEY_Q (row 0, column 0)
        self.handler.press_key('KEY_Q')
        
        # Select row 0 by setting KOL bit 0
        self.handler.handle_register_write(IMEMRegisters.KOL, 0x01)
        
        # Read KIL - should have bit 0 cleared (active low)
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFE)  # 11111110
        
        # Select different row - KEY_Q should not be detected
        self.handler.handle_register_write(IMEMRegisters.KOL, 0x02)
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFF)  # No keys detected
    
    def test_multiple_keys_same_row(self):
        """Test multiple keys pressed in the same row."""
        # Press KEY_Q (row 0, col 0) and KEY_W (row 0, col 1)
        self.handler.press_key('KEY_Q')
        self.handler.press_key('KEY_W')
        
        # Select row 0
        self.handler.handle_register_write(IMEMRegisters.KOL, 0x01)
        
        # Read KIL - should have bits 0 and 1 cleared
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFC)  # 11111100
    
    def test_multiple_rows_selected(self):
        """Test multiple rows selected simultaneously."""
        # Press keys in different rows
        self.handler.press_key('KEY_Q')  # Row 0, col 0
        self.handler.press_key('KEY_A')  # Row 1, col 0
        
        # Select rows 0 and 1
        self.handler.handle_register_write(IMEMRegisters.KOL, 0x03)  # bits 0 and 1
        
        # Both keys should be detected in column 0
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFE)  # bit 0 cleared
    
    def test_extended_rows_koh(self):
        """Test extended rows using KOH register."""
        # Press KEY_CALC (row 8, col 0)
        self.handler.press_key('KEY_CALC')
        
        # Select row 8 by setting KOH bit 0
        self.handler.handle_register_write(IMEMRegisters.KOH, 0x01)
        
        # Read KIL - should have bit 0 cleared
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFE)
    
    def test_register_read_write(self):
        """Test register read/write operations."""
        # Write to KOL
        result = self.handler.handle_register_write(IMEMRegisters.KOL, 0x55)
        self.assertTrue(result)
        self.assertEqual(self.handler._last_kol, 0x55)
        
        # Read back KOL
        value = self.handler.handle_register_read(IMEMRegisters.KOL)
        self.assertEqual(value, 0x55)
        
        # Write to KOH
        result = self.handler.handle_register_write(IMEMRegisters.KOH, 0xAA)
        self.assertTrue(result)
        self.assertEqual(self.handler._last_koh, 0xAA)
        
        # Read back KOH
        value = self.handler.handle_register_read(IMEMRegisters.KOH)
        self.assertEqual(value, 0xAA)
        
        # Writing to KIL should be ignored (read-only)
        result = self.handler.handle_register_write(IMEMRegisters.KIL, 0xFF)
        self.assertTrue(result)  # Returns True but doesn't actually write
    
    def test_invalid_key_code(self):
        """Test handling of invalid key codes."""
        # Press invalid key - should be ignored
        self.handler.press_key('INVALID_KEY')
        self.assertEqual(len(self.handler.pressed_keys), 0)
        
        # KIL should still read 0xFF
        self.handler.handle_register_write(IMEMRegisters.KOL, 0xFF)
        kil_value = self.handler.handle_register_read(IMEMRegisters.KIL)
        self.assertEqual(kil_value, 0xFF)
    
    def test_debug_info(self):
        """Test debug information output."""
        # Set up some state
        self.handler.press_key('KEY_A')
        self.handler.press_key('KEY_B')
        self.handler.handle_register_write(IMEMRegisters.KOL, 0x03)
        self.handler.handle_register_write(IMEMRegisters.KOH, 0x01)
        
        # Get debug info
        debug = self.handler.get_debug_info()
        
        # Verify debug info structure
        self.assertIn('pressed_keys', debug)
        self.assertIn('KEY_A', debug['pressed_keys'])
        self.assertIn('KEY_B', debug['pressed_keys'])
        self.assertEqual(debug['kol'], '0x03')
        self.assertEqual(debug['koh'], '0x01')
        self.assertIn('kil', debug)
        self.assertIn('selected_rows', debug)
        self.assertIsInstance(debug['selected_rows'], list)
    
    def test_non_keyboard_register(self):
        """Test handling of non-keyboard registers."""
        # Read non-keyboard register should return None
        result = self.handler.handle_register_read(0x00)
        self.assertIsNone(result)
        
        # Write to non-keyboard register should return False
        result = self.handler.handle_register_write(0x00, 0xFF)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()