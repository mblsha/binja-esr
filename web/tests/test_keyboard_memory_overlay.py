"""Unit tests for PC-E500 keyboard memory overlay."""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from keyboard_memory_overlay import KeyboardMemoryOverlay
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START


class TestKeyboardMemoryOverlay(unittest.TestCase):
    """Test keyboard memory overlay functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock keyboard handler
        self.mock_handler = Mock()
        self.mock_handler.handle_register_read.return_value = 0x00
        self.mock_handler.handle_register_write.return_value = True
        
        # Create overlay
        self.overlay = KeyboardMemoryOverlay.create_overlay(self.mock_handler)
    
    def test_overlay_properties(self):
        """Test overlay basic properties."""
        # Check address range covers keyboard registers
        self.assertEqual(self.overlay.start, INTERNAL_MEMORY_START + IMEMRegisters.KOL)
        self.assertEqual(self.overlay.end, INTERNAL_MEMORY_START + IMEMRegisters.KIL)
        self.assertEqual(self.overlay.name, "keyboard_io")
        self.assertFalse(self.overlay.read_only)
        self.assertEqual(self.overlay.perfetto_thread, "I/O")
        
        # Should have handlers, not data
        self.assertIsNone(self.overlay.data)
        self.assertIsNotNone(self.overlay.read_handler)
        self.assertIsNotNone(self.overlay.write_handler)
    
    def test_read_kol(self):
        """Test reading KOL register through overlay."""
        self.mock_handler.handle_register_read.return_value = 0x55
        
        # Read KOL
        address = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        value = self.overlay.read_handler(address, None)
        
        # Verify handler was called with correct register
        self.mock_handler.handle_register_read.assert_called_once_with(IMEMRegisters.KOL)
        self.assertEqual(value, 0x55)
    
    def test_read_koh(self):
        """Test reading KOH register through overlay."""
        self.mock_handler.handle_register_read.return_value = 0xAA
        
        # Read KOH
        address = INTERNAL_MEMORY_START + IMEMRegisters.KOH
        value = self.overlay.read_handler(address, None)
        
        # Verify handler was called with correct register
        self.mock_handler.handle_register_read.assert_called_once_with(IMEMRegisters.KOH)
        self.assertEqual(value, 0xAA)
    
    def test_read_kil(self):
        """Test reading KIL register through overlay."""
        self.mock_handler.handle_register_read.return_value = 0xFF
        
        # Read KIL
        address = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        value = self.overlay.read_handler(address, None)
        
        # Verify handler was called with correct register
        self.mock_handler.handle_register_read.assert_called_once_with(IMEMRegisters.KIL)
        self.assertEqual(value, 0xFF)
    
    def test_write_kol(self):
        """Test writing KOL register through overlay."""
        # Write KOL
        address = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        self.overlay.write_handler(address, 0x33, None)
        
        # Verify handler was called with correct register and value
        self.mock_handler.handle_register_write.assert_called_once_with(IMEMRegisters.KOL, 0x33)
    
    def test_write_koh(self):
        """Test writing KOH register through overlay."""
        # Write KOH
        address = INTERNAL_MEMORY_START + IMEMRegisters.KOH
        self.overlay.write_handler(address, 0xCC, None)
        
        # Verify handler was called with correct register and value
        self.mock_handler.handle_register_write.assert_called_once_with(IMEMRegisters.KOH, 0xCC)
    
    def test_write_kil(self):
        """Test writing KIL register through overlay (should be ignored)."""
        # Write KIL (read-only register)
        address = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        self.overlay.write_handler(address, 0x00, None)
        
        # Handler should still be called (it decides to ignore)
        self.mock_handler.handle_register_write.assert_called_once_with(IMEMRegisters.KIL, 0x00)
    
    def test_read_non_keyboard_register(self):
        """Test reading non-keyboard register returns 0."""
        self.mock_handler.handle_register_read.return_value = None
        
        # Read some random address within overlay range
        address = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        value = self.overlay.read_handler(address, None)
        
        # Should return 0x00 when handler returns None
        self.assertEqual(value, 0x00)
    
    def test_cpu_pc_parameter(self):
        """Test CPU PC parameter is passed through."""
        # Test read with PC
        address = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        cpu_pc = 0x123456
        self.overlay.read_handler(address, cpu_pc)
        
        # PC should be passed but handler doesn't use it
        self.mock_handler.handle_register_read.assert_called_once_with(IMEMRegisters.KIL)
        
        # Test write with PC
        self.mock_handler.reset_mock()
        self.overlay.write_handler(address, 0xFF, cpu_pc)
        
        # PC should be passed but handler doesn't use it
        self.mock_handler.handle_register_write.assert_called_once_with(IMEMRegisters.KIL, 0xFF)
    
    def test_overlay_address_calculation(self):
        """Test correct address to register mapping."""
        test_cases = [
            (INTERNAL_MEMORY_START + 0xF0, IMEMRegisters.KOL),
            (INTERNAL_MEMORY_START + 0xF1, IMEMRegisters.KOH),
            (INTERNAL_MEMORY_START + 0xF2, IMEMRegisters.KIL),
        ]
        
        for address, expected_register in test_cases:
            self.mock_handler.reset_mock()
            self.overlay.read_handler(address, None)
            self.mock_handler.handle_register_read.assert_called_once_with(expected_register)


if __name__ == '__main__':
    unittest.main()