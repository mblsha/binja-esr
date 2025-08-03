"""Integration tests for keyboard to CPU communication."""

import unittest
import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set FORCE_BINJA_MOCK before imports
os.environ['FORCE_BINJA_MOCK'] = '1'

from keyboard_handler import PCE500KeyboardHandler
from sc62015.pysc62015.emulator import Emulator as SC62015Emulator
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START
from pce500.memory import PCE500Memory


class MockMemoryForKeyboard(PCE500Memory):
    """Mock memory that supports keyboard overlay for testing."""
    
    def __init__(self):
        super().__init__()
        self.keyboard_handler = None
    
    def set_keyboard_handler(self, handler):
        """Set keyboard handler for testing."""
        self.keyboard_handler = handler
    
    def read_byte(self, address: int, cpu_pc=None) -> int:
        """Override to handle keyboard registers."""
        # Check if this is a keyboard register address
        if address >= INTERNAL_MEMORY_START + IMEMRegisters.KOL and address <= INTERNAL_MEMORY_START + IMEMRegisters.KIL:
            if self.keyboard_handler:
                register = address - INTERNAL_MEMORY_START
                result = self.keyboard_handler.handle_register_read(register)
                if result is not None:
                    return result
        
        # Otherwise use parent implementation
        return super().read_byte(address, cpu_pc)
    
    def write_byte(self, address: int, value: int, cpu_pc=None) -> None:
        """Override to handle keyboard registers."""
        # Check if this is a keyboard register address
        if address >= INTERNAL_MEMORY_START + IMEMRegisters.KOL and address <= INTERNAL_MEMORY_START + IMEMRegisters.KIL:
            if self.keyboard_handler:
                register = address - INTERNAL_MEMORY_START
                self.keyboard_handler.handle_register_write(register, value)
                return
        
        # Otherwise use parent implementation
        super().write_byte(address, value, cpu_pc)


class TestKeyboardCPUIntegration(unittest.TestCase):
    """Test keyboard input reaches CPU through memory system."""
    
    def setUp(self):
        """Set up integrated test environment."""
        # Create memory and CPU
        self.memory = MockMemoryForKeyboard()
        self.cpu = SC62015Emulator(self.memory, reset_on_init=True)
        
        # Create keyboard handler
        self.keyboard_handler = PCE500KeyboardHandler(self.cpu)
        self.memory.set_keyboard_handler(self.keyboard_handler)
    
    def test_keyboard_registers_accessible(self):
        """Test CPU can access keyboard registers through memory."""
        # Write to KOL through memory
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        self.memory.write_byte(kol_addr, 0x55)
        
        # Read back through memory
        value = self.memory.read_byte(kol_addr)
        self.assertEqual(value, 0x55)
        
        # Verify keyboard handler received the write
        self.assertEqual(self.keyboard_handler._last_kol, 0x55)
    
    def test_key_press_affects_kil(self):
        """Test key press changes KIL register value."""
        # Select row 0
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        self.memory.write_byte(kol_addr, 0x01)
        
        # Read KIL before key press
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        kil_before = self.memory.read_byte(kil_addr)
        self.assertEqual(kil_before, 0xFF)  # No keys pressed
        
        # Press KEY_Q (row 0, col 0)
        self.keyboard_handler.press_key('KEY_Q')
        
        # Read KIL after key press
        kil_after = self.memory.read_byte(kil_addr)
        self.assertEqual(kil_after, 0xFE)  # Bit 0 cleared
    
    def test_keyboard_scanning_pattern(self):
        """Test typical keyboard scanning pattern."""
        # Press multiple keys
        self.keyboard_handler.press_key('KEY_Q')  # Row 0, col 0
        self.keyboard_handler.press_key('KEY_A')  # Row 1, col 0
        self.keyboard_handler.press_key('KEY_W')  # Row 0, col 1
        
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        
        # Scan row 0
        self.memory.write_byte(kol_addr, 0x01)
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xFC)  # Bits 0 and 1 cleared (Q and W)
        
        # Scan row 1
        self.memory.write_byte(kol_addr, 0x02)
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xFE)  # Bit 0 cleared (A)
        
        # Scan row 2 (no keys pressed)
        self.memory.write_byte(kol_addr, 0x04)
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xFF)  # No keys
    
    def test_cpu_instruction_reads_keyboard(self):
        """Test CPU can execute instructions that read keyboard state."""
        # This would require setting up a small test program
        # For now, we'll test direct register access
        
        # Press a key
        self.keyboard_handler.press_key('KEY_A')
        
        # Select the row containing KEY_A
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        self.memory.write_byte(kol_addr, 0x02)  # Row 1
        
        # CPU should be able to read KIL
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        kil_value = self.memory.read_byte(kil_addr)
        
        # Verify key is detected
        self.assertEqual(kil_value, 0xFE)  # Bit 0 cleared
    
    def test_multiple_overlays_coexist(self):
        """Test keyboard overlay works with other memory overlays."""
        # Add a ROM overlay
        rom_data = bytes([0xFF] * 1024)
        self.memory.add_rom(0x1000, rom_data, "Test ROM")
        
        # Keyboard should still work
        self.keyboard_handler.press_key('KEY_B')
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        
        # KEY_B is in row 2, column 4
        self.memory.write_byte(kol_addr, 0x04)  # Select row 2
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xEF)  # Bit 4 cleared
        
        # ROM should also be readable
        rom_value = self.memory.read_byte(0x1000)
        self.assertEqual(rom_value, 0xFF)
    
    def test_keyboard_state_persistence(self):
        """Test keyboard state persists across multiple reads."""
        # Press and hold a key
        self.keyboard_handler.press_key('KEY_SPACE')
        
        # Select the row
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        self.memory.write_byte(kol_addr, 0x40)  # Row 6
        
        # Read multiple times
        for _ in range(10):
            kil = self.memory.read_byte(kil_addr)
            self.assertEqual(kil, 0xFD)  # Bit 1 cleared (SPACE)
        
        # Release key
        self.keyboard_handler.release_key('KEY_SPACE')
        
        # Now should read as not pressed
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xFF)
    
    def test_simultaneous_key_presses(self):
        """Test handling multiple simultaneous key presses."""
        # Simulate pressing Ctrl+A
        self.keyboard_handler.press_key('KEY_CTRL')  # Row 6, col 7
        self.keyboard_handler.press_key('KEY_A')     # Row 1, col 0
        
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        
        # Check CTRL
        self.memory.write_byte(kol_addr, 0x40)  # Row 6
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0x7F)  # Bit 7 cleared
        
        # Check A
        self.memory.write_byte(kol_addr, 0x02)  # Row 1
        kil = self.memory.read_byte(kil_addr)
        self.assertEqual(kil, 0xFE)  # Bit 0 cleared
        
        # Both keys remain pressed
        self.assertEqual(len(self.keyboard_handler.pressed_keys), 2)
    
    def test_keyboard_register_addresses(self):
        """Verify keyboard registers are at correct addresses."""
        # Test addresses match expected values
        kol_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOL
        koh_addr = INTERNAL_MEMORY_START + IMEMRegisters.KOH
        kil_addr = INTERNAL_MEMORY_START + IMEMRegisters.KIL
        
        self.assertEqual(kol_addr, INTERNAL_MEMORY_START + 0xF0)
        self.assertEqual(koh_addr, INTERNAL_MEMORY_START + 0xF1)
        self.assertEqual(kil_addr, INTERNAL_MEMORY_START + 0xF2)
        
        # Verify we can read/write to these addresses
        self.memory.write_byte(kol_addr, 0xAA)
        self.assertEqual(self.keyboard_handler._last_kol, 0xAA)
        
        self.memory.write_byte(koh_addr, 0xBB)
        self.assertEqual(self.keyboard_handler._last_koh, 0xBB)


if __name__ == '__main__':
    unittest.main()