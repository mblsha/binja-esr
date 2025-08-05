"""Integration tests for keyboard to CPU communication."""

import unittest
import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set FORCE_BINJA_MOCK before imports
os.environ['FORCE_BINJA_MOCK'] = '1'

from pce500 import PCE500Emulator
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START
from sc62015.pysc62015.emulator import RegisterName


class TestKeyboardCPUIntegration(unittest.TestCase):
    """Test keyboard integration with CPU via memory-mapped I/O."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a full PC-E500 emulator with integrated keyboard
        self.emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        # Create a simple test ROM that reads keyboard state
        # This program:
        # 1. Sets KOL to 0xFE (select column 0 with active-low logic)
        # 2. Reads KIL to register A
        # 3. Loops forever
        test_program = bytes([
            # MV A, 0xFE   - Load 0xFE into A (bit 0 low = column 0 active)
            0x08, 0xFE,
            # MV (0xF0), A - Store A to KOL (internal memory address 0xF0)
            0xA0, 0xF0,
            # MV A, (0xF2) - Load KIL into A (internal memory address 0xF2)
            0x80, 0xF2,
            # JP 0x0006    - Jump back to the MV A, (0xF2) instruction
            0x02, 0x06, 0x00
        ])
        
        # Load the test program as ROM at a simple address
        self.emulator.load_rom(test_program, start_address=0x0000)
        
        # Set PC to start of program
        self.emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
    def test_keyboard_register_access(self):
        """Test that CPU can access keyboard registers."""
        # Step through the program to set KOL
        self.emulator.step()  # MV A, 0xFE
        self.emulator.step()  # MV (0xF0), A - stores to KOL
        
        # Verify KOL was set
        kol_value = self.emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOL)
        self.assertEqual(kol_value, 0xFE)
        
        # No keys pressed, so KIL should read 0xFF (all bits high with active-low logic)
        self.emulator.step()  # MV A, (0xF2) - reads KIL
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        self.assertEqual(reg_a, 0xFF)
        
    def test_key_press_detection(self):
        """Test that key presses are detected by CPU."""
        # Step through to set KOL
        self.emulator.step()  # MV A, 0xFE
        self.emulator.step()  # MV (0xF0), A
        
        # Press KEY_Q (column 0, row 1)
        self.emulator.press_key('KEY_Q')
        
        # Read KIL
        self.emulator.step()  # MV A, (0xF2)
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        
        # With KEY_Q pressed in column 0 row 1, bit 1 should be low (active-low)
        # Since the key is pressed, result will be 0xFD (bit 1 clear)
        self.assertEqual(reg_a, 0xFD)
        
    def test_key_release(self):
        """Test that key releases are properly handled."""
        # Step through to set KOL
        self.emulator.step()  # MV A, 0xFE
        self.emulator.step()  # MV (0xF0), A
        
        # Press and release KEY_Q
        self.emulator.press_key('KEY_Q')
        self.emulator.release_key('KEY_Q')
        
        # Read KIL
        self.emulator.step()  # MV A, (0xF2)
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        
        # Key was released, hardware simulation doesn't have debouncing
        # Should read 0xFF (no keys pressed)
        self.assertEqual(reg_a, 0xFF)
        
    def test_multiple_rows(self):
        """Test scanning different keyboard rows."""
        # Create a new emulator for this test since we need a different program
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        # Create a program that scans column 1
        # MV A, 0xFD   - Load 0xFD into A (bit 1 low = column 1 active)
        # MV (0xF0), A - Store A to KOL
        # MV A, (0xF2) - Load KIL into A
        test_program = bytes([0x08, 0xFD, 0xA0, 0xF0, 0x80, 0xF2])
        
        # Load and run the program
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press KEY_W (column 1, row 0)
        emulator.press_key('KEY_W')
        
        # Step through program
        emulator.step()  # MV A, 0xFD
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # With KEY_W pressed in column 1 row 0, bit 0 should be low
        # Since the key is pressed, result will be 0xFE (bit 0 clear)
        self.assertEqual(reg_a, 0xFE)


if __name__ == '__main__':
    unittest.main()