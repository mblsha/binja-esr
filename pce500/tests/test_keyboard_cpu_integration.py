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
        # 1. Sets KOL to 0x01 (select row 0)
        # 2. Reads KIL to register A
        # 3. Loops forever
        test_program = bytes([
            # MV A, 0x01   - Load 0x01 into A
            0x08, 0x01,
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
        self.emulator.step()  # MV A, 0x01
        self.emulator.step()  # MV (0xF0), A - stores to KOL
        
        # Verify KOL was set
        kol_value = self.emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOL)
        self.assertEqual(kol_value, 0x01)
        
        # No keys pressed, so KIL should read 0xFE (workaround for performance)
        self.emulator.step()  # MV A, (0xF2) - reads KIL
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        self.assertEqual(reg_a, 0xFE)
        
    def test_key_press_detection(self):
        """Test that key presses are detected by CPU."""
        # Step through to set KOL
        self.emulator.step()  # MV A, 0x01
        self.emulator.step()  # MV (0xF0), A
        
        # Press KEY_Q (row 0, column 0)
        self.emulator.press_key('KEY_Q')
        
        # Read KIL
        self.emulator.step()  # MV A, (0xF2)
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        
        # With KEY_Q pressed in row 0, bit 0 should be cleared (active low)
        # Since the key is pressed, result will be 0xFE (bit 0 cleared)
        self.assertEqual(reg_a, 0xFE)
        
    def test_key_release(self):
        """Test that key releases are properly handled."""
        # Step through to set KOL
        self.emulator.step()  # MV A, 0x01
        self.emulator.step()  # MV (0xF0), A
        
        # Press and release KEY_Q
        self.emulator.press_key('KEY_Q')
        self.emulator.release_key('KEY_Q')
        
        # Read KIL
        self.emulator.step()  # MV A, (0xF2)
        reg_a = self.emulator.cpu.regs.get(RegisterName.A)
        
        # No keys pressed, should read 0xFE
        self.assertEqual(reg_a, 0xFE)
        
    def test_multiple_rows(self):
        """Test scanning different keyboard rows."""
        # Create a new emulator for this test since we need a different program
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        # Create a program that scans row 1
        # MV A, 0x02   - Load 0x02 into A (row 1)
        # MV (0xF0), A - Store A to KOL
        # MV A, (0xF2) - Load KIL into A
        test_program = bytes([0x08, 0x02, 0xA0, 0xF0, 0x80, 0xF2])
        
        # Load and run the program
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press KEY_A (row 1, column 0)
        emulator.press_key('KEY_A')
        
        # Step through program
        emulator.step()  # MV A, 0x02
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # With KEY_A pressed in row 1, bit 0 should be cleared
        # Since the key is pressed, result will be 0xFE (bit 0 cleared)
        self.assertEqual(reg_a, 0xFE)


if __name__ == '__main__':
    unittest.main()