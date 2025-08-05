"""Test keyboard hardware integration with correct active-low strobing."""

import pytest
from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


@pytest.mark.skip(reason="Hardware keyboard implementation not currently in use")
class TestKeyboardHardwareIntegration:
    """Test keyboard hardware with correct active-low strobing behavior."""
    
    def test_active_low_column_strobing(self):
        """Test that keyboard uses active-low column strobing."""
        # Create emulator
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        # Program that properly strobes column 0 (active-low)
        # KOL = 0xFE means KO0 is low (active), KO1-KO7 are high (inactive)
        test_program = bytes([
            # MV A, 0xFE   - Load 0xFE into A (strobe column 0)
            0x08, 0xFE,
            # MV (0xF0), A - Store A to KOL
            0xA0, 0xF0,
            # MV A, (0xF2) - Load KIL into A
            0x80, 0xF2,
        ])
        
        # Load and run program
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press KEY_Q (column 0, row 1)
        emulator.press_key('KEY_Q')
        
        # Execute program
        emulator.step()  # MV A, 0xFE
        emulator.step()  # MV (0xF0), A - strobe column 0
        emulator.step()  # MV A, (0xF2) - read KIL
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # With KEY_Q pressed and column 0 strobed, KI1 should be low
        # So KIL should be 0xFD (bit 1 clear, all others set)
        assert reg_a == 0xFD, f"Expected 0xFD but got 0x{reg_a:02X}"
    
    def test_no_keys_pressed_returns_ff(self):
        """Test that KIL returns 0xFF when no keys are pressed."""
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        test_program = bytes([
            # MV A, 0x00   - Load 0x00 into A (strobe all columns)
            0x08, 0x00,
            # MV (0xF0), A - Store A to KOL
            0xA0, 0xF0,
            # MV A, (0xF2) - Load KIL into A
            0x80, 0xF2,
        ])
        
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Execute without pressing any keys
        emulator.step()  # MV A, 0x00
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # No keys pressed, all KI lines should be high (0xFF)
        assert reg_a == 0xFF, f"Expected 0xFF but got 0x{reg_a:02X}"
    
    def test_wrong_column_strobe(self):
        """Test that keys are not detected when wrong column is strobed."""
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        test_program = bytes([
            # MV A, 0xFD   - Load 0xFD into A (strobe column 1, not 0)
            0x08, 0xFD,
            # MV (0xF0), A - Store A to KOL
            0xA0, 0xF0,
            # MV A, (0xF2) - Load KIL into A
            0x80, 0xF2,
        ])
        
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press KEY_Q (column 0, row 1)
        emulator.press_key('KEY_Q')
        
        # Execute program
        emulator.step()  # MV A, 0xFD
        emulator.step()  # MV (0xF0), A - strobe column 1
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # Column 0 not strobed, so KEY_Q should not be detected
        assert reg_a == 0xFF, f"Expected 0xFF but got 0x{reg_a:02X}"
    
    def test_multiple_keys_same_row(self):
        """Test multiple keys pressed in same row."""
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        test_program = bytes([
            # MV A, 0xFC   - Load 0xFC (strobe columns 0 and 1)
            0x08, 0xFC,
            # MV (0xF0), A - Store A to KOL
            0xA0, 0xF0,
            # MV A, (0xF2) - Load KIL into A
            0x80, 0xF2,
        ])
        
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press Q (col 0, row 1) and E (col 1, row 1)
        emulator.press_key('KEY_Q')
        emulator.press_key('KEY_E')
        
        # Execute
        emulator.step()  # MV A, 0xFC
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # Both keys on row 1, so KI1 should be low
        assert reg_a == 0xFD, f"Expected 0xFD but got 0x{reg_a:02X}"
    
    def test_koh_register_strobing(self):
        """Test strobing columns controlled by KOH."""
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        test_program = bytes([
            # MV A, 0xFF   - All KOL columns inactive
            0x08, 0xFF,
            # MV (0xF0), A - Store to KOL
            0xA0, 0xF0,
            # MV A, 0xFB   - KO10 active (bit 2 low)
            0x08, 0xFB,
            # MV (0xF1), A - Store to KOH
            0xA0, 0xF1,
            # MV A, (0xF2) - Read KIL
            0x80, 0xF2,
        ])
        
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press P (col 10, row 0)
        emulator.press_key('KEY_P')
        
        # Execute
        emulator.step()  # MV A, 0xFF
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, 0xFB
        emulator.step()  # MV (0xF1), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # P pressed on row 0, so KI0 should be low
        assert reg_a == 0xFE, f"Expected 0xFE but got 0x{reg_a:02X}"
    
    def test_keyboard_strobe_disable(self):
        """Test KSD bit disables keyboard scanning."""
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
        
        test_program = bytes([
            # Set KSD bit in LCC register
            # MV A, 0x04   - Load 0x04 (KSD bit set)
            0x08, 0x04,
            # MV (0xFE), A - Store to LCC
            0xA0, 0xFE,
            # Now try to scan keyboard
            # MV A, 0xFE   - Strobe column 0
            0x08, 0xFE,
            # MV (0xF0), A - Store to KOL
            0xA0, 0xF0,
            # MV A, (0xF2) - Read KIL
            0x80, 0xF2,
        ])
        
        emulator.load_rom(test_program, start_address=0x0000)
        emulator.cpu.regs.set(RegisterName.PC, 0x0000)
        
        # Press a key
        emulator.press_key('KEY_Q')
        
        # Execute
        emulator.step()  # MV A, 0x04
        emulator.step()  # MV (0xFE), A - set KSD
        emulator.step()  # MV A, 0xFE
        emulator.step()  # MV (0xF0), A
        emulator.step()  # MV A, (0xF2)
        
        reg_a = emulator.cpu.regs.get(RegisterName.A)
        
        # KSD set, keyboard disabled, should read 0xFF
        assert reg_a == 0xFF, f"Expected 0xFF but got 0x{reg_a:02X}"