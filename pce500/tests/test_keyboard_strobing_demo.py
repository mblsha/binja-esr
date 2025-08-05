"""Demonstration of correct keyboard strobing behavior."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def demonstrate_keyboard_strobing():
    """Demonstrate the correct keyboard strobing behavior."""
    print("PC-E500 Keyboard Strobing Demonstration")
    print("=" * 50)
    
    # Create emulator
    emu = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
    
    # Program that demonstrates proper keyboard scanning
    # This is similar to what the real PC-E500 ROM would do
    program = bytes([
        # Scan column 0
        0x08, 0xFE,      # MV A, 0xFE (KO0 low, others high)
        0xA0, 0xF0,      # MV (KOL), A
        0x08, 0xFF,      # MV A, 0xFF (all KOH high)
        0xA0, 0xF1,      # MV (KOH), A
        0x80, 0xF2,      # MV A, (KIL) - read keyboard
        0xC0, 0x00,      # MV (0x00), A - store result
        
        # Scan column 1
        0x08, 0xFD,      # MV A, 0xFD (KO1 low, others high)
        0xA0, 0xF0,      # MV (KOL), A
        0x80, 0xF2,      # MV A, (KIL)
        0xC0, 0x01,      # MV (0x01), A
        
        # Scan column 10 (using KOH)
        0x08, 0xFF,      # MV A, 0xFF (all KOL high)
        0xA0, 0xF0,      # MV (KOL), A
        0x08, 0xFB,      # MV A, 0xFB (KO10 low - bit 2)
        0xA0, 0xF1,      # MV (KOH), A
        0x80, 0xF2,      # MV A, (KIL)
        0xC0, 0x02,      # MV (0x02), A
        
        0x01,            # RETI (use as halt)
    ])
    
    # Load program
    emu.load_rom(program, start_address=0x0000)
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    
    print("\n1. No keys pressed - scanning all columns should return 0xFF")
    print("-" * 50)
    
    # Run scan without any keys pressed
    for i in range(9):
        emu.step()
    
    col0_result = emu.memory.read_byte(0x00)
    col1_result = emu.memory.read_byte(0x01)
    print(f"Column 0 scan result: 0x{col0_result:02X} (expected 0xFF)")
    print(f"Column 1 scan result: 0x{col1_result:02X} (expected 0xFF)")
    
    # Continue scanning column 10
    for i in range(6):
        emu.step()
    col10_result = emu.memory.read_byte(0x02)
    print(f"Column 10 scan result: 0x{col10_result:02X} (expected 0xFF)")
    
    print("\n2. Press KEY_Q (column 0, row 1) and rescan")
    print("-" * 50)
    
    # Reset and press key
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.press_key('KEY_Q')
    
    # Run scan again
    for i in range(9):
        emu.step()
    
    col0_result = emu.memory.read_byte(0x00)
    col1_result = emu.memory.read_byte(0x01)
    print(f"Column 0 scan result: 0x{col0_result:02X} (expected 0xFD - bit 1 low)")
    print(f"Column 1 scan result: 0x{col1_result:02X} (expected 0xFF - no key in this column)")
    
    print("\n3. Press KEY_P (column 10, row 0) and continue scan")
    print("-" * 50)
    
    emu.press_key('KEY_P')
    
    # Continue to scan column 10
    for i in range(6):
        emu.step()
    col10_result = emu.memory.read_byte(0x02)
    print(f"Column 10 scan result: 0x{col10_result:02X} (expected 0xFE - bit 0 low)")
    
    print("\n4. Test Key Strobe Disable (KSD)")
    print("-" * 50)
    
    # Set KSD bit in LCC register
    emu.memory.write_byte(0x1000FE, 0x04)  # Set bit 2 of LCC
    
    # Reset and scan again
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    for i in range(9):
        emu.step()
    
    col0_result = emu.memory.read_byte(0x00)
    print(f"Column 0 scan with KSD enabled: 0x{col0_result:02X} (expected 0xFF - keyboard disabled)")
    
    print("\n5. Keyboard state debugging")
    print("-" * 50)
    debug = emu.keyboard.get_debug_info()
    print(f"Current KOL: {debug['kol']}")
    print(f"Current KOH: {debug['koh']}")
    print(f"Active columns: {debug['active_columns']}")
    print(f"Pressed keys: {debug['pressed_keys']}")
    print(f"KSD enabled: {debug['ksd_enabled']}")
    
    print("\nDemonstration complete!")
    print("\nKey points:")
    print("- Columns are selected by setting bits LOW (active-low)")
    print("- When a key is pressed and its column is selected, its row bit goes LOW")
    print("- KSD bit in LCC register disables all keyboard scanning")
    print("- This matches real PC-E500 hardware behavior")


if __name__ == "__main__":
    demonstrate_keyboard_strobing()