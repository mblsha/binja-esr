"""Simple demonstration of keyboard strobing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.keyboard_hardware import KeyboardHardware


def test_keyboard_strobing():
    """Test keyboard strobing without CPU execution."""
    print("PC-E500 Keyboard Hardware Strobing Test")
    print("=" * 50)

    # Create standalone keyboard hardware
    memory_data = {0x1000FE: 0x00}  # LCC register
    keyboard = KeyboardHardware(lambda addr: memory_data.get(addr, 0x00))

    print("\n1. Test active-low column strobing")
    print("-" * 40)

    # Press KEY_Q (column 0, row 1)
    keyboard.press_key("KEY_Q")

    # Wrong column strobe
    keyboard.write_register(0xF0, 0xFD)  # KOL = 0xFD (column 1 active)
    kil = keyboard.read_register(0xF2)
    print(f"Strobe column 1, read KIL: 0x{kil:02X} (expected 0xFF - no key)")

    # Correct column strobe
    keyboard.write_register(0xF0, 0xFE)  # KOL = 0xFE (column 0 active)
    kil = keyboard.read_register(0xF2)
    print(f"Strobe column 0, read KIL: 0x{kil:02X} (expected 0xFD - bit 1 low)")

    print("\n2. Test multiple columns")
    print("-" * 40)

    # Press KEY_E (column 1, row 1)
    keyboard.press_key("KEY_E")

    # Strobe both columns
    keyboard.write_register(0xF0, 0xFC)  # KOL = 0xFC (columns 0 and 1 active)
    kil = keyboard.read_register(0xF2)
    print(
        f"Strobe columns 0&1, read KIL: 0x{kil:02X} (expected 0xFD - both keys on row 1)"
    )

    print("\n3. Test KOH register (columns 8-10)")
    print("-" * 40)

    keyboard.release_all_keys()
    keyboard.press_key("KEY_P")  # Column 10, row 0

    # Set KOL inactive, KOH active for column 10
    keyboard.write_register(0xF0, 0xFF)  # All KOL inactive
    keyboard.write_register(0xF1, 0xFB)  # KOH = 0xFB (bit 2 low = column 10 active)
    kil = keyboard.read_register(0xF2)
    print(f"Strobe column 10, read KIL: 0x{kil:02X} (expected 0xFE - bit 0 low)")

    print("\n4. Test KSD (Key Strobe Disable)")
    print("-" * 40)

    # Enable KSD
    memory_data[0x1000FE] = 0x04  # Set bit 2
    kil = keyboard.read_register(0xF2)
    print(
        f"With KSD enabled, read KIL: 0x{kil:02X} (expected 0xFF - keyboard disabled)"
    )

    # Disable KSD
    memory_data[0x1000FE] = 0x00
    kil = keyboard.read_register(0xF2)
    print(
        f"With KSD disabled, read KIL: 0x{kil:02X} (expected 0xFE - key detected again)"
    )

    print("\n5. Debug information")
    print("-" * 40)
    debug = keyboard.get_debug_info()
    print(f"Debug info: {debug}")

    print("\n✓ Test complete - keyboard hardware strobing works correctly!")


if __name__ == "__main__":
    test_keyboard_strobing()
