#!/usr/bin/env python3
"""Test elegant LCD address decoding for PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the display module to avoid emulator dependencies
from pce500.display.hd61202u_toolkit import HD61202U, HD61202UController, ChipSelect, AddressDecode


def test_address_decoding():
    """Test the elegant address decoding system."""
    controller = HD61202UController()
    
    print("=== HD61202U Address Decoding Test ===\n")
    
    # Test cases with different address patterns
    test_addresses = [
        # Address,    Description
        (0x20000, "Base address - write instruction to both chips"),
        (0x20001, "Read instruction from both chips"),
        (0x20002, "Write data to both chips"),
        (0x20003, "Read data from both chips"),
        (0x20004, "Write instruction to chip 2 only"),
        (0x20008, "Write instruction to chip 1 only"),
        (0x2000C, "Write instruction to no chips"),
        (0x22000, "CS3=1, write instruction to both chips"),
        (0x23000, "CS2=1 (inactive), disables chip 2"),
        (0x10000, "Wrong range (0x1xxxx)"),
        (0x30000, "Wrong range (0x3xxxx)"),
    ]
    
    for addr, desc in test_addresses:
        print(f"Address: 0x{addr:05X} - {desc}")
        
        if controller.contains_address(addr):
            decode = controller.decode_address(addr)
            print(f"  ✓ Valid LCD access")
            print(f"  - R/W: {'Read' if decode.rw else 'Write'}")
            print(f"  - D/I: {'Data' if decode.di else 'Instruction'}")
            print(f"  - Chip Select: {decode.chip_select.name}")
            print(f"  - CS2: {'High (inactive)' if decode.cs2 else 'Low (active)'}")
            print(f"  - CS3: {'High (active)' if decode.cs3 else 'Low (inactive)'}")
            
            chips = controller._get_selected_chips(decode)
            if chips:
                chip_names = []
                if controller.chips['left'] in chips:
                    chip_names.append("Left")
                if controller.chips['right'] in chips:
                    chip_names.append("Right")
                print(f"  - Active chips: {', '.join(chip_names)}")
            else:
                print(f"  - Active chips: None")
        else:
            print(f"  ✗ Invalid LCD access")
        print()


def demonstrate_lcd_commands():
    """Demonstrate sending commands to the LCD."""
    controller = HD61202UController()
    
    print("\n=== LCD Command Examples ===\n")
    
    # Turn on display for both chips
    addr = 0x22000  # CS3=1, A1=0 (instruction), A3:A2=00 (both chips)
    cmd = 0x3F  # Display ON command
    print(f"1. Turn on both displays:")
    print(f"   Write 0x{cmd:02X} to address 0x{addr:05X}")
    controller.write(addr, cmd)
    print(f"   Left display on: {controller.chips['left'].display_on}")
    print(f"   Right display on: {controller.chips['right'].display_on}")
    
    # Set page on left chip only
    addr = 0x22008  # CS3=1, A1=0 (instruction), A3:A2=10 (left chip)
    cmd = 0xB3  # Set page 3
    print(f"\n2. Set page 3 on left chip:")
    print(f"   Write 0x{cmd:02X} to address 0x{addr:05X}")
    
    # Debug: check what command is parsed
    parsed_cmd = HD61202U.Parser.parse(False, False, cmd)
    print(f"   Parsed command: {parsed_cmd}")
    
    controller.write(addr, cmd)
    print(f"   Left chip page: {controller.chips['left'].page}")
    print(f"   Right chip page: {controller.chips['right'].page}")
    
    # Write data to right chip only
    addr = 0x22006  # CS3=1, A1=1 (data), A3:A2=01 (right chip)
    data = 0xAA  # Data pattern
    print(f"\n3. Write data to right chip:")
    print(f"   Write 0x{data:02X} to address 0x{addr:05X}")
    controller.write(addr, data)
    print(f"   Right chip VRAM[0]: 0x{controller.chips['right'].vram[0]:02X}")
    
    # Read status from both chips
    addr = 0x22001  # CS3=1, A1=0 (instruction), A3:A2=00 (both), A0=1 (read)
    print(f"\n4. Read status from both chips:")
    print(f"   Read from address 0x{addr:05X}")
    status = controller.read(addr)
    print(f"   Combined status: 0x{status:02X}")


if __name__ == "__main__":
    test_address_decoding()
    demonstrate_lcd_commands()