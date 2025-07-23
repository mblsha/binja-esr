#!/usr/bin/env python3
"""Test LCD controller for PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the display module to avoid emulator dependencies
from pce500.display import HD61202Controller


def test_basic_operations():
    """Test basic LCD controller operations."""
    controller = HD61202Controller()
    
    print("=== HD61202 Controller Test ===\n")
    
    # Test initial state
    print("1. Initial state:")
    print(f"   Display on: {controller.display_on}")
    print(f"   Page: {controller.page}")
    print(f"   Column: {controller.column}")
    print(f"   Display size: {controller.width}x{controller.height}")
    
    # Test display on/off commands
    print("\n2. Testing display control:")
    
    # Turn on left display (chip 0)
    addr = 0x20000  # Base address, instruction mode
    cmd = 0x3F  # Display ON
    print(f"   Sending display ON command (0x{cmd:02X}) to address 0x{addr:05X}")
    controller.write(addr, cmd)
    print(f"   Left display on: {controller.display_on[0]}")
    print(f"   Right display on: {controller.display_on[1]}")
    
    # Turn on right display (chip 1)  
    addr = 0x20004  # Chip 1 select
    controller.write(addr, cmd)
    print(f"   After turning on right chip: {controller.display_on}")
    
    # Test page selection
    print("\n3. Testing page selection:")
    addr = 0x20000  # Both chips
    cmd = 0xB2  # Set page 2
    print(f"   Setting page 2 (command 0x{cmd:02X})")
    controller.write(addr, cmd)
    print(f"   Page registers: {controller.page}")
    
    # Test column selection
    print("\n4. Testing column selection:")
    cmd = 0x15  # Set column to 21 (0x15)
    print(f"   Setting column to 21 (command 0x{cmd:02X})")
    controller.write(addr, cmd)
    print(f"   Column registers: {controller.column}")
    
    # Test data write
    print("\n5. Testing data write:")
    addr = 0x20002  # Data write to both chips
    data = 0xFF  # All pixels on
    print(f"   Writing data 0x{data:02X} to current position")
    controller.write(addr, data)
    
    # Check VRAM
    left_offset = controller.page[0] * 120 + controller.column[0]
    right_offset = controller.page[1] * 120 + controller.column[1]
    print(f"   Left VRAM[{left_offset}]: 0x{controller.left_vram[left_offset]:02X}")
    print(f"   Right VRAM[{right_offset}]: 0x{controller.right_vram[right_offset]:02X}")
    print(f"   Column after write: {controller.column}")
    
    # Test status read
    print("\n6. Testing status read:")
    addr = 0x20001  # Status read
    status = controller.read(addr)
    print(f"   Status byte: 0x{status:02X}")
    print(f"   Busy: {bool(status & 0x80)}")
    print(f"   Display ON: {bool(status & 0x20)}")
    print(f"   Reset: {bool(status & 0x10)}")
    
    # Test reset
    print("\n7. Testing reset:")
    controller.reset()
    print("   After reset:")
    print(f"   - Display on: {controller.display_on}")
    print(f"   - Page: {controller.page}")
    print(f"   - Column: {controller.column}")


def test_address_patterns():
    """Test different address patterns."""
    controller = HD61202Controller()
    
    print("\n=== Address Pattern Test ===\n")
    
    test_cases = [
        # (address, description)
        (0x20000, "Base: instruction to both chips"),
        (0x20001, "Base+1: status read from both chips"),
        (0x20002, "Base+2: data write to both chips"),
        (0x20003, "Base+3: data read from both chips"),
        (0x20004, "Base+4: instruction to right chip only"),
        (0x20006, "Base+6: data write to right chip only"),
        (0x20008, "Base+8: instruction to left chip only"),
        (0x2000A, "Base+A: data write to left chip only"),
        (0x2000C, "Base+C: no chip selected"),
        (0x22000, "Different base: still works"),
        (0x30000, "Out of range: ignored"),
    ]
    
    # Turn on displays first
    controller.write(0x20000, 0x3F)
    
    for addr, desc in test_cases:
        print(f"Address 0x{addr:05X}: {desc}")
        
        # Try writing a command
        if addr <= 0x2FFFF:
            old_page = controller.page.copy()
            controller.write(addr, 0xB1)  # Set page 1
            
            # Check which chips were affected
            if controller.page != old_page:
                changed = []
                if controller.page[0] != old_page[0]:
                    changed.append("left")
                if controller.page[1] != old_page[1]:
                    changed.append("right")
                print(f"  -> Changed: {', '.join(changed)}")
            else:
                print("  -> No change")
        else:
            print("  -> Out of range")
        print()


def test_drawing_pattern():
    """Test drawing a simple pattern."""
    controller = HD61202Controller()
    
    print("\n=== Drawing Pattern Test ===\n")
    
    # Turn on both displays
    controller.write(0x20000, 0x3F)
    
    # Draw a checkerboard pattern on page 0
    controller.write(0x20000, 0xB0)  # Set page 0
    
    print("Drawing checkerboard pattern on page 0...")
    addr = 0x20002  # Data write
    for col in range(120):
        # Set column
        controller.write(0x20000, col)
        
        # Write pattern (alternating 0xAA and 0x55)
        pattern = 0xAA if col % 2 == 0 else 0x55
        controller.write(addr, pattern)
    
    # Check a few bytes
    print("Sample of VRAM contents:")
    for i in range(0, 10):
        print(f"  Column {i}: Left=0x{controller.left_vram[i]:02X}, "
              f"Right=0x{controller.right_vram[i]:02X}")
    
    # Calculate fill percentage
    filled_left = sum(1 for b in controller.left_vram[:120] if b != 0)
    filled_right = sum(1 for b in controller.right_vram[:120] if b != 0)
    print("\nFill stats for page 0:")
    print(f"  Left chip: {filled_left}/120 columns filled")
    print(f"  Right chip: {filled_right}/120 columns filled")


if __name__ == "__main__":
    test_basic_operations()
    test_address_patterns()
    test_drawing_pattern()