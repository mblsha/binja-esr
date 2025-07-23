#!/usr/bin/env python3
"""Example usage of the simplified PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500.simple_emulator import SimplifiedPCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def main():
    """Demonstrate simplified emulator usage."""
    print("=== Simplified PC-E500 Emulator Example ===\n")
    
    # Create emulator with optional tracing
    emulator = SimplifiedPCE500Emulator(trace_enabled=True)
    print("✓ Created emulator")
    
    # Create a simple test ROM
    rom_data = bytearray(256 * 1024)
    
    # Set reset vector at 0xFFFFD to point to 0xC0000
    rom_data[0x3FFFD] = 0x00  # Low byte
    rom_data[0x3FFFE] = 0x00  # Middle byte  
    rom_data[0x3FFFF] = 0x0C  # High byte (0xC0000)
    
    # Add some simple code at 0xC0000
    # Just a series of NOPs for simplicity
    rom_data[0x0000] = 0x00  # NOP
    rom_data[0x0001] = 0x00  # NOP
    rom_data[0x0002] = 0x00  # NOP
    rom_data[0x0003] = 0x00  # NOP
    rom_data[0x0004] = 0x00  # NOP
    rom_data[0x0005] = 0x00  # NOP
    rom_data[0x0006] = 0x00  # NOP
    
    # Load ROM
    emulator.load_rom(bytes(rom_data))
    print("✓ Loaded ROM")
    
    # Optional: Load a memory card
    card_data = bytes([i % 256 for i in range(8192)])  # 8KB test pattern
    emulator.load_memory_card(card_data, 8192)
    print("✓ Loaded 8KB memory card")
    
    # Reset emulator (sets PC to reset vector)
    emulator.reset()
    print("✓ Reset emulator")
    
    # Check initial state
    state = emulator.get_cpu_state()
    print(f"\nInitial state:")
    print(f"  PC: 0x{state['pc']:06X}")
    print(f"  A:  0x{state['a']:02X}")
    print(f"  Flags: Z={state['flags']['z']}, C={state['flags']['c']}")
    
    # Add a breakpoint
    emulator.add_breakpoint(0xC0006)
    print(f"\n✓ Added breakpoint at 0xC0006")
    
    # Run until breakpoint
    print("\nRunning emulation...")
    instructions = emulator.run()
    
    # Check final state
    state = emulator.get_cpu_state()
    print(f"\nStopped after {instructions} instructions")
    print(f"Final state:")
    print(f"  PC: 0x{state['pc']:06X}")
    print(f"  A:  0x{state['a']:02X}")
    
    # Note: Since we only executed NOPs, register values remain unchanged
    
    # Show performance stats
    stats = emulator.get_performance_stats()
    print(f"\nPerformance:")
    print(f"  Instructions/sec: {stats['instructions_per_second']:.0f}")
    print(f"  Speed ratio: {stats['speed_ratio']:.2f}x")
    
    # Show trace (if enabled)
    if emulator.trace:
        print(f"\nExecution trace ({len(emulator.trace)} entries):")
        for entry in emulator.trace[:5]:  # Show first 5
            event_type, pc, *extra = entry
            print(f"  {event_type}: PC=0x{pc:06X}")
        if len(emulator.trace) > 5:
            print(f"  ... and {len(emulator.trace) - 5} more")
    
    # Show memory configuration
    print(f"\n{emulator.get_memory_info()}")
    
    # LCD example
    print("\n=== LCD Controller Example ===")
    
    # Turn on display (left chip)
    emulator.memory.write_byte(0x20008, 0x3F)  # Display on command
    print("✓ Display turned on")
    
    # Write some data to LCD
    emulator.memory.write_byte(0x2000A, 0xFF)  # Data write
    
    # Get display buffer
    buffer = emulator.get_display_buffer()
    print(f"✓ Display buffer: {buffer.shape[1]}x{buffer.shape[0]} pixels")
    

if __name__ == "__main__":
    main()