#!/usr/bin/env python3
"""Test Perfetto tracing in the simplified PC-E500 emulator."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable Perfetto tracing
os.environ['PCE500_ENABLE_TRACING'] = '1'

from pce500.simple_emulator import SimplifiedPCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def main():
    """Test Perfetto tracing functionality."""
    print("=== Testing Perfetto Tracing in Simplified Emulator ===\n")
    
    # Create emulator with Perfetto tracing enabled
    emulator = SimplifiedPCE500Emulator(trace_enabled=False, perfetto_trace=True)
    print(f"✓ Created emulator with Perfetto tracing enabled")
    
    # Create test ROM with some instructions
    rom_data = bytearray(256 * 1024)
    
    # Set reset vector at 0xFFFFD to point to 0xC0000
    rom_data[0x3FFFD] = 0x00  # Low byte
    rom_data[0x3FFFE] = 0x00  # Middle byte  
    rom_data[0x3FFFF] = 0x0C  # High byte (0xC0000)
    
    # Add some instructions at 0xC0000
    rom_data[0x0000] = 0x00  # NOP
    rom_data[0x0001] = 0x00  # NOP
    rom_data[0x0002] = 0x00  # NOP
    
    # Load ROM
    emulator.load_rom(bytes(rom_data))
    print("✓ Loaded test ROM")
    
    # Reset emulator
    emulator.reset()
    print("✓ Reset emulator")
    
    # Test memory write tracing
    print("\nTesting memory write tracing...")
    emulator.memory.write_byte(0xB8000, 0x42)
    emulator.memory.write_byte(0xB8001, 0x43)
    print("✓ Wrote to RAM (should be traced)")
    
    # Try to write to ROM (should be traced as ignored)
    emulator.memory.write_byte(0xC0000, 0xFF)
    print("✓ Attempted ROM write (should be traced as ignored)")
    
    # Test LCD command tracing
    print("\nTesting LCD command tracing...")
    emulator.memory.write_byte(0x20008, 0x3F)  # Display on
    emulator.memory.write_byte(0x20008, 0xB0)  # Set page 0
    emulator.memory.write_byte(0x20008, 0x40)  # Set Y address 0
    print("✓ Sent LCD commands (should be traced)")
    
    # Execute some instructions
    print("\nExecuting instructions...")
    for i in range(3):
        emulator.step()
        print(f"  Step {i+1}: PC=0x{emulator.cpu.regs.get(RegisterName.PC):06X}")
    
    # Show performance
    stats = emulator.get_performance_stats()
    print(f"\n✓ Executed {stats['instructions_executed']} instructions")
    
    # Stop tracing and save
    emulator.stop_tracing()
    print("\n✓ Stopped tracing")
    
    # Check if trace file exists
    trace_path = Path("pc-e500.trace")
    if trace_path.exists():
        size_kb = trace_path.stat().st_size / 1024
        print(f"✓ Trace file created: {trace_path} ({size_kb:.1f} KB)")
        print("\nTo view the trace:")
        print("  1. Open Chrome/Edge browser")
        print("  2. Navigate to: chrome://tracing")
        print("  3. Click 'Load' and select: pc-e500.trace")
    else:
        print("⚠️  Trace file not found - check if retrobus_perfetto is working")
    

if __name__ == "__main__":
    main()