#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from pce500.debug import DisplayRenderer, MemoryInspector
from pce500.config import MachineConfig


def main():
    # Create emulator
    emu = PCE500Emulator()
    
    # Example: Load ROM if available
    rom_path = "rom.bin"  # Replace with actual ROM path
    if Path(rom_path).exists():
        print(f"Loading ROM from {rom_path}")
        emu.load_rom(rom_path)
    else:
        print(f"ROM file '{rom_path}' not found. Running with empty ROM space.")
    
    # Print memory map
    print("\n" + emu.machine.get_memory_info())
    
    # Reset and run a few instructions
    print("\nResetting CPU...")
    emu.reset()
    
    # Enable tracing for debugging
    emu.trace_enabled = True
    
    # Run for a few cycles
    print("\nRunning emulation for 100 cycles...")
    emu.run(max_cycles=100)
    
    # Get CPU state
    state = emu.get_cpu_state()
    print(f"\nCPU State after {state['cycles']} cycles:")
    print(f"  PC: {state['pc']:06X}")
    print(f"  A: {state['a']:02X}  B: {state['b']:02X}  BA: {state['ba']:04X}")
    print(f"  X: {state['x']:06X}  Y: {state['y']:06X}")
    print(f"  Flags: Z={state['flags']['z']} C={state['flags']['c']}")
    
    # Create debug image
    renderer = DisplayRenderer()
    print("\nCreating debug image...")
    renderer.create_debug_image(
        emu.machine.main_lcd,
        emu.machine.sub_lcd,
        state,
        "pce500_debug.png"
    )
    print("Debug image saved to pce500_debug.png")
    
    # Memory inspection example
    inspector = MemoryInspector(emu.machine.memory)
    print("\nMemory dump at PC:")
    print(inspector.dump_memory(state['pc'], 32))
    
    # Performance stats
    stats = emu.get_performance_stats()
    print(f"\nPerformance: {stats['emulated_mhz']:.2f} MHz emulated "
          f"({stats['speed_ratio']:.1f}x target speed)")


def example_with_config():
    """Example using configuration file."""
    # Create and save a config
    config = MachineConfig.for_model("PC-E500S")
    config.save("pce500s.json")
    
    # Load config
    loaded_config = MachineConfig.load("pce500s.json")
    print(f"Loaded config: {loaded_config.name}")
    print(f"CPU Frequency: {loaded_config.cpu_frequency:,} Hz")
    print("Memory regions:")
    for region in loaded_config.memory_map:
        print(f"  {region.name}: {region.start:06X}-{region.start+region.size-1:06X}")


if __name__ == "__main__":
    main()
    # example_with_config()