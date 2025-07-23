#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def main():
    # Create emulator
    emu = PCE500Emulator()
    
    # Example: Load ROM if available
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
    if rom_path.exists():
        print(f"Loading ROM from {rom_path}")
        with open(rom_path, "rb") as f:
            rom_data = f.read()
            # Load ROM portion (256KB at 0xC0000)
            if len(rom_data) >= 0x100000:
                rom_portion = rom_data[0xC0000:0x100000]
                emu.load_rom(rom_portion)
                print(f"Loaded {len(rom_portion)} bytes of ROM")
    else:
        print(f"ROM file '{rom_path}' not found. Running with empty ROM space.")
    
    # Print memory map
    print("\n" + emu.memory.get_memory_info())
    
    # Reset and set entry point
    print("\nResetting CPU...")
    emu.reset()
    
    # Run for a few instructions
    print("\nRunning emulation for 100 instructions...")
    for i in range(100):
        pc = emu.cpu.regs.get(RegisterName.PC)
        if not emu.step():
            print(f"Breakpoint hit at PC={pc:06X}")
            break
    
    # Get CPU state
    print(f"\nCPU State after {emu.cycle_count} cycles:")
    print(f"  PC: {emu.cpu.regs.get(RegisterName.PC):06X}")
    print(f"  A: {emu.cpu.regs.get(RegisterName.A):02X}  B: {emu.cpu.regs.get(RegisterName.B):02X}")
    print(f"  X: {emu.cpu.regs.get(RegisterName.X):06X}  Y: {emu.cpu.regs.get(RegisterName.Y):06X}")
    print(f"  S: {emu.cpu.regs.get(RegisterName.S):06X}  U: {emu.cpu.regs.get(RegisterName.U):06X}")
    print(f"  Flags: Z={emu.cpu.regs.get_flag('Z')} C={emu.cpu.regs.get_flag('C')}")
    
    # Display LCD state
    print("\nLCD Controller State:")
    print(f"  Display on: {emu.lcd.display_on}")
    print(f"  Page: {emu.lcd.page}")
    print(f"  Column: {emu.lcd.column}")
    
    # Performance stats
    stats = emu.get_performance_stats()
    print(f"\nPerformance: {stats['emulated_mhz']:.2f} MHz emulated "
          f"({stats['speed_ratio']:.1f}x target speed)")


def example_with_tracing():
    """Example with Perfetto tracing enabled."""
    # Create emulator with tracing
    emu = PCE500Emulator(trace_enabled=True, perfetto_trace=True)
    
    print("Created emulator with Perfetto tracing enabled")
    print("Trace will be saved to pc-e500.trace")
    
    # Load ROM and run
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
    if rom_path.exists():
        with open(rom_path, "rb") as f:
            rom_data = f.read()
            if len(rom_data) >= 0x100000:
                rom_portion = rom_data[0xC0000:0x100000]
                emu.load_rom(rom_portion)
    
    # Reset and run
    emu.reset()
    
    # Run for 1000 instructions
    print("Running 1000 instructions with tracing...")
    for _ in range(1000):
        emu.step()
    
    print(f"Executed {emu.instruction_count} instructions")
    print(f"Memory reads: {emu.memory_read_count}")
    print(f"Memory writes: {emu.memory_write_count}")
    
    # Note: Perfetto trace is automatically saved when emulator is destroyed


def example_with_breakpoints():
    """Example using breakpoints."""
    emu = PCE500Emulator()
    
    # Set some breakpoints
    breakpoints = [0xF10C2, 0xF10D0, 0xF10E0]
    for bp in breakpoints:
        emu.add_breakpoint(bp)
        print(f"Set breakpoint at {bp:06X}")
    
    # Load ROM
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
    if rom_path.exists():
        with open(rom_path, "rb") as f:
            rom_data = f.read()
            if len(rom_data) >= 0x100000:
                rom_portion = rom_data[0xC0000:0x100000]
                emu.load_rom(rom_portion)
    
    # Reset and run until breakpoint
    emu.reset()
    
    print("\nRunning until breakpoint...")
    steps = 0
    while steps < 10000:  # Safety limit
        pc = emu.cpu.regs.get(RegisterName.PC)
        if not emu.step():
            print(f"Hit breakpoint at PC={pc:06X} after {steps} steps")
            break
        steps += 1
    else:
        print(f"Completed {steps} steps without hitting breakpoint")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PC-E500 Emulator Example")
    parser.add_argument("--trace", action="store_true", help="Enable Perfetto tracing")
    parser.add_argument("--breakpoints", action="store_true", help="Run breakpoint example")
    
    args = parser.parse_args()
    
    if args.trace:
        example_with_tracing()
    elif args.breakpoints:
        example_with_breakpoints()
    else:
        main()