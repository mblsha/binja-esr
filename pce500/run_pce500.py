#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def main():
    """Example with Perfetto tracing enabled."""
    # Note: Perfetto trace is automatically saved when exiting the context manager
    # Create emulator with tracing
    with PCE500Emulator(trace_enabled=True, perfetto_trace=True) as emu:
        print("Created emulator with Perfetto tracing enabled")
        print("Trace will be saved to pc-e500.trace")

        # Load ROM and run
        rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
        if rom_path.exists():
            with open(rom_path, "rb") as f:
                rom_data = f.read()
                assert len(rom_data) >= 0x100000
                rom_portion = rom_data[0xC0000:0x100000]
                emu.load_rom(rom_portion)

        # Reset and run
        emu.reset()

        num_steps = 1000
        print(f"Running {num_steps} instructions with tracing...")
        for _ in range(num_steps):
            emu.step()

        print(f"Executed {emu.instruction_count} instructions")
        print(f"Memory reads: {emu.memory_read_count}")
        print(f"Memory writes: {emu.memory_write_count}")

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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PC-E500 Emulator Example")
    args = parser.parse_args()
    main()
