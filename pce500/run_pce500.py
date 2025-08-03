#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def main(dump_pc=None, no_dump=False, save_lcd=True):
    """Example with Perfetto tracing enabled."""
    # Note: Perfetto trace is automatically saved when exiting the context manager
    # Create emulator with tracing
    with PCE500Emulator(trace_enabled=True, perfetto_trace=True, save_lcd_on_exit=save_lcd) as emu:
        print("Created emulator with Perfetto tracing enabled")
        print("Trace will be saved to pc-e500.trace")

        # Handle memory dump configuration
        if no_dump:
            # Disable dumps by setting an impossible PC value
            emu.set_memory_dump_pc(0xFFFFFF)
            print("Internal memory dumps disabled")
        elif dump_pc is not None:
            emu.set_memory_dump_pc(dump_pc)
            print(f"Internal memory dump will trigger at PC=0x{dump_pc:06X}")

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
        print(f"PC after reset: {emu.cpu.regs.get(RegisterName.PC):06X}")

        num_steps = 20000
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
        
        # Display detailed statistics for each chip
        print("\nDetailed LCD Chip Statistics:")
        print(f"  Chip select usage: BOTH={emu.lcd.cs_both_count}, LEFT={emu.lcd.cs_left_count}, RIGHT={emu.lcd.cs_right_count}")
        stats = emu.lcd.get_chip_statistics()
        for stat in stats:
            chip_name = "Left" if stat['chip'] == 0 else "Right"
            print(f"\n  {chip_name} Chip (Chip {stat['chip']}):")
            print(f"    Display ON: {stat['on']}")
            print(f"    Instructions received: {stat['instructions']}")
            print(f"    ON/OFF commands: {stat['on_off_commands']}")
            print(f"    Data bytes written: {stat['data_written']}")
            print(f"    Data bytes read: {stat['data_read']}")
            print(f"    Current page: {stat['page']}")
            print(f"    Current column: {stat['column']}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PC-E500 Emulator Example")
    parser.add_argument("--dump-pc", type=lambda x: int(x, 0),
                        help="PC address to trigger internal memory dump (hex or decimal, e.g., 0x0F119C)")
    parser.add_argument("--no-dump", action='store_true',
                        help="Disable internal memory dumps entirely")
    parser.add_argument("--no-lcd", action='store_true',
                        help="Don't save LCD displays as PNG files on exit")
    args = parser.parse_args()
    main(dump_pc=args.dump_pc, no_dump=args.no_dump, save_lcd=not args.no_lcd)
