#!/usr/bin/env python3
"""Debug script to understand keyboard performance issues."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName

# Create emulator with hardware keyboard
print("Creating emulator with hardware keyboard...")
emu = PCE500Emulator(
    trace_enabled=False,
    perfetto_trace=False,
    save_lcd_on_exit=False,
    keyboard_impl="hardware",
)

# Load ROM
rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500.bin"
if rom_path.exists():
    with open(rom_path, "rb") as f:
        rom_data = f.read()
        rom_portion = rom_data[0xC0000:0x100000]
        emu.load_rom(rom_portion)

# Reset
emu.reset()
print(f"PC after reset: {emu.cpu.regs.get(RegisterName.PC):06X}")

# Track KIL reads
kil_reads = 0
original_read_register = emu.keyboard.read_register


def tracked_read_register(offset):
    global kil_reads
    if offset == 0xF2:  # KIL
        kil_reads += 1
        if kil_reads <= 10:
            print(f"KIL read #{kil_reads}")
    return original_read_register(offset)


emu.keyboard.read_register = tracked_read_register

# Run a few instructions
print("\nRunning 10 instructions...")
for i in range(10):
    pc = emu.cpu.regs.get(RegisterName.PC)
    print(f"Instruction {i + 1}: PC=0x{pc:06X}")

    # Get instruction
    instr = emu.cpu.decode_instruction(pc)
    print(f"  Opcode: {instr.name()}")

    # Track memory reads before and after
    reads_before = emu.memory_read_count
    emu.step()
    reads_after = emu.memory_read_count

    print(f"  Memory reads: {reads_after - reads_before}")
    print(f"  Total KIL reads so far: {kil_reads}")

    if reads_after - reads_before > 10:
        print("  WARNING: High memory read count!")
        break

print(f"\nTotal KIL reads: {kil_reads}")
print(f"Total memory reads: {emu.memory_read_count}")
print(f"Total instructions: {emu.instruction_count}")
