#!/usr/bin/env python3
"""Trace memory accesses to understand performance issues."""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
repo_root = Path(__file__).parent.parent.parent
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from pce500 import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


class MemoryTracer:
    def __init__(self, original_read):
        self.original_read = original_read
        self.reads = []

    def __call__(self, address, cpu_pc=None):
        self.reads.append(address)
        return self.original_read(address, cpu_pc)


# Create emulator (default keyboard handler)
print("Creating emulator...")
emu = PCE500Emulator(
    trace_enabled=False,
    perfetto_trace=False,
    save_lcd_on_exit=False,
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
print(f"PC after reset: {emu.cpu.regs.get(RegisterName.PC):06X}\n")

# Wrap memory read
tracer = MemoryTracer(emu.memory.read_byte)
emu.memory.read_byte = tracer

# Execute one instruction
pc = emu.cpu.regs.get(RegisterName.PC)
print(f"Executing instruction at PC=0x{pc:06X}")

# Get the instruction
instr = emu.cpu.decode_instruction(pc)
print(f"Instruction: {instr.name()}")

# Clear reads and execute
tracer.reads.clear()
emu.step()

print(f"\nMemory reads during execution: {len(tracer.reads)}")

# Group reads by address
read_counts = Counter(tracer.reads)
print("\nMemory access pattern:")
for addr, count in sorted(read_counts.items(), key=lambda x: -x[1])[:10]:
    if addr >= 0x100000:
        print(f"  Internal 0x{addr - 0x100000:02X}: {count} reads")
    else:
        print(f"  0x{addr:06X}: {count} reads")
