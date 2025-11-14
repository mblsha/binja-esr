"""Test PC-E500 emulator execution trace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from typing import List

from pce500 import PCE500Emulator
from sc62015.pysc62015.instr import decode, OPCODES
from sc62015.pysc62015 import RegisterName


def test_pce500_execution_trace():
    """Test executing 100 instructions from PC-E500 ROM entry point."""
    rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM file {rom_path} not found")

    # Create emulator and load full memory image
    emu = PCE500Emulator()
    with open(rom_path, "rb") as f:
        rom_data = f.read()

    # Validate file size - must be exactly 1MB
    assert len(rom_data) == 0x100000, (
        f"ROM file must be exactly 1MB (1048576 bytes), but is {len(rom_data)} bytes"
    )

    # Load ROM from 0xC0000-0xFFFFF (256KB)
    rom_portion = rom_data[0xC0000:0x100000]
    emu.load_rom(rom_portion, start_address=0xC0000)

    # Load RAM from dump as well
    ram_portion = rom_data[0xB8000:0xC0000]
    for i, byte in enumerate(ram_portion):
        if byte != 0:  # Only write non-zero bytes to save time
            emu.memory.write_byte(0xB8000 + i, byte)

    # Read entry point from 0xFFFFD (3 bytes, little-endian)
    entry_point = (
        emu.memory.read_byte(0xFFFFD)
        | (emu.memory.read_byte(0xFFFFE) << 8)
        | (emu.memory.read_byte(0xFFFFF) << 16)
    )

    # Validate entry point
    assert entry_point != 0xFFFFFF, "Entry point cannot be 0xFFFFFF (invalid)"
    assert entry_point != 0x000000, "Entry point cannot be 0x000000 (invalid)"
    assert entry_point <= 0xFFFFF, (
        f"Entry point {entry_point:06X} exceeds 20-bit address space"
    )

    print(f"Entry point: {entry_point:06X}")
    emu.cpu.regs.set(RegisterName.PC, entry_point)

    # Execute and log 100 instructions
    trace: List[str] = []
    for _ in range(100):
        pc = emu.cpu.regs.get(RegisterName.PC)

        # Read up to 10 bytes for instruction decoding
        instr_bytes = bytes(emu.memory.read_byte(pc + i) for i in range(10))

        # Decode the instruction
        instr = decode(instr_bytes, pc, OPCODES)

        if instr:
            # Format instruction bytes (up to 4 bytes for display)
            hex_bytes = " ".join(
                f"{instr_bytes[i]:02X}" for i in range(min(instr.length(), 4))
            )
            hex_bytes = hex_bytes.ljust(
                11
            )  # Pad to 11 chars (4 bytes * 2 chars + 3 spaces)

            # Get disassembly by rendering tokens to string
            tokens = instr.render()
            disasm = "".join(str(token) for token in tokens)

            trace.append(f"{pc:06X}: {hex_bytes} {disasm}")
        else:
            # Invalid instruction
            trace.append(f"{pc:06X}: {instr_bytes[0]:02X}          <invalid>")

        # Step the CPU
        emu.step()

    # Create snapshot string
    snapshot = "\n".join(trace)

    # Expected trace snapshot for verification
    # This would be filled in once we have a known good ROM
    # expected_snapshot = """FFFFFF: 00 00 00    NOP
    # 000000: 00 00 00    NOP
    # 000001: 00 00 00    NOP
    # 000002: 00 00 00    NOP
    # 000003: 00 00 00    NOP
    # 000004: 00 00 00    NOP
    # 000005: 00 00 00    NOP
    # 000006: 00 00 00    NOP
    # 000007: 00 00 00    NOP
    # 000008: 00 00 00    NOP"""

    # For now, just print the trace and verify basic properties
    print("\nExecution Trace (First 100 Instructions):")
    print("=" * 50)
    print(snapshot)

    # Basic validation
    assert len(trace) == 100
    assert all(":" in line for line in trace)
    assert trace[0].startswith(f"{entry_point:06X}:")

    # Optionally save to file for analysis
    trace_file = Path("data/pce500_trace.txt")
    trace_file.parent.mkdir(exist_ok=True)
    with open(trace_file, "w") as f:
        f.write(snapshot)
    print(f"\nTrace saved to {trace_file}")
