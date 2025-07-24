"""Debug PC advancement issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500 import PCE500Emulator
from sc62015.pysc62015.instr import decode, OPCODES
from sc62015.pysc62015.emulator import RegisterName


def test_debug_pc():
    """Debug why PC isn't advancing."""
    rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500.bin"
    
    # Create emulator and load ROM
    emu = PCE500Emulator()
    with open(rom_path, "rb") as f:
        rom_data = f.read()
    
    # Load ROM
    rom_portion = rom_data[0xC0000:0x100000]
    emu.load_rom(rom_portion, start_address=0xC0000)
    
    # Set PC to a specific address
    test_pc = 0xF10C2
    emu.cpu.regs.set(RegisterName.PC, test_pc)
    
    print(f"Initial PC: {emu.cpu.regs.get(RegisterName.PC):06X}")
    
    # Read and decode instruction
    instr_bytes = bytes(emu.memory.read_byte(test_pc + i) for i in range(10))
    instr = decode(instr_bytes, test_pc, OPCODES)
    
    print(f"Instruction at {test_pc:06X}: {instr}")
    print(f"Instruction length: {instr.length()}")
    
    # Execute one step
    print("\nBefore execute_instruction:")
    print(f"  PC: {emu.cpu.regs.get(RegisterName.PC):06X}")
    
    # Call execute_instruction directly
    _ = emu.cpu.execute_instruction(test_pc)
    
    print("\nAfter execute_instruction:")
    print(f"  PC: {emu.cpu.regs.get(RegisterName.PC):06X}")
    print(f"  Expected PC: {test_pc + instr.length():06X}")
    
    # Try stepping through emulator wrapper
    emu.cpu.regs.set(RegisterName.PC, test_pc)
    print("\n\nUsing emu.step():")
    print(f"Before: PC = {emu.cpu.regs.get(RegisterName.PC):06X}")
    emu.step()
    print(f"After:  PC = {emu.cpu.regs.get(RegisterName.PC):06X}")


if __name__ == "__main__":
    test_debug_pc()