from .emulator import (
    Registers,
    RegisterName,
    Emulator,
)
# from typing import Dict, List, Union, Optional

def test_registers() -> None:
    regs = Registers()

    # Test initial state
    assert regs.get(RegisterName.A) == 0
    assert regs.get(RegisterName.B) == 0
    assert regs.get(RegisterName.IL) == 0
    assert regs.get(RegisterName.IH) == 0
    assert regs.get(RegisterName.I) == 0
    assert regs.get(RegisterName.BA) == 0
    assert regs.get(RegisterName.X) == 0
    assert regs.get(RegisterName.Y) == 0
    assert regs.get(RegisterName.U) == 0
    assert regs.get(RegisterName.S) == 0
    assert regs.get(RegisterName.PC) == 0
    assert regs.get(RegisterName.FC) == 0
    assert regs.get(RegisterName.FZ) == 0
    assert regs.get(RegisterName.F) == 0

    regs.set(RegisterName.A, 0x42)
    assert regs.get(RegisterName.A) == 0x42
    assert regs.get(RegisterName.BA) == 0x42
    regs.set(RegisterName.B, 0x84)
    assert regs.get(RegisterName.B) == 0x84
    assert regs.get(RegisterName.BA) == 0x8442

    regs.set(RegisterName.IL, 0x12)
    assert regs.get(RegisterName.IL) == 0x12
    assert regs.get(RegisterName.I) == 0x12
    regs.set(RegisterName.IH, 0x34)
    assert regs.get(RegisterName.IH) == 0x34
    assert regs.get(RegisterName.I) == 0x3412

    regs.set(RegisterName.FC, 1)
    assert regs.get(RegisterName.FC) == 1
    assert regs.get(RegisterName.F) == 1
    regs.set(RegisterName.FZ, 1)
    assert regs.get(RegisterName.FZ) == 1
    assert regs.get(RegisterName.F) == 3  # FC + FZ bits set


def test_decode_instruction() -> None:
    memory = bytearray([0x00] * 255)
    def read_mem(addr: int) -> int:
        if addr < len(memory):
            return memory[addr]
        raise IndexError(f"Address out of bounds: {addr:04X}")
    def write_mem(addr: int, value: int) -> None:
        assert 0 <= value < 256, "Value must be a byte (0-255)"
        if addr < len(memory):
            memory[addr] = value & 0xFF
        else:
            raise IndexError("Address out of bounds")
    cpu = Emulator()
    cpu.read_mem = read_mem
    cpu.write_mem = write_mem

    instr = cpu.execute_instruction(0x00)
    assert instr is not None, "Instruction should not be None"
