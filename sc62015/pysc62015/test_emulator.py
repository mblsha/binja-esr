from .emulator import (
    Registers,
    RegisterName,
    Emulator,
    Memory,
)
from .instr import MAX_ADDR
from .mock_llil import MockLowLevelILFunction
from .test_instr import opcode_generator


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
    raw_memory = bytearray([0x00] * 255)

    def read_mem(addr: int) -> int:
        return raw_memory[addr]

    def write_mem(addr: int, value: int) -> None:
        raw_memory[addr] = value

    memory = Memory(read_mem, write_mem)
    cpu = Emulator(memory)

    instr = cpu.decode_instruction(0x00)
    assert instr is not None, "Instruction should not be None"


def test_decode_all_opcodes() -> None:
    # return
    raw_memory = bytearray([0x00] * MAX_ADDR)

    # enumerate all opcodes, want index for each opcode
    for i, (b, s) in enumerate(opcode_generator()):
        if b is None:
            continue

        for j, byte in enumerate(b):
            raw_memory[j] = byte

        def read_mem(addr: int) -> int:
            if addr < 0 or addr >= len(raw_memory):
                raise IndexError(f"Address out of bounds: {addr:04x}")
            return raw_memory[addr]

        def write_mem(addr: int, value: int) -> None:
            if addr < 0 or addr >= len(raw_memory):
                raise IndexError(f"Address out of bounds: {addr:04x}")
            raw_memory[addr] = value

        memory = Memory(read_mem, write_mem)
        cpu = Emulator(memory)

        address = 0x00
        try:
            cpu.execute_instruction(address)
        except Exception as e:
            instr = cpu.decode_instruction(address)
            il = MockLowLevelILFunction()
            assert instr is not None
            instr.lift(il, address)

            print(f"{s}:")
            for llil in il.ils:
                print(f"  {llil}")
            raise ValueError(f"Failed to evaluate {s} at line {i+1}") from e
