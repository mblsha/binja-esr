from .emulator import (
    Registers,
    RegisterName,
    Emulator,
    Memory,
)
from .instr import MAX_ADDR
from .mock_llil import MockLowLevelILFunction
from .test_instr import opcode_generator
from typing import Dict, Tuple, List
from .tokens import asm_str


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


def _make_cpu_and_mem(
    size: int, init_data: Dict[int, int], instr_bytes: bytes
) -> Tuple[Emulator, bytearray, List[int], List[Tuple[int, int]]]:
    """
    Create a bytearray-backed mock memory, preload it with `init_data` and
    `instr_bytes`, then return (cpu, raw_memory).
    """
    raw = bytearray(size)
    for addr, val in init_data.items():
        raw[addr] = val & 0xFF
    raw[: len(instr_bytes)] = instr_bytes
    reads: List[int] = []
    writes: List[Tuple[int, int]] = []

    def read_mem(addr: int) -> int:
        reads.append(addr)
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        print(f"Writing {value:02x} to address {addr:04x}")
        raw[addr] = value & 0xFF

    cpu = Emulator(Memory(read_mem, write_mem))
    return cpu, raw, reads, writes


def debug_instruction(cpu: Emulator, address: int) -> None:
    il = MockLowLevelILFunction()
    instr = cpu.decode_instruction(address)
    assert instr is not None, f"Failed to decode instruction at {address:04x}"
    instr.lift(il, address)

    rendered = asm_str(instr.render())

    print(f"Decoded instruction at {address:04x}: {rendered}")
    for llil in il.ils:
        print(f"  {llil}")


def test_load_from_external_memory() -> None:
    # 1-byte register load from external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(
        0x40, {0x10: 0xAB}, bytes.fromhex("88100000")
    )
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    A, [00010]"

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.A) == 0xAB
    assert writes == []

    # 2-byte register load from external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(
        0x40, {0x20: 0x12, 0x21: 0x34}, bytes.fromhex("8A200000")
    )
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    BA, [00020]"

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.BA) == 0x3412  # Little-endian order
    assert writes == []

    # 3-byte register load from external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(
        0x40, {0x30: 0x01, 0x31: 0x02, 0x32: 0x03}, bytes.fromhex("8C300000")
    )
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    X, [00030]"

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.X) == 0x030201  # Little-endian order
    assert writes == []


def test_store_to_external_memory() -> None:
    # 1-byte register store to external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("A8200000"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    [00020], A"

    cpu.regs.set(RegisterName.A, 0xCD)
    cpu.execute_instruction(0x00)
    assert writes == [(0x20, 0xCD)]

    # 2-byte register store to external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("AA200000"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    [00020], BA"

    cpu.regs.set(RegisterName.BA, 0x1234)
    cpu.execute_instruction(0x00)
    assert writes == [(0x20, 0x34), (0x21, 0x12)]  # Little-endian order

    # 3-byte register store to external memory
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("AC200000"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "MV    [00020], X"

    cpu.regs.set(RegisterName.X, 0x010203)
    cpu.execute_instruction(0x00)
    assert writes == [(0x20, 0x03), (0x21, 0x02), (0x22, 0x01)]  # Little-endian order


def test_add() -> None:
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("4001"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "ADD   A, 01"

    cpu.regs.set(RegisterName.A, 0x10)
    cpu.execute_instruction(0x00)
    assert writes == []
    assert cpu.regs.get(RegisterName.A) == 0x11
    assert cpu.regs.get(RegisterName.FZ) == 0  # Zero flag not set
    assert cpu.regs.get(RegisterName.FC) == 0  # Carry flag not set

    cpu.regs.set(RegisterName.A, 0xFF)
    cpu.execute_instruction(0x00)
    debug_instruction(cpu, 0x00)
    assert cpu.regs.get(RegisterName.A) == 0x00  # Wraps around
    assert cpu.regs.get(RegisterName.FZ) == 1  # Zero flag set
    assert cpu.regs.get(RegisterName.FC) == 1  # Carry flag set


def test_sub() -> None:
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("4801"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "SUB   A, 01"

    cpu.regs.set(RegisterName.A, 0x10)
    cpu.execute_instruction(0x00)
    assert writes == []
    assert cpu.regs.get(RegisterName.A) == 0x0F
    assert cpu.regs.get(RegisterName.FZ) == 0  # Zero flag not set
    assert cpu.regs.get(RegisterName.FC) == 0  # Carry flag not set

    cpu.regs.set(RegisterName.A, 0x00)
    cpu.execute_instruction(0x00)
    debug_instruction(cpu, 0x00)
    assert cpu.regs.get(RegisterName.A) == 0xFF  # Wraps around
    assert cpu.regs.get(RegisterName.FZ) == 0  # Zero flag not set
    assert cpu.regs.get(RegisterName.FC) == 1  # Carry flag set


def test_pushs_pops() -> None:
    cpu, raw, reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("4F"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHS F"

    cpu.regs.set(RegisterName.F, 0x0)
    cpu.regs.set(RegisterName.S, 0x20)
    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1F
    assert writes == [(0x1F, 0x0)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1E
    assert writes == [(0x1E, 0x2)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 0)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1D
    assert writes == [(0x1D, 0x1)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1C
    assert writes == [(0x1C, 0x3)]
    writes.clear()

    cpu.regs.set(RegisterName.F, 0)
    raw[0] = 0x5F  # Change to POPS instruction
    assert asm_str(cpu.decode_instruction(0x00).render()) == "POPS  F"
    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1D
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 1

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1E
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 1

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1F
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0

    cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x20
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 0


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
        cpu.regs.set(RegisterName.S, 0x1000)  # Set stack pointer to a valid location
        cpu.regs.set(RegisterName.U, 0x2000)  # Set stack pointer to a valid location
        try:
            cpu.execute_instruction(address)
        except Exception as e:
            # if UNIMPL its not an error, just not implemented
            if b == bytearray([0x20]):
                continue

            debug_instruction(cpu, address)
            raise ValueError(f"Failed to evaluate {s} at line {i+1}") from e
