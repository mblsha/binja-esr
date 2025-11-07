from __future__ import annotations

import pytest
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE


def _make_memory(*bytes_seq: int) -> Memory:
    if not bytes_seq:
        raise ValueError("memory initialiser requires at least one byte")
    raw = bytearray(ADDRESS_SPACE_SIZE)
    for idx, value in enumerate(bytes_seq):
        raw[idx] = value & 0xFF

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF

    memory = Memory(read, write)
    setattr(memory, "_raw", raw)
    return memory


def test_cpu_facade_executes_nop(cpu_backend: str) -> None:
    """Ensure all enabled backends can execute a trivial instruction."""

    memory = _make_memory(0x00)  # NOP

    try:
        cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    cpu.regs.set(RegisterName.PC, 0x0000)
    info = cpu.execute_instruction(0x0000)

    assert info.instruction.name() == "NOP"
    assert cpu.regs.get(RegisterName.PC) == 0x0001


def test_cpu_executes_simple_program(cpu_backend: str) -> None:
    """Run a tiny program across backends to exercise register/flag paths."""

    # Program: LD A,#1; ADD A,#2; HALT
    program = (0x08, 0x01, 0x40, 0x02, 0xDE, 0x00)
    memory = _make_memory(*program)

    try:
        cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x00)
    steps = 0
    while not getattr(cpu.state, "halted", False):
        steps += 1
        assert steps < 8, "HALT not reached within expected instruction budget"
        pc = cpu.regs.get(RegisterName.PC)
        cpu.execute_instruction(pc)

    assert cpu.regs.get(RegisterName.A) == 0x03
    assert cpu.regs.get(RegisterName.FC) in (0, 1)
    # PC should advance past the HALT instruction
    assert cpu.regs.get(RegisterName.PC) == 5
