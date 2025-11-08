from __future__ import annotations

import pytest
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.stepper import CPURegistersSnapshot
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


def test_cpu_snapshot_roundtrip(cpu_backend: str) -> None:
    memory = _make_memory(0x00)

    try:
        cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    snapshot = CPURegistersSnapshot(
        pc=0x12345,
        ba=0x201,
        i=0x3456,
        x=0xAAAAAA,
        y=0xBBBBBB,
        u=0xCCCCCC,
        s=0xDDDDDD,
        f=0x12,
        temps={3: 0xABCD, 7: 0xEEFF},
        call_sub_level=3,
    )
    cpu.apply_snapshot(snapshot)
    round_trip = cpu.snapshot_registers()

    assert round_trip.to_dict() == snapshot.to_dict()


def test_cpu_step_snapshot(cpu_backend: str) -> None:
    memory = _make_memory(0x08, 0x01, 0xDE)  # MV A,#1 ; HALT

    try:
        cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    snapshot = CPURegistersSnapshot(pc=0x0000)
    result = cpu.step_snapshot(snapshot, {0x0000: 0x08, 0x0001: 0x01})

    assert result.instruction_name == "MV"
    assert result.instruction_length == 2
    assert result.registers.ba & 0xFF == 0x01
    assert result.registers.pc == 0x0002
    assert result.changed_registers["PC"] == (0x0000, 0x0002)
