from __future__ import annotations

import os

import pytest
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
from sc62015.pysc62015.stepper import CPURegistersSnapshot


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
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
    return memory


def test_cpu_facade_executes_nop(cpu_backend: str) -> None:
    """Ensure all enabled backends can execute a trivial instruction."""

    memory = _make_memory(0x00)  # NOP

    cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)

    cpu.regs.set(RegisterName.PC, 0x0000)
    info = cpu.execute_instruction(0x0000)

    assert info.instruction.name() == "NOP"
    assert cpu.backend_stats()["execution_scope"] == "cpu-only"
    assert cpu.backend_stats()["scheduler_owner"] == "python-caller"


def test_cpu_stepper_round_trip(cpu_backend: str) -> None:
    """Verify snapshot-based stepping matches per-backend execution."""

    memory = _make_memory(0x08, 0x5A)  # MV A,n

    cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)

    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x00)

    snapshot = CPURegistersSnapshot.from_registers(cpu.regs)
    result = cpu.step_snapshot(snapshot, {0x0000: 0x08, 0x0001: 0x5A})

    assert result.registers.ba & 0xFF == 0x5A


@pytest.mark.skipif(not os.environ.get("CI"), reason="CI backend-availability guard")
def test_ci_parity_requires_llama_backend() -> None:
    assert "llama" in available_backends(), (
        "CI parity tests require the built LLAMA backend; absence must not become a skip"
    )
