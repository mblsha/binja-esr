from __future__ import annotations

import pytest
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName


def _make_memory(opcode: int) -> Memory:
    raw = bytearray([opcode, 0x00, 0x00, 0x00])

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

    if cpu.backend == "rust":
        pytest.skip("Rust backend is currently a stub implementation")

    cpu.regs.set(RegisterName.PC, 0x0000)
    info = cpu.execute_instruction(0x0000)

    assert info.instruction.name() == "NOP"
    assert cpu.regs.get(RegisterName.PC) == 0x0001
