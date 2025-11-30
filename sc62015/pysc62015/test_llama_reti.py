from __future__ import annotations

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _make_memory(imr: int, f: int, ret_bytes: tuple[int, int, int], sp: int) -> Memory:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0x01  # RETI opcode
    raw[sp] = imr & 0xFF
    raw[sp + 1] = f & 0xFF
    raw[sp + 2] = ret_bytes[0] & 0xFF
    raw[sp + 3] = ret_bytes[1] & 0xFF
    raw[sp + 4] = ret_bytes[2] & 0xFF

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


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_reti_restores_imr_exactly(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    sp = 0x0100
    imr_saved = 0x00  # deliberately clear IRM bit to detect forced setting
    f_saved = 0x7C
    ret_bytes = (0x12, 0x34, 0x05)  # little-endian PC
    memory = _make_memory(imr_saved, f_saved, ret_bytes, sp)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.S, sp)

    cpu.execute_instruction(0x0000)

    expected_pc = ret_bytes[0] | (ret_bytes[1] << 8) | (ret_bytes[2] << 16)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    assert memory._raw[imr_addr] == imr_saved
    assert cpu.regs.get(RegisterName.PC) == expected_pc
    assert cpu.regs.get(RegisterName.S) == (sp + 5)
