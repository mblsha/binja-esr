"""Low-byte arithmetic must preserve the word used by the native keyword matcher."""

import pytest
from binja_test_mocks.eval_llil import Memory

from . import CPU, RegisterName
from .constants import ADDRESS_SPACE_SIZE


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("carry", [0, 1])
@pytest.mark.parametrize(
    "opcode,initial,expected",
    [
        (0x6C, 0x8264, 0x8265),
        (0x7C, 0x8265, 0x8264),
        (0x6C, 0x82FF, 0x8200),
        (0x7C, 0x8200, 0x82FF),
    ],
)
def test_inc_dec_il_preserves_ih_carry_and_uses_byte_zero(
    backend, carry, opcode, initial, expected
):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000:0x1002] = bytes([opcode, 1])
    memory = Memory(raw.__getitem__, raw.__setitem__)
    memory.peek_byte_for_preflight = lambda address, _pc=None: raw[address]
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, initial)
    cpu.regs.set(RegisterName.FC, carry)
    cpu.execute_instruction(0x1000)
    assert cpu.regs.get(RegisterName.I) == expected
    assert cpu.regs.get(RegisterName.FC) == carry
    assert cpu.regs.get(RegisterName.FZ) == int(expected & 0xFF == 0)


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_keyword_probe_keeps_the_word_for_the_following_comparison(backend):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    # INC IL; DEC IL; SUB BA,I: the complete 8264 word must still compare equal.
    raw[0x1000:0x1006] = bytes.fromhex("6c 01 7c 01 4c 23")
    memory = Memory(raw.__getitem__, raw.__setitem__)
    memory.peek_byte_for_preflight = lambda address, _pc=None: raw[address]
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 0x8264)
    cpu.regs.set(RegisterName.BA, 0x8264)
    for pc in (0x1000, 0x1002, 0x1004):
        cpu.execute_instruction(pc)
    assert cpu.regs.get(RegisterName.I) == 0x8264
    assert cpu.regs.get(RegisterName.BA) == 0
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0
