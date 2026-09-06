"""Independent stored-byte Z reference; selected cases are silicon-backed."""

import pytest
from binja_test_mocks.eval_llil import Memory

from . import CPU, RegisterName
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("opcode,initial", [(0xEC, 0x10), (0xFC, 0x01)])
@pytest.mark.parametrize("carry", [0, 1])
def test_discarded_nibble_does_not_clear_zero(backend, opcode, initial, carry):
    # PC-E500 Au1/FT600 readback, 2026-09-05: result=00, I=0, F=02/03.
    _check(backend, opcode, [initial], carry)


def _check(backend, opcode, source, carry):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000:0x1002] = bytes((opcode, 0x40))
    direction = -1 if opcode == 0xEC else 1
    addresses = [0x40 + direction * i for i in range(len(source))]
    for address, value in zip(addresses, source):
        raw[INTERNAL_MEMORY_START + address] = value
    memory = Memory(raw.__getitem__, raw.__setitem__)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, len(source))
    cpu.regs.set(RegisterName.F, carry)
    # Ordinary-RAM arithmetic reference, independent of either lifter.
    expected = []
    nibble = 0
    for value in source:
        if opcode == 0xEC:
            expected.append(((value << 4) & 0xFF) | nibble)
            nibble = value >> 4
        else:
            expected.append((value >> 4) | (nibble << 4))
            nibble = value & 0xF
    cpu.execute_instruction(0x1000)
    assert [raw[INTERNAL_MEMORY_START + a] for a in addresses] == expected
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.F) == carry | (2 if not any(expected) else 0)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("opcode", [0xEC, 0xFC])
def test_short_decimal_shifts_match_stored_byte_reference(backend, opcode):
    for carry in (0, 1):
        for value in range(256):
            _check(backend, opcode, [value], carry)
        for source in (
            [0, 0],
            [0x10, 0],
            [0, 0x10],
            [1, 0],
            [0, 1],
            [0x12, 0x34, 0x56],
        ):
            _check(backend, opcode, source, carry)
