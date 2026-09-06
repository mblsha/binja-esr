"""Decoder/LLIL PRE1 consistency for one-selector MVL; not hardware evidence."""

import pytest
from binja_test_mocks.eval_llil import Memory

from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from . import CPU, RegisterName


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "pre,source",
    [
        (None, 0x60),
        (0x22, 0x60),
        (0x30, 0x20),
        (0x31, 0x20),
        (0x32, 0x20),
        (0x33, 0x20),
        (0x34, 0x90),
        (0x36, 0x90),
        (0x24, 0xB0),
        (0x26, 0xB0),
    ],
)
@pytest.mark.parametrize(
    "mode,first,step",
    [(0x34, 0x801FF, -1), (0x24, 0x80200, 1), (None, 0x80200, 1)],
)
@pytest.mark.parametrize("count", [1, 2])
def test_mvl_external_destination_uses_pre1_for_its_only_imem_selector(
    backend, pre, source, mode, first, step, count
):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    code = (b"" if pre is None else bytes([pre])) + (
        bytes.fromhex("db00020820") if mode is None else bytes([0xEB, mode, 0x20])
    )
    raw[0x1000 : 0x1000 + len(code)] = code
    raw[INTERNAL_MEMORY_START + 0xEC] = 0x40
    raw[INTERNAL_MEMORY_START + 0xED] = 0x70
    raw[INTERNAL_MEMORY_START + 0x20 : INTERNAL_MEMORY_START + 0x22] = b"\xa5\x5a"
    raw[INTERNAL_MEMORY_START + 0x60 : INTERNAL_MEMORY_START + 0x62] = b"\x17\x29"
    raw[INTERNAL_MEMORY_START + 0x90 : INTERNAL_MEMORY_START + 0x92] = b"\x36\x48"
    raw[INTERNAL_MEMORY_START + 0xB0 : INTERNAL_MEMORY_START + 0xB2] = b"\x69\x7b"
    expected = bytearray(raw)
    for index in range(count):
        expected[first + step * index] = raw[INTERNAL_MEMORY_START + source + index]
    memory = Memory(raw.__getitem__, raw.__setitem__)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    for name, value in (
        (RegisterName.X, 0x80200),
        (RegisterName.I, count),
        (RegisterName.F, 3),
    ):
        cpu.regs.set(name, value)
    cpu.execute_instruction(0x1000)
    assert raw == expected  # Only the intended external destination bytes may change.
    assert cpu.regs.get(RegisterName.X) == (
        0x80200 if mode is None else 0x80200 + step * count
    )
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.F) == 3
