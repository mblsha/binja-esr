"""SC62015 regressions based on September 5 PC-E500 bus/readback captures.

Measured BCD cases list their actually captured incoming C values explicitly.
The complementary C values are tested separately as model extensions, not
as additional hardware observations. Initial-C independence was measured on
selected inputs; that does not qualify every input in both states. Neither
group claims exhaustive invalid-BCD hardware coverage.
"""

import pytest
from binja_test_mocks.eval_llil import Memory

from . import CPU, RegisterName
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


def _execute(backend, raw, initial, expected, *, count, flags, expected_flags, a=0):
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[0x1000 : 0x1000 + len(raw)] = raw
    for address, value in initial.items():
        backing[INTERNAL_MEMORY_START + address] = value
    expected_imem = bytearray(backing[INTERNAL_MEMORY_START:])
    for address, value in expected.items():
        expected_imem[address] = value
    memory = Memory(backing.__getitem__, backing.__setitem__)
    setattr(
        memory, "peek_byte_for_preflight", lambda address, _pc=None: backing[address]
    )
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, count)
    cpu.regs.set(RegisterName.F, flags)
    cpu.regs.set(RegisterName.A, a)
    cpu.execute_instruction(0x1000)
    assert backing[INTERNAL_MEMORY_START:] == expected_imem
    assert cpu.regs.get(RegisterName.F) == expected_flags
    return cpu


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "raw,initial,expected",
    [
        ("32c02050", {0x20: 0x20, 0x50: 0x60}, {0x20: 0x60, 0x50: 0x20}),
        (
            "32c12050",
            {0x20: 0x20, 0x21: 0x40, 0x50: 0x60, 0x51: 0x80},
            {0x20: 0x60, 0x21: 0x80, 0x50: 0x20, 0x51: 0x40},
        ),
        ("30c0ec30", {0xEC: 0x20, 0x50: 0x60}, {0xEC: 0x60, 0x50: 0x20}),
        (
            "30c1ec30",
            {0xEC: 0x20, 0xED: 0x40, 0x50: 0x60, 0x51: 0x80},
            {0xEC: 0x60, 0xED: 0x80, 0x50: 0x20, 0x51: 0x40},
        ),
        ("34c00030", {0xEC: 0x20, 0xED: 0xED, 0x50: 0x60}, {0xED: 0x60, 0x50: 0xED}),
        (
            "34c10030",
            {0xEC: 0x20, 0xED: 0xED, 0xEE: 0x40, 0x50: 0x60, 0x51: 0x80},
            {0xED: 0x60, 0xEE: 0x80, 0x50: 0xED, 0x51: 0x40},
        ),
        ("33c0ee30", {0xEE: 0x20, 0x50: 0x60}, {0xEE: 0x60, 0x50: 0x20}),
        (
            "33c1ed30",
            {0xED: 0x40, 0xEE: 0x20, 0x50: 0x80, 0x51: 0x60},
            {0xED: 0x80, 0xEE: 0x60, 0x50: 0x40, 0x51: 0x20},
        ),
    ],
)
def test_exchange_latches_both_base_addresses(backend, raw, initial, expected):
    # Guard the historical wrong destinations; full-IMEM comparison also catches
    # writes outside either model's predicted range.
    initial = {**{a: 0xA5 for a in (0x60, 0x61, 0x80, 0x81, 0x90, 0x91)}, **initial}
    cpu = _execute(
        backend,
        bytes.fromhex(raw),
        initial,
        expected,
        count=1,
        flags=0,
        expected_flags=0,
    )
    assert cpu.regs.get(RegisterName.I) == 1


# (opcode, left bytes, right bytes, result bytes, F, measured incoming C).
# Byte tuples follow instruction order, which is descending for decimal ops.
# The paired private capture suite checks this declaration against exact raw
# target/input/readback witnesses; do not expand it by carry-independence analogy.
BCD_VECTORS = [
    (0xC4, (0x09,), (1,), (0x10,), 0, (0,)),
    (0xC4, (0x0A,), (0,), (0x10,), 0, (0,)),
    (0xC4, (0x0F,), (0x0F,), (0x14,), 0, (0,)),
    (0xC4, (0x99,), (1,), (0,), 3, (0,)),
    (0xC4, (0xF0,), (0xF0,), (0x40,), 1, (0,)),
    (0xC4, (0xFF,), (0xFF,), (0x54,), 1, (0,)),
    (0xD4, (0,), (0x0F,), (0x9B,), 1, (0, 1)),
    (0xD4, (0,), (0,), (0,), 2, (0, 1)),
    (0xD4, (1,), (0,), (1,), 0, (0, 1)),
    # Independently reproduced after the byte matrix found a shared bug:
    # an invalid positive digit result also triggers decimal correction.
    (0xD4, (0x0A,), (0,), (0x94,), 1, (0, 1)),
    (0xD4, (0xA0,), (0,), (0x40,), 1, (0,)),
    (0xD4, (0xFF,), (0,), (0x89,), 1, (0,)),
    (0xD4, (0x0F,), (5,), (0x94,), 1, (0,)),
    (0xD4, (0x0A, 0), (0, 0), (0x94, 0x99), 1, (0,)),
    (0xD5, (0x0A,), (0,), (0x94,), 1, (0, 1)),
    (0xC4, (0xFF, 0), (0xFF, 0), (0x54, 1), 0, (0,)),
    (0xC4, (0xF0, 0), (0xF0, 0), (0x40, 1), 0, (0,)),
    (0xC4, (0x99, 0x0F), (1, 0x0F), (0, 0x15), 0, (0, 1)),
    (0xD4, (0, 0), (1, 0x0F), (0x99, 0x9A), 1, (0,)),
    (0xD4, (0, 0), (0xFF, 0), (0xAB, 0x99), 1, (0,)),
    (0xD5, (0,), (0,), (0,), 2, (1,)),
    (0xD5, (0, 0), (1,), (0x99, 0x99), 1, (0, 1)),
    (0xC5, (0x0F,), (0x0F,), (0x14,), 0, (0,)),
]


def _check_bcd_vector(backend, carry, opcode, left, right, expected, flags):
    count = len(left)
    destination = 0x20 + count - 1
    initial = {destination - i: value for i, value in enumerate(left)}
    expected_imem = {destination - i: value for i, value in enumerate(expected)}
    raw = bytes((opcode, destination))
    if opcode in (0xC4, 0xD4):
        source = 0x30 + count - 1
        initial.update({source - i: value for i, value in enumerate(right)})
        raw += bytes((source,))
    cpu = _execute(
        backend,
        raw,
        initial,
        expected_imem,
        count=count,
        flags=carry,
        expected_flags=flags,
        a=right[0],
    )
    assert cpu.regs.get(RegisterName.I) == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "opcode,left,right,expected,flags,carry",
    [(*row[:5], carry) for row in BCD_VECTORS for carry in row[5]],
)
def test_bcd_silicon_vectors(backend, carry, opcode, left, right, expected, flags):
    _check_bcd_vector(backend, carry, opcode, left, right, expected, flags)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "opcode,left,right,expected,flags,carry",
    [
        (*row[:5], carry)
        for row in BCD_VECTORS
        for carry in (0, 1)
        if carry not in row[5]
    ],
)
def test_bcd_unmeasured_carry_model_extensions(
    backend, carry, opcode, left, right, expected, flags
):
    # This deliberately retains useful regression coverage without claiming an
    # exact capture for this incoming-C/input combination.
    _check_bcd_vector(backend, carry, opcode, left, right, expected, flags)


# Exact HW-024 measured input states; ascending byte order for binary ops.
# (opcode, destination, source/A, incoming C, stored result, final F).
# Do not expand C=0 two-byte cases into unmeasured C=1 hardware claims.
BINARY_VECTORS = [
    (0x54, (0,), (0,), 0, (0,), 2),
    (0x54, (0,), (0,), 1, (0,), 2),
    (0x55, (0,), (0,), 0, (0,), 2),
    (0x55, (0,), (0,), 1, (0,), 2),
    (0x55, (0, 0), (1,), 0, (1, 0), 0),
    (0x55, (0xFF, 2), (1,), 0, (0, 3), 0),
    (0x5C, (0,), (0,), 0, (0,), 2),
    (0x5C, (0,), (0,), 1, (0,), 2),
    (0x5D, (0,), (0,), 0, (0,), 2),
    (0x5D, (0,), (0,), 1, (0,), 2),
    (0x5D, (2, 2), (1,), 0, (1, 2), 0),
    (0x5D, (0, 2), (1,), 0, (0xFF, 1), 0),
]


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("opcode,left,right,carry,expected,flags", BINARY_VECTORS)
def test_binary_silicon_vectors(backend, opcode, left, right, carry, expected, flags):
    initial = {0x20 + i: value for i, value in enumerate(left)}
    raw = bytes((opcode, 0x20))
    if opcode in (0x54, 0x5C):
        raw += bytes((0x30,))
        initial.update({0x30 + i: value for i, value in enumerate(right)})
    cpu = _execute(
        backend,
        raw,
        initial,
        {0x20 + i: value for i, value in enumerate(expected)},
        count=len(left),
        flags=carry,
        expected_flags=flags,
        a=right[0],
    )
    assert cpu.regs.get(RegisterName.I) == 0
