"""Narrow ESR-H component contracts; full CPU/board support is not asserted."""

from copy import deepcopy

import pytest

from pce500.pa9500.compat_probe import StartupState, step


class Bus:
    def __init__(self, data):
        self.data = data

    def fetch(self, address):
        try:
            return self.data[address]
        except IndexError as error:
            raise ValueError("unmapped fetch") from error

    def read(self, address):
        raise ValueError("unmapped data")

    def writ(self, dp, p, value):
        raise ValueError("unqualified WRIT")


@pytest.mark.parametrize(
    "data", [b"\x10\x40", b"\x56", b"\xd3", b"\x57", b"\x7b\x10\0\0"]
)
def test_unsupported_access_and_opcode_keep_state(data):
    state = StartupState(0)
    state.internal[0] = 0xA5
    original = deepcopy(state)
    with pytest.raises(ValueError):
        step(state, Bus(data))
    assert state == original


def test_compare_flags_and_mask_independently_expected_branch():
    state = StartupState(0)
    bus = Bus(bytes.fromhex("02 80 67 a5 64 0f 7c 12 34"))
    step(state, bus)
    step(state, bus)
    assert state.carry and not state.zero
    step(state, bus)
    assert state.carry and state.zero
    step(state, bus)
    assert state.pc == 9
    state.pc, state.zero = 6, False
    step(state, bus)
    assert state.pc == 0x1234


def test_far_transfer_preserves_twenty_bits():
    state = StartupState(0)
    assert step(state, Bus(bytes.fromhex("7b 0f 3f c1"))) == 0xF3FC1
    assert state.pc == 4


def test_call_return_stack_order():
    state = StartupState(0, r=0x5C)
    bus = Bus(bytes.fromhex("78 00 03 37"))
    step(state, bus)
    assert (state.pc, state.r, state.internal[0x5A:0x5C]) == (3, 0x5A, b"\x03\x00")
    step(state, bus)
    assert (state.pc, state.r) == (3, 0x5C)


def test_counted_copy_overlap_exchange_and_strict_pointer_boundary():
    state = StartupState(0, p=8, q=2, carry=True, zero=False)
    state.internal[1:4] = bytes([1, 0xCC, 0xC3])
    bus = Bus(bytes([0x0A, 0x0B]))
    step(state, bus)
    assert state.internal[8:10] == bytes([0xCC, 0xC3])
    assert (state.p, state.q, state.carry, state.zero) == (10, 4, True, False)
    for pc, expected in [(0, [1, 1, 1, 1]), (1, [2, 3, 4, 1])]:
        state.pc, state.p, state.q, state.internal[1] = pc, 3, 2, 2
        state.internal[2:6] = bytes([1, 2, 3, 4])
        step(state, bus)
        assert state.internal[2:6] == bytes(expected)
    state.pc, state.p, state.q = 0, 127, 2
    before = deepcopy(state)
    with pytest.raises(ValueError):
        step(state, bus)
    assert state == before


@pytest.mark.parametrize(
    "opcode,reg",
    [
        (0x40, 0),
        (0x42, 2),
        (0x48, 8),
        (0x4A, 10),
        (0xC0, 1),
        (0xC2, 3),
        (0xC8, 9),
        (0xCA, 11),
    ],
)
def test_count_register_flags_and_q_side_effect(opcode, reg):
    for op, old, result, zero, carry in [
        (opcode, 255, 0, True, True),
        (opcode, 0, 1, False, False),
        (opcode + 1, 0, 255, False, True),
        (opcode + 1, 1, 0, True, False),
    ]:
        state = StartupState(0, p=39)
        state.internal[reg] = old
        step(state, Bus(bytes([op])))
        assert (state.internal[reg], state.q, state.p, state.zero, state.carry) == (
            result,
            reg,
            39,
            zero,
            carry,
        )


@pytest.mark.parametrize(
    "a,target,zero", [(0x10, 0x20, True), (0x11, 0x30, True), (0x12, 0x40, False)]
)
def test_counted_table_cases_default_and_return(a, target, zero):
    state = StartupState(0, r=0x5C, carry=True)
    state.internal[2] = a
    bus = Bus(bytes.fromhex("7a 02 12 34 69 10 00 20 11 00 30 00 40"))
    step(state, bus)
    assert state.internal[0x5A:0x5C] == b"\x34\x12"
    step(state, bus)
    assert (state.pc, state.r, state.zero, state.carry) == (target, 0x5A, zero, True)
