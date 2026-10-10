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
