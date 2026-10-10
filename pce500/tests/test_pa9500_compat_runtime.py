"""Explicit legacy mapping hypotheses retain strict failures outside their scope."""

import pytest
from copy import deepcopy

from pce500.pa9500.bootstrap import Hardware
from pce500.pa9500.compat_probe import StartupState, step
from pce500.pa9500.compat_runtime import (
    LegacyBus,
    entered_legacy,
    legacy_flags,
    native_flags,
)


def test_split_fetch_and_page_selection_are_distinct_from_data():
    banks = {0: bytearray(0x20000), 6: bytearray(0x20000)}
    banks[0][0x10000], banks[0][0x2000], banks[6][0x1C000] = 0x4D, 0x79, 0x10
    hw = Hardware(banks)
    hw.gate[6] = 0x37
    bus = LegacyBus(hw)
    assert [bus.fetch(a) for a in (0, 0x2000, 0xC000)] == [0x4D, 0x79, 0x10]
    with pytest.raises(ValueError):
        bus.read(0)
    bus.write(0x4E20, 0xA5)
    assert hw.ram[0xE20] == bus.read(0x4E20) == 0xA5
    bus.writ(0x4006, 2, 3)
    banks[0][0xC000] = 0x7A
    assert bus.fetch(0xC000) == 0x7A
    for a in (0x4000, 0x8000):
        with pytest.raises(ValueError):
            bus.fetch(a)


def test_absent_media_is_limited_to_explicit_port04_aperture():
    hw = Hardware({})
    strict, absent = LegacyBus(hw), LegacyBus(hw, absent_legacy_media=True)
    with pytest.raises(ValueError):
        absent.read(0x8000)
    hw.gate[4] = 1
    assert absent.read(0x8000) == absent.read(0xBFFF) == 0xFF
    for bus, a in ((strict, 0x8000), (absent, 0xC000)):
        with pytest.raises(ValueError):
            bus.read(a)
    before = bytes(hw.gate)
    with pytest.raises(ValueError):
        absent.writ(0x4006, 2, 0x38)
    assert bytes(hw.gate) == before


def test_entry_requires_instruction_gate_and_low_target():
    assert entered_legacy(3, 0xC000, 3)
    assert not entered_legacy(3, 0xC000, 1)
    assert not entered_legacy(2, 0xC000, 3)
    assert not entered_legacy(3, 0x10000, 3)
    for f in range(4):
        assert native_flags(*legacy_flags(f)) == f


def test_rom_fill_loop_exact_span_and_atomic_rejection():
    bank = bytearray(0x20000)
    bank[0x10000:0x10010] = bytes.fromhex(
        "10 40 00 00 ff 02 00 03 10 1f c3 29 03 00 0f 1f"
    )
    hw = Hardware({0: bank})
    hw.ram[:] = b"\xa5" * len(hw.ram)
    bus, state = LegacyBus(hw), StartupState(0)
    for _ in range(54):
        step(state, bus)
    assert (state.pc, state.dp, state.q, state.p, state.internal[3]) == (
        16,
        0x4FFF,
        3,
        0,
        0,
    )
    assert hw.ram[:0x1000] == b"\0" * 0x1000
    assert hw.ram[0x1000:] == b"\xa5" * (len(hw.ram) - 0x1000)
    state.pc, state.dp, state.internal[0] = 9, 0x7FFF, 1
    before, ram, events = deepcopy(state), bytes(hw.ram), deepcopy(bus.events)
    with pytest.raises(ValueError):
        step(state, bus)
    assert state == before and bytes(hw.ram) == ram and bus.events == events
    state.internal[0], state.carry, state.zero = 0, True, False
    step(state, bus)
    assert state.dp == 0x7FFF and state.carry and not state.zero
    state.pc, state.dp, state.internal[0] = 9, 0xFFFF, 1
    before = deepcopy(state)
    with pytest.raises(ValueError):
        step(state, bus)
    assert state == before
