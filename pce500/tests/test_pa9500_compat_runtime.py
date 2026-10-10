"""Explicit legacy mapping hypotheses retain strict failures outside their scope."""

import pytest

from pce500.pa9500.bootstrap import Hardware
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
