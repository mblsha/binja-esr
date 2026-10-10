"""PA reset-bus component controls; these do not claim foreground boot."""

from pce500.pa9500.bootstrap import Hardware


def test_selector_aliases_keep_latch_identity_and_unacquired_banks_are_missing():
    bank = bytes([0x88]) * 0x20000
    hw = Hardware({0xF8: bank, 0xFA: bank})
    hw.write(0x4021, 0xFA)
    assert hw.read(0x4021) == 0xFA
    assert hw.read(0xC0000) == hw.read(0xDFFFF) == 0x88
    hw.write(0x4021, 0x08)
    assert hw.read(0xC0000) is None


def test_fresh_high_ram_has_no_implicit_oz_workspace_or_peripherals():
    hw = Hardware({})
    assert not any(hw.ram)
    hw.write(0xBFDA8, 0xA5)
    assert hw.read(0xBFDA8) == 0xA5
    assert hw.read(0x1FDA8) is None
    assert hw.read(0x8401) is None
    assert hw.read(0) is None


def test_pa_geometry_clips_set_pattern_at_independently_expected_last_pixels():
    hw = Hardware({}, controller_hypothesis=True)
    for offset, value in [
        (31, 2),
        (3, 255),
        (8, 0),
        (16, 190),
        (17, 0),
        (18, 144),
        (19, 0),
        (2, 0xAA),
    ]:
        hw.write(0x8420 + offset, value)
    assert hw.lcd is not None
    for y in range(145):
        for x in range(195):
            assert hw.lcd.pixel(x, y) == (y == 144 and x in (190, 192, 194))
    assert not hw.lcd.pixel(195, 144)
    assert not hw.lcd.pixel(194, 145)
