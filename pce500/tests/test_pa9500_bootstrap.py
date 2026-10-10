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


def test_explicit_absence_and_single_zero_byte_keep_unknown_space():
    from pce500.pa9500.bootstrap import Hardware

    hw = Hardware({})
    assert hw.read(0x40000) is None and hw.read(0) is None
    hw.external_media_enabled = True
    for address in (0x20000, 0x30000, 0x40000, 0x7FFFF):
        assert hw.read(address) == 0xFF
        hw.write(address, 0x55)
        assert hw.read(address) == 0xFF
    hw.native_zero_data = 0xE0
    assert hw.read(0) == 0xE0 and hw.read(1) is None


def test_raw_tablet_requires_contact_and_does_not_invent_other_adc_channels():
    hw = Hardware({}, controller_hypothesis=True)
    hw.tablet = (37, 129, 224, True)
    for control, value in [(6, 9), (7, 32), (22, 224), (14, 0), (30, 0)]:
        hw.write(0x8800, control)
        assert hw.read(0x8C00) == value
    hw.tablet = (37, 129, 224, False)
    hw.write(0x8800, 6)
    assert hw.read(0x8C00) == 0


def test_tablet_conversion_holds_low_bits_and_peeks_do_not_advance():
    hw = Hardware({}, controller_hypothesis=True)
    hw.tablet = (403, 601, 255, True)
    hw.write(0x8800, 6)
    assert [hw.read(0x8C00) for _ in range(4)] == [100] * 4
    assert hw.adc_data_reads == 0
    assert hw.architectural_read(0x8C00) == 100
    hw.tablet = (404, 602, 255, True)
    assert [hw.read(0x8C00) for _ in range(4)] == [3] * 4
    assert hw.architectural_read(0x8C00) == 3
    hw.write(0x8800, 7)
    assert hw.architectural_read(0x8C00) == 150
    assert hw.architectural_read(0x8C00) == 2
    assert hw.adc_data_reads == 4


def test_scratch_candidate_is_separate_bounded_and_explicitly_command_gated():
    hw = Hardware({}, controller_hypothesis=True)
    hw.scratch_candidate = bytearray(4096)
    assert hw.read(0xE000) is None
    hw.compatibility_command = 1
    hw.write(0xE000, 0xA5)
    hw.write(0xEFFF, 0x5A)
    assert hw.read(0xE000) == 0xA5 and hw.read(0xEFFF) == 0x5A
    assert hw.read(0xDFFF) is None and hw.read(0xF000) is None
    assert not any(hw.ram)
    hw.compatibility_command = 0
    assert hw.read(0xE000) is None
