"""Shared ADC conversion with explicit model pressure candidates."""

import pytest

from pce500.oz9600.tablet import Tablet


@pytest.mark.parametrize("pressure", [513, 1023])
def test_pressure_is_held_across_reads_and_does_not_replace_xy_routes(pressure):
    tablet = Tablet(contact_sample=pressure)
    tablet.set_contact(403, 601, True)
    tablet.drive_control = 0x80
    tablet.write_conversion_control(6)
    assert tablet.peek_data() == pressure >> 2
    assert tablet.data_reads == 0
    assert tablet.read_data() == pressure >> 2
    tablet.set_contact(404, 602, False)
    assert tablet.read_data() == pressure & 3
    tablet.set_contact(403, 601, True)
    tablet.drive_control = 0xA8
    tablet.write_conversion_control(6)
    assert [tablet.read_data(), tablet.read_data()] == [100, 3]
    tablet.drive_control = 0x8A
    tablet.write_conversion_control(7)
    assert [tablet.read_data(), tablet.read_data()] == [150, 1]


def test_default_pressure_and_invalid_configuration():
    assert Tablet().contact_sample == 1023
    for pressure in [-1, 1024]:
        with pytest.raises(ValueError, match="ten-bit"):
            Tablet(contact_sample=pressure)
