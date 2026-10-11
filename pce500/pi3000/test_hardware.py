"""PI strict bus and opt-in raw ADC contracts, independent of ROM fixtures."""

import pytest

from pce500.pi3000.hardware import Hardware
from pce500.pi3000.auxiliary import Auxiliary
from pce500.pi3000.input import parse_replay


@pytest.mark.parametrize(
    "route,control,data,other",
    [
        ("external", 0x8800, 0x8C00, 0x8BC0),
        ("integrated", 0x8B80, 0x8BC0, 0x8C00),
    ],
)
def test_adc_route_is_explicit_and_conversion_cannot_tear(route, control, data, other):
    h = Hardware({})
    with pytest.raises(ValueError, match="unqualified"):
        h.set_raw_contact(403, 601, True)
    h.configure_tablet_route(route)
    h.set_raw_contact(403, 601, True)
    assert h.read(0x8B1B) == int(route == "external")
    assert h.read(other) is None
    h.write(control, 6)
    h.write(0x8B04, 0xA8)
    assert [h.read(data) for _ in range(8)] == [100] * 8
    assert h.tablet.data_reads == 0
    assert h.architectural_read(data) == 100
    h.set_raw_contact(404, 602, True)
    assert [h.read(data) for _ in range(8)] == [3] * 8
    assert h.architectural_read(data) == 3
    h.write(control, 7)
    h.write(0x8B04, 0x8A)
    assert [h.architectural_read(data), h.architectural_read(data)] == [150, 2]
    assert h.fault is None
    with pytest.raises(ValueError, match="ten-bit"):
        h.set_raw_contact(1024, 0, False)
    assert (h.tablet.x, h.tablet.y, h.tablet.pressed) == (404, 602, True)
    with pytest.raises(ValueError, match="cannot change"):
        h.configure_tablet_route("integrated" if route == "external" else "external")


def test_auxiliary_absence_retains_request_without_claiming_completion():
    h = Hardware({}, absent_auxiliary=True)
    h.write(0x8A20, 0xFE)
    h.write(0x8A21, 1)
    h.write(0x8A02, 0x80)
    assert h.auxiliary_command_count == 1
    assert h.auxiliary_commands == [b"\xfe\x01" + bytes(30)]
    assert h.read(0x8A00) == 2 and h.read(0x8A40) is None
    h.write(0x8A03, 0xBF)
    assert h.read(0x8A00) == 2
    assert h.architectural_read(0x8A40) == 255
    first = h.fault
    h.write(0x8801, 3)
    assert h.fault == first


def test_auxiliary_completion_is_timed_acknowledged_and_status_only():
    a = Auxiliary()
    request = b"\xfe\x01" + bytes(30)
    assert a.status() == 2
    a.mark_written(0)
    a.mark_written(1)
    a.start(request, 100)
    assert a.status() == 0x82 and a.response(0) is None
    a.advance(131)
    assert a.status() == 0x82
    a.advance(132)
    assert a.status() == 0x42 and a.response(0) == 1 and a.response(1) is None
    with pytest.raises(ValueError, match="prior completion"):
        a.start(request, 132)
    with pytest.raises(ValueError, match="acknowledgment"):
        a.acknowledge(255)
    assert a.status() == 0x42
    a.acknowledge(0xBF)
    assert a.status() == 2 and a.response(0) is None
    a.mark_written(0)
    with pytest.raises(ValueError, match="exactly two"):
        a.start(request, 133)
    a.mark_written(1)
    with pytest.raises(ValueError, match="14FB"):
        a.start(b"\xfb\x14" + bytes(30), 133)
    assert a.status() == 2 and a.accepted_count == 1


def test_raw_press_uses_candidate_gate_and_replay_rejects_all_invalid_data_first():
    h = Hardware({}, auxiliary_control=True)
    h.configure_tablet_route("integrated")
    h.set_raw_contact(403, 601, True)
    assert h.gate[0x12] == 8 and not h.external_irq()
    h.gate[0x10] = 8
    assert h.external_irq()
    h.gate[0x12] = 0
    h.set_raw_contact(404, 602, True)
    assert h.gate[0x12] == 0
    for data in [
        '{"steps":[{"boundaries":1},{"boundaries":1,"tablet":{"raw_x":1024,"raw_y":0,"pressed":true}}]}',
        '{"steps":[{"boundaries":1,"tablet":{"raw_x":2,"raw_y":3,"pressed":1}}]}',
        '{"steps":[{"boundaries":1,"native_ram_event":2}]}',
        '{"steps":[]}',
    ]:
        with pytest.raises(ValueError):
            parse_replay(data)
