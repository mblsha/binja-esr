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


def test_foreground_contact_source_is_selected_before_input_and_conversion():
    h = Hardware({}, auxiliary_control=True)
    h.configure_tablet_route("integrated")
    h.configure_tablet_interrupt_source("gate1")
    h.set_raw_contact(66, 66, True)
    assert h.gate[0x12] == 2
    with pytest.raises(ValueError):
        h.configure_tablet_interrupt_source("gate3")

    h.gate[0x12] = 0
    h.set_raw_contact(66, 67, True)
    assert h.gate[0x12] == 0
    h.set_raw_contact(66, 67, False)
    h.tablet.write_conversion_control(6)
    h.tablet.read_data()
    with pytest.raises(ValueError):
        h.configure_tablet_interrupt_source("gate3")


def test_halt_clock_candidate_requires_lcc7_and_never_off():
    h = Hardware({})
    args = dict(halted=True, off=False, lcc=0x80, next_mti=17)
    assert h.halt_main_deadline(**args) is None
    h.halt_main_timer_candidate = True
    assert h.halt_main_deadline(**args) == 17
    for replacement in [dict(halted=False), dict(off=True), dict(lcc=0)]:
        assert h.halt_main_deadline(**(args | replacement)) is None


def test_mapper_readbacks_follow_saved_output_latches():
    h = Hardware({})
    for i, value in enumerate((0x12, 0x34, 0x56)):
        h.write(0xF00C + i, value)
        assert h.read(0xF01C + i) == value
    assert h.startup_latches == bytes((0x12, 0x34, 0x56))


def test_periodic_rtc_candidate_coalesces_and_requires_device_and_gate_ack():
    h = Hardware({})
    h.rtc_tick_period = 10
    h.advance_rtc_ticks(9)
    assert h.rtc.read(10) == 0
    h.advance_rtc_ticks(21)
    assert (h.rtc_ticks, h.rtc_tick_phase) == (3, 0)
    assert h.rtc.read(10) == 4 and not h.external_irq()
    assert h.gate[0x12] == 0
    h.rtc.write(8, 4)
    assert not h.external_irq() and h.gate[0x12] == 1
    h.gate[0x10] = 1
    assert h.external_irq()
    h.gate[0x12] = 0
    assert h.external_irq()
    h.rtc.write(10, 0)
    assert h.external_irq()
    h.gate[0x12] = 0
    assert not h.external_irq()
    assert not any(h.ram) and h.rtc.registers()[:5] == bytes(5)


def test_calendar_candidate_crosses_year_and_hold_freezes_only_calendar_phase():
    from pce500.oz9600.rtc import Clock

    h = Hardware({})
    h.rtc_second_period, h.rtc_second_phase = 10, 7
    with pytest.raises(ValueError, match="invalid ordinary RTC"):
        h.advance_rtc_calendar(3)
    assert (h.rtc_seconds, h.rtc_second_phase) == (0, 7)
    assert h.rtc.registers()[:5] == bytes(5)
    before = Clock(1994, 12, 31, 23, 59, 59, 3)
    for offset, value in enumerate(before.encode(bytes([192, 0, 0, 0, 128]))):
        h.rtc.write(offset, value)
    h.rtc.write(12, 128)
    h.advance_rtc_calendar(25)
    for _ in range(100):
        assert h.read(0x8000) == 251
    assert h.rtc.clock() == before
    assert (h.rtc_seconds, h.rtc_second_phase, h.rtc_held_units) == (0, 7, 25)
    h.rtc_tick_period = 5
    h.advance_rtc_ticks(25)
    assert h.rtc.read(10) == 4
    h.rtc.write(12, 0)
    h.advance_rtc_calendar(3)
    assert h.rtc.clock() == Clock(1995, 1, 1, 0, 0, 0, 4)
    assert (h.rtc_seconds, h.rtc_second_phase) == (1, 0)
    assert h.rtc.read(0) & 192 == 192 and h.rtc.read(4) & 128 == 128
    assert not any(h.ram)


def test_unmapped_read_candidate_is_bounded_readonly_and_keeps_default_strict():
    h = Hardware({})
    assert h.read(0x5844) is None
    for value in (0, 0xE0, 0xFF):
        h = Hardware({})
        h.unmapped_low_read = value
        for address in (0x5000, 0x5844, 0x7FFF):
            for _ in range(10):
                assert h.read(address) == value
        assert h.read(0x4FFF) is None and h.read(0x8000) == 0
        assert not any(h.ram) and not any(h.scratch) and h.fault is None
        h.write(0x5844, 0x55)
        assert "unqualified PI write 05844" in h.fault
        assert h.read(0x5844) == value and not any(h.ram) and not any(h.scratch)


def test_auxiliary_initialization_control_matches_single_byte_acknowledgment():
    from pce500.pi3000.auxiliary import Auxiliary

    a = Auxiliary()
    a.mark_written(0)
    a.mark_written(1)
    a.start(b"\xfe\x00" + bytes(30), 40)
    assert a.status() == 0x82
    a.advance(72)
    assert a.response(0) == 1 and a.response(1) is None
    a.acknowledge(0xBF)
    assert a.status() == 2 and a.accepted == [0x00FE]


@pytest.mark.parametrize(
    "contact",
    [
        {"column": 11, "row": 0, "pressed": True},
        {"column": 0, "row": 8, "pressed": True},
        {"column": 0, "row": 7, "pressed": 1},
    ],
)
def test_matrix_replay_rejects_bad_late_contacts_before_execution(contact):
    import json

    with pytest.raises(ValueError):
        parse_replay(
            json.dumps(
                {"steps": [{"boundaries": 10}, {"boundaries": 1, "contact": contact}]}
            )
        )
    assert (
        parse_replay(
            '{"steps":[{"boundaries":0,"contact":{"column":0,"row":7,"pressed":true}}]}'
        )[0]["contact"]["row"]
        == 7
    )
