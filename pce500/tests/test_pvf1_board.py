"""PV-specific board boundaries; the real-ROM consumers live in acquisition QA."""

import pytest

from pce500.pvf1.bundle import parse_bundle
from pce500.pvf1.compatibility import entered_legacy
from pce500.pvf1.hardware import Hardware


def test_rtc_a2_candidate_latches_through_the_native_gate_and_not_guest_ram():
    board = Hardware({}, "control-candidate")
    board.configure_rtc_half_second_candidate(4)
    board.advance_rtc_timing(3)
    assert board.rtc.read(10) == 0
    with pytest.raises(ValueError, match="positive"):
        board.configure_rtc_half_second_candidate(0)
    board.advance_rtc_timing(1)
    assert board.rtc.read(10) == 4 and not board.external_irq()
    board.write(0x8008, 4)
    board.write(0x4010, 1)
    assert board.external_irq() and board.gate[0x12] == 1
    board.write(0x4012, 0)
    assert (
        board.external_irq()
    )  # RTC cause is a held level until native acknowledgement.
    board.write(0x800A, 0)
    board.write(0x4012, 0)
    assert not board.external_irq()
    board.advance_rtc_timing(8)
    assert board.rtc.read(10) == 4 and board.rtc_clock.snapshot()["ticks"] == 3
    assert board.rtc.registers()[:5] == bytes(5)
    assert board.ram[0x3FE32:0x3FE35] == bytes(3)
    board.configure_rtc_half_second_candidate(None)
    board.advance_rtc_timing(100)
    assert board.rtc_clock.snapshot()["ticks"] == 3
    with pytest.raises(ValueError, match="explicit board"):
        Hardware({}).configure_rtc_half_second_candidate(4)


def test_absent_cd_input_is_firmware_only_and_not_gpio_output_feedback():
    for profile in ("strict", "absent-auxiliary", "control-candidate"):
        assert Hardware({}, profile).peek(0x8B1B) is None
    board = Hardware({}, "firmware")
    assert board.read_mutated(0x8B1B) == 0x81
    board.write(0x8B03, 0x5A)
    assert board.gpio[3] == 0x5A
    assert board.peek(0x8B1B) == 0x81
    assert board.peek(0x8B18) is None and board.peek(0x8B19) is None
    assert board.peek(0x8B1B) & 2 == 0
    assert board.fault is None


def test_auxiliary_control_readback_matches_the_native_diagnostic():
    board = Hardware({}, "control-candidate")
    for value in (0xAA, 0x55):
        board.write(0x8A06, value)
        assert board.peek(0x8A04) == board.read_mutated(0x8A04) == value


def test_registration_directory_query_is_empty_timed_and_acknowledged():
    board = Hardware({}, "control-candidate")
    for i, byte in enumerate((0xFB, 0x13, 9)):
        board.auxiliary.write(0x8A20 + i, byte, 100)
    board.auxiliary.write(0x8A02, 0x80, 100)
    assert board.peek(0x8A41) is None
    board.auxiliary.advance(132)
    assert [board.peek(a) for a in (0x8A40, 0x8A41, 0x8A42)] == [0, 0, 0]
    assert board.peek(0x8A43) is None
    board.auxiliary.write(0x8A03, 0xBF, 132)
    assert board.peek(0x8A41) is None


def test_registered_controller_backing_extends_restart_and_validates_before_mutation():
    import hashlib
    import json
    from pce500.pvf1.auxiliary import Auxiliary
    from pce500.pvf1.retained import IMAGE_SIZE, retained_state, restore_retained_state

    external = bytes(0x100000)
    source = Hardware({}, "control-candidate")
    old = retained_state(external, source)
    assert old[:8] == b"PVRAM01\0" and len(old) == IMAGE_SIZE
    glyphs = {
        "0": {
            "code": 0x7382,
            "width": 138,
            "height": 222,
            "strokes": [[[39.0, 40.0], [97.0, 40.0]], [[67.0, 40.0], [67.0, 159.0]]],
        }
    }
    source.auxiliary.restore_registered_state(Auxiliary.decode_registered_state(glyphs))
    image = retained_state(external, source)
    fresh = Hardware({}, "control-candidate")
    restore_retained_state(external, fresh, image)
    assert image[:8] == b"PVRAM02\0"
    assert fresh.auxiliary.registered_state() == glyphs
    fresh.auxiliary.write(0x8A01, 0xFD, 0)
    fresh.auxiliary.write(0x8A01, 2, 100)
    assert fresh.auxiliary.registered_state() == glyphs
    bad = json.dumps({"10": glyphs["0"]}).encode()
    payload = image[104:IMAGE_SIZE] + bad
    corrupt = image[:72] + hashlib.sha256(payload).digest() + payload
    fresh.ram[0] = 0x55
    with pytest.raises(ValueError, match="outside directory"):
        restore_retained_state(external, fresh, corrupt)
    assert fresh.ram[0] == 0x55 and fresh.auxiliary.registered_state() == glyphs


def test_clear_all_handwritten_registrations_removes_labels_and_ink_after_restart():
    from pce500.pvf1.auxiliary import Auxiliary

    a = Auxiliary(control=True)
    glyph = {
        "code": 0x7382,
        "width": 138,
        "height": 222,
        "strokes": [[[1.0, 2.0], [3.0, 4.0]]],
    }
    a.restore_registered_state(
        Auxiliary.decode_registered_state({"0": glyph, "9": glyph})
    )
    a.registration_channel, a.registration_response = 9, b"\x82\x73"
    a.stroke_read, a.raw_confirmation = (9, b"\x01\x02", 0), True
    a.input_flags = 7
    for i, value in enumerate((0xFB, 0x31)):
        a.write(0x8A20 + i, value, 100)
    a.write(0x8A02, 0x80, 100)
    assert a.peek(0x8A40) is None
    a.advance(132)
    assert a.peek(0x8A40) == 1 and a.peek(0x8A41) is None
    assert not a.registrations and not a.registered_glyphs
    assert a.registration_channel is None and a.registration_response is None
    assert a.stroke_read is None and not a.raw_confirmation
    assert (
        a.input_flags == 7
        and not a.registration_exchange
        and not a.engine.capture_active
    )
    a.write(0x8A03, 0xBF, 132)
    a.write(0x8A01, 0xFD, 200)
    a.write(0x8A01, 2, 300)
    assert a.registered_state() == {}


def test_nominal_ram_card_guard_write_protection_and_unaliased_bounds():
    board = Hardware({}, "control-candidate")
    with pytest.raises(ValueError, match="exact"):
        board.configure_ram_card_candidate(bytes(65536))
    image = bytearray([0xFF]) * 0x20000
    image[0], image[-1] = 0x12, 0x34
    board.configure_ram_card_candidate(image, False)
    assert board.card_presence_sample(0x40) == 0x42
    board.write(0x20000, 0x56)
    assert board.ram_card_image() == image
    assert [board.peek(a) for a in (0x20000, 0x3FFFF, 0x40000, 0x7FFFF)] == [
        0x12,
        0x34,
        0xFF,
        0xFF,
    ]
    with pytest.raises(ValueError, match="once before"):
        board.configure_ram_card_candidate(bytes(0x20000))
    assert board.ram_card_image() == image


def test_capture_clock_is_opt_in_latched_and_acknowledged_without_query_data():
    board = Hardware({}, "control-candidate")
    aux = board.auxiliary
    for i, byte in enumerate(b"\xfc\x03"):
        board.write(0x8A20 + i, byte)
    board.write(0x8A02, 0x80)
    aux.advance(32)
    board.write(0x8A03, 0xBF)
    board.write(0x8A06, 0x80)
    assert not aux.advance(10000) and board.peek(0x8A05) == 0
    aux.configure_sample_period(4096)
    assert not aux.advance(100) and not aux.advance(4195)
    assert aux.advance(4196)
    for _ in range(3):
        assert board.peek(0x8A05) == 0x80 and board.peek(0x8A41) is None
    assert not aux.advance(100000)
    board.write(0x8A07, 0xFF)
    assert board.peek(0x8A05) == 0x80
    board.write(0x8A07, 0x7F)
    assert board.peek(0x8A05) == 0 and not aux.advance(100000)
    assert aux.advance(104096)
    assert aux.snapshot()["sample_events"] == 2
    # Disabling the configured candidate cancels its latch; normal controls persist.
    aux.configure_sample_period(None)
    assert board.peek(0x8A04) == 0x80 and board.peek(0x8A05) == 0
    assert not aux.advance(200000)
    aux.configure_sample_period(4096)
    board.write(0x8A01, 0xFD)
    board.write(0x8A01, 2)
    assert aux.snapshot()["sample_period_candidate"] == 4096
    assert not aux.snapshot()["capture_active"] and not aux.advance(500000)


def test_firmware_capture_clock_asserts_only_the_native_auxiliary_gate_source():
    board = Hardware({}, "firmware")
    board.write(0x4010, 8)
    for i, byte in enumerate(b"\xfc\x03"):
        board.write(0x8A20 + i, byte)
    board.write(0x8A02, 0x80)
    board.write(0x8A06, 0x80)
    assert not board.external_irq()
    board.cycle = 4096
    assert board.external_irq() and board.gate[0x12] == 8
    assert board.tablet.data_reads == 0
    board.write(0x4012, 0)
    assert (
        not board.external_irq()
    )  # held ready status does not reassert a cleared edge
    assert board.peek(0x8A05) == 0x80
    board.write(0x8A07, 0x7F)
    assert not board.external_irq()
    board.cycle = 8192
    assert board.external_irq() and board.gate[0x12] == 8


def test_cpu_hypotheses_belong_to_explicit_pvf1_candidates():
    assert Hardware({}).cpu_options == {
        "block_transfer_policy": "independent",
        "byte_arithmetic_source_policy": "strict",
    }
    for profile in ("absent-auxiliary", "control-candidate", "firmware"):
        assert Hardware({}, profile).cpu_options == {
            "block_transfer_policy": "coupled_predecrement",
            "byte_arithmetic_source_policy": "low_byte",
        }
    for profile in ("strict", "absent-auxiliary", "control-candidate", "firmware"):
        assert Hardware({}, profile).lcd.ordered_window_clipping_candidate == (
            profile == "firmware"
        )


def test_lcd_contact_uses_live_coefficients_and_rejects_storage_rows():
    board = Hardware({}, "firmware")
    ws = 0x80000
    for i, value in enumerate((1132, 650, -123, 812, 737, -68)):
        for j, byte in enumerate(value.to_bytes(2, "little", signed=True)):
            board.write(ws + 0x8E + 2 * i + j, byte)
    assert board.lcd_to_raw(0, 0, ws) == (71, 63)
    assert board.lcd_to_raw(222, 16, ws) == (581, 121)
    assert board.lcd_to_raw(335, 239, ws) == (841, 931)
    for x, y in ((336, 0), (0, 240), (-1, 0)):
        with pytest.raises(ValueError, match="main window"):
            board.lcd_to_raw(x, y, ws)
    board.write(ws + 0x90, 0x8B)
    assert board.lcd_to_raw(222, 16, ws) == (582, 121)
    assert board.tablet.data_reads == 0 and not board.tablet.pressed


@pytest.mark.parametrize("header,count", [(0x04FE, 3), (0x03FE, 8)])
def test_editor_parameter_controls_require_exact_native_packet_lengths(header, count):
    board = Hardware({}, "firmware")
    packet = header.to_bytes(2, "little") + bytes(range(count - 2))
    for i, byte in enumerate(packet[:-1]):
        board.write(0x8A20 + i, byte)
    with pytest.raises(ValueError, match="written bytes"):
        board.write(0x8A02, 0x80)
    board.write(0x8A20 + count - 1, packet[-1])
    board.write(0x8A02, 0x80)
    assert board.peek(0x8A40) is None
    board.auxiliary.advance(32)
    assert board.peek(0x8A40) == 1 and board.peek(0x8A41) is None
    board.write(0x8A03, 0xBF)
    for i, byte in enumerate(packet):
        board.write(0x8A20 + i, byte)
    board.write(0x8A20 + count, 0)
    with pytest.raises(ValueError, match="written bytes"):
        board.write(0x8A02, 0x80)


@pytest.mark.parametrize(
    "value, expanded", [(0, 0), (255, 65535), (129, 49155), (90, 13260)]
)
def test_font_pixel_expansion_is_little_endian_and_peeks_do_not_advance(
    value, expanded
):
    board = Hardware({})
    board.write(0x401D, 0xA5)
    board.write(0x4020, 0x5A)
    board.write(0x401E, value)
    for _ in range(3):
        assert board.peek(0x401E) == board.read_mutated(0x401E) == expanded & 255
        assert board.peek(0x401F) == board.read_mutated(0x401F) == expanded >> 8
    assert board.peek(0x401D) == 0xA5 and board.peek(0x4020) == 0x5A
    assert board.selector == 7 and board.fault is None


def test_gate_read_aliases_use_live_selector_expansion_and_latches():
    board = Hardware({}, "control-candidate")
    board.write(0x402F, 0xA5)
    board.write(0x401E, 0x81)
    board.write(0x4021, 0xFE)
    for base in (0x4000, 0x4040, 0x4080, 0x4FC0):
        for offset, expected in ((0x2F, 0xA5), (0x21, 0xFE), (0x1E, 3), (0x1F, 0xC0)):
            assert board.peek(base + offset) == expected
            assert board.read_mutated(base + offset) == expected
    board.write(0x402F, 0x5A)
    assert board.peek(0x40AF) == 0x5A
    assert board.peek(0x4FFF) == 0 and board.peek(0x5000) is None
    with pytest.raises(ValueError, match="write 04061"):
        board.write(0x4061, 7)
    assert board.selector == 0xFE and "write 04061" in board.fault


def test_adc_peeks_hold_conversion_and_gpio_feedback_is_model_specific():
    board = Hardware({}, "control-candidate")
    board.set_raw_contact(403, 601, True)
    assert board.gate[0x12] == 2 and not board.external_irq()
    board.write(0x4010, 2)
    assert board.external_irq()
    board.write(0x8B04, 0xA8)
    board.write(0x8800, 6)
    assert [board.peek(0x8C00) for _ in range(4)] == [100] * 4
    assert board.tablet.data_reads == 0
    assert board.read_mutated(0x8C00) == 100
    board.set_raw_contact(404, 602, True)
    assert board.peek(0x8C00) == board.read_mutated(0x8C00) == 3
    for write, read in [(0x8B00, 0x8B1C), (0x8B02, 0x8B1A), (0x8B05, 0x8B1D)]:
        board.write(write, 0x5A)
        assert board.peek(read) == 0x5A


def test_control_readback_preserves_pixel_pipeline_and_unknown_expansion_traps():
    board = Hardware({})
    board.write(0x842E, 0x11)
    assert board.peek(0x842E) == board.read_mutated(0x842E) == 0x11
    assert board.lcd.data_reads == 0
    assert board.peek(0x20000) is None
    with pytest.raises(ValueError, match="unresolved PV-F1 read"):
        board.read_mutated(0x20000)
    assert board.fault is not None


def test_auxiliary_controls_have_exact_lengths_and_never_supply_query_stubs():
    board = Hardware({}, "control-candidate")
    board.write(0x8A20, 0xFE)
    board.write(0x8A21, 5)
    board.write(0x8A02, 0x80)
    assert board.peek(0x8A40) is None
    board.auxiliary.advance(32)
    assert board.peek(0x8A40) == 1 and board.peek(0x8A41) is None
    board.write(0x8A03, 0xBF)
    board.write(0x8A20, 0xFB)
    board.write(0x8A21, 0x14)
    with pytest.raises(ValueError, match="unrecovered PV-F1 auxiliary command"):
        board.write(0x8A02, 0x80)
    assert board.peek(0x8A40) is None


@pytest.mark.parametrize(
    "data",
    [b"", b"PIROM01\0\0\0\0\0", b"PVROM01\0\0\0\0\0", b"PVROM01\0\xff\xff\xff\xff"],
)
def test_bundle_rejects_wrong_model_and_unbounded_or_empty_lengths(data):
    with pytest.raises(ValueError):
        parse_bundle(data)


def test_mode_handoff_requires_the_encoded_target_and_the_enabled_gate():
    assert entered_legacy(3, 0, 0, 3)
    assert not entered_legacy(3, 0, 1, 3)
    assert not entered_legacy(3, 0, 0, 0)
    assert not entered_legacy(3, 0xD0000, 0xD0000, 3)


def test_pvf1_on_edge_latch_is_independent_of_live_ssr_contact():
    from pce500.pvf1.interrupts import on_irq_should_reassert, sample_on_input

    assert on_irq_should_reassert(True)
    assert not on_irq_should_reassert(True, contact_edge_candidate=True)
    assert not on_irq_should_reassert(False, contact_edge_candidate=True)
    assert sample_on_input(4, True, live_candidate=True) == 12
    assert sample_on_input(12, False, live_candidate=True) == 4


def test_pvf1_on_edge_keeps_alarm_level_separate():
    from pce500.pvf1.interrupts import on_irq_should_reassert

    assert on_irq_should_reassert(False, contact_edge_candidate=True, alarm_wake=True)
