"""PV-specific board boundaries; the real-ROM consumers live in acquisition QA."""

import pytest

from pce500.pvf1.bundle import parse_bundle
from pce500.pvf1.compatibility import entered_legacy
from pce500.pvf1.hardware import Hardware


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
