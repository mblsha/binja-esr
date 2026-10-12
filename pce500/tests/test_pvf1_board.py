"""PV-specific board boundaries; the real-ROM consumers live in acquisition QA."""

import pytest

from pce500.pvf1.bundle import parse_bundle
from pce500.pvf1.compatibility import entered_legacy
from pce500.pvf1.hardware import Hardware


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


@pytest.mark.parametrize("data", [b"", b"PIROM01\0\0\0\0\0", b"PVROM01\0\0\0\0\0", b"PVROM01\0\xff\xff\xff\xff"])
def test_bundle_rejects_wrong_model_and_unbounded_or_empty_lengths(data):
    with pytest.raises(ValueError):
        parse_bundle(data)


def test_mode_handoff_requires_the_encoded_target_and_the_enabled_gate():
    assert entered_legacy(3, 0, 0, 3)
    assert not entered_legacy(3, 0, 1, 3)
    assert not entered_legacy(3, 0, 0, 0)
    assert not entered_legacy(3, 0xD0000, 0xD0000, 3)
