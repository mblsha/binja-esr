"""Reference corpus shared with core Rust tests; no private captures needed."""

import json
from pathlib import Path
import pytest
from pce500.oz9600.parity import corpus
from pce500.oz9600.profile import (
    ExecutionProfile,
    on_irq_should_reassert,
    handler_blocks_irq,
)
from pce500.oz9600.input import KEYS, parse_replay
from pce500.oz9600.lcd import LcdController
from pce500.oz9600.hardware import guest_internal_write_is_allowed
from pce500.oz9600.hardware import ON_KEY_SSR_MASK
from pce500.oz9600.rtc import Clock, Rtc
from pce500.oz9600.hardware import Hardware, WORKSPACE_RAM_OFFSET
from pce500.oz9600.retained import (
    _merged_legacy_payload,
    MAIN_RAM_SIZE,
    WORKSPACE_SIZE,
    PAYLOAD_SIZE,
)
from pce500.oz9600.boundary import ExecutionState


def test_on_input_ssr_reference_matches_the_rom_mask():
    assert ON_KEY_SSR_MASK == 8


def test_workspace_top_ram_reference_has_one_backing_for_both_directions():
    hw = Hardware()
    for low, high in [(0x10000, 0xB0000), (0x12345, 0xB2345), (0x1FFFF, 0xBFFFF)]:
        hw.write(low, 0x5A)
        assert hw.read(high) == hw.architectural_read(high) == 0x5A
        hw.write(high, 0xA5)
        assert hw.read(low) == hw.architectural_read(low) == 0xA5
    assert len(hw.ram) == MAIN_RAM_SIZE


def legacy_payload_fixture():
    data = bytearray(WORKSPACE_SIZE + PAYLOAD_SIZE)
    for a, v in [(0xFF38, 0x10300), (0xFD00, 0x80100), (0xFD03, 0xB0300)]:
        data[a : a + 3] = v.to_bytes(3, "little")
    data[WORKSPACE_SIZE + WORKSPACE_RAM_OFFSET + 0x100] = 0xFB
    data[0x300], data[0xFFFF], data[-1] = 0x17, 0x52, 0xA5
    return data


def test_legacy_payload_conversion_preserves_filesystem_workspace_and_rtc():
    converted = _merged_legacy_payload(legacy_payload_fixture())
    assert len(converted) == PAYLOAD_SIZE
    assert converted[WORKSPACE_RAM_OFFSET + 0x100] == 0xFB
    assert converted[WORKSPACE_RAM_OFFSET + 0x300] == 0x17
    assert converted[MAIN_RAM_SIZE - 1] == 0x52 and converted[-1] == 0xA5
    assert _merged_legacy_payload(bytes(WORKSPACE_SIZE + PAYLOAD_SIZE)) == bytes(
        PAYLOAD_SIZE
    )


@pytest.mark.parametrize("a", [0xFF38, 0xFD00, 0xFD03, 0x100])
def test_legacy_payload_conversion_rejects_inconsistent_or_conflicting_images(a):
    data = legacy_payload_fixture()
    data[a] ^= 1
    if a == 0xFD00:
        data[a : a + 3] = bytes(3)
    with pytest.raises(ValueError):
        _merged_legacy_payload(data)


def test_guest_input_write_reference_only_claims_eil_and_eih():
    for address in (0xF5, 0xF6, 0x1000F4, 0x1000F7, 0x1000FF):
        assert guest_internal_write_is_allowed(address)
    for address in (0x1000F5, 0x1000F6):
        assert not guest_internal_write_is_allowed(address)


def test_controller_reference_fixture_matches_current_python():
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "sc62015/core/data/oz9600_controller_reference.json").read_text()
    )
    assert corpus() == fixture


def test_profiles_do_not_claim_automatic_experimental_settings():
    assert ExecutionProfile.configure("strict").initial_bp == 0
    assert not ExecutionProfile.configure("strict").timers_enabled
    assert ExecutionProfile.configure("experimental").initial_bp == 0xD0
    clear_only = ExecutionProfile.configure("experimental-isr-clear-only")
    assert clear_only.initial_bp == 0
    assert clear_only.isr_software_write == "clear_only" and clear_only.timers_enabled
    mti_writable = ExecutionProfile.configure("experimental-isr-mti-writable")
    assert mti_writable.initial_bp == 0
    assert mti_writable.isr_software_write == "clear_only_except_mti"
    assert mti_writable.timers_enabled
    clock = ExecutionProfile.configure("experimental-rtc")
    assert clock.initial_bp == 0 and clock.isr_software_write == "clear_only_except_mti"
    assert clock.rtc_a2_period == 512_000 and clock.rtc_second_period == 1_024_000
    edge = ExecutionProfile.configure("experimental-on-edge")
    assert edge.initial_bp == 0 and edge.isr_software_write == "clear_only_except_mti"
    assert edge.on_irq_edge_only and edge.rtc_a2_period == 1_000_000
    assert edge.rtc_second_period is None
    assert not mti_writable.on_irq_edge_only and not clock.on_irq_edge_only
    irq_imr = ExecutionProfile.configure("experimental-irq-imr")
    assert irq_imr.on_irq_edge_only and irq_imr.irq_imr_only
    assert irq_imr.rtc_second_period is None and irq_imr.rtc_a2_period == 1_000_000
    assert (
        irq_imr.initial_bp == 0
        and irq_imr.isr_software_write == "clear_only_except_mti"
    )
    assert not edge.irq_imr_only and not clock.irq_imr_only
    combined = ExecutionProfile.configure("experimental-rtc-irq-imr")
    assert combined.on_irq_edge_only and combined.irq_imr_only
    assert combined.rtc_second_period == 1_024_000 and combined.rtc_a2_period == 512_000
    assert (
        combined.initial_bp == 0
        and combined.isr_software_write == "clear_only_except_mti"
    )
    assert ExecutionProfile.configure("strict").rtc_second_period is None
    assert ExecutionProfile.configure("strict").isr_software_write == "replace"
    assert (
        ExecutionProfile.configure("experimental", retained_loaded=True).initial_bp == 0
    )
    with pytest.raises(ValueError):
        ExecutionProfile.configure("experimental", instructions=1)
    assert KEYS["SPACE"] == 4 * 8 + 6
    assert "CALENDAR" not in KEYS


def test_v1_profile_exposes_provisional_power_on_and_saved_memory_reset_contract():
    cold = ExecutionProfile.configure("provisional-v1")
    warm = ExecutionProfile.configure("provisional-v1", retained_loaded=True)
    reloaded_empty = ExecutionProfile.configure(
        "provisional-v1", retained_loaded=True, empty_sram=True
    )
    assert reloaded_empty.initial_bp == 0xD0
    assert cold.initial_bp == 0xD0 and warm.initial_bp == 0
    assert cold.block_transfer == warm.block_transfer == "coupled_predecrement"
    assert cold.byte_arithmetic == warm.byte_arithmetic == "low_byte"
    assert cold.isr_software_write == warm.isr_software_write == "clear_only_except_mti"
    assert cold.on_irq_edge_only and cold.irq_imr_only
    assert cold.rtc_second_period == 1_024_000 and cold.rtc_a2_period == 512_000
    with pytest.raises(ValueError):
        ExecutionProfile.configure("provisional-v1", instructions=1)


def test_on_irq_reference_separates_latch_reassertion_from_held_ssr():
    assert not ExecutionState().diagnostic_on_irq_edge_only
    assert on_irq_should_reassert(True, edge_only=False)
    assert not on_irq_should_reassert(True, edge_only=True)
    assert not on_irq_should_reassert(False, edge_only=False)


def test_irq_handler_guard_is_opt_in_and_does_not_replace_architectural_mask_checks():
    assert not ExecutionState().diagnostic_irq_imr_only
    assert handler_blocks_irq(True, imr_only=False)
    assert not handler_blocks_irq(True, imr_only=True)
    assert not handler_blocks_irq(False, imr_only=False)


def test_experimental_calendar_and_boundary_reference_handles_crossed_deadline_and_hold():
    from pce500.oz9600.hardware import Hardware

    rtc = Rtc()
    start = Clock(1993, 12, 31, 23, 59, 58, 5)
    rtc.backing[:5] = start.encode(bytes(5))
    rtc.backing[16:22] = Clock(1994, 1, 1, 0, 0, 0, 0).encode(bytes(5)) + bytes([7])
    rtc.advance_experimental_seconds(3)
    assert rtc.clock() == Clock(1994, 1, 1, 0, 0, 1, 6)
    assert rtc.read(10) == 0x79 and not rtc.interrupt_asserted()
    rtc.write(8, 1)
    assert rtc.interrupt_asserted()
    rtc.write(10, 0)
    rtc.advance_experimental_seconds(1)
    assert rtc.read(10) == 8
    hw = Hardware()
    hw.rtc = rtc
    boundary = ExecutionState(experimental_rtc_second_period=10)
    boundary.after_boundary(hw, 9, 9)
    assert boundary.rtc_second_phase_units == 9
    hw.rtc.write(12, 128)
    saved = rtc.registers()
    boundary.after_boundary(hw, 39, 30)
    assert boundary.rtc_held_units == 30 and boundary.rtc_second_phase_units == 9
    assert rtc.registers() == saved
    hw.rtc.write(12, 0)
    boundary.after_boundary(hw, 40, 1)
    assert boundary.rtc_elapsed_seconds == 1 and boundary.rtc_second_phase_units == 0
    assert rtc.clock().second == 3


def test_schedule_special_seconds_match_the_programmed_minute_without_relaxing_clock():
    rtc = Rtc()
    rtc.backing[16:22] = bytes.fromhex("3F 8F 78 D2 05 07")
    start = Clock(1993, 2, 15, 2, 14, 58, 0)
    rtc.backing[:5] = start.encode(bytes(5))
    for seconds, alarm in [(1, False), (1, True), (1, True), (58, True), (1, False)]:
        rtc.write(10, 0)
        rtc.advance_experimental_seconds(seconds)
        assert bool(rtc.read(10) & 1) == alarm
    rtc.backing[:5] = start.encode(bytes(5))
    rtc.write(10, 0)
    rtc.advance_experimental_seconds(64)
    assert rtc.read(10) & 1 and not rtc.interrupt_asserted()
    rtc.write(8, 1)
    assert rtc.interrupt_asserted()
    with pytest.raises(ValueError):
        Clock.decode(bytes.fromhex("3F 8F 78 D2 05"))
    for second in [0x3C, 0x3D, 0x3E]:
        rtc.write(16, second)
        rtc.backing[:5] = start.encode(bytes(5))
        rtc.write(10, 0)
        rtc.advance_experimental_seconds(64)
        assert not rtc.read(10) & 1


def test_off_time_advances_only_the_opt_in_rtc_source_and_preserves_phase_during_hold():
    from pce500.oz9600.hardware import Hardware

    hw = Hardware()
    start = Clock(1993, 1, 1, 23, 59, 58, 0)
    hw.rtc.backing[:5] = start.encode(bytes(5))
    hw.rtc.backing[16:22] = start.advance_seconds(2).encode(bytes(5)) + bytes([7])
    b = ExecutionState(experimental_rtc_second_period=10, diagnostic_rtc_a2_period=5)
    for time in range(27):
        b.before_boundary(hw, 0, pc=0xE0000, off=True, elapsed_timing_units=time)
        b.after_boundary(hw, 0, 0, elapsed_timing_units=time + 1)
    assert hw.rtc.clock() == start.advance_seconds(2)
    assert hw.rtc.read(10) == 0x7D
    assert b.rtc_timing_units == b.rtc_off_elapsed_units == 27
    assert b.rtc_second_phase_units == 7 and b.rtc_a2_ticks == 5
    hw.rtc.write(12, 128)
    b.before_boundary(hw, 0, pc=0xE0000, off=True, elapsed_timing_units=27)
    b.after_boundary(hw, 0, 0, elapsed_timing_units=38)
    assert b.rtc_held_units == 11 and b.rtc_second_phase_units == 7
    legacy = ExecutionState(diagnostic_rtc_a2_period=5)
    legacy.before_boundary(hw, 0, pc=0xE0000, off=True, elapsed_timing_units=38)
    legacy.after_boundary(hw, 0, 0, elapsed_timing_units=50)
    assert legacy.rtc_a2_phase_units == legacy.rtc_a2_ticks == 0


@pytest.mark.parametrize(
    "data",
    [
        '{"steps":[{"boundaries":0},{"boundaries":1,"contact":{"column":11,"row":0,"pressed":true}}]}',
        '{"steps":[{"boundaries":1,"tablet":{"raw_x":1024,"raw_y":0,"pressed":true}}]}',
        '{"steps":[{"boundaries":1,"rtc_causes":{"a":4,"b":0}}]}',
        '{"steps":[{"boundaries":0,"on_key":1}]}',
        '{"steps":[{"boundaries":-1}]}',
        '{"steps":[],"pc":983040}',
    ],
)
def test_invalid_physical_replay_is_rejected_before_application(data):
    with pytest.raises(ValueError):
        parse_replay(data)


def test_full_frame_owns_pixels_and_invalid_increment_is_atomic():
    lcd = LcdController()
    lcd.write(31, 2)
    lcd.write(3, 255)
    lcd.set_word(16, 335)
    lcd.set_word(18, 239)
    lcd.write(2, 128)
    frame = lcd.matrix_frame()
    assert (frame.cols, frame.rows) == (336, 240)
    assert frame.pixels[-1] == 1
    lcd.write(5, 0x53)
    before = (bytes(lcd.registers), lcd.pbm(), lcd.read_latch, lcd.data_writes)
    with pytest.raises(ValueError):
        lcd.write(2, 255)
    assert before == (bytes(lcd.registers), lcd.pbm(), lcd.read_latch, lcd.data_writes)


@pytest.mark.parametrize("mode,data", [(1, 255), (5, 0)])
def test_rom_highlight_operations_preserve_and_restore_mixed_pixels(mode, data):
    # Calendar's solid pattern uses mode 1/FF; Notebook uses mode 5/00.
    # Both complement existing glyph pixels, then restore on a second pass.
    lcd = LcdController()
    lcd.write(31, 2)  # explicit full-window bypass for this component fixture
    lcd.write(3, 255)
    lcd.set_word(16, 24)
    lcd.set_word(18, 40)
    lcd.write(2, 0xA5)
    lcd.write(8, mode)
    for expected in (0x5A, 0xA5):
        lcd.write(2, data)
        assert [lcd.pixel(24 + bit, 40) for bit in range(8)] == [
            bool(expected & (128 >> bit)) for bit in range(8)
        ]


@pytest.mark.parametrize("vertical", [0, 128])
def test_scrapbook_mode_three_adds_stroke_without_erasing_existing_ink(vertical):
    # The ROM's live line uses mode 3, while its saved bitmap ORs the same
    # stroke. Previously AND left an editor blank until Store/reset redraw.
    lcd = LcdController()
    lcd.write(31, 2)
    lcd.write(3, 255)
    lcd.set_word(16, 48)
    lcd.set_word(18, 88)
    lcd.write(8, vertical)
    lcd.write(2, 0xA5)
    lcd.write(8, vertical | 3)
    lcd.write(2, 0x42)
    lcd.write(2, 0)
    assert [
        lcd.pixel(48 + (0 if vertical else bit), 88 + (bit if vertical else 0))
        for bit in range(8)
    ] == [bool(0xE7 & (128 >> bit)) for bit in range(8)]
