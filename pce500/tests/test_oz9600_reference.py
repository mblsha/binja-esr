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


@pytest.mark.parametrize(
    "increment,source,destination,delta", [(5, 208, 212, 1), (6, 209, 213, -1)]
)
def test_physical_completed_block_consumes_count_and_requires_reload(
    increment, source, destination, delta
):
    lcd = LcdController()
    lcd.windows[0][:] = bytes([0, 0, 0, 0, 79, 1, 239, 0])
    lcd.write(31, 2)
    lcd.write(3, 255)
    lcd.write(4, increment)
    lcd.write(5, increment)
    for y, value in [(source, 0xA5), (source + delta, 0x5A)]:
        lcd.set_word(16, 300)
        lcd.set_word(18, y)
        lcd.write(2, value)
    lcd.set_word(18, destination)
    lcd.set_word(6, 2)
    lcd.write(0, 0x3C)
    assert lcd.word(6) == 0
    lcd.set_word(20, 300)
    lcd.set_word(22, source)
    lcd.set_word(18, destination)
    lcd.read(0, 0)
    assert (lcd.word(22), lcd.word(18)) == (source, destination)
    assert lcd.read_pixels(300, destination) == 0x3C
    lcd.set_word(6, 2)
    lcd.read(0, 0)
    assert lcd.word(6) == 0
    assert [lcd.read_pixels(300, destination + n * delta) for n in range(2)] == [
        0xA5,
        0x5A,
    ]
    coordinates = bytes(lcd.registers[16:24])
    lcd.read(0, 0)
    assert bytes(lcd.registers[16:24]) == coordinates


@pytest.mark.parametrize("vertical", [0, 128])
@pytest.mark.parametrize("missing", [1, 2])
def test_signed_pixel_stream_crosses_zero_on_both_axes(vertical, missing):
    lcd = LcdController()
    # Signed-coordinate coverage requires an enabled, full drawing/read window.
    lcd.windows[0][:] = bytes([0, 0, 0, 0, 79, 1, 239, 0])
    lcd.write(31, 2)
    lcd.write(3, 255)
    lcd.write(8, vertical)
    axis, other = (16, 18) if vertical == 0 else (18, 16)
    lcd.set_word(axis, 0)
    lcd.set_word(other, 10)
    lcd.write(2, 0xA5)
    lcd.set_word(axis + 4, -missing)
    lcd.set_word(other + 4, 10)
    lcd.read(2, 0)  # prefetch dummy
    assert lcd.read(2, 0) == 0xA5 >> missing
    lcd.set_word(axis, -missing)
    lcd.write(2, 0x53)
    lcd.set_word(axis + 4, 0)
    lcd.read(2, 0)
    assert lcd.read(2, 0) == ((0x53 << missing) & 255) | (0xA5 & ((1 << missing) - 1))


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
    assert clock.initial_bp == 0 and clock.isr_software_write == "replace"
    assert clock.rtc_a2_period == 512_000 and clock.rtc_second_period == 1_024_000
    edge = ExecutionProfile.configure("experimental-on-edge")
    assert edge.initial_bp == 0 and edge.isr_software_write == "replace"
    assert edge.on_irq_edge_only and edge.rtc_a2_period == 1_000_000
    assert edge.rtc_second_period is None
    assert not mti_writable.on_irq_edge_only and not clock.on_irq_edge_only
    irq_imr = ExecutionProfile.configure("experimental-irq-imr")
    assert irq_imr.on_irq_edge_only and irq_imr.irq_imr_only
    assert irq_imr.rtc_second_period is None and irq_imr.rtc_a2_period == 1_000_000
    assert irq_imr.initial_bp == 0 and irq_imr.isr_software_write == "replace"
    assert not edge.irq_imr_only and not clock.irq_imr_only
    combined = ExecutionProfile.configure("experimental-rtc-irq-imr")
    assert combined.on_irq_edge_only and combined.irq_imr_only
    assert combined.rtc_second_period == 1_024_000 and combined.rtc_a2_period == 512_000
    assert combined.initial_bp == 0 and combined.isr_software_write == "replace"
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
    assert cold.isr_software_write == warm.isr_software_write == "replace"
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


def full_lcd_window(lcd):
    # A component needs a real full descriptor, not the disproved bypass bit.
    for n, word in enumerate((0, 0, 335, 239)):
        lcd.write(24 + 2 * n, word & 255)
        lcd.write(25 + 2 * n, word >> 8)


def test_full_frame_owns_pixels_and_invalid_increment_is_atomic():
    lcd = LcdController()
    full_lcd_window(lcd)
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


def test_ordered_window_candidate_preserves_earlier_region_and_advances_masked_blocks():
    lcd = LcdController(16, 8)

    def define(index, bounds, bypass=0):
        lcd.write(9, index)
        for n, value in enumerate(bounds):
            lcd.write(24 + n * 2, value & 255)
            lcd.write(25 + n * 2, (value >> 8) | (bypass if n == 3 else 0))

    define(1, [0, 0, 15, 7], 2)
    lcd.write(3, 255)
    lcd.set_word(16, 2)
    lcd.set_word(18, 2)
    lcd.write(2, 255)
    define(2, [4, 2, 6, 3])
    define(4, [8, 2, 9, 3])
    define(3, [2, 1, 9, 5])
    lcd.ordered_window_clipping_candidate = True
    lcd.write(5, 0x10)
    lcd.write(2, 0)
    assert [lcd.pixel(x, 2) for x in range(2, 10)] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert lcd.word(16) == 3
    lcd.write(3, 128)
    lcd.set_word(16, 4)
    lcd.set_word(18, 3)
    lcd.set_word(6, 2)
    lcd.write(0, 255)
    assert not lcd.pixel(4, 3) and not lcd.pixel(5, 3)
    assert lcd.word(16) == 6
    lcd.write(9, 2)
    lcd.write(2, 255)
    assert lcd.pixel(6, 3)
    lcd.write(9, 1)
    lcd.set_word(16, 4)
    lcd.set_word(18, 2)
    lcd.write(2, 0)
    assert not lcd.pixel(4, 2)


@pytest.mark.parametrize("mode,data", [(1, 255), (5, 0)])
def test_rom_highlight_operations_preserve_and_restore_mixed_pixels(mode, data):
    # Calendar's solid pattern uses mode 1/FF; Notebook uses mode 5/00.
    # Both complement existing glyph pixels, then restore on a second pass.
    lcd = LcdController()
    full_lcd_window(lcd)
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
    full_lcd_window(lcd)
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


def test_external_off_wake_is_explicit_and_cpu_external_masked():
    from pce500.oz9600.profile import off_wake_mask

    ordinary = ExecutionProfile.configure("provisional-v1")
    candidate = ExecutionProfile.configure("experimental-external-wake")
    assert not ordinary.external_off_wake and candidate.external_off_wake
    assert candidate.initial_bp == 0xD0 and candidate.irq_imr_only
    assert off_wake_mask(external_enabled=True, external_experiment=True) == 0x48
    assert off_wake_mask(external_enabled=False, external_experiment=True) == 0x08
    assert off_wake_mask(external_enabled=True, external_experiment=False) == 0x08


def test_tablet_release_readiness_requires_completed_pair_and_idle_drive():
    from pce500.oz9600.tablet import Tablet

    t = Tablet()
    t.set_contact(403, 601, True)
    for drive, control, result in [(0xA8, 6, [100, 3]), (0x8A, 7, [150, 1])]:
        t.drive_control = drive
        t.write_conversion_control(control)
        assert not t.contact_release_ready()
        assert [t.read_data(), t.read_data()] == result
        assert not t.contact_release_ready()
    t.drive_control = 0x10
    assert t.contact_release_ready()
    t.drive_control = 0x8A
    assert (
        not t.contact_release_ready()
    )  # An earlier pair cannot authorize this release.
    t.drive_control = 0x10
    t.set_contact(403, 601, False)
    t.set_contact(403, 601, True)
    assert not t.contact_release_ready()


def test_native_pen_release_grace_starts_after_the_actual_host_release():
    from pce500.oz9600.ui import Contacts

    c = Contacts(minimum_hold=40000)
    assert c.sync(set(), (61, 141), 0, ready=False) == [("tablet", 61, 141, True)]
    assert c.sync(set(), (61, 141), 1000000, ready=False) == []
    assert c.sync(set(), None, 1000000, ready=False) == []
    assert c.sync(set(), None, 1399999, ready=False) == []
    assert c.sync(set(), None, 1400000, ready=False) == [("tablet", 61, 141, False)]
    c.sync(set(), (61, 141), 1500000, ready=False)
    assert c.cancel() == [("tablet", 61, 141, False)]
    raw = Contacts(minimum_hold=0)
    raw.sync(set(), (61, 141), 0, ready=False)
    assert raw.sync(set(), None, 1, ready=False) == [("tablet", 61, 141, False)]


def test_native_short_pen_click_waits_for_rom_mailbox_and_raw_mode_bypasses_it():
    from pce500.oz9600.ui import Contacts

    c = Contacts(minimum_hold=40000)
    assert c.sync(set(), (627, 130), 0, busy=True) == []
    assert c.sync(set(), None, 1, busy=True) == []
    assert c.sync(set(), None, 40960, busy=False) == [("tablet", 627, 130, True)]
    assert c.sync(set(), None, 80959, busy=False) == []
    assert c.sync(set(), None, 80960, busy=False) == [("tablet", 627, 130, False)]
    raw = Contacts()
    assert raw.sync(set(), (627, 130), 0, ready=False, busy=True) == [
        ("tablet", 627, 130, True)
    ]
    assert raw.sync(set(), None, 1, ready=False, busy=True) == [
        ("tablet", 627, 130, False)
    ]


def test_rom_tablet_mailbox_observer_is_read_only_and_pointer_bounded():
    from types import SimpleNamespace
    from pce500.oz9600.host_input import tablet_mailbox_busy

    h = SimpleNamespace(ram=bytearray(0x40000))
    h.ram[0x3FF47:0x3FF4A] = bytes.fromhex("49 E5 01")
    for offset, code in [(0, 0xC8), (0x0B, 0xC9), (0x1A, 0xCA)]:
        h.ram[0x3E549 + offset] = code
        before = bytes(h.ram)
        assert tablet_mailbox_busy(h)
        assert bytes(h.ram) == before
        h.ram[0x3E549 + offset] = 0
        assert not tablet_mailbox_busy(h)
    h.ram[0x3FF47:0x3FF4A] = bytes.fromhex("FF FF 01")
    assert not tablet_mailbox_busy(h)


def test_physical_irq_recognition_latch_survives_mask_clear_and_repeated_request():
    from pce500.oz9600.profile import IrqRecognition

    ordinary = ExecutionProfile.configure("provisional-v1")
    experimental = ExecutionProfile.configure("experimental-irq-recognition")
    assert not ordinary.irq_recognition_delay and experimental.irq_recognition_delay
    assert experimental.irq_imr_only and experimental.rtc_second_period == 1_024_000
    for next_imr, next_isr in [(0, 1), (0x81, 0), (0x81, 1)]:
        latch = IrqRecognition(enabled=True)
        assert not latch.transfer_selected()
        latch.recognize_before_instruction(0x81, 1)
        # The next boundary sees the preceding instruction's completed writes.
        latch.recognize_before_instruction(next_imr, next_isr)
        assert latch.recognized_mask == 1 and latch.transfer_selected()
        latch.entered_or_reset()
        assert not latch.transfer_selected()
    masked = IrqRecognition(enabled=True)
    masked.recognize_before_instruction(1, 1)
    assert not masked.transfer_selected()
    masked.recognize_before_instruction(0x82, 2)
    assert masked.recognized_mask == 2

    priority = IrqRecognition(enabled=True)
    priority.recognize_before_instruction(0xE0, 0x60)
    assert priority.recognized_mask == 0x20  # Existing RX-before-EX dispatcher priority


def test_recognized_irq_session_uses_schema_two_to_fail_closed_in_older_readers():
    from pce500.oz9600.session import decode, encode

    identity = bytes(range(64))
    data = {"schema": 2, "timer": {"recognition_delay": True, "recognized_mask": 1}}
    assert decode(encode(data, identity), identity) == data
    for bad in [
        {"schema": 1, "timer": {"recognition_delay": True}},
        {"schema": 2, "timer": {}},
    ]:
        with pytest.raises(ValueError, match="policy/schema"):
            encode(bad, identity)


@pytest.mark.parametrize(
    "bounds,source,backing",
    [
        ((305, 208, 306, 215), 0x00, 0xFF),
        ((304, 210, 304, 213), 0x00, 0xC3),
        ((302, 208, 303, 0x200), 0x00, 0xFF),
        ((304, 216, 304, 223), 0x3C, 0xFF),
        ((304, 216, 304, 735), 0x3C, 0xFF),
        ((304, 0, 304, 207), 0x3C, 0xFF),
        ((304, 0, 304, 719), 0x3C, 0xFF),
    ],
)
def test_physical_window_readback_preserves_excluded_backing_pixels(
    bounds, source, backing
):
    lcd = LcdController()
    lcd.write(9, 2)

    def window(values):
        for n, word in enumerate(values):
            lcd.write(24 + 2 * n, word & 255)
            lcd.write(25 + 2 * n, word >> 8)

    def read_group():
        lcd.read(2, 0)
        return lcd.read(2, 0)

    window((0, 0, 319, 239))
    lcd.write(3, 255)
    lcd.write(8, 128)
    for offset, word in ((16, 304), (18, 208), (20, 304), (22, 208)):
        lcd.write(offset, word & 255)
        lcd.write(offset + 1, word >> 8)
    lcd.write(2, 255)
    assert read_group() == 255
    window(bounds)
    lcd.write(2, source)
    assert read_group() == 0
    window((0, 0, 319, 239))
    assert read_group() == backing
