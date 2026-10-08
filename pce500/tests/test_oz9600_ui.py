"""Full controller geometry and host-contact lifetime regression checks."""

import pytest

from pce500.display.lcd_frame import LcdFrame
from pce500.oz9600.ui import (
    HEIGHT,
    INK,
    LCD,
    PAPER,
    SOUND,
    WIDTH,
    Contacts,
    CloseGate,
    HostKeys,
    PointerGate,
    contains,
    window_point,
    window_title,
    lcd_into,
    lcd_tablet,
    sound_artwork,
)


def test_pointer_conversion_inverts_scaling_and_rejects_outside_samples():
    for factor in (1, 2, 3, 4):
        for x, y in ((0, 0), (125, 95), (WIDTH - 1, HEIGHT - 1)):
            assert window_point(
                x * factor, y * factor, WIDTH * factor, HEIGHT * factor
            ) == (x, y)
    assert window_point(float("nan"), 1, WIDTH, HEIGHT) is None
    assert window_point(-1, 1, WIDTH, HEIGHT) is None
    assert window_point(WIDTH, 1, WIDTH, HEIGHT) is None
    assert window_point(1, 1, 0, HEIGHT) is None


def test_short_host_taps_keep_edges_and_repeat_does_not_create_new_tap():
    keys = HostKeys()
    assert keys.events([("Q", True), ("Q", False)]) == {"Q"}
    assert keys.down == set()
    assert keys.events([]) == set()
    assert keys.events(
        [("LeftShift", True), ("LeftShift", True), ("RightShift", True)]
    ) == {"LeftShift", "RightShift"}
    assert keys.events([("LeftShift", False)]) == set()
    assert keys.down == {"RightShift"}


def test_host_artwork_never_changes_guest_pixels_or_clips_last_row_and_column():
    pixels = bytes(int(n % 7 == 0 or n == 336 * 240 - 1) for n in range(336 * 240))
    frame = LcdFrame(336, 240, pixels)
    host = [0x333B40] * (WIDTH * HEIGHT)
    output = lcd_into(frame, host)
    for i, pixel in enumerate(pixels):
        y, x = divmod(i, 336)
        assert output[(LCD[1] + y) * WIDTH + LCD[0] + x] == (INK if pixel else PAPER)
    assert frame.pixels == pixels
    assert output[(LCD[1] + 239) * WIDTH + LCD[0] + 335] == INK
    assert host == [0x333B40] * (WIDTH * HEIGHT)
    assert output[0] == host[0]
    with pytest.raises(ValueError):
        lcd_into(LcdFrame(320, 240, bytes(320 * 240)), host)


def test_screen_contacts_use_all_controller_pixels_and_half_open_edges():
    assert lcd_tablet(0, 0) == (227, 68)
    assert lcd_tablet(335, 239) == (956, 944)
    assert lcd_tablet(999, 999) == (956, 944)
    assert contains(LCD, LCD[0], LCD[1])
    assert not contains(LCD, LCD[0] + 336, LCD[1])
    assert not contains(LCD, LCD[0], LCD[1] + 240)


def test_combined_owners_short_click_and_focus_loss_release_all_contacts():
    contacts = Contacts(minimum_hold=40_000)
    assert contacts.sync({3}, (61, 141), 0) == [
        ("matrix", 3, True),
        ("tablet", 61, 141, True),
    ]
    assert contacts.sync({3}, None, 40_000) == [("tablet", 61, 141, False)]
    assert set(contacts.keys) == {3}
    assert contacts.sync(set(), None, 40_001) == [("matrix", 3, False)]
    assert len(contacts.sync({3}, (227, 68), 50_000)) == 2
    assert contacts.sync(set(), None, 50_001) == []
    assert contacts.cancel() == [("matrix", 3, False), ("tablet", 227, 68, False)]
    assert contacts.cancel() == []


def test_pen_motion_retains_original_hold_deadline_and_zero_assistance_is_raw():
    contacts = Contacts()
    assert contacts.sync(set(), (227, 68), 0) == [("tablet", 227, 68, True)]
    assert contacts.sync(set(), (956, 944), 1) == [("tablet", 956, 944, True)]
    assert contacts.sync(set(), None, 2) == [("tablet", 956, 944, False)]


def test_on_contact_has_independent_ownership_deadline_and_immediate_cancellation():
    contacts = Contacts(minimum_hold=40_000)
    assert contacts.sync(set(), None, 0, True) == [("on", True)]
    assert contacts.sync(set(), None, 1, False) == []
    assert contacts.on_deadline == 40_000 and contacts.keys == {}
    assert contacts.sync(set(), None, 40_000, True) == []
    assert contacts.sync(set(), None, 40_001, False) == [("on", False)]
    assert contacts.on_deadline is None
    assert contacts.sync(set(), None, 50_000, True) == [("on", True)]
    assert contacts.cancel() == [("on", False)]
    assert contacts.on_deadline is None and contacts.cancel() == []
    raw = Contacts()
    assert raw.sync(set(), None, 0, True) == [("on", True)]
    assert raw.sync(set(), None, 1, False) == [("on", False)]


def test_sound_control_remains_outside_guest_lcd_and_contact_owners():
    x, y, width, height = SOUND
    assert not any(
        contains(LCD, px, py) for px in (x, x + width - 1) for py in (y, y + height - 1)
    )
    assert sound_artwork(False) == ("SOUND OFF", 0x94ACBB)
    assert sound_artwork(True) == ("SOUND ON", 0xB2C9A5)
    contacts = Contacts()
    assert contacts.sync(set(), None, 0) == []
    assert contacts.keys == {} and contacts.tablet is None


def test_close_gate_seals_event_acceptance_and_is_idempotent():
    gate = CloseGate()
    assert not gate.closing
    assert gate.begin_close()
    assert gate.closing
    assert not gate.begin_close()
    assert gate.closing


def test_keyboard_focus_does_not_discard_the_first_fresh_pointer_press():
    gate = PointerGate(active=True)
    gate.focus(False)
    gate.focus(True)
    assert gate.button(True)
    assert gate.accepts_target(host_control=False, fault=False)
    assert not gate.button(False)
    assert not gate.down and not gate.blocked


@pytest.mark.parametrize("held", [False, True])
def test_held_or_background_pointer_cannot_resume_without_release(held):
    gate = PointerGate(active=True)
    if held:
        assert gate.button(True)
    gate.focus(False)
    assert not gate.button(True)
    gate.focus(True)
    assert not gate.button(True)
    assert not gate.button(False)
    assert gate.button(True)


def test_fault_allows_host_controls_and_rejects_guest_targets():
    gate = PointerGate(active=True)
    assert gate.button(True)
    assert gate.accepts_target(host_control=True, fault=True)
    assert not gate.accepts_target(host_control=False, fault=True)
    gate.focus(False)
    assert not gate.accepts_target(host_control=True, fault=True)


def test_fault_keeps_successful_host_feedback_visible():
    assert window_title("Strict", "paused", "Saved backup.ozbat", "guest fault") == (
        "OZ-9600 | Strict | faulted | Saved backup.ozbat"
    )
    assert window_title("Strict", "paused", "guest fault", "guest fault") == (
        "OZ-9600 | Strict | faulted | guest fault"
    )
    assert window_title("Strict", "paused", "Captured guest", "") == (
        "OZ-9600 | Strict | faulted | Captured guest"
    )
    assert window_title("Strict", "paused", "Saved backup.ozbat", None) == (
        "OZ-9600 | Strict | paused | Saved backup.ozbat"
    )
