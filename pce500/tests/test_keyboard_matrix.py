"""Snapshot-style tests for the pure keyboard matrix."""

from __future__ import annotations

from pce500.keyboard_matrix import KeyboardMatrix, KEY_LOCATIONS


def _select_column(matrix: KeyboardMatrix, key_code: str) -> None:
    loc = KEY_LOCATIONS[key_code]
    kol = 0
    koh = 0
    if loc.column < 8:
        kol = 1 << loc.column
    else:
        koh = 1 << (loc.column - 8)
    matrix.write_kol(kol)
    matrix.write_koh(koh)


def _matrix_byte(key_code: str) -> int:
    loc = KEY_LOCATIONS[key_code]
    return (loc.column << 3) | loc.row


def test_single_press_and_release_events():
    matrix = KeyboardMatrix(
        press_threshold=2,
        release_threshold=2,
        repeat_delay=5,
        repeat_interval=3,
    )
    _select_column(matrix, "KEY_A")
    assert matrix.press_key("KEY_A")

    # Debounce: first tick primes, second tick enqueues
    matrix.scan_tick()
    events = matrix.scan_tick()
    assert events and events[0].code == _matrix_byte("KEY_A") and not events[0].release

    matrix.release_key("KEY_A")
    matrix.scan_tick()
    events = matrix.scan_tick()
    assert events and events[0].release


def test_repeat_events_marked():
    matrix = KeyboardMatrix(
        press_threshold=1,
        release_threshold=1,
        repeat_delay=2,
        repeat_interval=2,
    )
    _select_column(matrix, "KEY_F1")
    matrix.press_key("KEY_F1")

    # Initial press event
    events = matrix.scan_tick()
    assert events and not events[0].repeat

    # Hold long enough for repeat
    matrix.scan_tick()
    events = matrix.scan_tick()
    assert events and events[0].repeat


def test_simultaneous_keys_share_fifo():
    matrix = KeyboardMatrix(press_threshold=1, release_threshold=1)

    for key in ("KEY_Q", "KEY_W"):
        matrix.press_key(key)
    # Strobe both columns simultaneously
    loc_q = KEY_LOCATIONS["KEY_Q"].column
    loc_w = KEY_LOCATIONS["KEY_W"].column
    kol = 0
    for col in (loc_q, loc_w):
        if col < 8:
            kol |= 1 << col
    matrix.write_kol(kol)
    matrix.scan_tick()
    snapshot = matrix.fifo_snapshot()
    codes = {_matrix_byte("KEY_Q"), _matrix_byte("KEY_W")}
    assert set(snapshot) == codes


def test_scan_disable_suppresses_events():
    matrix = KeyboardMatrix(press_threshold=1, release_threshold=1)
    matrix.scan_enabled = False
    matrix.press_key("KEY_Z")
    _select_column(matrix, "KEY_Z")
    events = matrix.scan_tick()
    assert events == []
    assert matrix.fifo_snapshot() == []


def test_trace_kio_uses_separate_hook():
    matrix = KeyboardMatrix(press_threshold=1, release_threshold=1)
    scan_calls = []
    loc = KEY_LOCATIONS["KEY_Q"]
    matrix._trace_hook = lambda col, row, pressed: scan_calls.append(
        (col, row, pressed)
    )
    kio_calls = []

    def kio_hook(name, kol, koh, kil, pc=None):
        kio_calls.append((name, kol, koh, kil, pc))
        return True

    matrix._kio_trace_hook = kio_hook

    matrix.write_kol(0x01)
    matrix.press_key("KEY_Q")
    matrix.scan_tick()
    assert scan_calls == [(loc.column, loc.row, True)]

    expected_kil = matrix._compute_kil()
    matrix.trace_kio("read_kil", pc=0x123456)

    assert kio_calls == [
        ("read_kil", matrix.kol & 0xFF, matrix.koh & 0x0F, expected_kil, 0x123456)
    ]
    # KIO tracing should not reuse the scan hook signature.
    assert scan_calls == [(loc.column, loc.row, True)]
