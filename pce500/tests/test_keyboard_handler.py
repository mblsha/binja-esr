"""Tests for the high-level keyboard handler and FIFO buffering."""

from __future__ import annotations


from typing import List
from pce500.keyboard_handler import PCE500KeyboardHandler
from pce500.keyboard_matrix import KEY_LOCATIONS
from pce500.memory import PCE500Memory, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _select_column(handler: PCE500KeyboardHandler, key_code: str) -> None:
    loc = KEY_LOCATIONS[key_code]
    kol = 0
    koh = 0
    if loc.column < 8:
        kol = 1 << loc.column
    else:
        koh = 1 << (loc.column - 8)
    handler.handle_register_write(0xF0, kol)
    handler.handle_register_write(0xF1, koh)


def _read_fifo(handler: PCE500KeyboardHandler, count: int) -> List[int]:
    return handler.fifo_snapshot()[:count]


class TestKeyboardHandler:
    def setup_method(self) -> None:
        self.memory = PCE500Memory()
        self.handler = PCE500KeyboardHandler(self.memory)
        # Speed up debouncing/repeat for tests
        self.handler._matrix.press_threshold = 2
        self.handler._matrix.release_threshold = 2
        self.handler._matrix.repeat_delay = 2
        self.handler._matrix.repeat_interval = 2

    def test_register_roundtrip(self) -> None:
        # Writes to KOL/KOH should be readable via handler
        self.handler.handle_register_write(0xF0, 0x12)
        self.handler.handle_register_write(0xF1, 0x05)
        assert self.handler.handle_register_read(0xF0) == 0x12
        assert self.handler.handle_register_read(0xF1) == 0x05

    def test_fifo_enqueues_press_and_release(self) -> None:
        _select_column(self.handler, "KEY_A")
        assert self.handler.press_key("KEY_A")

        # First scan tick primes debounce, second one enqueues press
        self.handler.scan_tick()
        events = self.handler.scan_tick()
        assert events and not events[0].release

        fifo = _read_fifo(self.handler, 2)
        assert fifo == [events[0].to_byte()]

        # Release the key and tick again to generate release event
        self.handler.release_key("KEY_A")
        self.handler.scan_tick()
        events = self.handler.scan_tick()
        assert events and events[0].release
        fifo = _read_fifo(self.handler, 2)
        assert fifo == [
            KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row,
            0x80 | (KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row),
        ]

    def test_fifo_drains_into_rom_iocs_workspace(self) -> None:
        workspace = 0x0BFC80
        self.memory.write_long(0x1000E6, workspace)
        self.memory.write_byte(workspace + 0x02, 0x50)
        self.memory.write_byte(workspace + 0x04, 0)
        self.memory.write_byte(workspace + 0x05, 0)

        _select_column(self.handler, "KEY_F1")
        self.handler._matrix.press_threshold = 1
        assert self.handler.press_key("KEY_F1")
        assert self.handler.scan_tick()

        assert self.handler.drain_fifo_to_pce500_iocs_workspace(True) == 1
        assert self.memory.read_byte(workspace + 0x50) == (
            KEY_LOCATIONS["KEY_F1"].column << 3 | KEY_LOCATIONS["KEY_F1"].row
        )
        assert self.memory.read_byte(workspace + 0x04) == 1
        assert self.handler.fifo_snapshot() == []

    def test_kil_read_preserves_debounced_event_in_rom_iocs_workspace(self) -> None:
        workspace = 0x0BFC80
        self.memory.write_long(0x1000E6, workspace)
        self.memory.write_byte(workspace + 0x02, 0x50)
        self.memory.write_byte(workspace + 0x04, 0)
        self.memory.write_byte(workspace + 0x05, 0)

        _select_column(self.handler, "KEY_F1")
        self.handler._matrix.press_threshold = 1
        self.handler._matrix.release_threshold = 1
        assert self.handler.press_key("KEY_F1")
        assert self.handler.scan_tick()
        assert self.handler.drain_fifo_to_pce500_iocs_workspace(False) == 1

        self.handler.handle_register_write(0xF0, 0)
        self.handler.handle_register_write(0xF1, 0)
        assert self.handler.handle_register_read(0xF2) == 0

        matrix_code = KEY_LOCATIONS["KEY_F1"].column << 3 | KEY_LOCATIONS["KEY_F1"].row
        assert self.memory.read_byte(workspace + 0x51) == (matrix_code | 0x80)
        assert self.memory.read_byte(workspace + 0x04) == 2
        assert self.handler.fifo_snapshot() == []

    def test_scan_respects_ksd_mask(self) -> None:
        # Assert initial read matches press
        _select_column(self.handler, "KEY_B")
        self.handler.press_key("KEY_B")
        self.handler.scan_tick()
        assert self.handler.handle_register_read(0xF2) != 0x00

        # Set KSD bit in LCC and ensure handler returns 0 and scanning pauses
        lcc_addr = INTERNAL_MEMORY_START + IMEMRegisters.LCC
        self.memory.write_byte(lcc_addr, 0x04)
        assert self.handler.handle_register_read(0xF2) == 0x00

        fifo_before = _read_fifo(self.handler, 4)
        self.handler.scan_tick()
        fifo_after = _read_fifo(self.handler, 4)
        assert fifo_before == fifo_after

    def test_repeat_events_marked(self) -> None:
        _select_column(self.handler, "KEY_F1")
        self.handler.press_key("KEY_F1")
        # Initial debounce
        self.handler.scan_tick()
        first_events = self.handler.scan_tick()
        assert first_events and not first_events[0].repeat

        # Hold the key to trigger repeat (repeat_delay=3)
        self.handler.scan_tick()
        repeat_events = self.handler.scan_tick()
        assert repeat_events and repeat_events[0].repeat

    def test_press_bounce_below_threshold_does_not_enqueue(self) -> None:
        self.handler._matrix.press_threshold = 3
        _select_column(self.handler, "KEY_A")

        assert self.handler.press_key("KEY_A")
        assert self.handler.scan_tick() == []
        self.handler.release_key("KEY_A")
        assert self.handler.scan_tick() == []

        assert self.handler.fifo_snapshot() == []

    def test_release_debounce_holds_until_threshold(self) -> None:
        self.handler._matrix.release_threshold = 3
        _select_column(self.handler, "KEY_A")
        assert self.handler.press_key("KEY_A")
        self.handler.scan_tick()
        press_events = self.handler.scan_tick()
        assert press_events and not press_events[0].release

        self.handler.release_key("KEY_A")
        assert self.handler.scan_tick() == []
        assert self.handler.scan_tick() == []
        release_events = self.handler.scan_tick()

        assert release_events and release_events[0].release
        assert self.handler.fifo_snapshot() == [
            KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row,
            0x80 | (KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row),
        ]

    def test_repeat_interval_is_stable_after_first_repeat(self) -> None:
        self.handler._matrix.repeat_delay = 1
        self.handler._matrix.repeat_interval = 3
        _select_column(self.handler, "KEY_F1")
        assert self.handler.press_key("KEY_F1")
        self.handler.scan_tick()
        self.handler.scan_tick()

        first_repeat = self.handler.scan_tick()
        assert first_repeat and first_repeat[0].repeat
        assert self.handler.scan_tick() == []
        assert self.handler.scan_tick() == []
        next_repeat = self.handler.scan_tick()
        assert next_repeat and next_repeat[0].repeat
