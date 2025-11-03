"""Tests for the high-level keyboard handler and FIFO mirroring."""

from __future__ import annotations


from typing import List
from pce500.keyboard_handler import PCE500KeyboardHandler
from pce500.keyboard_matrix import (
    FIFO_BASE,
    FIFO_HEAD_ADDR,
    FIFO_TAIL_ADDR,
    KEY_LOCATIONS,
)
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


def _read_fifo(memory: PCE500Memory, count: int) -> List[int]:
    data: List[int] = []
    head = memory.read_byte(FIFO_HEAD_ADDR)
    tail = memory.read_byte(FIFO_TAIL_ADDR)
    idx = head
    while idx != tail and len(data) < count:
        data.append(memory.read_byte(FIFO_BASE + idx))
        idx = (idx + 1) % 8
    return data


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

        fifo = _read_fifo(self.memory, 2)
        assert fifo == [events[0].to_byte()]

        # Release the key and tick again to generate release event
        self.handler.release_key("KEY_A")
        self.handler.scan_tick()
        events = self.handler.scan_tick()
        assert events and events[0].release
        fifo = _read_fifo(self.memory, 2)
        assert fifo == [
            KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row,
            0x80 | (KEY_LOCATIONS["KEY_A"].column << 3 | KEY_LOCATIONS["KEY_A"].row),
        ]

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

        fifo_before = _read_fifo(self.memory, 4)
        self.handler.scan_tick()
        fifo_after = _read_fifo(self.memory, 4)
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
