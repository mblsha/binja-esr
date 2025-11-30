from __future__ import annotations

import pytest

from pce500.display.controller_wrapper import HD61202Controller
from binja_test_mocks.eval_llil import Memory
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE


def _make_memory():
    raw = bytearray(ADDRESS_SPACE_SIZE)

    def read(addr: int) -> int:
        return raw[addr]

    def write(addr: int, value: int) -> None:
        raw[addr] = value & 0xFF

    mem = Memory(read, write)
    setattr(mem, "_raw", raw)
    return mem


def test_lcd_reads_return_ff():
    _ = _make_memory()  # retained for parity with other tests if needed later
    lcd = HD61202Controller()

    # LCD address (instruction/data bits don't matter since we always return 0xFF)
    base_addr = 0x2000
    for offset in range(0, 4):
        addr = base_addr | offset
        assert lcd.read(addr) == 0xFF
