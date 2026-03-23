"""Tests for the LCD text decoder."""

from __future__ import annotations

from pathlib import Path

import pytest

from pce500 import PCE500Emulator
from pce500.display.font import GLYPH_HEIGHT, GLYPH_STRIDE, JP_FONT_ATLAS_BASE
from pce500.display.text_decoder import decode_display_text


def _load_rom() -> bytes:
    rom_path = Path(__file__).resolve().parent.parent.parent / "data" / "pc-e500-en.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM image missing at {rom_path}")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip(f"Expected 1MB pc-e500-en.bin at {rom_path}")
    return data


class _SparseMemory:
    def __init__(self):
        self._data = {}

    def write_bytes(self, address: int, values: bytes) -> None:
        for offset, value in enumerate(values):
            self._data[address + offset] = value

    def read_byte(self, address: int) -> int:
        return self._data.get(address, 0)


class _FakeBuffer:
    def __init__(self, height: int = 32, width: int = 240):
        self.shape = (height, width)
        self._rows = [[True for _ in range(width)] for _ in range(height)]

    def __getitem__(self, index):
        row, col = index
        return self._rows[row][col]

    def light_pixel(self, row: int, col: int) -> None:
        self._rows[row][col] = False


class _FakeController:
    def __init__(self, buffer: _FakeBuffer):
        self._buffer = buffer

    def get_display_buffer(self):
        return self._buffer


def _paint_cell(buffer: _FakeBuffer, cell_x: int, cell_y: int, pattern: tuple[int, ...]) -> None:
    row_base = cell_y * 8
    col_base = cell_x * GLYPH_STRIDE
    for col_idx, column in enumerate(pattern):
        for row in range(GLYPH_HEIGHT):
            if (column >> row) & 1:
                buffer.light_pixel(row_base + row, col_base + col_idx)


def test_boot_screen_text_decodes() -> None:
    """Ensure the decoder reproduces the familiar boot menu."""

    rom_image = _load_rom()
    emu = PCE500Emulator(
        save_lcd_on_exit=False, perfetto_trace=False, trace_enabled=False
    )
    emu.load_rom(rom_image[0xC0000:0x100000], start_address=0xC0000)
    emu.reset()
    emu.run(20_000)

    lines = decode_display_text(emu.lcd, emu.memory)

    # Boot banner should match the ROM string resources.
    assert lines[0] == "S2(CARD):NEW CARD"
    assert lines[1] == ""
    assert lines[2].startswith("   PF1 --- INITIALIZE")
    assert lines[3].startswith("   PF2 --- DO NOT INITIALIZE")


def test_decoder_supports_jp_halfwidth_kana_cells() -> None:
    memory = _SparseMemory()
    # JP font-layout sentinels used by the decoder.
    memory.write_bytes(0x00F232B, bytes.fromhex("7c1211127c"))
    memory.write_bytes(0x00F2589, bytes.fromhex("0a4a4a2a1e"))

    wo_addr = JP_FONT_ATLAS_BASE + 0xA6 * GLYPH_STRIDE
    memory.write_bytes(wo_addr, bytes.fromhex("0a4a4a2a1e00"))

    buffer = _FakeBuffer()
    _paint_cell(buffer, 0, 0, (0x0A, 0x4A, 0x4A, 0x2A, 0x1E))
    lines = decode_display_text(_FakeController(buffer), memory)

    assert lines[0] == "ｦ"
