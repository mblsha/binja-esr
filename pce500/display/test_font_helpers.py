"""Tests for the ROM font helper utilities."""

from __future__ import annotations

from pathlib import Path
import pytest

from pce500.display.font import (
    FONT_BASE,
    GLYPH_HEIGHT,
    GLYPH_STRIDE,
    GLYPH_WIDTH,
    JP_FONT_ATLAS_BASE,
    font_reverse_lookup,
    glyph_bitmap,
    glyph_columns,
    text_columns,
)


class _RomMemory:
    def __init__(self, data: bytes):
        self._data = data

    def read_byte(self, address: int) -> int:
        return self._data[address]


class _SparseMemory:
    def __init__(self):
        self._data = {}

    def write_bytes(self, address: int, values: bytes) -> None:
        for offset, value in enumerate(values):
            self._data[address + offset] = value

    def read_byte(self, address: int) -> int:
        return self._data.get(address, 0)


def _load_rom() -> bytes:
    rom_path = Path(__file__).resolve().parent.parent.parent / "data" / "pc-e500-en.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM image missing at {rom_path}")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip("Expected 1MB pc-e500-en.bin ROM image")
    return data


def test_glyph_columns_match_rom_bytes():
    rom = _load_rom()
    memory = _RomMemory(rom)

    for char in "A9? ":
        columns = glyph_columns(memory, char)
        index = ord(char) - 0x20
        expected = tuple(
            rom[FONT_BASE + index * GLYPH_STRIDE + i] & 0x7F for i in range(GLYPH_WIDTH)
        )
        assert columns == expected


def test_glyph_bitmap_dimensions_and_bits():
    rom = _load_rom()
    memory = _RomMemory(rom)
    bitmap = glyph_bitmap(memory, "S")

    assert len(bitmap) == GLYPH_HEIGHT
    assert all(len(row) == GLYPH_WIDTH for row in bitmap)
    # At least one dark pixel should be present.
    assert sum(pixel == 0 for row in bitmap for pixel in row) > 0


def test_text_columns_appends_spacers():
    rom = _load_rom()
    memory = _RomMemory(rom)
    columns = text_columns(memory, "AB")

    assert len(columns) == GLYPH_STRIDE * 2
    first = columns[:GLYPH_WIDTH]
    second = columns[GLYPH_STRIDE : GLYPH_STRIDE + GLYPH_WIDTH]

    assert first == list(glyph_columns(memory, "A"))
    assert second == list(glyph_columns(memory, "B"))
    # Spacer columns should be present (zeroed) between glyphs.
    assert columns[GLYPH_WIDTH] == 0


def test_jp_kana_and_fullwidth_aliases_decode_from_jp_atlas():
    memory = _SparseMemory()
    # JP font-layout sentinels used by the decoder.
    memory.write_bytes(0x00F232B, bytes.fromhex("7c1211127c"))
    memory.write_bytes(0x00F2589, bytes.fromhex("0a4a4a2a1e"))

    wo_addr = JP_FONT_ATLAS_BASE + 0xA6 * GLYPH_STRIDE
    o_small_addr = JP_FONT_ATLAS_BASE + 0xAB * GLYPH_STRIDE
    memory.write_bytes(wo_addr, bytes.fromhex("0a4a4a2a1e00"))
    memory.write_bytes(o_small_addr, bytes.fromhex("4828187c0800"))

    assert glyph_columns(memory, "ｦ") == (0x0A, 0x4A, 0x4A, 0x2A, 0x1E)
    assert glyph_columns(memory, "ヲ") == (0x0A, 0x4A, 0x4A, 0x2A, 0x1E)
    assert glyph_columns(memory, "ｫ") == (0x48, 0x28, 0x18, 0x7C, 0x08)
    assert glyph_columns(memory, "ォ") == (0x48, 0x28, 0x18, 0x7C, 0x08)


def test_display_aliases_decode_menu_brackets_and_updown_arrow():
    memory = _SparseMemory()
    memory.write_bytes(0x00F232B, bytes.fromhex("7c1211127c"))
    memory.write_bytes(0x00F2589, bytes.fromhex("0a4a4a2a1e"))

    reverse = font_reverse_lookup(memory)

    assert reverse[(0x00, 0x28, 0x6C, 0x6C, 0x28)] == "↕"
    assert reverse[(0x00, 0x7F, 0x7F, 0x41, 0x00)] == "["
    assert reverse[(0x00, 0x41, 0x7F, 0x7F, 0x00)] == "]"
