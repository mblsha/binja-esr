"""Tests for the ROM font helper utilities."""

from __future__ import annotations

from pathlib import Path
import pytest

from pce500.display.font import (
    FONT_BASE,
    GLYPH_HEIGHT,
    GLYPH_STRIDE,
    GLYPH_WIDTH,
    glyph_bitmap,
    glyph_columns,
    text_columns,
)


class _RomMemory:
    def __init__(self, data: bytes):
        self._data = data

    def read_byte(self, address: int) -> int:
        return self._data[address]


def _load_rom() -> bytes:
    rom_path = Path(__file__).resolve().parent.parent.parent / "data" / "pc-e500.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM image missing at {rom_path}")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip("Expected 1MB pc-e500.bin ROM image")
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
