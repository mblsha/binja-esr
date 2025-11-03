"""Shared helpers for working with the PC-E500 ROM font."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

FONT_BASE = 0x00F2215
GLYPH_WIDTH = 5
GLYPH_STRIDE = 6  # five data columns + spacer
GLYPH_COUNT = 96  # ASCII 0x20-0x7F
GLYPH_HEIGHT = 7


def _read_glyph_columns(memory, index: int) -> Tuple[int, ...]:
    """Return the raw column bytes for a glyph index."""
    if not (0 <= index < GLYPH_COUNT):
        raise ValueError(f"Glyph index out of range: {index}")

    offset = FONT_BASE + index * GLYPH_STRIDE
    return tuple(memory.read_byte(offset + i) & 0x7F for i in range(GLYPH_WIDTH))


def _font_lookup(memory) -> Dict[str, Tuple[int, ...]]:
    """Build (and cache) the mapping from ASCII characters to column tuples."""
    key = id(memory)

    @lru_cache(maxsize=8)
    def _build(_: int) -> Dict[str, Tuple[int, ...]]:
        mapping: Dict[str, Tuple[int, ...]] = {}
        for index in range(GLYPH_COUNT):
            char_code = 0x20 + index
            mapping[chr(char_code)] = _read_glyph_columns(memory, index)
        return mapping

    return _build(key)


def glyph_columns(memory, char: str) -> Tuple[int, ...]:
    """Return the ROM column data for a single ASCII character."""
    if not char:
        raise ValueError("Character must be non-empty")
    lookup = _font_lookup(memory)
    if char not in lookup:
        raise KeyError(f"Character {char!r} is not available in the ROM font")
    return lookup[char]


def text_columns(memory, text: str) -> List[int]:
    """Return the concatenated column bytes (including spacers) for text."""
    columns: List[int] = []
    for char in text:
        glyph = glyph_columns(memory, char)
        columns.extend(glyph)
        columns.append(0)  # Implicit spacer column between glyphs
    return columns


def glyph_bitmap(memory, char: str) -> List[List[int]]:
    """Decode the ROM glyph into a 2D bitmap (0 = pixel on, 1 = pixel off)."""
    columns = glyph_columns(memory, char)
    bitmap: List[List[int]] = [[1] * GLYPH_WIDTH for _ in range(GLYPH_HEIGHT)]
    for col_idx, column in enumerate(columns):
        for row in range(GLYPH_HEIGHT):
            if (column >> row) & 1:
                bitmap[row][col_idx] = 0
    return bitmap
