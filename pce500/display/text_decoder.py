"""Utilities to decode LCD controller VRAM into ASCII text."""

from __future__ import annotations

from typing import Dict, List, Tuple

FONT_BASE = 0x00F2215
GLYPH_WIDTH = 5
GLYPH_STRIDE = 6
GLYPH_COUNT = 96  # ASCII 0x20-0x7F
CHAR_COLUMNS = 40
CHAR_ROWS = 4
PIXELS_PER_CHAR_COL = 6  # 5 glyph columns + 1 spacer
CHAR_HEIGHT = 7


def _read_glyph_columns(memory) -> Dict[Tuple[int, ...], str]:
    lookup: Dict[Tuple[int, ...], str] = {}
    for index in range(GLYPH_COUNT):
        code = 0x20 + index
        offset = FONT_BASE + index * GLYPH_STRIDE
        columns = [memory.read_byte(offset + i) & 0x7F for i in range(GLYPH_WIDTH)]
        lookup[tuple(columns)] = chr(code)
    return lookup


def _font_lookup(memory) -> Dict[Tuple[int, ...], str]:
    # Avoid caching complexities; reading 96 glyphs is cheap
    return _read_glyph_columns(memory)


def decode_display_text(controller, memory) -> List[str]:
    """Decode the current LCD VRAM into textual rows."""

    lookup = _font_lookup(memory)
    buffer = controller.get_display_buffer()
    if buffer is None:
        return []

    height, width = buffer.shape
    lines: List[str] = []

    for page in range(CHAR_ROWS):
        row_chars: List[str] = []
        row_base = page * 8
        if row_base + CHAR_HEIGHT > height:
            break

        for char_index in range(CHAR_COLUMNS):
            col_base = char_index * PIXELS_PER_CHAR_COL
            if col_base + GLYPH_WIDTH > width:
                row_chars.append(" ")
                continue

            columns: List[int] = []
            for glyph_col in range(GLYPH_WIDTH):
                bits = 0
                col = col_base + glyph_col
                for row in range(CHAR_HEIGHT):
                    if not buffer[row_base + row, col]:
                        bits |= 1 << row
                columns.append(bits & 0x7F)

            glyph = tuple(columns)
            row_chars.append(lookup.get(glyph, "?"))

        lines.append("".join(row_chars).rstrip())

    return lines


__all__ = ["decode_display_text"]
