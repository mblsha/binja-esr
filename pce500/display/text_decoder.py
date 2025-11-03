"""Utilities to decode LCD controller VRAM into ASCII text."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .font import GLYPH_HEIGHT, GLYPH_WIDTH, glyph_columns

CHAR_COLUMNS = 40
CHAR_ROWS = 4
PIXELS_PER_CHAR_COL = 6  # 5 glyph columns + 1 spacer
CHAR_HEIGHT = GLYPH_HEIGHT


def decode_display_text(controller, memory) -> List[str]:
    """Decode the current LCD VRAM into textual rows."""

    reverse_lookup: Dict[Tuple[int, ...], str] = {}
    for code in range(0x20, 0x20 + 96):
        char = chr(code)
        reverse_lookup[glyph_columns(memory, char)] = char

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
                for row in range(GLYPH_HEIGHT):
                    if not buffer[row_base + row, col]:
                        bits |= 1 << row
                columns.append(bits & 0x7F)

            glyph = tuple(columns)
            row_chars.append(reverse_lookup.get(glyph, "?"))

        lines.append("".join(row_chars).rstrip())

    return lines


__all__ = ["decode_display_text"]
