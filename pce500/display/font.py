"""Shared helpers for working with the PC-E500 ROM font."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

FONT_BASE = 0x00F2215
JP_FONT_ATLAS_BASE = 0x00F21A5
GLYPH_WIDTH = 5
GLYPH_STRIDE = 6  # five data columns + spacer
GLYPH_COUNT = 96  # ASCII 0x20-0x7F
GLYPH_HEIGHT = 7

_JP_SENTINELS = (
    (0x00F232B, (0x7C, 0x12, 0x11, 0x12, 0x7C)),  # 'A'
    (0x00F2589, (0x0A, 0x4A, 0x4A, 0x2A, 0x1E)),  # ｦ / ヲ
)
_JP_SYMBOL_LABELS = {
    0xE8: "✳",
    0xEC: "●",
    0xEF: "/",
}
_JP_DISPLAY_ALIASES = {
    (0x00, 0x28, 0x6C, 0x6C, 0x28): "↕",
    (0x00, 0x7F, 0x7F, 0x41, 0x00): "[",
    (0x00, 0x41, 0x7F, 0x7F, 0x00): "]",
}
_FULLWIDTH_TO_HALFWIDTH = {
    unicodedata.normalize("NFKC", bytes([code]).decode("cp932")): bytes([code]).decode("cp932")
    for code in range(0xA1, 0xE0)
}
_FONT_LOOKUP_CACHE: Dict[Tuple[object, ...], Dict[str, Tuple[int, ...]]] = {}
_FONT_REVERSE_LOOKUP_CACHE: Dict[Tuple[object, ...], Dict[Tuple[int, ...], str]] = {}


def _read_bytes(memory, address: int, count: int) -> Tuple[int, ...]:
    values: List[int] = []
    for offset in range(count):
        try:
            values.append(memory.read_byte(address + offset) & 0xFF)
        except Exception:
            values.append(0)
    return tuple(values)


def _cache_key(memory) -> Tuple[object, ...]:
    return (
        id(memory),
        _read_bytes(memory, FONT_BASE, GLYPH_WIDTH),
        _read_bytes(memory, 0x00F232B, GLYPH_WIDTH),
        _read_bytes(memory, 0x00F2589, GLYPH_WIDTH),
    )


def _looks_like_jp_font(memory) -> bool:
    return all(
        _read_bytes(memory, address, GLYPH_WIDTH) == expected
        for address, expected in _JP_SENTINELS
    )


def _is_blank(pattern: Tuple[int, ...]) -> bool:
    return all((byte & 0x7F) == 0 for byte in pattern)


def _read_glyph_columns(memory, index: int, *, font_base: int = FONT_BASE) -> Tuple[int, ...]:
    """Return the raw column bytes for a glyph index from a specific font base."""
    if not (0 <= index <= 0xFF):
        raise ValueError(f"Glyph index out of range: {index}")

    offset = font_base + index * GLYPH_STRIDE
    return tuple(memory.read_byte(offset + i) & 0x7F for i in range(GLYPH_WIDTH))


def _font_lookup(memory) -> Dict[str, Tuple[int, ...]]:
    """Build (and cache) the mapping from display characters to column tuples."""
    key = _cache_key(memory)
    cached = _FONT_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached

    mapping: Dict[str, Tuple[int, ...]] = {}

    if _looks_like_jp_font(memory):
        for code in range(0x20, 0x7F):
            pattern = _read_glyph_columns(memory, code, font_base=JP_FONT_ATLAS_BASE)
            if _is_blank(pattern) and code != 0x20:
                continue
            mapping[chr(code)] = pattern

        for code in range(0xA1, 0xE0):
            pattern = _read_glyph_columns(memory, code, font_base=JP_FONT_ATLAS_BASE)
            if _is_blank(pattern):
                continue
            mapping[bytes([code]).decode("cp932")] = pattern

        for code, label in _JP_SYMBOL_LABELS.items():
            pattern = _read_glyph_columns(memory, code, font_base=JP_FONT_ATLAS_BASE)
            if _is_blank(pattern):
                continue
            mapping[label] = pattern
    else:
        for index in range(GLYPH_COUNT):
            char_code = 0x20 + index
            pattern = _read_glyph_columns(memory, index)
            if _is_blank(pattern) and char_code != 0x20:
                continue
            mapping[chr(char_code)] = pattern

    _FONT_LOOKUP_CACHE[key] = mapping
    return mapping


def font_reverse_lookup(memory) -> Dict[Tuple[int, ...], str]:
    """Return the glyph-pattern to character mapping for the active ROM layout."""
    key = _cache_key(memory)
    cached = _FONT_REVERSE_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached

    reverse: Dict[Tuple[int, ...], str] = {}
    for char, pattern in _font_lookup(memory).items():
        reverse.setdefault(pattern, char)
        inverted = tuple((~column) & 0x7F for column in pattern)
        reverse.setdefault(inverted, char)
    for pattern, char in _JP_DISPLAY_ALIASES.items():
        reverse.setdefault(pattern, char)
        inverted = tuple((~column) & 0x7F for column in pattern)
        reverse.setdefault(inverted, char)

    _FONT_REVERSE_LOOKUP_CACHE[key] = reverse
    return reverse


def glyph_columns(memory, char: str) -> Tuple[int, ...]:
    """Return the ROM column data for a single display character."""
    if not char:
        raise ValueError("Character must be non-empty")
    lookup = _font_lookup(memory)
    if char not in lookup:
        alias = _FULLWIDTH_TO_HALFWIDTH.get(char)
        if alias and alias in lookup:
            return lookup[alias]
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
