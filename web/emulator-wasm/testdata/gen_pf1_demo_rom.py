#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sc62015.pysc62015.sc_asm import Assembler

ROM_WINDOW_LEN = 0x40000
ROM_WINDOW_START = 0x0C0000
ROM_RESET_VECTOR_ADDR = 0x0FFFFD

FONT_BASE = 0x00F2215
FONT_WINDOW_OFFSET = FONT_BASE - ROM_WINDOW_START
GLYPH_WIDTH = 5
GLYPH_STRIDE = 6
GLYPH_COUNT = 96


def _glyph_bytes(ch: str) -> list[int]:
    idx = ord(ch) - 0x20
    if not (0 <= idx < GLYPH_COUNT):
        raise ValueError(f"unsupported char: {ch!r}")
    return [idx] * GLYPH_WIDTH + [0]


def _emit_write_bytes(values: list[int]) -> str:
    out: list[str] = []
    for value in values:
        out.append(f"    MV A, 0x{value:02X}")
        out.append("    MV [0x2006], A")
    return "\n".join(out)


def build_rom_window() -> bytes:
    boot: list[int] = []
    for ch in "BOOT":
        boot.extend(_glyph_bytes(ch))

    menu: list[int] = []
    for ch in "MENU":
        menu.extend(_glyph_bytes(ch))

    source = "\n".join(
        [
            ".ORG 0x0000",
            "start:",
            "    MV A, 0x00",
            "    MV (KOL), A",
            "    MV A, 0x04",
            "    MV (KOH), A",
            "",
            "    ; LCD init (right chip)",
            "    MV A, 0x01",
            "    MV [0x2004], A",
            "    MV A, 0x80",
            "    MV [0x2004], A",
            "    MV A, 0x40",
            "    MV [0x2004], A",
            "",
            _emit_write_bytes(boot),
            "",
            "poll_pf1:",
            "    MV A, (KIL)",
            "    AND A, 0x40",
            "    JPZ poll_pf1",
            "",
            "    ; Rewrite as 'MENU' after PF1",
            "    MV A, 0x80",
            "    MV [0x2004], A",
            "    MV A, 0x40",
            "    MV [0x2004], A",
            "",
            _emit_write_bytes(menu),
            "",
            "    HALT",
            "",
        ]
    )

    asm = Assembler()
    binfile = asm.assemble(source)
    program = binfile.as_binary()

    rom = bytearray([0] * ROM_WINDOW_LEN)
    if len(program) > ROM_WINDOW_LEN:
        raise ValueError(f"program too large ({len(program)} bytes)")
    rom[0 : len(program)] = program

    # Font table used by the LCD text decoder.
    table_len = GLYPH_COUNT * GLYPH_STRIDE
    if FONT_WINDOW_OFFSET + table_len > ROM_WINDOW_LEN:
        raise ValueError("font table would not fit in ROM window")
    for idx in range(GLYPH_COUNT):
        base = FONT_WINDOW_OFFSET + idx * GLYPH_STRIDE
        value = idx & 0x7F
        for i in range(GLYPH_WIDTH):
            rom[base + i] = value
        rom[base + GLYPH_WIDTH] = 0

    # Reset vector points to 0xC0000 (start of ROM window).
    vector = ROM_WINDOW_START.to_bytes(3, byteorder="little", signed=False)
    reset_offset = ROM_RESET_VECTOR_ADDR - ROM_WINDOW_START
    rom[reset_offset : reset_offset + 3] = vector

    return bytes(rom)


def main() -> None:
    out_path = Path(__file__).with_name("pf1_demo_rom_window.rom")
    out_path.write_bytes(build_rom_window())
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
