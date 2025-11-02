"""Tests for the LCD text decoder."""

from __future__ import annotations

from pathlib import Path

import pytest

from pce500 import PCE500Emulator
from pce500.display.text_decoder import decode_display_text


def _load_rom() -> bytes:
    rom_path = Path(__file__).resolve().parent.parent.parent / "data" / "pc-e500.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM image missing at {rom_path}")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip(f"Expected 1MB pc-e500.bin at {rom_path}")
    return data


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
