from __future__ import annotations

from pce500.display.controller_wrapper import HD61202Controller
from pce500.memory import PCE500Memory


def test_lcd_reads_return_status_and_data():
    lcd = HD61202Controller()

    # Status read: left chip, instruction, read (0x2009).
    lcd.chips[0].state.on = False
    lcd.chips[0].state.busy = True
    assert lcd.read(0x2009) == 0xA0

    # Data read: left chip, data, read (0x200B) returns buffered column.
    lcd.chips[0].state.page = 0
    lcd.chips[0].state.y_address = 1
    lcd.chips[0].vram[0][0] = 0x12
    assert lcd.read(0x200B) == 0x12


def test_lcd_reads_fall_back_for_cs_both():
    mem = PCE500Memory()
    lcd = HD61202Controller()
    mem.set_lcd_controller(lcd, enable_overlay=True)

    addr = 0x2001  # CS=both, instruction, read
    mem.external_memory[addr] = 0x55
    assert mem.read_byte(addr) == 0x55
