from pce500.display.controller_wrapper import HD61202Controller


def test_write_tracks_chip_select_counters_without_magic_values() -> None:
    lcd = HD61202Controller()

    lcd.write(0xA000, 0x00)  # BOTH
    lcd.write(0xA004, 0x00)  # RIGHT
    lcd.write(0xA008, 0x00)  # LEFT

    assert lcd.cs_both_count == 1
    assert lcd.cs_right_count == 1
    assert lcd.cs_left_count == 1
