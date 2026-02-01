// PY_SOURCE: pce500/tests/test_peripherals.py
// PY_SOURCE: pce500/tests/test_lcd_read_parity.py
use sc62015_core::keyboard::KeyboardMatrix;
use sc62015_core::memory::MemoryImage;

#[test]
fn eport_reads_return_zero_and_writes_ignored() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();
    // E-port offsets 0xF5/0xF6 fall back to memory so the host can drive ON/ONK inputs.
    assert_eq!(kb.handle_read(0xF5, &mut mem), None);
    assert_eq!(kb.handle_read(0xF6, &mut mem), None);
    assert!(!kb.handle_write(0xF5, 0xAA, &mut mem));
    assert!(!kb.handle_write(0xF6, 0xBB, &mut mem));
    // Bus fallback writes should stick in IMEM.
    let _ = mem.store(sc62015_core::INTERNAL_MEMORY_START + 0xF5, 8, 0xAA);
    let _ = mem.store(sc62015_core::INTERNAL_MEMORY_START + 0xF6, 8, 0xBB);
    assert_eq!(mem.read_internal_byte(0xF5), Some(0xAA));
    assert_eq!(mem.read_internal_byte(0xF6), Some(0xBB));
}

#[test]
fn lcd_reads_return_vram_and_advance() {
    use sc62015_core::lcd::LcdController;

    let mut lcd = LcdController::new();
    let instr_right = 0x2004;
    let data_right = 0x2006;
    let read_right = 0x2007;

    // Page 0, Y=0, write then read should return the prior column.
    lcd.write(instr_right, 0x80); // SetPage 0
    lcd.write(instr_right, 0x40); // SetY 0
    lcd.write(data_right, 0x12); // write to col 0 (Y -> 1)
    assert_eq!(lcd.read(read_right), Some(0x12));

    // Y=63 write should wrap and read back from col 63 when Y==0.
    lcd.write(instr_right, 0x7F); // SetY 63
    lcd.write(data_right, 0xAB); // write to col 63 (Y -> 0)
    assert_eq!(lcd.read(read_right), Some(0xAB));
}
