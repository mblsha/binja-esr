// PY_SOURCE: pce500/tests/test_peripherals.py
// PY_SOURCE: pce500/tests/test_lcd_read_parity.py
use sc62015_core::keyboard::KeyboardMatrix;
use sc62015_core::memory::MemoryImage;

#[test]
fn eport_reads_return_zero_and_writes_ignored() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();
    // E-port offsets 0xF5/0xF6 should return 0 and not mutate internal RAM.
    assert_eq!(kb.handle_read(0xF5, &mut mem), Some(0));
    assert_eq!(kb.handle_read(0xF6, &mut mem), Some(0));
    kb.handle_write(0xF5, 0xAA, &mut mem);
    kb.handle_write(0xF6, 0xBB, &mut mem);
    assert_eq!(mem.read_internal_byte(0xF5), Some(0));
    assert_eq!(mem.read_internal_byte(0xF6), Some(0));
}

#[test]
fn lcd_reads_stub_to_ff() {
    use sc62015_core::lcd::LcdController;

    let mut lcd = LcdController::new();
    // Instruction read path: returns ON flag only (default off => 0).
    assert_eq!(lcd.read(0x2001), Some(0x00));
    // Data read path: default VRAM returns 0 and advances Y.
    assert_eq!(lcd.read(0x2003), Some(0x00));
}
