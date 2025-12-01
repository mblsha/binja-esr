use sc62015_core::keyboard::KeyboardMatrix;
use sc62015_core::memory::MemoryImage;

#[test]
fn kol_write_triggers_scan_and_keyi() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Press a key; FIFO should contain one press entry.
    kb.press_matrix_code(0x10, &mut mem);
    assert_eq!(kb.fifo_len(), 1);

    // KOL/KOH writes should trigger scan_tick; a subsequent KIL read should reflect latch.
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);
    let kil = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    assert_ne!(kil, 0);
}
