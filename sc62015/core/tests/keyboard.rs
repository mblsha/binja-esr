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

#[test]
fn keyi_reasserts_when_isr_cleared_with_pending_fifo() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Seed a press and force a scan via KOL to populate FIFO and set KEYI.
    kb.press_matrix_code(0x10, &mut mem);
    kb.handle_write(0xF0, 0xFF, &mut mem);
    assert!(kb.fifo_len() > 0);
    let isr_after_scan = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(isr_after_scan & 0x04, 0, "KEYI should be set after scan");

    // Clear ISR (e.g., by firmware) while FIFO is still non-empty.
    mem.write_internal_byte(0xFC, 0x00);
    assert_eq!(mem.read_internal_byte(0xFC).unwrap_or(0) & 0x04, 0);

    // A subsequent KIL read should reassert KEYI because events remain queued.
    let _kil = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    let isr_reassert = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(
        isr_reassert & 0x04,
        0,
        "KEYI should reassert when FIFO still holds events"
    );
}
