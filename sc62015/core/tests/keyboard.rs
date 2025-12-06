// PY_SOURCE: pce500/tests/test_keyboard_handler.py
// PY_SOURCE: pce500/tests/test_keyboard_matrix.py
use sc62015_core::keyboard::KeyboardMatrix;
use sc62015_core::memory::MemoryImage;

#[test]
fn scan_tick_populates_fifo_and_sets_keyi() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Press a key; no FIFO events yet (scan is timer-driven).
    kb.press_matrix_code(0x10, &mut mem);
    assert_eq!(kb.fifo_len(), 0);

    // Strobe columns, then run a scan tick and mirror FIFO/KEYI to memory.
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);
    let mut events = 0;
    for _ in 0..8 {
        events += kb.scan_tick();
        if events > 0 {
            break;
        }
    }
    assert!(
        events > 0,
        "scan_tick should detect the pressed key after debounce"
    );
    kb.write_fifo_to_memory(&mut mem, true);
    assert!(kb.fifo_len() > 0);
    let isr = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(isr & 0x04, 0, "KEYI should be set after scan");
}

#[test]
fn keyi_is_not_reasserted_by_kil_read_without_host() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Seed a press and force a scan via KOL to populate FIFO and set KEYI.
    kb.press_matrix_code(0x10, &mut mem);
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);
    for _ in 0..8 {
        if kb.scan_tick() > 0 {
            break;
        }
    }
    kb.write_fifo_to_memory(&mut mem, true);
    assert!(kb.fifo_len() > 0);
    let isr_after_scan = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(isr_after_scan & 0x04, 0, "KEYI should be set after scan");

    // Clear ISR (e.g., by firmware) while FIFO is still non-empty.
    mem.write_internal_byte(0xFC, 0x00);
    assert_eq!(mem.read_internal_byte(0xFC).unwrap_or(0) & 0x04, 0);

    // A subsequent KIL read alone should not reassert KEYI; bus/host logic handles that.
    let _kil = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    let isr_reassert = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_eq!(
        isr_reassert & 0x04,
        0,
        "Keyboard read should not reassert KEYI without host involvement"
    );
}

#[test]
fn kil_read_does_not_generate_events_without_scan() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Press a key but do not run scan_tick yet.
    kb.press_matrix_code(0x10, &mut mem);
    assert_eq!(kb.fifo_len(), 0);
    assert_eq!(mem.read_internal_byte(0xFC).unwrap_or(0) & 0x04, 0);

    // KIL read should reflect latch but not enqueue events or assert KEYI.
    let _ = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    assert_eq!(kb.fifo_len(), 0);
    assert_eq!(mem.read_internal_byte(0xFC).unwrap_or(0) & 0x04, 0);
}
