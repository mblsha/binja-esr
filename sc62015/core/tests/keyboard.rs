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

    // Strobe columns, then run a scan tick and assert KEYI.
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);
    let mut events = 0;
    for _ in 0..8 {
        events += kb.scan_tick(&mut mem, true);
        if events > 0 {
            break;
        }
    }
    assert!(
        events > 0,
        "scan_tick should detect the pressed key after debounce"
    );
    assert!(
        kb.fifo_len() > 0,
        "FIFO should hold the event before KEYI assert"
    );
    kb.write_fifo_to_memory(&mut mem, true);
    assert!(
        kb.fifo_len() > 0,
        "FIFO should remain pending until the ROM consumes it"
    );
    let isr = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(isr & 0x04, 0, "KEYI should be set after scan");
    let _ = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    assert_eq!(kb.fifo_len(), 0, "FIFO should drain after KIL read");
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
        if kb.scan_tick(&mut mem, true) > 0 {
            break;
        }
    }
    kb.write_fifo_to_memory(&mut mem, true);
    assert!(
        kb.fifo_len() > 0,
        "FIFO should remain queued until KIL read"
    );
    let isr_after_scan = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_ne!(isr_after_scan & 0x04, 0, "KEYI should be set after scan");

    // Clear ISR (e.g., by firmware) after the KEYI latch was asserted.
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

#[test]
fn write_fifo_to_memory_keeps_events_when_irq_masked() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Populate one event into the FIFO and assert KEYI.
    kb.press_matrix_code(0x10, &mut mem);
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);
    for _ in 0..8 {
        if kb.scan_tick(&mut mem, true) > 0 {
            break;
        }
    }
    assert!(kb.fifo_len() > 0);
    kb.write_fifo_to_memory(&mut mem, false);
    assert!(
        kb.fifo_len() > 0,
        "FIFO should remain queued while IRQs are masked"
    );
    let isr = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_eq!(isr & 0x04, 0, "KEYI should not assert when masked");
    let _ = kb.handle_read(0xF2, &mut mem).unwrap_or(0);
    assert_eq!(kb.fifo_len(), 0, "KIL read should consume pending events");
}

#[test]
fn defaults_match_python_keyboard_matrix() {
    // Parity guard: ensure Rust defaults mirror pce500/keyboard_matrix.py.
    let kb = KeyboardMatrix::new();
    let snap = kb.snapshot_state();
    assert!(snap.columns_active_high);
    assert_eq!(snap.press_threshold, 6);
    assert_eq!(snap.release_threshold, 6);
    assert_eq!(snap.repeat_delay, 24);
    assert_eq!(snap.repeat_interval, 6);
    assert!(snap.scan_enabled);
    assert_eq!(snap.kol, 0x00);
    assert_eq!(snap.koh, 0x00);
    assert_eq!(snap.kil_latch, 0x00);
    assert_eq!(snap.fifo_len, 0);
    assert_eq!(snap.head, 0);
    assert_eq!(snap.tail, 0);
    assert_eq!(snap.fifo_len, 0);
    assert!(snap.fifo.iter().all(|b| *b == 0));
}

#[test]
fn held_key_does_not_generate_spurious_release_events() {
    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();
    // Disable repeats so FIFO length reflects only press/release transitions.
    kb.set_repeat_enabled(false);

    // Press a key and strobe all columns so it stays visible throughout the hold.
    kb.press_matrix_code(0x10, &mut mem);
    kb.handle_write(0xF0, 0xFF, &mut mem);
    kb.handle_write(0xF1, 0x07, &mut mem);

    // Debounce until the initial press event is enqueued.
    for _ in 0..8 {
        if kb.scan_tick(&mut mem, true) > 0 {
            break;
        }
    }
    assert_eq!(
        kb.fifo_len(),
        1,
        "expected exactly one press event after debounce"
    );

    // Hold the key for many more scan ticks: no release event should be generated.
    for _ in 0..200 {
        kb.scan_tick(&mut mem, true);
    }
    assert_eq!(
        kb.fifo_len(),
        1,
        "held key should not enqueue release events"
    );
    let fifo = kb.fifo_snapshot();
    assert_eq!(fifo.len(), 1);
    assert_eq!(
        fifo[0] & 0x80,
        0,
        "FIFO entry should be a press, not release"
    );
}
