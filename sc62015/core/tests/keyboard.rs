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
        events += kb.scan_tick(&mut mem, true);
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
        if kb.scan_tick(&mut mem, true) > 0 {
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

#[test]
fn fifo_syncs_head_from_memory_and_clears_keyi_when_drained() {
    const FIFO_HEAD_ADDR: u32 = 0x00BFC9D;

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
    kb.write_fifo_to_memory(&mut mem, true);
    let snap = kb.snapshot_state();
    assert!(snap.fifo_len > 0);

    // Firmware consumes the entry: advance head pointer in RAM and clear ISR.
    mem.write_external_byte(FIFO_HEAD_ADDR, snap.tail as u8);
    mem.write_internal_byte(0xFC, 0x00);

    // Next mirror should notice the drained head and avoid reasserting KEYI.
    kb.write_fifo_to_memory(&mut mem, true);
    assert_eq!(
        kb.fifo_len(),
        0,
        "FIFO should drop consumed events when head advances in RAM"
    );
    let isr = mem.read_internal_byte(0xFC).unwrap_or(0);
    assert_eq!(isr & 0x04, 0, "KEYI should stay clear after drain");
}

#[test]
fn inject_event_normalizes_fifo_head_only_for_main_menu_snapshot_shape() {
    const FIFO_HEAD_ADDR: u32 = 0x00BFC9D;
    const FIFO_TAIL_ADDR: u32 = 0x00BFC9E;
    const FIFO_BASE_ADDR: u32 = 0x00BFC96;

    let mut mem = MemoryImage::new();
    let mut kb = KeyboardMatrix::new();

    // Boot prompt shape: head can be 0x28 and tail=0x04; do not touch it (boot PF1 relies on it).
    mem.write_external_byte(FIFO_HEAD_ADDR, 0x28);
    mem.write_external_byte(FIFO_TAIL_ADDR, 0x04);
    kb.inject_matrix_event(0x56, false, &mut mem, true);
    assert_eq!(
        mem.read_byte(FIFO_HEAD_ADDR).unwrap_or(0),
        0x28,
        "boot prompt should not force-normalize FIFO_HEAD_ADDR"
    );

    // Main-menu snapshot shape (from `py_main_ready_555000`): FIFO bytes contain the PF1 press/release
    // history and head retains stale high bits.
    mem.write_external_byte(FIFO_HEAD_ADDR, 0x28);
    mem.write_external_byte(FIFO_TAIL_ADDR, 0x03);
    for (i, byte) in [0x56u8, 0xD6, 0xD6, 0, 0, 0, 0].into_iter().enumerate() {
        mem.write_external_byte(FIFO_BASE_ADDR + i as u32, byte);
    }
    kb.inject_matrix_event(0x56, false, &mut mem, true);
    assert_eq!(
        mem.read_byte(FIFO_HEAD_ADDR).unwrap_or(0),
        0x00,
        "main-menu snapshot should normalize FIFO_HEAD_ADDR to 0..7"
    );
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
