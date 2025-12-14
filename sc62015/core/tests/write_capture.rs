// PY_SOURCE: pce500/memory.py:PCE500Memory

use sc62015_core::memory::MemoryImage;
use sc62015_core::INTERNAL_MEMORY_START;

#[test]
fn write_capture_records_last_value_for_external_internal_and_overlay_writes() {
    let mut mem = MemoryImage::new();
    mem.add_ram_overlay(0x005000, 8, "Test_RAM");

    mem.begin_write_capture();

    // External
    let _ = mem.store(0x001234, 8, 0x11);
    let _ = mem.store(0x001234, 8, 0x22);

    // Internal
    let _ = mem.store(INTERNAL_MEMORY_START + 0x10, 8, 0x33);

    // Overlay
    let _ = mem.store(0x005001, 8, 0x44);
    let _ = mem.store(0x005001, 8, 0x55);

    let captured = mem.take_write_capture();
    assert_eq!(
        captured,
        vec![(0x001234, 0x22), (0x005001, 0x55), (INTERNAL_MEMORY_START + 0x10, 0x33)]
    );
}

