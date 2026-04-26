mod common;

use common::{boot_pce500, call_iocs, call_with_sentinel, load_pce500_font, write_u24};
use sc62015_core::keyboard::KeyboardMatrix;
use sc62015_core::lcd_text::decode_display_text;
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::memory::{IMEM_EIL_OFFSET, IMEM_RXD_OFFSET};

#[test]
fn stdo_stream_output_0d_brackets_immediate_and_maintenance_lcd_writes() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };

    rt.lcd
        .as_deref_mut()
        .expect("lcd present")
        .begin_display_write_capture();

    rt.state.set_reg(RegName::A, b'@' as u32);
    let result = call_iocs(&mut rt, 0x00, 0x000D, 50_000);
    assert!(result.returned, "STDO 0Dh did not return: {result:?}");

    let immediate_writes = rt
        .lcd
        .as_deref_mut()
        .expect("lcd present")
        .take_display_write_capture();

    rt.lcd
        .as_deref_mut()
        .expect("lcd present")
        .begin_display_write_capture();
    rt.step(5_000).expect("run deferred display maintenance");
    let deferred_writes = rt
        .lcd
        .as_deref_mut()
        .expect("lcd present")
        .take_display_write_capture();

    assert!(
        !immediate_writes.is_empty(),
        "expected stream output 0Dh to perform immediate LCD writes before the maintenance window"
    );
    assert!(
        deferred_writes.is_empty(),
        "stream output 0Dh should not require extra deferred LCD writes after its immediate flush; got {} writes",
        deferred_writes.len()
    );
}

#[test]
fn stdo_stream_output_wraps_and_scrolls_multiple_characters() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };
    let Some(font) = load_pce500_font() else {
        return;
    };

    let clear = call_iocs(&mut rt, 0x00, 0x0051, 80_000);
    assert!(clear.returned, "STDO 51h did not return: {clear:?}");
    assert!(!clear.carry, "STDO 51h should clear the display: {clear:?}");

    let before = rt
        .lcd
        .as_deref()
        .map(|lcd| decode_display_text(lcd, &font))
        .expect("lcd present");
    let height = before.len();
    let width = before
        .iter()
        .map(|line| line.len())
        .max()
        .filter(|width| *width > 0)
        .unwrap_or(40);
    assert!(height >= 2, "expected at least two text lines: {before:?}");
    assert!(width >= 6, "expected a usable text width: {before:?}");

    let start_col = (width - 2) as u32;
    let start_row = (height - 1) as u32;
    let _ = rt.memory.store(0x00BFC27, 8, start_col);
    let _ = rt.memory.store(0x00BFC28, 8, start_row);
    let _ = rt.memory.store(0x00BFC9B, 8, start_col);
    let _ = rt.memory.store(0x00BFC9C, 8, start_row);

    for byte in b"ABCDEF" {
        rt.state.set_reg(RegName::A, u32::from(*byte));
        let result = call_iocs(&mut rt, 0x00, 0x000D, 80_000);
        assert!(
            result.returned,
            "STDO 0Dh did not return while writing {byte:02X}: {result:?}"
        );
        assert!(
            !result.carry,
            "STDO 0Dh should accept printable bytes: {result:?}"
        );
    }

    let after = rt
        .lcd
        .as_deref()
        .map(|lcd| decode_display_text(lcd, &font))
        .expect("lcd present");
    let penultimate = after
        .get(height - 2)
        .expect("penultimate line should be present")
        .as_bytes();
    let bottom = after
        .get(height - 1)
        .expect("bottom line should be present")
        .as_bytes();

    assert_eq!(
        penultimate.get(width - 2..width),
        Some(&b"AB"[..]),
        "wrapping at the bottom edge should scroll the old bottom-row tail up: {after:?}"
    );
    assert_eq!(
        bottom.get(0..4),
        Some(&b"CDEF"[..]),
        "wrapped stream output should continue at the left edge of the new bottom row: {after:?}"
    );
}

#[test]
fn stdi_keyboard_commands_consume_booted_rom_fifo() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };

    // The booted ROM initialises IOCS_WS to 0xBF9B4. Seed the ROM FIFO directly so
    // these calls exercise the STDI consumer path without relying on scan timing.
    write_u24(&mut rt, 0x00BFD17, 0x00BF9B4);
    write_u24(&mut rt, 0x1000E6, 0x00BF9B4);
    let _ = rt.memory.store(0x00BF9B6, 8, 0x50);
    let _ = rt.memory.store(0x00BF9B8, 8, 0x01);
    let _ = rt.memory.store(0x00BF9B9, 8, 0x00);
    let _ = rt.memory.store(0x00BFA04, 8, 0x21);

    let peek = call_iocs(&mut rt, 0x01, 0x0042, 50_000);
    assert!(peek.returned, "STDI 42h did not return: {peek:?}");
    assert!(
        !peek.carry,
        "STDI 42h should succeed for a queued key: {peek:?}"
    );
    assert_eq!(peek.a & 0xff, 0x21);
    assert_eq!(
        rt.memory.load(0x00BF9B9, 8),
        Some(0x00),
        "42h is non-destructive and must not advance FIFO head"
    );

    let read_matrix = call_iocs(&mut rt, 0x01, 0x0041, 50_000);
    assert!(
        read_matrix.returned,
        "STDI 41h did not return: {read_matrix:?}"
    );
    assert!(
        !read_matrix.carry,
        "STDI 41h should consume queued key: {read_matrix:?}"
    );
    assert_eq!(read_matrix.a & 0xff, 0x21);
    assert_eq!(
        rt.memory.load(0x00BF9B9, 8),
        Some(0x01),
        "41h must advance FIFO head"
    );

    let _ = rt.memory.store(0x00BF9B8, 8, 0x02);
    let _ = rt.memory.store(0x00BFA05, 8, 0x22);
    let read_key = call_iocs(&mut rt, 0x01, 0x0043, 50_000);
    assert!(read_key.returned, "STDI 43h did not return: {read_key:?}");
    assert!(
        !read_key.carry,
        "STDI 43h should translate/read queued key: {read_key:?}"
    );
    assert_ne!(rt.memory.load(0x00BF9B9, 8), Some(0x01));
}

#[test]
fn stdi_physical_key_timer_scan_reaches_booted_rom_fifo() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };

    write_u24(&mut rt, 0x00BFD17, 0x00BF9B4);
    write_u24(&mut rt, 0x1000E6, 0x00BF9B4);
    let _ = rt.memory.store(0x00BF9B6, 8, 0x50);
    let _ = rt.memory.store(0x00BF9B8, 8, 0x00);
    let _ = rt.memory.store(0x00BF9B9, 8, 0x00);

    let code = KeyboardMatrix::matrix_code_for_key_name("KEY_Q").expect("KEY_Q matrix code");
    let col = code >> 3;
    {
        let keyboard = rt.keyboard.as_mut().expect("keyboard present");
        keyboard.set_press_threshold(1);
        keyboard.set_repeat_enabled(false);
        if col < 8 {
            keyboard.handle_write(0xF0, 1 << col, &mut rt.memory);
        } else {
            keyboard.handle_write(0xF1, 1 << (col - 8), &mut rt.memory);
        }
        keyboard.press_matrix_code(code, &mut rt.memory);
        let events = keyboard.scan_tick(&mut rt.memory, true);
        assert!(
            events > 0,
            "physical KEY_Q press should debounce into a scanner FIFO event"
        );
        let drained = keyboard.drain_fifo_to_pce500_iocs_workspace(&mut rt.memory, true);
        assert_eq!(
            drained, 1,
            "scanner FIFO event should be mirrored into the PC-E500 IOCS ring buffer"
        );
    }

    assert_eq!(rt.memory.load(0x00BFA04, 8), Some(u32::from(code)));
    assert_eq!(rt.memory.load(0x00BF9B8, 8), Some(0x01));

    let read_matrix = call_iocs(&mut rt, 0x01, 0x0041, 50_000);
    assert!(
        read_matrix.returned,
        "STDI 41h did not return after physical scan: {read_matrix:?}"
    );
    assert!(
        !read_matrix.carry,
        "STDI 41h should consume the physical scanner event: {read_matrix:?}"
    );
    assert_eq!(read_matrix.a & 0xff, u32::from(code));
    assert_eq!(
        rt.memory.load(0x00BF9B9, 8),
        Some(0x01),
        "ROM FIFO head should advance after consuming the physical key event"
    );
}

#[test]
fn com_device_commands_41h_to_4bh_use_rom_dispatch_and_sio_contract() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };
    rt.enable_sio_stub();

    rt.state.set_reg(RegName::A, 0x5A);
    let direct_output = call_iocs(&mut rt, 0x02, 0x0041, 80_000);
    assert!(
        direct_output.returned,
        "COM 41h direct output did not return: {direct_output:?}"
    );

    rt.sio
        .as_mut()
        .expect("sio enabled")
        .set_auto_response(0x33);
    let direct_input = call_iocs(&mut rt, 0x02, 0x0042, 80_000);
    assert!(
        direct_input.returned,
        "COM 42h direct input did not return: {direct_input:?}"
    );
    assert_eq!(
        rt.memory.read_internal_byte(IMEM_RXD_OFFSET),
        Some(0x33),
        "COM 42h should observe the configured direct-input response"
    );

    rt.sio
        .as_mut()
        .expect("sio enabled")
        .set_direct_input_timeout(true);
    let timeout = call_iocs(&mut rt, 0x02, 0x0042, 80_000);
    assert!(
        timeout.returned,
        "COM 42h timeout did not return: {timeout:?}"
    );
    assert!(
        timeout.carry,
        "COM 42h timeout should return carry set: {timeout:?}"
    );
    assert_eq!(timeout.a & 0xff, 0x00);
    rt.sio
        .as_mut()
        .expect("sio enabled")
        .set_direct_input_timeout(false);

    for command in 0x0044..=0x0049 {
        let result = call_iocs(&mut rt, 0x02, command, 80_000);
        assert!(
            result.returned,
            "COM {command:02X}h line-control command did not return: {result:?}"
        );
    }

    rt.sio
        .as_ref()
        .expect("sio enabled")
        .set_input_lines(&mut rt.memory, Some(true), Some(false));
    let read_cs = call_iocs(&mut rt, 0x02, 0x004A, 80_000);
    assert!(read_cs.returned, "COM 4Ah did not return: {read_cs:?}");
    assert_ne!(
        rt.memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0) & 0x04,
        0
    );

    let read_cd = call_iocs(&mut rt, 0x02, 0x004B, 80_000);
    assert!(read_cd.returned, "COM 4Bh did not return: {read_cd:?}");
    assert_eq!(
        rt.memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0) & 0x02,
        0
    );
}

#[test]
fn cas_block_commands_use_rom_dispatch_and_deterministic_tape_bridge() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };
    rt.enable_pce500_peripheral_bridge(64 * 1024);

    let buffer = 0x00B9000;
    rt.state.set_reg(RegName::X, buffer);
    for offset in 0..0x30u32 {
        let _ = rt.memory.store(buffer + offset, 8, offset & 0xff);
    }
    let write_header = call_iocs(&mut rt, 0x04, 0x0044, 120_000);
    assert!(
        write_header.returned,
        "CAS 44h did not return: {write_header:?}"
    );
    assert!(
        !write_header.carry,
        "CAS 44h should append a header block: {write_header:?}"
    );

    rt.state.set_reg(RegName::X, buffer + 0x100);
    let read_header = call_iocs(&mut rt, 0x04, 0x0045, 120_000);
    assert!(
        read_header.returned,
        "CAS 45h did not return: {read_header:?}"
    );
    assert!(
        !read_header.carry,
        "CAS 45h should read the queued header block: {read_header:?}"
    );
    assert_eq!(rt.memory.load(buffer + 0x100 + 0x2f, 8), Some(0x2f));

    rt.state.set_reg(RegName::X, buffer + 0x200);
    rt.state.set_reg(RegName::Y, 5);
    for (offset, byte) in b"HELLO".iter().enumerate() {
        let _ = rt
            .memory
            .store(buffer + 0x200 + offset as u32, 8, u32::from(*byte));
    }
    let write_data = call_iocs(&mut rt, 0x04, 0x0041, 120_000);
    assert!(
        write_data.returned,
        "CAS 41h did not return: {write_data:?}"
    );
    assert!(
        !write_data.carry,
        "CAS 41h should append a data block: {write_data:?}"
    );

    rt.state.set_reg(RegName::X, buffer + 0x300);
    rt.state.set_reg(RegName::Y, 5);
    let read_data = call_iocs(&mut rt, 0x04, 0x0042, 120_000);
    assert!(read_data.returned, "CAS 42h did not return: {read_data:?}");
    assert!(
        !read_data.carry,
        "CAS 42h should read the queued data block: {read_data:?}"
    );
    assert_eq!(rt.memory.load(buffer + 0x300, 8), Some(u32::from(b'H')));

    rt.pce500_peripherals
        .as_mut()
        .expect("cassette bridge enabled")
        .cassette
        .append_data(b"HELLO")
        .expect("seed verify block");
    for (offset, byte) in b"HELLO".iter().enumerate() {
        let _ = rt
            .memory
            .store(buffer + 0x400 + offset as u32, 8, u32::from(*byte));
    }
    rt.state.set_reg(RegName::X, buffer + 0x400);
    rt.state.set_reg(RegName::Y, 5);
    let verify_data = call_iocs(&mut rt, 0x04, 0x0043, 120_000);
    assert!(
        verify_data.returned,
        "CAS 43h did not return: {verify_data:?}"
    );
    assert!(
        !verify_data.carry,
        "CAS 43h should verify the queued data block: {verify_data:?}"
    );
}

#[test]
fn storage_iocs_commands_use_rom_dispatch_and_deterministic_card_bridge() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };
    rt.enable_pce500_peripheral_bridge(64);

    let name = 0x00B9000;
    for (offset, byte) in b"FOO        ".iter().enumerate() {
        let _ = rt.memory.store(name + offset as u32, 8, u32::from(*byte));
    }
    rt.state.set_reg(RegName::X, name);
    rt.state.set_reg(RegName::Y, 8);

    let create = call_iocs(&mut rt, 0x06, 0x0045, 120_000);
    assert!(create.returned, "S1 45h did not return: {create:?}");
    assert!(!create.carry, "S1 45h should create the block: {create:?}");

    let duplicate = call_iocs(&mut rt, 0x06, 0x0045, 120_000);
    assert!(
        duplicate.returned,
        "duplicate S1 45h did not return: {duplicate:?}"
    );
    assert!(
        duplicate.carry,
        "duplicate S1 45h should report an error: {duplicate:?}"
    );

    let search = call_iocs(&mut rt, 0x06, 0x0041, 120_000);
    assert!(search.returned, "S1 41h did not return: {search:?}");
    assert!(
        !search.carry,
        "S1 41h should find the created block: {search:?}"
    );

    rt.state.set_reg(RegName::Y, 4);
    let resize = call_iocs(&mut rt, 0x06, 0x0042, 120_000);
    assert!(resize.returned, "S1 42h did not return: {resize:?}");
    assert!(!resize.carry, "S1 42h should resize the block: {resize:?}");

    let delete = call_iocs(&mut rt, 0x06, 0x0046, 120_000);
    assert!(delete.returned, "S1 46h did not return: {delete:?}");
    assert!(!delete.carry, "S1 46h should delete the block: {delete:?}");

    let missing = call_iocs(&mut rt, 0x06, 0x0041, 120_000);
    assert!(
        missing.returned,
        "missing S1 41h did not return: {missing:?}"
    );
    assert!(
        missing.carry,
        "S1 41h should report missing block after delete: {missing:?}"
    );

    rt.state.set_reg(RegName::Y, 6);
    let create_top = call_iocs(&mut rt, 0x06, 0x0048, 120_000);
    assert!(create_top.returned, "S1 48h did not return: {create_top:?}");
    assert!(
        !create_top.carry,
        "S1 48h should create a top block: {create_top:?}"
    );

    let condense = call_iocs(&mut rt, 0x06, 0x0047, 120_000);
    assert!(condense.returned, "S1 47h did not return: {condense:?}");
    assert!(
        !condense.carry,
        "S1 47h condense should succeed: {condense:?}"
    );

    rt.pce500_peripherals
        .as_mut()
        .expect("storage bridge enabled")
        .card
        .create("RAMFILE", 5, true)
        .expect("seed RAMFILE");

    rt.state.set_reg(RegName::X, name + 0x100);
    rt.state.set_reg(RegName::Y, 5);
    for (offset, byte) in b"DISK!".iter().enumerate() {
        let _ = rt
            .memory
            .store(name + 0x100 + offset as u32, 8, u32::from(*byte));
    }
    rt.state.set_reg(RegName::I, 0x0013);
    let ram_write = call_with_sentinel(&mut rt, 0x00E493E, 120_000);
    assert!(
        ram_write.returned,
        "E:F:G 13h did not return: {ram_write:?}"
    );
    assert!(
        !ram_write.carry,
        "E:F:G 13h should write RAMFILE bytes: {ram_write:?}"
    );

    rt.state.set_reg(RegName::X, name + 0x200);
    rt.state.set_reg(RegName::Y, 5);
    rt.state.set_reg(RegName::I, 0x0012);
    let ram_read = call_with_sentinel(&mut rt, 0x00E493E, 120_000);
    assert!(ram_read.returned, "E:F:G 12h did not return: {ram_read:?}");
    assert!(
        !ram_read.carry,
        "E:F:G 12h should read RAMFILE bytes: {ram_read:?}"
    );
    assert_eq!(rt.memory.load(name + 0x200, 8), Some(u32::from(b'D')));

    rt.state.set_reg(RegName::Y, 7);
    rt.state.set_reg(RegName::I, 0x003F);
    let ram_format = call_with_sentinel(&mut rt, 0x00E493E, 120_000);
    assert!(
        ram_format.returned,
        "E:F:G 3Fh did not return: {ram_format:?}"
    );
    assert!(
        !ram_format.carry,
        "E:F:G 3Fh should resize/create RAMFILE: {ram_format:?}"
    );
    assert_eq!(
        rt.pce500_peripherals
            .as_ref()
            .expect("storage bridge enabled")
            .card
            .find("RAMFILE")
            .expect("RAMFILE exists")
            .size(),
        7
    );
}

#[test]
fn storage_file_style_20h_to_2fh_use_rom_dispatch_and_card_bridge() {
    let Some(mut rt) = boot_pce500() else {
        return;
    };
    rt.enable_pce500_peripheral_bridge(128);

    let name = 0x00B9000;
    let new_name = 0x00B9020;
    let buffer = 0x00B9100;
    let out = 0x00B9200;
    for (offset, byte) in b"BAR        ".iter().enumerate() {
        let _ = rt.memory.store(name + offset as u32, 8, u32::from(*byte));
    }
    for (offset, byte) in b"BAZ        ".iter().enumerate() {
        let _ = rt
            .memory
            .store(new_name + offset as u32, 8, u32::from(*byte));
    }
    for (offset, byte) in b"DATA!".iter().enumerate() {
        let _ = rt.memory.store(buffer + offset as u32, 8, u32::from(*byte));
    }

    rt.state.set_reg(RegName::X, name);
    rt.state.set_reg(RegName::Y, 5);
    let create = call_iocs(&mut rt, 0x06, 0x0020, 120_000);
    assert!(create.returned, "S1 20h did not return: {create:?}");
    assert!(
        !create.carry,
        "S1 20h should create/open a file: {create:?}"
    );

    rt.state.set_reg(RegName::X, name);
    let open = call_iocs(&mut rt, 0x06, 0x0021, 120_000);
    assert!(open.returned, "S1 21h did not return: {open:?}");
    assert!(!open.carry, "S1 21h should open the file: {open:?}");

    rt.state.set_reg(RegName::X, buffer);
    rt.state.set_reg(RegName::Y, 5);
    let write_block = call_iocs(&mut rt, 0x06, 0x0024, 120_000);
    assert!(
        write_block.returned,
        "S1 24h did not return: {write_block:?}"
    );
    assert!(
        !write_block.carry,
        "S1 24h should write a data block: {write_block:?}"
    );

    rt.state.set_reg(RegName::Y, 0);
    let seek = call_iocs(&mut rt, 0x06, 0x0029, 120_000);
    assert!(seek.returned, "S1 29h did not return: {seek:?}");
    assert!(!seek.carry, "S1 29h should seek to file start: {seek:?}");

    rt.state.set_reg(RegName::X, out);
    rt.state.set_reg(RegName::Y, 5);
    let peek = call_iocs(&mut rt, 0x06, 0x0028, 120_000);
    assert!(peek.returned, "S1 28h did not return: {peek:?}");
    assert!(
        !peek.carry,
        "S1 28h should non-destructively read bytes: {peek:?}"
    );
    assert_eq!(rt.memory.load(out, 8), Some(u32::from(b'D')));

    rt.state.set_reg(RegName::X, out + 0x20);
    rt.state.set_reg(RegName::Y, 5);
    let read_block = call_iocs(&mut rt, 0x06, 0x0023, 120_000);
    assert!(read_block.returned, "S1 23h did not return: {read_block:?}");
    assert!(
        !read_block.carry,
        "S1 23h should read and advance: {read_block:?}"
    );
    assert_eq!(rt.memory.load(out + 0x20, 8), Some(u32::from(b'D')));

    rt.state.set_reg(RegName::Y, 0);
    let seek_again = call_iocs(&mut rt, 0x06, 0x0029, 120_000);
    assert!(
        seek_again.returned,
        "second S1 29h did not return: {seek_again:?}"
    );
    assert!(!seek_again.carry);

    rt.state.set_reg(RegName::X, buffer);
    rt.state.set_reg(RegName::Y, 5);
    let verify = call_iocs(&mut rt, 0x06, 0x0027, 120_000);
    assert!(verify.returned, "S1 27h did not return: {verify:?}");
    assert!(!verify.carry, "S1 27h should verify file bytes: {verify:?}");

    rt.state.set_reg(RegName::Y, 0);
    let seek_for_byte = call_iocs(&mut rt, 0x06, 0x0029, 120_000);
    assert!(seek_for_byte.returned);
    let read_byte = call_iocs(&mut rt, 0x06, 0x0025, 120_000);
    assert!(read_byte.returned, "S1 25h did not return: {read_byte:?}");
    assert!(
        !read_byte.carry,
        "S1 25h should read one byte: {read_byte:?}"
    );
    assert_eq!(read_byte.a & 0xff, u32::from(b'D'));

    rt.state.set_reg(RegName::A, u32::from(b'?'));
    let write_byte = call_iocs(&mut rt, 0x06, 0x0026, 120_000);
    assert!(write_byte.returned, "S1 26h did not return: {write_byte:?}");
    assert!(
        !write_byte.carry,
        "S1 26h should write one byte: {write_byte:?}"
    );

    rt.state.set_reg(RegName::X, out + 0x40);
    let info = call_iocs(&mut rt, 0x06, 0x002A, 120_000);
    assert!(info.returned, "S1 2Ah did not return: {info:?}");
    assert!(!info.carry, "S1 2Ah should report file info: {info:?}");

    let change_dir = call_iocs(&mut rt, 0x06, 0x002B, 120_000);
    assert!(change_dir.returned, "S1 2Bh did not return: {change_dir:?}");
    assert!(!change_dir.carry, "S1 2Bh should be accepted by the bridge");

    rt.state.set_reg(RegName::X, name);
    let search = call_iocs(&mut rt, 0x06, 0x002C, 120_000);
    assert!(search.returned, "S1 2Ch did not return: {search:?}");
    assert!(!search.carry, "S1 2Ch should find the file: {search:?}");

    rt.state.set_reg(RegName::X, name);
    rt.state.set_reg(RegName::Y, new_name);
    let rename = call_iocs(&mut rt, 0x06, 0x002D, 120_000);
    assert!(rename.returned, "S1 2Dh did not return: {rename:?}");
    assert!(!rename.carry, "S1 2Dh should rename the file: {rename:?}");

    rt.state.set_reg(RegName::X, new_name);
    let delete = call_iocs(&mut rt, 0x06, 0x002E, 120_000);
    assert!(delete.returned, "S1 2Eh did not return: {delete:?}");
    assert!(!delete.carry, "S1 2Eh should delete the file: {delete:?}");

    let free = call_iocs(&mut rt, 0x06, 0x002F, 120_000);
    assert!(free.returned, "S1 2Fh did not return: {free:?}");
    assert!(!free.carry, "S1 2Fh should report free space: {free:?}");
}
