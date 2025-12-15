// PY_SOURCE: pce500/run_pce500.py

use sc62015_core::lcd_text::decode_display_text;
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::pce500::{
    load_pce500_rom_window, pce500_font_map_from_rom, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD,
};
use sc62015_core::CoreRuntime;
use std::fs;
use std::path::PathBuf;

const IOCS_PUBLIC_ENTRY_ADDR: u32 = 0x00FFFE8;

// TRM: "One character output to arbitrary position (41H)".
// Entry: (cx)=0000h, i=0041h, (bl)=x, (bh)=y, a=data.
const IOCS_PUTCHAR_XY: u16 = 0x0041;

fn default_rom_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pc-e500.bin")
}

fn call_with_sentinel(rt: &mut CoreRuntime, addr: u32, max_instructions: u32) {
    let before_pc = rt.state.pc();
    let before_sp = rt.state.get_reg(RegName::S);

    let sentinel_low16: u32 = 0xD00D;
    let sentinel_pc = ((addr & 0x0f_0000) | sentinel_low16) & 0x000f_ffff;

    let new_sp = before_sp.wrapping_sub(3) & 0x00ff_ffff;
    for i in 0..3u32 {
        let byte = (sentinel_pc >> (8 * i)) & 0xff;
        let _ = rt.memory.store(new_sp.wrapping_add(i), 8, byte);
    }
    rt.state.set_reg(RegName::S, new_sp);
    rt.state.set_pc(addr & 0x000f_ffff);

    let mut steps = 0;
    while steps < max_instructions {
        if rt.state.pc() == sentinel_pc {
            break;
        }
        if rt.step(1).is_err() {
            break;
        }
        steps += 1;
    }

    rt.state.set_pc(before_pc);
    rt.state.set_reg(RegName::S, before_sp);
}

fn iocs_putchar_xy(rt: &mut CoreRuntime, x: u8, y: u8, ch: u8) {
    // (cx)=0000h is passed via IMEM (cl/ch) on the SC62015: D6/D7.
    rt.memory.write_internal_byte(0xD6, 0);
    rt.memory.write_internal_byte(0xD7, 0);

    // (bl)/(bh) are IMEM D4/D5 per TRM register map.
    rt.memory.write_internal_byte(0xD4, x);
    rt.memory.write_internal_byte(0xD5, y);

    rt.state.set_reg(RegName::A, ch as u32);
    rt.state
        .set_reg(RegName::I, IOCS_PUTCHAR_XY as u32);
    call_with_sentinel(rt, IOCS_PUBLIC_ENTRY_ADDR, 50_000);

    // Allow the driver to complete any deferred LCD writes.
    rt.step(5_000).expect("execute after IOCS call");
}

fn byte_at_line_col(line: &str, col: usize) -> u8 {
    // Decoded LCD text is expected to be ASCII-like single-byte characters.
    // If a line is shorter than expected, treat missing cells as spaces.
    *line.as_bytes().get(col).unwrap_or(&b' ')
}

#[test]
fn pce500_iocs_putchar_xy_writes_expected_screen_cells() {
    let rom_path = default_rom_path();
    if !rom_path.exists() {
        eprintln!("Skipping: ROM not present at {}", rom_path.display());
        return;
    }

    let rom = fs::read(&rom_path).expect("read ROM");
    let font = pce500_font_map_from_rom(&rom).expect("load font map");

    let mut rt = CoreRuntime::new();
    rt.timer.enabled = true;
    rt.timer.mti_period = DEFAULT_MTI_PERIOD;
    rt.timer.sti_period = DEFAULT_STI_PERIOD;
    rt.timer.reset(rt.cycle_count());
    load_pce500_rom_window(&mut rt, &rom).expect("load ROM window");
    rt.power_on_reset();

    rt.step(20_000).expect("boot");

    let before_lines = rt
        .lcd
        .as_ref()
        .map(|lcd| decode_display_text(lcd, &font))
        .expect("lcd present");
    assert!(
        !before_lines.is_empty(),
        "expected at least one decoded text line"
    );

    let height = before_lines.len();
    let width = before_lines[0].len();
    assert!(height >= 1, "expected non-zero display height");
    assert!(width >= 2, "expected display width >= 2");

    let top_y = 0u8;
    let bottom_y = (height - 1) as u8;
    let left_x = 0u8;
    let right_x = (width - 1) as u8;

    rt.lcd
        .as_mut()
        .expect("lcd present")
        .begin_display_write_capture();

    iocs_putchar_xy(&mut rt, left_x, top_y, b'1');
    iocs_putchar_xy(&mut rt, right_x, top_y, b'2');
    iocs_putchar_xy(&mut rt, left_x, bottom_y, b'3');
    iocs_putchar_xy(&mut rt, right_x, bottom_y, b'4');

    let lcd_writes = rt
        .lcd
        .as_mut()
        .map(|lcd| lcd.take_display_write_capture())
        .expect("lcd present");
    assert!(
        !lcd_writes.is_empty(),
        "expected IOCS putchar_xy to write to LCD, got 0 writes"
    );

    let after_lines = rt
        .lcd
        .as_ref()
        .map(|lcd| decode_display_text(lcd, &font))
        .expect("lcd present");
    assert_eq!(
        after_lines.len(),
        height,
        "expected stable decoded line count"
    );

    assert_eq!(
        byte_at_line_col(&after_lines[top_y as usize], left_x as usize),
        b'1',
        "expected top-left cell to be '1'"
    );
    assert_eq!(
        byte_at_line_col(&after_lines[top_y as usize], right_x as usize),
        b'2',
        "expected top-right cell to be '2'"
    );
    assert_eq!(
        byte_at_line_col(&after_lines[bottom_y as usize], left_x as usize),
        b'3',
        "expected bottom-left cell to be '3'"
    );
    assert_eq!(
        byte_at_line_col(&after_lines[bottom_y as usize], right_x as usize),
        b'4',
        "expected bottom-right cell to be '4'"
    );
}
