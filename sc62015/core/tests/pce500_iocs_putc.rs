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
const IOCS_LCD_PUTC: u8 = 0x0D;

fn default_rom_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pc-e500.bin")
}

fn call_with_sentinel(rt: &mut CoreRuntime, addr: u32, max_instructions: u32) {
    let before_pc = rt.state.pc();
    let before_sp = rt.state.get_reg(RegName::S);

    // Use a sentinel return address so we can detect when the called routine returns.
    // Keep the sentinel within the same 64KiB bank to avoid altering the bank bits.
    let sentinel_low16: u32 = 0xD00D;
    let sentinel_pc = ((addr & 0x0f_0000) | sentinel_low16) & 0x000f_ffff;

    // Push the sentinel return address (little-endian 3-byte PC) onto the stack.
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

#[test]
fn pce500_iocs_putc_emits_lcd_writes() {
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
        .as_deref()
        .map(|lcd| decode_display_text(lcd, &font))
        .expect("lcd present");

    rt.lcd
        .as_deref_mut()
        .expect("lcd present")
        .begin_display_write_capture();
    rt.state.set_reg(RegName::A, b'~' as u32);
    rt.state.set_reg(RegName::IL, IOCS_LCD_PUTC as u32);
    rt.memory.write_internal_byte(0xD6, 0);
    rt.memory.write_internal_byte(0xD7, 0);
    call_with_sentinel(&mut rt, IOCS_PUBLIC_ENTRY_ADDR, 50_000);
    rt.step(5_000).expect("execute after IOCS call");

    let (lcd_writes, after_lines) = rt
        .lcd
        .as_deref_mut()
        .map(|lcd| {
            (
                lcd.take_display_write_capture(),
                decode_display_text(lcd, &font),
            )
        })
        .expect("lcd present");
    assert!(
        !lcd_writes.is_empty(),
        "expected IOCS putc to write to LCD, got 0 writes"
    );

    assert_ne!(
        before_lines, after_lines,
        "expected IOCS putc to change decoded LCD text"
    );
    assert!(
        after_lines.iter().any(|l| l.contains('~')),
        "expected IOCS putc to draw '~', got {after_lines:?}"
    );
}
