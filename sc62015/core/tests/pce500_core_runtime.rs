// PY_SOURCE: pce500/display/test_text_decoder.py

use sc62015_core::lcd_text::decode_display_text;
use sc62015_core::pce500::{
    load_pce500_rom_window, pce500_font_map_from_rom, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD,
    PF1_CODE,
};
use sc62015_core::CoreRuntime;
use std::fs;
use std::path::PathBuf;

fn default_rom_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pc-e500.bin")
}

#[test]
fn pce500_boots_and_decodes_lcd_text_via_core_runtime() {
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

    rt.step(20_000).expect("execute boot instructions");

    let lcd = rt.lcd.as_ref().expect("lcd present");
    let lines = decode_display_text(lcd, &font);
    assert!(
        lines
            .first()
            .is_some_and(|s| s.contains("S2(CARD):NEW CARD")),
        "expected boot header in row 0, got {lines:?}"
    );
    assert!(
        lines
            .get(2)
            .is_some_and(|s| s.contains("PF1 --- INITIALIZE")),
        "expected PF1 hint in row 2, got {lines:?}"
    );
    assert!(
        lines
            .get(3)
            .is_some_and(|s| s.contains("PF2 --- DO NOT INITIALIZE")),
        "expected PF2 hint in row 3, got {lines:?}"
    );
}

#[test]
fn pce500_pf1_changes_menu_header_via_core_runtime() {
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

    rt.step(15_000).expect("boot before key press");
    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, false, &mut rt.memory, rt.timer.kb_irq_enabled);
    }
    // Mirror the ROM test harness: hold PF1 for a while so firmware polling/IRQ delivery sees it.
    rt.step(40_000).expect("run while PF1 is held");
    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, true, &mut rt.memory, rt.timer.kb_irq_enabled);
    }
    rt.step(800_000 - 15_000 - 40_000)
        .expect("execute after PF1 release");

    let lcd = rt.lcd.as_ref().expect("lcd present");
    let lines = decode_display_text(lcd, &font);
    assert!(
        lines
            .first()
            .is_some_and(|s| s.contains("S1(MAIN):NEW CARD")),
        "expected main menu header after PF1, got {lines:?}"
    );
}
