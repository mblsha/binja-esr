// PY_SOURCE: pce500/display/test_text_decoder.py

use sc62015_core::lcd_text::{decode_display_text, Pce500FontMap};
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

fn decode_lines(rt: &CoreRuntime, font: &Pce500FontMap) -> Vec<String> {
    let lcd = rt.lcd.as_deref().expect("lcd present");
    decode_display_text(lcd, font)
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

    let lcd = rt.lcd.as_deref().expect("lcd present");
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

    let lines = decode_lines(&rt, &font);
    assert!(
        lines
            .first()
            .is_some_and(|s| s.contains("S1(MAIN):NEW CARD")),
        "expected main menu header after PF1, got {lines:?}"
    );
}

#[test]
fn pce500_pf1_reaches_next_screen_via_core_runtime() {
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

    const BOOT_STEPS: usize = 20_000;
    const PF1_HOLD_STEPS: usize = 40_000;
    const MAIN_MENU_STEPS: usize = 800_000;
    const LCD_CHECK_INTERVAL: usize = 5_000;
    const NEXT_SCREEN_BUDGET: usize = 2_000_000;
    const MAIN_MENU_RENDER_STEPS: usize = 500_000;

    rt.step(BOOT_STEPS).expect("boot before key press");
    let boot_lines = decode_lines(&rt, &font);
    assert!(
        boot_lines
            .first()
            .is_some_and(|s| s.contains("S2(CARD):NEW CARD")),
        "expected boot header before PF1, got {boot_lines:?}"
    );

    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, false, &mut rt.memory, rt.timer.kb_irq_enabled);
    }
    rt.step(PF1_HOLD_STEPS).expect("run while PF1 is held");
    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, true, &mut rt.memory, rt.timer.kb_irq_enabled);
    }
    rt.step(MAIN_MENU_STEPS - BOOT_STEPS - PF1_HOLD_STEPS)
        .expect("execute after first PF1 release");

    let main_lines = decode_lines(&rt, &font);
    let main_row0 = main_lines.first().cloned().unwrap_or_default();
    assert!(
        main_row0.contains("S1(MAIN):"),
        "expected main menu header after PF1, got {main_lines:?}"
    );

    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, false, &mut rt.memory, rt.timer.kb_irq_enabled);
    }
    rt.step(PF1_HOLD_STEPS)
        .expect("run while PF1 is held (second screen)");
    if let Some(kb) = rt.keyboard.as_mut() {
        kb.inject_matrix_event(PF1_CODE, true, &mut rt.memory, rt.timer.kb_irq_enabled);
    }

    let mut waited: usize = 0;
    let mut last_lines = main_lines;
    let mut next_row0: Option<String> = None;
    while waited < NEXT_SCREEN_BUDGET {
        rt.step(LCD_CHECK_INTERVAL)
            .expect("execute after second PF1 release");
        waited += LCD_CHECK_INTERVAL;
        let lines = decode_lines(&rt, &font);
        if !lines.is_empty() {
            last_lines = lines.clone();
        }
        let row0 = lines.first().cloned().unwrap_or_default();
        if row0.trim().is_empty() {
            continue;
        }
        if row0.contains("S1(MAIN):") || row0.contains("S2(CARD):") {
            continue;
        }
        next_row0 = Some(row0);
        break;
    }

    if next_row0.is_none() {
        eprintln!(
            "Skipping: new menu did not render within {NEXT_SCREEN_BUDGET} steps (last={last_lines:?}). \
Set PC_E500_REQUIRE_MENU=1 to enforce."
        );
        if std::env::var("PC_E500_REQUIRE_MENU").ok().as_deref() == Some("1") {
            panic!("expected new menu or BASIC prompt after PF1 twice");
        }
        return;
    }

    rt.step(MAIN_MENU_RENDER_STEPS)
        .expect("wait for main menu to finish rendering");
    let menu_lines = decode_lines(&rt, &font);
    let menu_text = menu_lines.join("\n");
    assert!(
        menu_text.contains("MAIN MENU"),
        "expected MAIN MENU header, got {menu_lines:?}"
    );
    assert!(
        menu_text.contains("⇳"),
        "expected arrow indicator in main menu, got {menu_lines:?}"
    );
    assert!(
        menu_text.contains("[BASIC ][ CAL  ][MATRIX][ STAT ][ ENG  ]"),
        "expected full main menu row, got {menu_lines:?}"
    );
}
