// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/display/text_decoder.py:decode_display_text
// PY_SOURCE: pce500/display/font.py

use js_sys::Uint8Array;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use sc62015_core::lcd_text::{decode_display_text, Pce500FontMap};
use sc62015_core::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET};
use sc62015_core::{CoreRuntime, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};

const ROM_WINDOW_START: usize = 0xC0000;
const ROM_WINDOW_LEN: usize = 0x40000;
const DEFAULT_MTI_PERIOD: u64 = 500;
const DEFAULT_STI_PERIOD: u64 = 5000;

#[derive(Debug, Default, Clone, Serialize)]
struct TimerState {
    enabled: bool,
    mti_period: u64,
    sti_period: u64,
    next_mti: u64,
    next_sti: u64,
    kb_irq_enabled: bool,
}

#[derive(Debug, Default, Clone, Serialize)]
struct IrqState {
    pending: bool,
    in_interrupt: bool,
    source: Option<String>,
    irq_total: u32,
    irq_key: u32,
    irq_mti: u32,
    irq_sti: u32,
}

#[derive(Debug, Default, Clone, Serialize)]
struct DebugState {
    instruction_count: u64,
    cycle_count: u64,
    halted: bool,
    call_depth: u32,
    call_sub_level: u32,
    imr: u8,
    isr: u8,
    timer: TimerState,
    irq: IrqState,
}

#[wasm_bindgen]
pub struct Pce500Emulator {
    runtime: CoreRuntime,
    rom_image: Vec<u8>,
    font_map: Option<Pce500FontMap>,
}

#[wasm_bindgen]
impl Pce500Emulator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut emulator = Self {
            runtime: CoreRuntime::new(),
            rom_image: Vec::new(),
            font_map: None,
        };
        emulator.configure_timer(true, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD);
        emulator
    }

    pub fn has_rom(&self) -> bool {
        !self.rom_image.is_empty()
    }

    pub fn load_rom(&mut self, rom: &[u8]) -> Result<(), JsValue> {
        if rom.is_empty() {
            return Err(JsValue::from_str("ROM is empty"));
        }
        self.rom_image = rom.to_vec();
        let font = Pce500FontMap::from_rom(&self.rom_image);
        self.font_map = if font.is_empty() { None } else { Some(font) };
        self.reset()
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        if self.rom_image.is_empty() {
            return Err(JsValue::from_str("ROM not loaded"));
        }
        let rom = self.rom_image.clone();
        self.runtime = CoreRuntime::new();
        self.configure_timer(true, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD);
        self.load_pce500_rom_window(&rom)?;
        self.runtime.power_on_reset();
        Ok(())
    }

    pub fn step(&mut self, instructions: u32) -> Result<(), JsValue> {
        self.runtime
            .step(instructions as usize)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn instruction_count(&self) -> u64 {
        self.runtime.instruction_count()
    }

    pub fn cycle_count(&self) -> u64 {
        self.runtime.cycle_count()
    }

    pub fn get_reg(&self, name: &str) -> u32 {
        self.runtime.get_reg(name)
    }

    pub fn set_reg(&mut self, name: &str, value: u32) {
        self.runtime.set_reg(name, value);
    }

    pub fn read_u8(&self, addr: u32) -> u8 {
        self.runtime.memory.load(addr, 8).unwrap_or(0) as u8
    }

    pub fn write_u8(&mut self, addr: u32, value: u8) {
        let _ = self.runtime.memory.store(addr, 8, value as u32);
    }

    pub fn imr(&self) -> u8 {
        self.runtime
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(0)
    }

    pub fn isr(&self) -> u8 {
        self.runtime
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(0)
    }

    pub fn press_matrix_code(&mut self, code: u8) {
        if let Some(kb) = self.runtime.keyboard.as_mut() {
            kb.press_matrix_code(code, &mut self.runtime.memory);
        }
    }

    pub fn release_matrix_code(&mut self, code: u8) {
        if let Some(kb) = self.runtime.keyboard.as_mut() {
            kb.release_matrix_code(code, &mut self.runtime.memory);
        }
    }

    pub fn inject_matrix_event(&mut self, code: u8, release: bool) -> usize {
        if let Some(kb) = self.runtime.keyboard.as_mut() {
            kb.inject_matrix_event(
                code,
                release,
                &mut self.runtime.memory,
                self.runtime.timer.kb_irq_enabled,
            )
        } else {
            0
        }
    }

    pub fn press_on_key(&mut self) {
        self.runtime.press_on_key();
    }

    pub fn release_on_key(&mut self) {
        self.runtime.release_on_key();
    }

    pub fn configure_timer(&mut self, enabled: bool, mti_period: u64, sti_period: u64) {
        self.runtime.timer.enabled = enabled;
        self.runtime.timer.mti_period = mti_period;
        self.runtime.timer.sti_period = sti_period;
        self.runtime.timer.reset(self.runtime.cycle_count());
    }

    pub fn lcd_pixels(&self) -> Uint8Array {
        let rows = LCD_DISPLAY_ROWS as usize;
        let cols = LCD_DISPLAY_COLS as usize;
        let mut flat = vec![0u8; rows * cols];
        if let Some(lcd) = self.runtime.lcd.as_ref() {
            let buf = lcd.display_buffer();
            for (row, row_buf) in buf.iter().enumerate().take(rows) {
                let start = row * cols;
                flat[start..start + cols].copy_from_slice(row_buf);
            }
        }
        Uint8Array::from(flat.as_slice())
    }

    pub fn lcd_text(&self) -> Result<JsValue, JsValue> {
        let Some(font) = self.font_map.as_ref() else {
            return serde_wasm_bindgen::to_value(&Vec::<String>::new())
                .map_err(|e| JsValue::from_str(&e.to_string()));
        };
        let Some(lcd) = self.runtime.lcd.as_ref() else {
            return serde_wasm_bindgen::to_value(&Vec::<String>::new())
                .map_err(|e| JsValue::from_str(&e.to_string()));
        };
        let lines = decode_display_text(lcd, font);
        serde_wasm_bindgen::to_value(&lines).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn regs(&self) -> Result<JsValue, JsValue> {
        let regs = sc62015_core::collect_registers(&self.runtime.state);
        serde_wasm_bindgen::to_value(&regs).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn debug_state(&self) -> Result<JsValue, JsValue> {
        let timer = TimerState {
            enabled: self.runtime.timer.enabled,
            mti_period: self.runtime.timer.mti_period,
            sti_period: self.runtime.timer.sti_period,
            next_mti: self.runtime.timer.next_mti,
            next_sti: self.runtime.timer.next_sti,
            kb_irq_enabled: self.runtime.timer.kb_irq_enabled,
        };
        let irq = IrqState {
            pending: self.runtime.timer.irq_pending,
            in_interrupt: self.runtime.timer.in_interrupt,
            source: self.runtime.timer.irq_source.clone(),
            irq_total: self.runtime.timer.irq_total,
            irq_key: self.runtime.timer.irq_key,
            irq_mti: self.runtime.timer.irq_mti,
            irq_sti: self.runtime.timer.irq_sti,
        };
        let state = DebugState {
            instruction_count: self.runtime.instruction_count(),
            cycle_count: self.runtime.cycle_count(),
            halted: self.runtime.state.is_halted(),
            call_depth: self.runtime.state.call_depth(),
            call_sub_level: self.runtime.state.call_sub_level(),
            imr: self.imr(),
            isr: self.isr(),
            timer,
            irq,
        };
        serde_wasm_bindgen::to_value(&state).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    fn load_pce500_rom_window(&mut self, rom: &[u8]) -> Result<(), JsValue> {
        let rom_len = ROM_WINDOW_LEN;
        let src_start = rom.len().saturating_sub(rom_len);
        let slice = &rom[src_start..];
        let copy_len = slice.len().min(rom_len);
        let start_in_slice = slice.len().saturating_sub(copy_len);
        self.runtime
            .memory
            .write_external_slice(ROM_WINDOW_START, &slice[start_in_slice..]);
        self.runtime.memory.set_readonly_ranges(vec![(
            ROM_WINDOW_START as u32,
            (ROM_WINDOW_START + rom_len - 1) as u32,
        )]);
        self.runtime.memory.set_keyboard_bridge(false);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    const PF1_CODE: u8 = 0x56;

    #[wasm_bindgen_test]
    fn reset_reads_rom_vector() {
        let mut emulator = Pce500Emulator::new();
        let mut rom = vec![0u8; ROM_WINDOW_LEN];
        // Reset vector at 0xFFFFD (last three bytes of the window).
        rom[ROM_WINDOW_LEN - 3] = 0x34;
        rom[ROM_WINDOW_LEN - 2] = 0x12;
        rom[ROM_WINDOW_LEN - 1] = 0x00;
        emulator.load_rom(&rom).expect("load rom");
        assert_eq!(emulator.get_reg("PC"), 0x001234);
    }

    #[wasm_bindgen_test]
    fn lcd_buffer_has_expected_size() {
        let emulator = Pce500Emulator::new();
        let pixels = emulator.lcd_pixels();
        assert_eq!(
            pixels.length(),
            (LCD_DISPLAY_ROWS as u32) * (LCD_DISPLAY_COLS as u32)
        );
    }

    #[wasm_bindgen_test]
    fn pf1_changes_lcd_text_in_synthetic_rom() {
        let rom = include_bytes!("../testdata/pf1_demo_rom_window.bin");
        let mut emulator = Pce500Emulator::new();
        emulator.load_rom(rom).expect("load synthetic ROM");

        emulator.step(5_000).expect("boot");
        let before = emulator
            .lcd_text()
            .ok()
            .and_then(|v| serde_wasm_bindgen::from_value::<Vec<String>>(v).ok())
            .unwrap_or_default();
        assert_eq!(before.get(0).map(|s| s.as_str()), Some("BOOT"));

        emulator.press_matrix_code(PF1_CODE);
        emulator.step(50_000).expect("poll PF1");

        let after = emulator
            .lcd_text()
            .ok()
            .and_then(|v| serde_wasm_bindgen::from_value::<Vec<String>>(v).ok())
            .unwrap_or_default();
        assert_eq!(after.get(0).map(|s| s.as_str()), Some("MENU"));
    }
}
