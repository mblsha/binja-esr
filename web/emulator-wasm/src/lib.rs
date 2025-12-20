// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/display/text_decoder.py:decode_display_text
// PY_SOURCE: pce500/display/font.py

use js_sys::Uint8Array;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use base64::Engine;
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET};
use sc62015_core::pce500::{DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD};
use sc62015_core::{CoreRuntime, LCD_CHIP_COLS, LCD_CHIP_ROWS, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
use sc62015_core::{DeviceModel, DeviceTextDecoder};

#[wasm_bindgen]
pub fn default_device_model() -> String {
    DeviceModel::DEFAULT.label().to_string()
}

#[derive(Debug, Clone, Serialize)]
struct BuildInfo {
    version: String,
    git_commit: String,
    build_timestamp: String,
}

fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: option_env!("GIT_COMMIT").unwrap_or("unknown").to_string(),
        build_timestamp: option_env!("BUILD_TIMESTAMP")
            .unwrap_or("unknown")
            .to_string(),
    }
}

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

#[derive(Debug, Clone, Serialize)]
struct MemoryWriteByte {
    addr: u32,
    value: u8,
}

#[derive(Debug, Clone, Serialize)]
struct CallReport {
    reason: String,
    steps: u32,
    pc: u32,
    sp: u32,
    halted: bool,
    fault: Option<CallFault>,
}

#[derive(Debug, Clone, Serialize)]
struct CallFault {
    kind: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeSample {
    pc: u32,
    count: u32,
    regs: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
struct CallOptions {
    #[serde(default)]
    trace: bool,
    #[serde(default)]
    probe_pc: Option<u32>,
    #[serde(default)]
    probe_max_samples: u32,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PerfettoFunctionSymbol {
    addr: u32,
    name: String,
}

#[derive(Debug, Clone, Serialize)]
struct CallArtifacts {
    address: u32,
    before_pc: u32,
    after_pc: u32,
    before_sp: u32,
    after_sp: u32,
    before_regs: std::collections::HashMap<String, u32>,
    after_regs: std::collections::HashMap<String, u32>,
    memory_writes: Vec<MemoryWriteByte>,
    lcd_writes: Vec<sc62015_core::lcd::LcdDisplayWrite>,
    probe_samples: Vec<ProbeSample>,
    perfetto_trace_b64: Option<String>,
    report: CallReport,
}

#[wasm_bindgen(js_name = Pce500Emulator)]
pub struct Sc62015Emulator {
    runtime: CoreRuntime,
    rom_image: Vec<u8>,
    model: DeviceModel,
    text_decoder: Option<DeviceTextDecoder>,
}

#[wasm_bindgen]
impl Sc62015Emulator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut emulator = Self {
            runtime: CoreRuntime::new(),
            rom_image: Vec::new(),
            model: DeviceModel::DEFAULT,
            text_decoder: None,
        };
        emulator.configure_timer(true, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD);
        emulator
    }

    pub fn device_model(&self) -> String {
        self.model.label().to_string()
    }

    pub fn has_rom(&self) -> bool {
        !self.rom_image.is_empty()
    }

    pub fn build_info(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&build_info()).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Install a symbol map used to label Call-UI Perfetto traces (the "Functions" track).
    pub fn set_perfetto_function_symbols(&mut self, symbols: JsValue) -> Result<(), JsValue> {
        let entries: Vec<PerfettoFunctionSymbol> = serde_wasm_bindgen::from_value(symbols)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut map = std::collections::HashMap::with_capacity(entries.len());
        for entry in entries {
            let name = entry.name.trim();
            if name.is_empty() {
                continue;
            }
            map.insert(entry.addr & 0x000f_ffff, name.to_string());
        }
        sc62015_core::perfetto::set_call_ui_function_names(map);
        Ok(())
    }

    pub fn load_rom(&mut self, rom: &[u8]) -> Result<(), JsValue> {
        if rom.is_empty() {
            return Err(JsValue::from_str("ROM is empty"));
        }
        self.rom_image = rom.to_vec();
        self.text_decoder = self.model.text_decoder(&self.rom_image);
        self.reset()
    }

    pub fn load_rom_with_model(&mut self, rom: &[u8], model: &str) -> Result<(), JsValue> {
        self.model = DeviceModel::parse(model).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unknown model '{model}' (expected: iq-7000|pc-e500)"
            ))
        })?;
        self.load_rom(rom)
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        if self.rom_image.is_empty() {
            return Err(JsValue::from_str("ROM not loaded"));
        }
        let rom = self.rom_image.clone();
        self.runtime = CoreRuntime::new();
        self.configure_timer(true, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD);
        self.model
            .configure_runtime(&mut self.runtime, &rom)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
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

    pub fn halted(&self) -> bool {
        self.runtime.state.is_halted()
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

    /// Run a ROM function at `addr` with a bounded instruction budget, capturing last-value
    /// memory writes and display-mapped LCD writes performed during the run.
    ///
    /// Notes:
    /// - This is a debugging helper for the web UI; it restores PC/SP/call-metrics on exit so
    ///   the harness does not perturb subsequent execution.
    /// - The function is entered by setting `PC=addr` and pushing a sentinel return address
    ///   onto the S stack; the loop stops when execution returns to that sentinel PC.
    pub fn call_function(&mut self, addr: u32, max_instructions: u32) -> Result<JsValue, JsValue> {
        self.call_function_ex(addr, max_instructions, JsValue::UNDEFINED)
    }

    /// Like `call_function`, but allows optional capture controls via `options`:
    /// - `{ trace: true }` captures a Perfetto trace for the duration of the call.
    /// - `{ probe_pc: 0x00F123, probe_max_samples: 100 }` records register snapshots whenever `PC == probe_pc`.
    #[wasm_bindgen]
    pub fn call_function_ex(
        &mut self,
        addr: u32,
        max_instructions: u32,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let addr = addr & 0x000f_ffff;
        let max_instructions = max_instructions.max(1);
        let mut opts: CallOptions = serde_wasm_bindgen::from_value(options).unwrap_or_default();
        if opts.probe_max_samples == 0 {
            opts.probe_max_samples = 256;
        }
        let before_pc = self.runtime.state.pc();
        let before_sp = self.runtime.state.get_reg(RegName::S);
        let before_call_metrics = self.runtime.state.snapshot_call_metrics();
        let before_regs = sc62015_core::collect_registers(&self.runtime.state);

        let sentinel_low16: u32 = 0xD00D;
        let sentinel_pc = ((addr & 0x0f_0000) | sentinel_low16) & 0x000f_ffff;

        // Push a 24-bit sentinel return address (little-endian) onto the S stack.
        let new_sp = before_sp.wrapping_sub(3) & 0x00ff_ffff;
        for i in 0..3u32 {
            let byte = ((sentinel_pc >> (8 * i)) & 0xff) as u32;
            let _ = self.runtime.memory.store(new_sp.wrapping_add(i), 8, byte);
        }
        self.runtime.state.set_reg(RegName::S, new_sp);
        // Bookkeeping for call-stack tracing; RET/RETF will unwind this.
        self.runtime.state.push_call_stack(addr);
        self.runtime.state.call_depth_inc();

        // Enable last-value write capture.
        self.runtime.memory.begin_write_capture();
        if let Some(lcd) = self.runtime.lcd.as_mut() {
            lcd.begin_display_write_capture();
        }

        // Optional perfetto capture for this call (serialized on wasm32).
        let mut previous_tracer: Option<sc62015_core::PerfettoTracer> = None;
        if opts.trace {
            let mut guard = sc62015_core::PERFETTO_TRACER.enter();
            previous_tracer = guard.replace(Some(sc62015_core::PerfettoTracer::new(
                std::path::PathBuf::from("call.perfetto-trace"),
            )));
        }

        // Enter the function.
        self.runtime.state.set_pc(addr);

        let mut steps: u32 = 0;
        let mut reason = "timeout".to_string();
        let mut fault: Option<CallFault> = None;
        let mut probe_samples: Vec<ProbeSample> = Vec::new();
        let mut probe_hits: u32 = 0;
        while steps < max_instructions {
            if self.runtime.state.is_halted() && !self.runtime.timer.irq_pending {
                reason = "halted".to_string();
                break;
            }
            if let Some(probe_pc) = opts.probe_pc {
                if self.runtime.state.pc() == (probe_pc & 0x000f_ffff)
                    && (probe_samples.len() as u32) < opts.probe_max_samples
                {
                    probe_hits = probe_hits.saturating_add(1);
                    probe_samples.push(ProbeSample {
                        pc: self.runtime.state.pc(),
                        count: probe_hits,
                        regs: sc62015_core::collect_registers(&self.runtime.state),
                    });
                }
            }
            if let Err(err) = self.runtime.step(1) {
                reason = "fault".to_string();
                fault = Some(CallFault {
                    kind: "CoreError".to_string(),
                    message: err.to_string(),
                });
                break;
            }
            steps += 1;
            if self.runtime.state.pc() == sentinel_pc {
                reason = "returned".to_string();
                break;
            }
        }

        let after_pc = self.runtime.state.pc();
        let after_sp = self.runtime.state.get_reg(RegName::S);
        let memory_writes = self
            .runtime
            .memory
            .take_write_capture()
            .into_iter()
            .map(|(addr, value)| MemoryWriteByte { addr, value })
            .collect();
        let lcd_writes = self
            .runtime
            .lcd
            .as_mut()
            .map(|lcd| lcd.take_display_write_capture())
            .unwrap_or_default();

        let perfetto_trace_b64 = if opts.trace {
            let mut guard = sc62015_core::PERFETTO_TRACER.enter();
            let trace_bytes = guard
                .take()
                .map(|tracer| tracer.serialize())
                .transpose()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            guard.replace(previous_tracer.take());
            trace_bytes.map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            None
        };

        // Restore state so the harness does not leave a sentinel PC/SP or call metrics behind.
        self.runtime.state.set_pc(before_pc);
        self.runtime.state.set_reg(RegName::S, before_sp);
        self.runtime.state.restore_call_metrics(before_call_metrics);

        let after_regs = sc62015_core::collect_registers(&self.runtime.state);

        let report = CallReport {
            reason,
            steps,
            pc: after_pc,
            sp: after_sp,
            halted: self.runtime.state.is_halted(),
            fault,
        };
        let artifacts = CallArtifacts {
            address: addr,
            before_pc,
            after_pc,
            before_sp,
            after_sp,
            before_regs,
            after_regs,
            memory_writes,
            lcd_writes,
            probe_samples,
            perfetto_trace_b64,
            report,
        };
        // Return JSON to avoid browser-specific structured-object aliasing errors observed with
        // `serde_wasm_bindgen` + `HashMap` (wasm-bindgen re-entrancy guard).
        let json =
            serde_json::to_string(&artifacts).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }

    pub fn lcd_pixels(&self) -> Uint8Array {
        let rows = LCD_DISPLAY_ROWS as usize;
        let cols = LCD_DISPLAY_COLS as usize;
        let mut flat = vec![0u8; rows * cols];
        if let Some(lcd) = self.runtime.lcd.as_deref() {
            let buf = lcd.display_buffer();
            for (row, row_buf) in buf.iter().enumerate().take(rows) {
                let start = row * cols;
                flat[start..start + cols].copy_from_slice(row_buf);
            }
        }
        Uint8Array::from(flat.as_slice())
    }

    pub fn lcd_chip_pixels(&self) -> Uint8Array {
        let rows = LCD_CHIP_ROWS as usize;
        let cols = LCD_CHIP_COLS as usize;
        let mut flat = vec![0u8; rows * cols * 2];
        if let Some(lcd) = self.runtime.lcd.as_deref() {
            for chip_index in 0..2 {
                let buf = lcd.chip_display_buffer(chip_index);
                let base = chip_index * rows * cols;
                for (row, row_buf) in buf.iter().enumerate().take(rows) {
                    let start = base + row * cols;
                    flat[start..start + cols].copy_from_slice(row_buf);
                }
            }
        }
        Uint8Array::from(flat.as_slice())
    }

    pub fn lcd_text(&self) -> Result<JsValue, JsValue> {
        let Some(text_decoder) = self.text_decoder.as_ref() else {
            return serde_wasm_bindgen::to_value(&Vec::<String>::new())
                .map_err(|e| JsValue::from_str(&e.to_string()));
        };
        let Some(lcd) = self.runtime.lcd.as_deref() else {
            return serde_wasm_bindgen::to_value(&Vec::<String>::new())
                .map_err(|e| JsValue::from_str(&e.to_string()));
        };
        let lines = text_decoder.decode_display_text(lcd);
        serde_wasm_bindgen::to_value(&lines).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn lcd_trace(&self) -> Result<JsValue, JsValue> {
        let Some(lcd) = self.runtime.lcd.as_deref() else {
            return serde_wasm_bindgen::to_value(
                &Vec::<Vec<sc62015_core::lcd::LcdWriteTrace>>::new(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()));
        };
        let grid = lcd.display_trace_buffer();
        let out: Vec<Vec<sc62015_core::lcd::LcdWriteTrace>> =
            grid.iter().map(|row| row.to_vec()).collect();
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn regs(&self) -> Result<JsValue, JsValue> {
        let regs = sc62015_core::collect_registers(&self.runtime.state);
        serde_wasm_bindgen::to_value(&regs).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn call_stack(&self) -> Result<JsValue, JsValue> {
        let frames = self.runtime.state.call_stack().to_vec();
        serde_wasm_bindgen::to_value(&frames).map_err(|e| JsValue::from_str(&e.to_string()))
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

    // ROM window loading lives in sc62015-core so the CLI runner and WASM wrapper share it.
}

pub type Pce500Emulator = Sc62015Emulator;

#[cfg(test)]
mod tests {
    use super::*;
    use sc62015_core::pce500::ROM_WINDOW_LEN;
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
    fn call_function_ex_can_capture_perfetto_trace() {
        #[derive(serde::Deserialize)]
        struct CallArtifactsDecoded {
            perfetto_trace_b64: Option<String>,
        }

        #[derive(Serialize)]
        struct Opts {
            trace: bool,
        }

        let rom: &[u8] = include_bytes!("../testdata/pf1_demo_rom_window.rom");
        let mut emulator = Pce500Emulator::new();
        emulator.load_rom(rom).expect("load");

        let pc = emulator.get_reg("PC");
        let js = emulator
            .call_function_ex(
                pc,
                64,
                serde_wasm_bindgen::to_value(&Opts { trace: true }).unwrap(),
            )
            .expect("call");
        let decoded: CallArtifactsDecoded =
            serde_json::from_str(&js.as_string().unwrap()).expect("decode");
        assert!(
            decoded
                .perfetto_trace_b64
                .as_ref()
                .is_some_and(|val| !val.is_empty()),
            "expected a non-empty perfetto trace when trace=true"
        );
    }

    #[wasm_bindgen_test]
    fn pf1_changes_lcd_text_in_synthetic_rom() {
        let rom = include_bytes!("../testdata/pf1_demo_rom_window.rom");
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
