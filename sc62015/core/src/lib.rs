// PY_SOURCE: sc62015/pysc62015/emulator.py:RegisterName
// PY_SOURCE: sc62015/pysc62015/emulator.py:Registers

pub mod keyboard;
pub mod lcd;
pub mod llama;
pub mod memory;
pub mod perfetto;
pub mod snapshot;
pub mod timer;

use crate::llama::{opcodes::RegName, state::LlamaState};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::SystemTime;
use thiserror::Error;

pub use keyboard::KeyboardMatrix;
pub use lcd::{LcdController, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
pub use llama::state::LlamaState as CpuState;
pub use memory::{
    MemoryImage, ADDRESS_MASK, EXTERNAL_SPACE, INTERNAL_ADDR_MASK, INTERNAL_MEMORY_START,
    INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE,
};
pub use perfetto::PerfettoTracer;
lazy_static::lazy_static! {
    pub static ref PERFETTO_TRACER: Mutex<Option<PerfettoTracer>> = Mutex::new(None);
}
pub use snapshot::{
    load_snapshot, pack_registers, save_snapshot, unpack_registers, SnapshotLoad, SNAPSHOT_MAGIC,
    SNAPSHOT_REGISTER_LAYOUT, SNAPSHOT_VERSION,
};
pub use timer::TimerContext;

use crate::keyboard::KeyboardSnapshot;
use crate::llama::eval::LlamaBus;
use crate::llama::state::mask_for;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("serialize error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("snapshot error: {0}")]
    InvalidSnapshot(String),
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerInfo {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mti_period: i32,
    #[serde(default)]
    pub sti_period: i32,
    #[serde(default)]
    pub next_mti: i32,
    #[serde(default)]
    pub next_sti: i32,
    #[serde(default = "default_true")]
    pub kb_irq_enabled: bool,
}

impl Default for TimerInfo {
    fn default() -> Self {
        Self {
            enabled: false,
            mti_period: 0,
            sti_period: 0,
            next_mti: 0,
            next_sti: 0,
            kb_irq_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InterruptInfo {
    #[serde(default)]
    pub pending: bool,
    #[serde(default)]
    pub in_interrupt: bool,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub stack: Vec<u32>,
    #[serde(default)]
    pub next_id: u32,
    #[serde(default)]
    pub imr: u8,
    #[serde(default)]
    pub isr: u8,
    #[serde(default)]
    pub irq_counts: Option<serde_json::Value>,
    #[serde(default)]
    pub last_irq: Option<serde_json::Value>,
    #[serde(default)]
    pub irq_bit_watch: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub magic: String,
    pub version: u32,
    pub backend: String,
    pub created: String,
    pub instruction_count: u64,
    pub cycle_count: u64,
    #[serde(default)]
    pub memory_reads: u64,
    #[serde(default)]
    pub memory_writes: u64,
    pub pc: u32,
    #[serde(default)]
    pub call_depth: u32,
    #[serde(default)]
    pub call_sub_level: u32,
    #[serde(default)]
    pub temps: HashMap<String, u32>,
    pub timer: TimerInfo,
    pub interrupts: InterruptInfo,
    #[serde(default)]
    pub keyboard: Option<serde_json::Value>,
    #[serde(default)]
    pub kb_metrics: Option<serde_json::Value>,
    #[serde(default)]
    pub fallback_ranges: Vec<(u32, u32)>,
    #[serde(default)]
    pub readonly_ranges: Vec<(u32, u32)>,
    #[serde(deserialize_with = "crate::snapshot::deserialize_range")]
    pub internal_ram: (u32, u32),
    #[serde(deserialize_with = "crate::snapshot::deserialize_range")]
    pub imem: (u32, u32),
    pub memory_dump_pc: u32,
    pub fast_mode: bool,
    pub memory_image_size: usize,
    pub lcd_payload_size: usize,
    pub lcd: Option<serde_json::Value>,
}

impl Default for SnapshotMetadata {
    fn default() -> Self {
        Self {
            magic: SNAPSHOT_MAGIC.to_string(),
            version: SNAPSHOT_VERSION,
            backend: "core".to_string(),
            created: now_timestamp(),
            instruction_count: 0,
            cycle_count: 0,
            memory_reads: 0,
            memory_writes: 0,
            pc: 0,
            call_depth: 0,
            call_sub_level: 0,
            temps: HashMap::new(),
            timer: TimerInfo::default(),
            interrupts: InterruptInfo::default(),
            keyboard: None,
            kb_metrics: None,
            fallback_ranges: Vec::new(),
            readonly_ranges: Vec::new(),
            internal_ram: (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32),
            imem: (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32),
            memory_dump_pc: 0,
            fast_mode: false,
            memory_image_size: EXTERNAL_SPACE,
            lcd_payload_size: 0,
            lcd: None,
        }
    }
}

    pub fn now_timestamp() -> String {
        match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(duration) => format!("{}Z", duration.as_secs()),
            Err(_) => "0Z".to_string(),
        }
    }

fn default_true() -> bool {
    true
}

pub const DEFAULT_REG_WIDTH: u8 = 24;

fn mask_for_width(bits: u8) -> u32 {
    if bits == 0 {
        0
    } else if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

fn reg_from_name(name: &str) -> Option<RegName> {
    match name.to_ascii_uppercase().as_str() {
        "A" => Some(RegName::A),
        "B" => Some(RegName::B),
        "BA" => Some(RegName::BA),
        "IL" => Some(RegName::IL),
        "IH" => Some(RegName::IH),
        "I" => Some(RegName::I),
        "X" => Some(RegName::X),
        "Y" => Some(RegName::Y),
        "U" => Some(RegName::U),
        "S" => Some(RegName::S),
        "PC" => Some(RegName::PC),
        "F" => Some(RegName::F),
        "FC" => Some(RegName::FC),
        "FZ" => Some(RegName::FZ),
        "IMR" => Some(RegName::IMR),
        _ => None,
    }
}

pub fn register_width(name: &str) -> u8 {
    match name.to_ascii_uppercase().as_str() {
        "A" | "B" | "IL" | "IH" => 8,
        "BA" | "I" => 16,
        "X" | "Y" | "U" | "S" => 24,
        "F" => 8,
        "FC" | "FZ" => 1,
        "PC" => 20,
        _ => DEFAULT_REG_WIDTH,
    }
}

pub fn collect_registers(state: &LlamaState) -> HashMap<String, u32> {
    let mut regs = HashMap::new();
    for (name, width_bytes) in snapshot::SNAPSHOT_REGISTER_LAYOUT.iter() {
        let bits = (width_bytes * 8) as u8;
        let value = reg_from_name(name)
            .map(|reg| state.get_reg(reg) & mask_for_width(bits))
            .unwrap_or(0);
        regs.insert((*name).to_string(), value);
    }
    // Capture temp registers (TEMP0..TEMP13) to align with Python snapshots.
    for idx in 0..14u8 {
        let name = format!("TEMP{idx}");
        let reg = RegName::Temp(idx);
        regs.insert(name, state.get_reg(reg) & mask_for_width(DEFAULT_REG_WIDTH));
    }
    regs
}

pub fn apply_registers(state: &mut LlamaState, regs: &HashMap<String, u32>) {
    for (name, _) in snapshot::SNAPSHOT_REGISTER_LAYOUT.iter() {
        let value = *regs.get(*name).unwrap_or(&0);
        if let Some(reg) = reg_from_name(name) {
            state.set_reg(reg, value & mask_for_width(register_width(name)));
        }
    }
    for idx in 0..14u8 {
        let key = format!("TEMP{idx}");
        if let Some(value) = regs.get(&key) {
            state.set_reg(RegName::Temp(idx), *value & mask_for_width(DEFAULT_REG_WIDTH));
        }
    }
}

/// Extremely small placeholder runtime for LLAMA-only execution.
pub struct CoreRuntime {
    metadata: SnapshotMetadata,
    pub memory: MemoryImage,
    pub state: LlamaState,
    pub fast_mode: bool,
    executor: crate::llama::eval::LlamaExecutor,
    pub keyboard: Option<KeyboardMatrix>,
    pub lcd: Option<LcdController>,
    pub timer: TimerContext,
}

impl Default for CoreRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreRuntime {
    pub fn new() -> Self {
        Self {
            metadata: SnapshotMetadata::default(),
            memory: MemoryImage::new(),
            state: LlamaState::new(),
            fast_mode: false,
            executor: crate::llama::eval::LlamaExecutor::new(),
            keyboard: Some(KeyboardMatrix::new()),
            lcd: Some(LcdController::new()),
            timer: TimerContext::new(false, 0, 0),
        }
    }

    pub fn load_rom(&mut self, blob: &[u8], start: usize) {
        let end = (start + blob.len()).min(self.memory.external_len());
        if start < end {
            self.memory
                .write_external_slice(start, &blob[..(end - start)]);
        }
    }

    pub fn step(&mut self, instructions: usize) -> Result<()> {
        // Execute real instructions through the LLAMA evaluator instead of bumping PC.
        struct RuntimeBus<'a> {
            mem: &'a mut MemoryImage,
        }
        impl<'a> LlamaBus for RuntimeBus<'a> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                self.mem.load(addr, bits).unwrap_or(0)
            }
            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                let _ = self.mem.store(addr, bits, value);
            }
            fn resolve_emem(&mut self, base: u32) -> u32 {
                base
            }
            fn peek_imem_silent(&mut self, offset: u32) -> u8 {
                self.mem.read_internal_byte_silent(offset).unwrap_or(0)
            }
            fn wait_cycles(&mut self, _cycles: u32) {}
        }

        for _ in 0..instructions {
            if self.state.is_halted() {
                break;
            }
            {
                let mut bus = RuntimeBus { mem: &mut self.memory };
                let pc = self.state.get_reg(RegName::PC) & ADDRESS_MASK;
                let opcode = bus.load(pc, 8) as u8;
                if let Err(e) = self
                    .executor
                    .execute(opcode, &mut self.state, &mut bus)
                {
                    return Err(CoreError::Other(format!("execute opcode 0x{opcode:02X}: {e}")));
                }
            }
            self.metadata.instruction_count = self.metadata.instruction_count.wrapping_add(1);
            // Python emulator counts instructions, not byte-length; keep parity by bumping once per opcode.
            self.metadata.cycle_count = self.metadata.cycle_count.wrapping_add(1);
            // Advance timers/keyboard each instruction to mirror Python runtime.
            let cycles = self.metadata.cycle_count;
            let (mti, sti, key_events) = self.timer.tick_timers_with_keyboard(
                &mut self.memory,
                cycles,
                |mem| {
                    if let Some(kb) = self.keyboard.as_mut() {
                        let events = kb.scan_tick();
                        if events > 0 {
                            kb.write_fifo_to_memory(mem);
                        }
                        (events, kb.fifo_len() > 0)
                    } else {
                        (0, false)
                    }
                },
            );
            let fifo_non_empty = self
                .keyboard
                .as_ref()
                .map(|kb| kb.fifo_len() > 0)
                .unwrap_or(false);
            if (mti && key_events > 0 && fifo_non_empty) || (sti && fifo_non_empty) {
                if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                    if (isr & 0x04) == 0 {
                        self.memory.write_internal_byte(0xFC, isr | 0x04);
                    }
                }
            }
            if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                self.timer.irq_isr = isr;
            }
            self.deliver_pending_irq();
        }
        Ok(())
    }

    pub fn save_snapshot(&self, path: &std::path::Path) -> Result<()> {
        let mut metadata = self.metadata.clone();
        metadata.instruction_count = self.metadata.instruction_count;
        metadata.cycle_count = self.metadata.cycle_count;
        metadata.pc = self.get_reg("PC");
        metadata.call_depth = self.state.call_depth();
        metadata.call_sub_level = self.state.call_sub_level();
        metadata.temps = collect_registers(&self.state)
            .into_iter()
            .filter(|(k, _)| k.starts_with("TEMP"))
            .collect();
        metadata.memory_dump_pc = 0;
        if let Some(kb) = self.keyboard.as_ref() {
            let kb_state = kb.snapshot_state();
            if let Ok(snapshot) = serde_json::to_value(&kb_state) {
                metadata.keyboard = Some(snapshot);
                metadata.kb_metrics = Some(json!({
                    "irq_count": kb_state.irq_count,
                    "strobe_count": kb_state.strobe_count,
                    "column_hist": kb_state.column_histogram,
                    "last_cols": kb_state.active_columns,
                    "last_kol": kb_state.kol,
                    "last_koh": kb_state.koh,
                    "kil_reads": kb_state.fifo_len,
                    "kb_irq_enabled": true,
                }));
            }
        }
        let mut lcd_payload: Option<Vec<u8>> = None;
        if let Some(lcd) = self.lcd.as_ref() {
            let (lcd_meta, payload) = lcd.export_snapshot();
            metadata.lcd = Some(lcd_meta);
            metadata.lcd_payload_size = payload.len();
            lcd_payload = Some(payload);
        }
        // Persist timer/interrupt mirrors to match Python snapshot expectations.
        let (timer_info, intr_info) = self.timer.snapshot_info();
        metadata.timer = timer_info;
        metadata.interrupts = intr_info;
        let regs = collect_registers(&self.state);
        snapshot::save_snapshot(
            path,
            &metadata,
            &regs,
            &self.memory,
            lcd_payload.as_deref(),
        )
    }

    pub fn load_snapshot(&mut self, path: &std::path::Path) -> Result<()> {
        let loaded = snapshot::load_snapshot(path, &mut self.memory)?;
        self.metadata = loaded.metadata;
        apply_registers(&mut self.state, &loaded.registers);
        self.fast_mode = self.metadata.fast_mode;
        self.timer
            .apply_snapshot_info(&self.metadata.timer, &self.metadata.interrupts);
        if self.keyboard.is_none() {
            self.keyboard = Some(KeyboardMatrix::new());
        }
        if let (Some(kb_meta), Some(kb)) = (self.metadata.keyboard.clone(), self.keyboard.as_mut()) {
            if let Ok(snapshot) = serde_json::from_value::<KeyboardSnapshot>(kb_meta) {
                kb.load_snapshot_state(&snapshot);
            }
        }
        if self.lcd.is_none() {
            self.lcd = Some(LcdController::new());
        }
        if let (Some(lcd_meta), Some(payload), Some(lcd)) =
            (self.metadata.lcd.clone(), loaded.lcd_payload.as_deref(), self.lcd.as_mut())
        {
            let _ = lcd.load_snapshot(&lcd_meta, payload);
        }
        // Restore call depth/sub-level and temps from metadata if present.
        if self.metadata.call_depth > 0 {
            for _ in 0..self.metadata.call_depth {
                self.state.call_depth_inc();
            }
        }
        self.state
            .set_call_sub_level(self.metadata.call_sub_level);
        for (name, value) in self.metadata.temps.iter() {
            if let Some(idx_str) = name.strip_prefix("TEMP") {
                if let Ok(idx) = idx_str.parse::<u8>() {
                    self.state
                        .set_reg(RegName::Temp(idx), *value & mask_for_width(DEFAULT_REG_WIDTH));
                }
            }
        }
        self.fast_mode = self.metadata.fast_mode;
        Ok(())
    }

    pub fn set_reg(&mut self, name: &str, value: u32) {
        if let Some(reg) = reg_from_name(name) {
            self.state.set_reg(reg, value);
        }
    }

    pub fn get_reg(&self, name: &str) -> u32 {
        reg_from_name(name)
            .map(|reg| self.state.get_reg(reg) & mask_for_width(register_width(name)))
            .unwrap_or(0)
    }

    pub fn set_flag(&mut self, name: &str, value: u8) {
        if let Some(reg) = reg_from_name(name) {
            self.state.set_reg(reg, value as u32);
        }
    }

    pub fn get_flag(&self, name: &str) -> u8 {
        reg_from_name(name)
            .map(|reg| self.state.get_reg(reg) as u8)
            .unwrap_or(0)
    }

    fn push_stack(&mut self, reg: RegName, value: u32, bits: u8) {
        let bytes = bits.div_ceil(8);
        let mask = mask_for(reg);
        let sp = self.state.get_reg(reg) & mask;
        let new_sp = sp.wrapping_sub(bytes as u32) & mask;
        for i in 0..bytes {
            let byte = (value >> (8 * i)) & 0xFF;
            let _ = self.memory.store(new_sp + i as u32, 8, byte);
        }
        self.state.set_reg(reg, new_sp);
    }

    fn deliver_pending_irq(&mut self) {
        if !self.timer.irq_pending {
            return;
        }
        let imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(0);
        let isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        if (imr & IMR_MASTER) == 0 {
            return;
        }
        let src = if (isr & ISR_ONKI != 0) && (imr & IMR_ONK != 0) {
            Some((ISR_ONKI, "ONK"))
        } else if (isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0) {
            Some((ISR_KEYI, "KEY"))
        } else if (isr & ISR_MTI != 0) && (imr & IMR_MTI != 0) {
            Some((ISR_MTI, "MTI"))
        } else if (isr & ISR_STI != 0) && (imr & IMR_STI != 0) {
            Some((ISR_STI, "STI"))
        } else {
            None
        };
        let Some((mask, src_name)) = src else { return };

        let pc = self.state.pc() & ADDRESS_MASK;
        // Stack push order mirrors IR intrinsic: PC (24 LE), F, IMR.
        self.push_stack(RegName::S, pc, 24);
        let f = self.state.get_reg(RegName::F) & 0xFF;
        self.push_stack(RegName::S, f, 8);
        let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
        let imr_mem = self.memory.load(imr_addr, 8).unwrap_or(0) & 0xFF;
        self.push_stack(RegName::S, imr_mem, 8);
        let cleared_imr = (imr_mem as u8) & 0x7F;
        let _ = self.memory.store(imr_addr, 8, cleared_imr as u32);
        self.state.set_reg(RegName::IMR, cleared_imr as u32);

        // Jump to vector.
        let vec = (self.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap_or(0))
            | (self
                .memory
                .load(INTERRUPT_VECTOR_ADDR + 1, 8)
                .unwrap_or(0)
                << 8)
            | (self
                .memory
                .load(INTERRUPT_VECTOR_ADDR + 2, 8)
                .unwrap_or(0)
                << 16);
        self.state.set_pc(vec & ADDRESS_MASK);
        self.state.set_halted(false);

        // Track interrupt metadata similar to Python snapshot fields.
        self.timer.in_interrupt = true;
        self.timer.irq_pending = false;
        self.timer.irq_source = Some(src_name.to_string());
        self.timer.interrupt_stack.push(pc);
        self.timer.last_fired = Some(src_name.to_string());
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        // Remember active mask to help RETI-like flows.
        self.timer.next_interrupt_id = self.timer.next_interrupt_id.saturating_add(1);
        self.timer.interrupt_stack.push(mask as u32);
    }
}

const IMEM_IMR_OFFSET: u32 = 0xFB;
const IMEM_ISR_OFFSET: u32 = 0xFC;
const IMR_MASTER: u8 = 0x80;
const IMR_MTI: u8 = 0x01;
const IMR_STI: u8 = 0x02;
const IMR_KEY: u8 = 0x04;
const IMR_ONK: u8 = 0x08;
const ISR_MTI: u8 = 0x01;
const ISR_STI: u8 = 0x02;
const ISR_KEYI: u8 = 0x04;
const ISR_ONKI: u8 = 0x08;
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;
    use std::fs;

    #[test]
    fn snapshot_roundtrip_preserves_call_and_temps() {
        let tmp = std::env::temp_dir().join("core_snapshot_test.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        rt.state.call_depth_inc();
        rt.state.call_depth_inc();
        rt.state.set_call_sub_level(3);
        rt.state.set_reg(RegName::Temp(0), 0x00AA_BB);
        rt.state.set_reg(RegName::Temp(5), 0x123456);
        rt.state.set_reg(RegName::PC, 0x12345);
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert_eq!(rt2.state.call_depth(), 2);
        assert_eq!(rt2.state.call_sub_level(), 3);
        assert_eq!(rt2.state.get_reg(RegName::Temp(0)) & 0xFFFFFF, 0x00AA_BB);
        assert_eq!(rt2.state.get_reg(RegName::Temp(5)) & 0xFFFFFF, 0x123456);
        assert_eq!(rt2.state.get_reg(RegName::PC) & 0x0F_FFFF, 0x12345);
    }

    #[test]
    fn snapshot_roundtrip_keeps_timer_and_fast_mode() {
        let tmp = std::env::temp_dir().join("core_snapshot_timer.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        rt.fast_mode = true;
        rt.metadata.fast_mode = true;
        rt.timer.enabled = true;
        rt.timer.mti_period = 7;
        rt.timer.sti_period = 11;
        rt.timer.next_mti = 123;
        rt.timer.next_sti = 456;
        rt.timer.kb_irq_enabled = false;
        rt.timer.set_interrupt_state(
            true,  // pending
            0xAA,  // imr
            0x55,  // isr
            200,   // next_mti
            300,   // next_sti
            Some("MTI".to_string()),
            true,              // in_interrupt
            Some(vec![0xDEAD]), // interrupt_stack
            5,                 // next_interrupt_id
        );
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert!(rt2.fast_mode, "fast_mode should round-trip");
        assert!(rt2.timer.enabled);
        assert_eq!(rt2.timer.mti_period, 7);
        assert_eq!(rt2.timer.sti_period, 11);
        assert_eq!(rt2.timer.next_mti, 200);
        assert_eq!(rt2.timer.next_sti, 300);
        assert!(!rt2.timer.kb_irq_enabled);
        assert!(rt2.timer.irq_pending);
        assert_eq!(rt2.timer.irq_imr, 0xAA);
        assert_eq!(rt2.timer.irq_isr, 0x55);
        assert!(rt2.timer.in_interrupt);
        assert_eq!(rt2.timer.interrupt_stack, vec![0xDEAD]);
        assert_eq!(rt2.timer.next_interrupt_id, 5);
        assert_eq!(rt2.timer.last_fired, None);
    }

    #[test]
    fn snapshot_roundtrip_captures_keyboard_and_lcd() {
        let tmp = std::env::temp_dir().join("core_snapshot_kb_lcd.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        let kb = rt.keyboard.as_mut().expect("keyboard present");
        // Simulate a key press (matrix code 0) to populate snapshot.
        kb.press_matrix_code(0, &mut rt.memory);
        // Exercise LCD writes to mutate vram and counters.
        let lcd = rt.lcd.as_mut().expect("lcd present");
        lcd.write(0x2000, 0b1100_0000); // ON instruction (turn on)
        lcd.write(0x2003, 0xAA); // Data write to advance Y/counts
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");

        // Keyboard state should round-trip (pressed key recorded).
        let kb_state = rt2
            .keyboard
            .as_ref()
            .expect("keyboard restored")
            .snapshot_state();
        assert!(
            !kb_state.pressed_keys.is_empty(),
            "pressed keys should persist across snapshot"
        );
        // LCD stats should reflect the writes we performed before saving.
        let lcd_stats = rt2.lcd.as_ref().expect("lcd restored").stats();
        assert!(
            lcd_stats.data_write_counts.iter().any(|&c| c > 0),
            "lcd data writes should persist across snapshot"
        );
        // Metadata should include kb_metrics for parity with Python.
        assert!(
            rt2.metadata.kb_metrics.is_some(),
            "keyboard metrics should be stored in snapshot metadata"
        );
    }

    #[test]
    fn step_ticks_timer_and_updates_isr() {
        let mut rt = CoreRuntime::new();
        // Enable timer with immediate MTI fire on first instruction boundary.
        rt.timer = TimerContext::new(true, 1, 0);
        let res = rt.step(1);
        assert!(res.is_ok(), "step should execute without error");
        let isr = rt.memory.read_internal_byte(0xFC).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01, "MTI should set ISR bit after first step");
        assert_eq!(rt.metadata.cycle_count, 1);
    }
}
