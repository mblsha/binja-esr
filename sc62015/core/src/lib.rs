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
use std::time::SystemTime;
use thiserror::Error;
use std::{cell::{Cell, UnsafeCell}, thread::ThreadId};

pub use keyboard::KeyboardMatrix;
pub use lcd::{LcdController, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
pub use llama::state::LlamaState as CpuState;
pub use memory::{
    AccessKind, MemoryAccessLog, MemoryImage, MemoryOverlay, ADDRESS_MASK, EXTERNAL_SPACE,
    INTERNAL_ADDR_MASK, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START,
    INTERNAL_SPACE,
};
pub use perfetto::PerfettoTracer;
pub struct PerfettoHandle {
    owner: Cell<Option<ThreadId>>,
    depth: Cell<usize>,
    inner: UnsafeCell<Option<PerfettoTracer>>,
    gate: std::sync::Mutex<()>,
}

pub struct PerfettoGuard<'a> {
    handle: &'a PerfettoHandle,
    gate: Option<std::sync::MutexGuard<'a, ()>>,
    released: bool,
}

impl PerfettoHandle {
    pub const fn new() -> Self {
        Self {
            owner: Cell::new(None),
            depth: Cell::new(0),
            inner: UnsafeCell::new(None),
            gate: std::sync::Mutex::new(()),
        }
    }

    pub fn enter(&self) -> PerfettoGuard<'_> {
        let tid = std::thread::current().id();
        if self.owner.get() == Some(tid) {
            self.depth.set(self.depth.get().saturating_add(1));
            return PerfettoGuard {
                handle: self,
                gate: None,
                released: false,
            };
        }
        let gate = self
            .gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        debug_assert!(self.owner.get().is_none());
        self.owner.set(Some(tid));
        self.depth.set(1);
        PerfettoGuard {
            handle: self,
            gate: Some(gate),
            released: false,
        }
    }
}

unsafe impl Sync for PerfettoHandle {}

impl<'a> PerfettoGuard<'a> {
    pub fn tracer_mut(&mut self) -> Option<&mut PerfettoTracer> {
        unsafe { &mut *self.handle.inner.get() }.as_mut()
    }

    pub fn take(&mut self) -> Option<PerfettoTracer> {
        std::mem::take(unsafe { &mut *self.handle.inner.get() })
    }
}

impl<'a> Drop for PerfettoGuard<'a> {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let depth = self.handle.depth.get();
        if depth <= 1 {
            self.handle.depth.set(0);
            self.handle.owner.set(None);
            // Drop the gate guard on the root acquisition to allow other threads in.
            drop(self.gate.take());
        } else {
            self.handle.depth.set(depth - 1);
        }
        self.released = true;
    }
}

impl<'a> std::ops::Deref for PerfettoGuard<'a> {
    type Target = Option<PerfettoTracer>;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.handle.inner.get() }
    }
}

impl<'a> std::ops::DerefMut for PerfettoGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.handle.inner.get() }
    }
}
lazy_static::lazy_static! {
    pub static ref PERFETTO_TRACER: PerfettoHandle = PerfettoHandle::new();
}
pub use snapshot::{
    load_snapshot, pack_registers, save_snapshot, unpack_registers, SnapshotLoad, SNAPSHOT_MAGIC,
    SNAPSHOT_REGISTER_LAYOUT, SNAPSHOT_VERSION,
};
pub use timer::TimerContext;

use crate::keyboard::KeyboardSnapshot;
use crate::llama::eval::{perfetto_last_pc, LlamaBus};
use crate::llama::state::mask_for;
use crate::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_KIL_OFFSET};

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
    #[serde(default)]
    pub delivered_masks: Vec<u8>,
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

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("True")
    )
}

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
            state.set_reg(
                RegName::Temp(idx),
                *value & mask_for_width(DEFAULT_REG_WIDTH),
            );
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
    pub timer: Box<TimerContext>,
    host_read: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
    host_write: Option<Box<dyn FnMut(u32, u8) + Send>>,
    onk_level: bool,
}

impl Default for CoreRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreRuntime {
    pub fn new() -> Self {
        let mut rt = Self {
            metadata: SnapshotMetadata::default(),
            memory: MemoryImage::new(),
            state: LlamaState::new(),
            fast_mode: false,
            executor: crate::llama::eval::LlamaExecutor::new(),
            keyboard: Some(KeyboardMatrix::new()),
            lcd: Some(LcdController::new()),
            timer: Box::new(TimerContext::new(false, 0, 0)),
            host_read: None,
            host_write: None,
            onk_level: false,
        };
        rt.install_imr_isr_hook();
        rt
    }

    /// Provide an optional host overlay reader for IMEM regions that require Python/device handling
    /// (e.g., E-port inputs, ONK). Called only when `MemoryImage::requires_python` flags an address.
    pub fn set_host_read<F>(&mut self, f: F)
    where
        F: FnMut(u32) -> Option<u8> + Send + 'static,
    {
        self.host_read = Some(Box::new(f));
    }

    /// Provide an optional host overlay writer for IMEM regions that require Python/device handling.
    pub fn set_host_write<F>(&mut self, f: F)
    where
        F: FnMut(u32, u8) + Send + 'static,
    {
        self.host_write = Some(Box::new(f));
    }

    /// Clear any host overlay handlers.
    pub fn clear_host_overlays(&mut self) {
        self.host_read = None;
        self.host_write = None;
    }

    fn install_imr_isr_hook(&mut self) {
        let timer_ptr: *mut TimerContext = self.timer.as_mut() as *mut TimerContext;
        self.memory.set_imr_isr_hook(Some(move |offset, prev, new| {
            let pc = crate::llama::eval::perfetto_instr_context()
                .map(|(_, pc)| pc)
                .unwrap_or_else(crate::llama::eval::perfetto_last_pc);
            unsafe {
                let timer = &mut *timer_ptr;
                let reg_name = if offset == IMEM_IMR_OFFSET { "IMR" } else { "ISR" };
                timer.record_bit_watch_transition(reg_name, prev, new, pc);
                if offset == IMEM_IMR_OFFSET {
                    timer.irq_imr = new;
                } else if offset == IMEM_ISR_OFFSET {
                    timer.irq_isr = new;
                }
                let mut guard = PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert("pc".to_string(), perfetto::AnnotationValue::Pointer(pc as u64));
                    payload.insert("prev".to_string(), perfetto::AnnotationValue::UInt(prev as u64));
                    payload.insert("value".to_string(), perfetto::AnnotationValue::UInt(new as u64));
                    payload.insert(
                        "imr".to_string(),
                        perfetto::AnnotationValue::UInt(timer.irq_imr as u64),
                    );
                    payload.insert(
                        "isr".to_string(),
                        perfetto::AnnotationValue::UInt(timer.irq_isr as u64),
                    );
                    let name = if offset == IMEM_IMR_OFFSET {
                        "IMR_Write"
                    } else {
                        "ISR_Write"
                    };
                    tracer.record_irq_event(name, payload);
                }
            }
        }));
    }

    /// Set the ON key level high and assert ISR.ONKI/IRQ pending to mirror Python KEY_ON handling.
    pub fn press_on_key(&mut self) {
        self.onk_level = true;
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if (isr & ISR_ONKI) == 0 {
            let new_isr = isr | ISR_ONKI;
            self.memory
                .write_internal_byte(IMEM_ISR_OFFSET, new_isr);
            self.timer
                .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
        }
        self.timer.irq_pending = true;
        self.timer.irq_source = Some("ONK".to_string());
        self.timer.last_fired = self.timer.irq_source.clone();
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            let mut payload = std::collections::HashMap::new();
            payload.insert(
                "pc".to_string(),
                perfetto::AnnotationValue::Pointer(perfetto_last_pc() as u64),
            );
            payload.insert(
                "imr".to_string(),
                perfetto::AnnotationValue::UInt(self.timer.irq_imr as u64),
            );
            payload.insert(
                "isr".to_string(),
                perfetto::AnnotationValue::UInt(self.timer.irq_isr as u64),
            );
            payload.insert(
                "src".to_string(),
                perfetto::AnnotationValue::Str("ONK".to_string()),
            );
            tracer.record_irq_event("KeyIRQ", payload);
        }
    }

    /// Clear the ON key level and deassert ONKI in ISR (firmware may also clear ONKI).
    pub fn release_on_key(&mut self) {
        self.onk_level = false;
        if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
            let new_isr = isr & !ISR_ONKI;
            self.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
            self.timer
                .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
            self.timer.irq_isr = new_isr;
        }
    }

    pub fn add_ram_overlay(&mut self, start: u32, size: usize, name: &str) {
        self.memory.add_ram_overlay(start, size, name);
    }

    pub fn add_rom_overlay(&mut self, start: u32, data: &[u8], name: &str) {
        self.memory.add_rom_overlay(start, data, name);
    }

    pub fn load_memory_card(&mut self, data: &[u8]) -> Result<()> {
        self.memory.load_memory_card(data)
    }

    pub fn overlay_read_log(&self) -> Vec<MemoryAccessLog> {
        self.memory.overlay_read_log()
    }

    pub fn overlay_write_log(&self) -> Vec<MemoryAccessLog> {
        self.memory.overlay_write_log()
    }

    pub fn clear_overlay_logs(&self) {
        self.memory.clear_overlay_logs();
    }

    /// Set the E-port input buffer values (EIL/EIH) to emulate external pin state.
    pub fn set_e_port_inputs(&mut self, low: u8, high: u8) {
        self.memory.write_internal_byte(0xF5, low);
        self.memory.write_internal_byte(0xF6, high);
    }

    fn maybe_force_keyboard_activity(&mut self, timer_fired: bool) {
        let force_strobe = env_flag("FORCE_STROBE_LLAMA");
        let force_keyi = env_flag("FORCE_KEYI_LLAMA");
        // Parity: Python only runs these debug hooks when the timer fires.
        if !timer_fired || (!force_strobe && !force_keyi) {
            return;
        }
        let Some(kb) = self.keyboard.as_mut() else {
            return;
        };
        if force_strobe {
            let _ = kb.handle_write(0xF0, 0xFF, &mut self.memory);
            let _ = kb.handle_write(0xF1, 0x0F, &mut self.memory);
            let events = kb.scan_tick(self.timer.kb_irq_enabled);
            let stats = kb.telemetry();
            if events > 0 {
                kb.write_fifo_to_memory(&mut self.memory, self.timer.kb_irq_enabled);
            }
            // Emit a scan marker to mirror Python's forced strobe diagnostics.
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "events".to_string(),
                    perfetto::AnnotationValue::UInt(events as u64),
                );
                payload.insert(
                    "strobe_count".to_string(),
                    perfetto::AnnotationValue::UInt(kb.strobe_count().into()),
                );
                payload.insert(
                    "imr".to_string(),
                    perfetto::AnnotationValue::UInt(
                        self.memory
                            .read_internal_byte(IMEM_IMR_OFFSET)
                            .unwrap_or(self.timer.irq_imr) as u64,
                    ),
                );
                payload.insert(
                    "isr".to_string(),
                    perfetto::AnnotationValue::UInt(
                        self.memory
                            .read_internal_byte(IMEM_ISR_OFFSET)
                            .unwrap_or(self.timer.irq_isr) as u64,
                    ),
                );
                payload.insert(
                    "kol".to_string(),
                    perfetto::AnnotationValue::UInt(stats.kol as u64),
                );
                payload.insert(
                    "koh".to_string(),
                    perfetto::AnnotationValue::UInt(stats.koh as u64),
                );
                payload.insert(
                    "pressed".to_string(),
                    perfetto::AnnotationValue::UInt(stats.pressed as u64),
                );
                payload.insert(
                    "active_cols".to_string(),
                    perfetto::AnnotationValue::Str(format!("{:?}", kb.active_columns())),
                );
                payload.insert(
                    "pc".to_string(),
                    perfetto::AnnotationValue::Pointer(perfetto_last_pc() as u64),
                );
                tracer.record_irq_event("KeyScanEvent", payload);
            }
        }
        let fifo_non_empty = kb.fifo_len() > 0;
        if force_keyi || (force_strobe && fifo_non_empty) {
            let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
            if (isr & ISR_KEYI) == 0 {
                let new_isr = isr | ISR_KEYI;
                self.memory
                    .write_internal_byte(IMEM_ISR_OFFSET, new_isr);
                self.timer
                    .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
            }
            self.timer.irq_pending = true;
            self.timer.irq_source = Some("KEY".to_string());
            self.timer.last_fired = self.timer.irq_source.clone();
            self.timer.irq_isr = self
                .memory
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(self.timer.irq_isr);
            self.timer.irq_imr = self
                .memory
                .read_internal_byte(IMEM_IMR_OFFSET)
                .unwrap_or(self.timer.irq_imr);
            self.timer.key_irq_latched = self.timer.key_irq_latched || fifo_non_empty;
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "pc".to_string(),
                    perfetto::AnnotationValue::Pointer(perfetto_last_pc() as u64),
                );
                payload.insert(
                    "imr".to_string(),
                    perfetto::AnnotationValue::UInt(self.timer.irq_imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    perfetto::AnnotationValue::UInt(self.timer.irq_isr as u64),
                );
                payload.insert(
                    "src".to_string(),
                    perfetto::AnnotationValue::Str("KEY".to_string()),
                );
                payload.insert(
                    "y".to_string(),
                    perfetto::AnnotationValue::Pointer(self.state.get_reg(RegName::Y) as u64),
                );
                tracer.record_irq_event("KeyIRQ", payload);
            }
        }
    }

    fn refresh_key_irq_latch(&mut self) {
        if self.timer.in_interrupt {
            // Parity: do not reassert KEYI while already in an interrupt handler.
            return;
        }
        if !self.timer.kb_irq_enabled {
            self.timer.key_irq_latched = false;
            return;
        }
        if let Some(kb) = self.keyboard.as_ref() {
            let fifo_len = kb.fifo_len();
            if fifo_len > 0 || self.timer.key_irq_latched {
                self.timer.key_irq_latched = true;
                let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                if (isr & ISR_KEYI) == 0 {
                    let new_isr = isr | ISR_KEYI;
                    self.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
                    self.timer
                        .record_bit_watch_transition("ISR", isr, new_isr, perfetto_last_pc());
                    self.timer.irq_isr = new_isr;
                } else {
                    self.timer.irq_isr = isr;
                }
                self.timer.irq_pending = true;
                if self.timer.irq_source.is_none() {
                    self.timer.irq_source = Some("KEY".to_string());
                }
                self.timer.last_fired = self.timer.irq_source.clone();
                self.timer.irq_imr = self
                    .memory
                    .read_internal_byte(IMEM_IMR_OFFSET)
                    .unwrap_or(self.timer.irq_imr);
            } else {
                self.timer.key_irq_latched = false;
            }
        }
    }

    fn arm_pending_irq_from_isr(&mut self) {
        if self.timer.in_interrupt {
            return;
        }
        if self.timer.irq_pending {
            return;
        }
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if isr == 0 {
            return;
        }
        let mut isr_effective = isr;
        if !self.timer.kb_irq_enabled {
            isr_effective &= !ISR_KEYI;
        }
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        let irm_enabled = (imr & IMR_MASTER) != 0;
        if !irm_enabled {
            return;
        }
        if (imr & isr_effective) == 0 {
            // Nothing enabled by IMR (including keyboard/on-key) — keep running.
            return;
        }
        self.timer.irq_pending = true;
        self.timer.irq_isr = isr_effective;
        self.timer.irq_imr = imr;
        // Prefer keyboard/ONK over timers, then MTI, then STI.
        let src = if (isr_effective & ISR_KEYI) != 0 {
            "KEY"
        } else if (isr_effective & ISR_ONKI) != 0 {
            "ONK"
        } else if (isr_effective & ISR_MTI) != 0 {
            "MTI"
        } else if (isr_effective & ISR_STI) != 0 {
            "STI"
        } else {
            "IRQ"
        };
        // Allow a newly latched KEY/ONK to override earlier timer sources to match Python priority.
        match self.timer.irq_source.as_deref() {
            None => self.timer.irq_source = Some(src.to_string()),
            Some(cur) => {
                if (src == "KEY" || src == "ONK") && cur != "KEY" && cur != "ONK" {
                    self.timer.irq_source = Some(src.to_string());
                }
            }
        }
        self.timer.last_fired = self.timer.irq_source.clone();
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            let kil = self.memory.read_internal_byte(IMEM_KIL_OFFSET).unwrap_or(0);
            let imr_reg = self.state.get_reg(RegName::IMR) as u8;
            tracer.record_irq_check(
                "IRQ_PendingArm",
                self.state.pc() & ADDRESS_MASK,
                imr,
                isr,
                self.timer.irq_pending,
                self.timer.in_interrupt,
                self.timer.irq_source.as_deref(),
                Some(kil),
                Some(imr_reg),
            );
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
            keyboard_ptr: *mut KeyboardMatrix,
            lcd_ptr: *mut LcdController,
            host_read: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            host_write: Option<*mut (dyn FnMut(u32, u8) + Send)>,
            onk_level: bool,
            #[allow(dead_code)]
            cycle: u64,
            #[allow(dead_code)]
            pc: u32,
            #[allow(dead_code)]
            meta_ptr: *const SnapshotMetadata,
            #[allow(dead_code)]
            state_ptr: *const LlamaState,
            overlay_fault_addr: Option<u32>,
        }
        impl<'a> LlamaBus for RuntimeBus<'a> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                // Route keyboard/LCD accesses to their devices for parity with Python overlays.
                unsafe {
                    let python_required = (*self.mem).requires_python(addr);
                    // Keyboard: internal IMEM offsets 0xF0-0xF2.
                    if !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK as u32
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK as u32;
                        if let Some(val) = (*self.keyboard_ptr).handle_read(offset, &mut *self.mem)
                        {
                            (*self.mem).bump_read_count();
                            (*self.mem).log_kio_read(offset, val);
                            return val as u32;
                        }
                        if offset <= 0x0F && !self.lcd_ptr.is_null() {
                            // IMEM overlay for the LCD controller; mirror to the 0x2000 map.
                            let mapped = crate::lcd::overlay_addr(offset);
                            if let Some(val) = (*self.lcd_ptr).read(mapped) {
                                (*self.mem).bump_read_count();
                                return val as u32;
                            }
                        }
                    }
                    // LCD controller mirrored at 0x2000/0xA000.
                    if !self.lcd_ptr.is_null() && (*self.lcd_ptr).handles(addr) {
                        if let Some(val) = (*self.lcd_ptr).read(addr) {
                            (*self.mem).bump_read_count();
                            return val as u32;
                        }
                    }
                    // Host overlay: delegate addresses flagged as Python-only.
                    if python_required {
                        if let Some(cb) = self.host_read {
                            if let Some(val) = (*cb)(addr) {
                                (*self.mem).bump_read_count();
                                return val as u32;
                            }
                        }
                        // If we reach here, the address requires Python handling but no callback is present.
                        // Emit a perfetto warning so traces surface the divergence and stop execution.
                        let mut guard = PERFETTO_TRACER.enter();
                        if let Some(tracer) = guard.as_mut() {
                            let mut payload = std::collections::HashMap::new();
                            payload.insert(
                                "addr".to_string(),
                                perfetto::AnnotationValue::Pointer(addr as u64),
                            );
                            payload.insert(
                                "pc".to_string(),
                                perfetto::AnnotationValue::Pointer(self.pc as u64),
                            );
                            tracer.record_irq_event("PythonOverlayMissing", payload);
                        }
                        if self.overlay_fault_addr.is_none() {
                            self.overlay_fault_addr = Some(addr);
                        }
                        return 0;
                    }
                    // SSR (0xFF) must reflect ONK level even without host overlays to match Python/Perfetto.
                    if MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK as u32
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK as u32;
                        if offset == 0xFF {
                            let mut val = (*self.mem).read_internal_byte(offset).unwrap_or(0);
                            if self.onk_level {
                                val |= 0x08;
                            }
                            return val as u32;
                        }
                    }
                    (*self.mem)
                        .load_with_pc(addr, bits, Some(self.pc))
                        .unwrap_or(0)
                }
            }
            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                unsafe {
                    let python_required = (*self.mem).requires_python(addr);
                    // Keyboard KOL/KOH/KIL writes.
                    if !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK as u32
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK as u32;
                        if (0xF0..=0xF2).contains(&offset) {
                            if (*self.keyboard_ptr).handle_write(
                                offset,
                                value as u8,
                                &mut *self.mem,
                            ) {
                                // Mirror writes into IMEM except when the handler already wrote KIL.
                                if offset != 0xF2 {
                                    let _ = (*self.mem).store(addr, bits, value);
                                }
                                return;
                            }
                        }
                        if offset <= 0x0F && !self.lcd_ptr.is_null() {
                            let mapped = crate::lcd::overlay_addr(offset);
                            (*self.mem).bump_write_count();
                            (*self.lcd_ptr).write(mapped, value as u8);
                            return;
                        }
                    }
                    // LCD writes.
                    if !self.lcd_ptr.is_null() && (*self.lcd_ptr).handles(addr) {
                        (*self.lcd_ptr).write(addr, value as u8);
                        let _ = (*self.mem).store(addr, bits, value);
                        return;
                    }
                    if python_required {
                        if let Some(cb) = self.host_write {
                            (*cb)(addr, value as u8);
                            // Parity: overlay writes should still count as memory writes and emit Perfetto traces.
                            (*self.mem).bump_write_count();
                            let mut guard = PERFETTO_TRACER.enter();
                            if let Some(tracer) = guard.as_mut() {
                                if let Some((op_idx, pc_ctx)) =
                                    crate::llama::eval::perfetto_instr_context()
                                {
                                    tracer.record_mem_write(
                                        op_idx,
                                        pc_ctx,
                                        addr,
                                        value as u32,
                                        "python_overlay",
                                        bits,
                                    );
                                } else {
                                    tracer.record_mem_write_at_cycle(
                                        self.cycle,
                                        Some(self.pc),
                                        addr,
                                        value as u32,
                                        "python_overlay",
                                        bits,
                                    );
                                }
                            }
                            return;
                        }
                        let mut guard = PERFETTO_TRACER.enter();
                        if let Some(tracer) = guard.as_mut() {
                            let mut payload = std::collections::HashMap::new();
                            payload.insert(
                                "addr".to_string(),
                                perfetto::AnnotationValue::Pointer(addr as u64),
                            );
                            payload.insert(
                                "pc".to_string(),
                                perfetto::AnnotationValue::Pointer(self.pc as u64),
                            );
                            payload.insert(
                                "value".to_string(),
                                perfetto::AnnotationValue::UInt(value as u64),
                            );
                            tracer.record_irq_event("PythonOverlayMissing", payload);
                        }
                        if self.overlay_fault_addr.is_none() {
                            self.overlay_fault_addr = Some(addr);
                        }
                        return;
                    }
                    let _ = (*self.mem).store_with_pc(addr, bits, value, Some(self.pc));
                }
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
            // Emit a diagnostic IRQ_Check parity marker mirroring Python’s early pending probe.
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
                let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                let kil = self.memory.read_internal_byte(IMEM_KIL_OFFSET).unwrap_or(0);
                let imr_reg = self.state.get_reg(RegName::IMR) as u8;
                let pending_src = self
                    .timer
                    .irq_source
                    .as_deref()
                    .map(str::to_string)
                    .or_else(|| {
                        if (isr & ISR_KEYI) != 0 {
                            Some("KEY".to_string())
                        } else if (isr & ISR_ONKI) != 0 {
                            Some("ONK".to_string())
                        } else if (isr & ISR_MTI) != 0 {
                            Some("MTI".to_string())
                        } else if (isr & ISR_STI) != 0 {
                            Some("STI".to_string())
                        } else {
                            None
                        }
                    });
                tracer.record_irq_check(
                    "IRQ_Check",
                    self.state.pc() & ADDRESS_MASK,
                    imr,
                    isr,
                    self.timer.irq_pending,
                    self.timer.in_interrupt,
                    pending_src.as_deref(),
                    Some(kil),
                    Some(imr_reg),
                );
            }
            // Reassert KEYI if the FIFO still holds events even after firmware clears ISR.
            self.refresh_key_irq_latch();
            // If ISR already has pending bits (e.g., host write) arm a pending IRQ so delivery can occur once IMR allows it.
            self.arm_pending_irq_from_isr();

            // HALT wake-up: exit low-power state when any ISR bit is set, even if IMR is masked.
            if self.state.is_halted() {
                if let Some(mut isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                    // When keyboard IRQs are disabled, ignore KEYI for HALT wake to mirror Python gating.
                    if !self.timer.kb_irq_enabled {
                        isr &= !ISR_KEYI;
                    }
                    if isr != 0 {
                        self.state.set_halted(false);
                        self.timer.irq_pending = true;
                        self.timer.irq_isr = isr;
                        self.timer.irq_imr = self
                            .memory
                            .read_internal_byte(IMEM_IMR_OFFSET)
                            .unwrap_or(self.timer.irq_imr);
                        if self.timer.irq_source.is_none() {
                            let src = if (isr & ISR_KEYI) != 0 {
                                "KEY"
                            } else if (isr & ISR_ONKI) != 0 {
                                "ONK"
                            } else if (isr & ISR_MTI) != 0 {
                                "MTI"
                            } else if (isr & ISR_STI) != 0 {
                                "STI"
                            } else {
                                "IRQ"
                            };
                            self.timer.irq_source = Some(src.to_string());
                        }
                        self.timer.last_fired = self.timer.irq_source.clone();
                    }
                }
            }

            // Halted cores still consume idle cycles and allow timers/IRQs to run.
            if self.state.is_halted() {
                let prev_cycle = self.metadata.cycle_count;
                let new_cycle = prev_cycle.wrapping_add(1);
                self.metadata.cycle_count = new_cycle;
                if !self.timer.in_interrupt {
                    let kb_irq_enabled = self.timer.kb_irq_enabled;
                    let (mti, sti, _key_events, _kb_stats) =
                        self.timer.tick_timers_with_keyboard(
                            &mut self.memory,
                            new_cycle,
                            |mem| {
                                if let Some(kb) = self.keyboard.as_mut() {
                                    let events = kb.scan_tick(kb_irq_enabled);
                                    if events > 0 {
                                        kb.write_fifo_to_memory(mem, kb_irq_enabled);
                                    }
                                    (events, kb.fifo_len() > 0, Some(kb.telemetry()))
                                } else {
                                    (0, false, None)
                                }
                            },
                            Some(self.state.get_reg(RegName::Y)),
                            Some(self.state.get_reg(RegName::PC)),
                        );
                    if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                        self.timer.irq_isr = isr;
                    }
                    self.maybe_force_keyboard_activity(mti || sti);
                    // Reassert KEYI latch only when enabled, mirroring Python HALT wake behavior.
                    self.refresh_key_irq_latch();
                }
                self.deliver_pending_irq()?;
                continue;
            }

            let keyboard_ptr = self
                .keyboard
                .as_mut()
                .map(|kb| kb as *mut KeyboardMatrix)
                .unwrap_or(std::ptr::null_mut());
            let lcd_ptr = self
                .lcd
                .as_mut()
                .map(|lcd| lcd as *mut LcdController)
                .unwrap_or(std::ptr::null_mut());
            let host_read = self
                .host_read
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
            let host_write = self
                .host_write
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32, u8) + Send));
            let pc = self.state.get_reg(RegName::PC) & ADDRESS_MASK;
            let mut bus = RuntimeBus {
                mem: &mut self.memory,
                keyboard_ptr,
                lcd_ptr,
                host_read,
                host_write,
                onk_level: self.onk_level,
                cycle: self.metadata.cycle_count,
                pc,
                meta_ptr: &self.metadata as *const SnapshotMetadata,
                state_ptr: &self.state as *const LlamaState,
                overlay_fault_addr: None,
            };
            let opcode = bus.load(pc, 8) as u8;
            if let Some(addr) = bus.overlay_fault_addr {
                return Err(CoreError::Other(format!(
                    "python overlay required for 0x{addr:06X} but no host handler is installed"
                )));
            }
            // Capture WAIT loop count before execution (executor clears I).
            let wait_loops = if opcode == 0xEF {
                self.state.get_reg(RegName::I) & mask_for(RegName::I)
            } else {
                0
            };
            if let Err(e) = self.executor.execute(opcode, &mut self.state, &mut bus) {
                return Err(CoreError::Other(format!(
                    "execute opcode 0x{opcode:02X}: {e}"
                )));
            }
            if let Some(addr) = bus.overlay_fault_addr {
                return Err(CoreError::Other(format!(
                    "python overlay required for 0x{addr:06X} but no host handler is installed"
                )));
            }
            if opcode == 0xFF {
                // RESET intrinsic: Python only adjusts IMEM + PC; preserve timer/counter state and
                // refresh mirrors from IMEM without clearing counters/bit-watch.
                self.timer.irq_imr = self
                    .memory
                    .read_internal_byte(IMEM_IMR_OFFSET)
                    .unwrap_or(self.timer.irq_imr);
                self.timer.irq_isr = self
                    .memory
                    .read_internal_byte(IMEM_ISR_OFFSET)
                    .unwrap_or(self.timer.irq_isr);
            }
            // IR intrinsic bookkeeping: align timer metadata with Python intrinsic IRQ handling.
            if opcode == 0xFE {
                self.timer.in_interrupt = true;
                self.timer.irq_pending = false;
                self.timer.irq_source = Some("IR".to_string());
                self.timer.last_fired = self.timer.irq_source.clone();
                self.timer.irq_isr = self
                    .memory
                    .read_internal_byte(IMEM_ISR_OFFSET)
                    .unwrap_or(self.timer.irq_isr);
                self.timer.irq_imr = self
                    .memory
                    .read_internal_byte(IMEM_IMR_OFFSET)
                    .unwrap_or(self.timer.irq_imr);
                self.timer.last_irq_src = Some("IR".to_string());
                self.timer.last_irq_pc = Some(pc & ADDRESS_MASK);
                let vec = (self.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap_or(0))
                    | (self.memory.load(INTERRUPT_VECTOR_ADDR + 1, 8).unwrap_or(0) << 8)
                    | (self.memory.load(INTERRUPT_VECTOR_ADDR + 2, 8).unwrap_or(0) << 16);
                self.timer.last_irq_vector = Some(vec & ADDRESS_MASK);
            }
            self.metadata.instruction_count = self.metadata.instruction_count.wrapping_add(1);

            // Advance cycles: one for the opcode plus simulated WAIT idle cycles, mirroring Python
            // _simulate_wait which burns I cycles and ticks timers/keyboard each iteration.
            let run_timer_cycles = true;
            let cycle_increment = 1u64.wrapping_add(wait_loops as u64);
            let prev_cycle = self.metadata.cycle_count;
            let new_cycle = prev_cycle.wrapping_add(cycle_increment);
            if run_timer_cycles {
                for cyc in prev_cycle + 1..=new_cycle {
                    let mut timer_fired = false;
                    if !self.timer.in_interrupt {
                        let _kb_irq_enabled = self.timer.kb_irq_enabled;
                        let kb_irq_enabled = self.timer.kb_irq_enabled;
                        let (mti, sti, _key_events, _kb_stats) = self.timer.tick_timers_with_keyboard(
                            &mut self.memory,
                            cyc,
                            |mem| {
                                if let Some(kb) = self.keyboard.as_mut() {
                                    let events = kb.scan_tick(kb_irq_enabled);
                                    if events > 0 {
                                        kb.write_fifo_to_memory(mem, kb_irq_enabled);
                                    }
                                    (events, kb.fifo_len() > 0, Some(kb.telemetry()))
                                } else {
                                    (0, false, None)
                                }
                            },
                            Some(self.state.get_reg(RegName::Y)),
                            Some(self.state.get_reg(RegName::PC)),
                        );
                        timer_fired = mti || sti;
                        // KEYI delivery is handled inside tick_timers_with_keyboard and respects kb_irq_enabled.
                        if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                            self.timer.irq_isr = isr;
                        }
                    }
                    self.maybe_force_keyboard_activity(timer_fired);
                }
            }
            self.metadata.cycle_count = new_cycle;
            if opcode == 0x01 {
                let irq_src = self.timer.irq_source.clone();
                // If irq_source was lost, fall back to the delivered mask stack or live ISR bits.
                let stack_mask = self.timer.delivered_masks.pop();
                let clear_mask = irq_src
                    .as_deref()
                    .and_then(src_mask_for_name)
                    .or(stack_mask)
                    .or_else(|| {
                        self.memory
                            .read_internal_byte(IMEM_ISR_OFFSET)
                            .and_then(|isr| {
                                if (isr & ISR_KEYI) != 0 {
                                    Some(ISR_KEYI)
                                } else if (isr & ISR_ONKI) != 0 {
                                    Some(ISR_ONKI)
                                } else if (isr & ISR_MTI) != 0 {
                                    Some(ISR_MTI)
                                } else if (isr & ISR_STI) != 0 {
                                    Some(ISR_STI)
                                } else {
                                    None
                                }
                            })
                    });
                self.timer.in_interrupt = false;
                if irq_src.as_deref().is_some_and(|s| s == "KEY") {
                    self.timer.key_irq_latched = false;
                }
                self.timer.irq_source = None;
                // Clear the delivered ISR bit, preferring explicit source but falling back to the saved mask.
                if let Some(src_mask) = clear_mask {
                    let isr_addr = INTERNAL_MEMORY_START + IMEM_ISR_OFFSET;
                    if let Some(isr_val) = self.memory.load(isr_addr, 8) {
                        let prev = isr_val as u8;
                        let cleared = prev & (!src_mask);
                        let _ = self.memory.store(isr_addr, 8, cleared as u32);
                        self.timer.record_bit_watch_transition(
                            "ISR",
                            prev,
                            cleared,
                            pc,
                        );
                    }
                }
                // Drop any stale interrupt-stack frames (used only for bookkeeping).
                let _ = self.timer.interrupt_stack.pop();
                let mut guard = PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert(
                        "pc".to_string(),
                        perfetto::AnnotationValue::Pointer(pc as u64),
                    );
                    payload.insert(
                        "ret".to_string(),
                        perfetto::AnnotationValue::Pointer(self.state.pc() as u64),
                    );
                    payload.insert(
                        "src".to_string(),
                        perfetto::AnnotationValue::Str(
                            irq_src.unwrap_or_else(|| "".to_string()),
                        ),
                    );
                    if let Some(mask) = clear_mask {
                        payload.insert(
                            "mask".to_string(),
                            perfetto::AnnotationValue::UInt(mask as u64),
                        );
                    }
                    payload.insert(
                        "imr".to_string(),
                        perfetto::AnnotationValue::UInt(self.state.get_reg(RegName::IMR) as u64),
                    );
                    tracer.record_irq_event("IRQ_Return", payload);
                }
            }
            self.deliver_pending_irq()?;
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                tracer.update_counters(
                    self.metadata.instruction_count,
                    self.state.call_depth(),
                    self.memory.memory_read_count(),
                    self.memory.memory_write_count(),
                );
            }
        }
        Ok(())
    }

    pub fn save_snapshot(&self, path: &std::path::Path) -> Result<()> {
        let mut metadata = self.metadata.clone();
        metadata.instruction_count = self.metadata.instruction_count;
        metadata.cycle_count = self.metadata.cycle_count;
        metadata.pc = self.get_reg("PC");
        metadata.memory_reads = self.memory.memory_read_count();
        metadata.memory_writes = self.memory.memory_write_count();
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
                    "kil_reads": kb_state.kil_read_count,
                    "kb_irq_enabled": self.timer.kb_irq_enabled,
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
        if metadata.interrupts.irq_bit_watch.is_none() {
            metadata.interrupts.irq_bit_watch = self
                .timer
                .irq_bit_watch
                .clone()
                .map(serde_json::Value::Object);
        }
        let regs = collect_registers(&self.state);
        snapshot::save_snapshot(path, &metadata, &regs, &self.memory, lcd_payload.as_deref())
    }

    pub fn load_snapshot(&mut self, path: &std::path::Path) -> Result<()> {
        let loaded = snapshot::load_snapshot(path, &mut self.memory)?;
        self.metadata = loaded.metadata;
        apply_registers(&mut self.state, &loaded.registers);
        self.fast_mode = self.metadata.fast_mode;
        self.memory
            .set_memory_counts(self.metadata.memory_reads, self.metadata.memory_writes);
        let allow_timer_scale = matches!(
            self.metadata.backend.as_str(),
            b if b.eq_ignore_ascii_case("core")
                || b.eq_ignore_ascii_case("llama")
                || b.eq_ignore_ascii_case("rust")
        );
        self.timer.apply_snapshot_info(
            &self.metadata.timer,
            &self.metadata.interrupts,
            self.metadata.cycle_count,
            allow_timer_scale,
        );
        if let Some(watch) = self.metadata.interrupts.irq_bit_watch.as_ref() {
            self.timer.irq_bit_watch = watch.as_object().map(|obj| obj.clone());
        }
        if self.keyboard.is_none() {
            self.keyboard = Some(KeyboardMatrix::new());
        }
        if let (Some(kb_meta), Some(kb)) = (self.metadata.keyboard.clone(), self.keyboard.as_mut())
        {
            if let Ok(snapshot) = serde_json::from_value::<KeyboardSnapshot>(kb_meta) {
                kb.load_snapshot_state(&snapshot);
            }
        }
        if self.lcd.is_none() {
            self.lcd = Some(LcdController::new());
        }
        if let (Some(lcd_meta), Some(payload), Some(lcd)) = (
            self.metadata.lcd.clone(),
            loaded.lcd_payload.as_deref(),
            self.lcd.as_mut(),
        ) {
            let _ = lcd.load_snapshot(&lcd_meta, payload);
        }
        // Restore call depth/sub-level and temps from metadata if present.
        if self.metadata.call_depth > 0 {
            for _ in 0..self.metadata.call_depth {
                self.state.call_depth_inc();
            }
        }
        self.state.set_call_sub_level(self.metadata.call_sub_level);
        for (name, value) in self.metadata.temps.iter() {
            if let Some(idx_str) = name.strip_prefix("TEMP") {
                if let Ok(idx) = idx_str.parse::<u8>() {
                    self.state.set_reg(
                        RegName::Temp(idx),
                        *value & mask_for_width(DEFAULT_REG_WIDTH),
                    );
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

    fn deliver_pending_irq(&mut self) -> Result<()> {
        if !self.timer.irq_pending {
            return Ok(());
        }
        let pc = self.state.pc() & ADDRESS_MASK;
        #[cfg(test)]
        {
            let _ = pc;
        }
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let mut irm_enabled = (imr & IMR_MASTER) != 0;
        // Parity: allow KEYI/ONKI delivery to proceed even if IRM is clear so level-triggered
        // keyboard/on-key events are not dropped before firmware enables the master bit.
        if !irm_enabled && (isr & (ISR_KEYI | ISR_ONKI)) != 0 {
            irm_enabled = true;
        }
        let kil = self.memory.read_internal_byte(IMEM_KIL_OFFSET).unwrap_or(0);
        let imr_reg = self.state.get_reg(RegName::IMR) as u8;
        let pending_src = self
            .timer
            .irq_source
            .as_deref()
            .map(str::to_string)
            .or_else(|| {
                if (isr & ISR_KEYI) != 0 {
                    Some("KEY".to_string())
                } else if (isr & ISR_ONKI) != 0 {
                    Some("ONK".to_string())
                } else if (isr & ISR_MTI) != 0 {
                    Some("MTI".to_string())
                } else if (isr & ISR_STI) != 0 {
                    Some("STI".to_string())
                } else {
                    None
                }
            });
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            tracer.record_irq_check(
                "IRQ_PendingCheck",
                pc,
                imr,
                isr,
                self.timer.irq_pending,
                self.timer.in_interrupt,
                pending_src.as_deref(),
                Some(kil),
                Some(imr_reg),
            );
            if imr == 0 {
                tracer.record_irq_check(
                    "IMR_ReadZero",
                    pc,
                    imr,
                    isr,
                    self.timer.irq_pending,
                    self.timer.in_interrupt,
                    pending_src.as_deref(),
                    Some(kil),
                    Some(imr_reg),
                );
            }
        }
        if !irm_enabled {
            return Ok(());
        }
        // Match Python ordering: deliver KEY before ONK when both are set/enabled.
        let src = if (isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0) {
            Some((ISR_KEYI, "KEY"))
        } else if (isr & ISR_ONKI != 0) && (imr & IMR_ONK != 0) {
            Some((ISR_ONKI, "ONK"))
        } else if (isr & ISR_MTI != 0) && (imr & IMR_MTI != 0) {
            Some((ISR_MTI, "MTI"))
        } else if (isr & ISR_STI != 0) && (imr & IMR_STI != 0) {
            Some((ISR_STI, "STI"))
        } else {
            None
        };
        let Some((mask, src_name)) = src else {
            return Ok(());
        };

        // Match Python guard: defer delivery until firmware initializes the stack pointer.
        let sp = self.state.get_reg(RegName::S) & ADDRESS_MASK;
        if sp < 5 {
            // Leave irq_pending intact and surface an error like Python does.
            return Err(CoreError::Other(
                "IRQ deferred: stack pointer not initialized".to_string(),
            ));
        }

        let pc = self.state.pc() & ADDRESS_MASK;
        let (op_idx, pc_trace) = crate::llama::eval::perfetto_instr_context()
            .unwrap_or_else(|| (crate::llama::eval::perfetto_last_instr_index(), pc));

        let record_stack_write = |addr: u32, bits: u8, value: u32| {
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let space = if MemoryImage::is_internal(addr) {
                    "internal"
                } else {
                    "external"
                };
                tracer.record_mem_write(op_idx, pc_trace, addr, value, space, bits);
            }
        };
        // Stack push order mirrors IR intrinsic: PC (24 LE), F, IMR.
        self.push_stack(RegName::S, pc, 24);
        record_stack_write(self.state.get_reg(RegName::S), 24, pc & ADDRESS_MASK);
        let f = self.state.get_reg(RegName::F) & 0xFF;
        self.push_stack(RegName::S, f, 8);
        record_stack_write(self.state.get_reg(RegName::S), 8, f);
        let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
        let imr_mem = self.memory.load(imr_addr, 8).unwrap_or(0) & 0xFF;
        self.push_stack(RegName::S, imr_mem, 8);
        record_stack_write(self.state.get_reg(RegName::S), 8, imr_mem);
        let cleared_imr = (imr_mem as u8) & 0x7F;
        let _ = self.memory.store(imr_addr, 8, cleared_imr as u32);
        self.timer
            .record_bit_watch_transition("IMR", imr_mem as u8, cleared_imr, pc);
        self.state.set_reg(RegName::IMR, cleared_imr as u32);
        record_stack_write(imr_addr, 8, cleared_imr as u32);

        // Jump to vector.
        let vec = (self.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap_or(0))
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 1, 8).unwrap_or(0) << 8)
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 2, 8).unwrap_or(0) << 16);
        // Emit a single delivery marker (matches Python tracer).
        if src_name == "KEY" {
            let mut guard = PERFETTO_TRACER.enter();
            if let Some(tracer) = guard.as_mut() {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "from".to_string(),
                    perfetto::AnnotationValue::Pointer(pc as u64),
                );
                payload.insert(
                    "vector".to_string(),
                    perfetto::AnnotationValue::Pointer(vec as u64),
                );
                payload.insert(
                    "imr".to_string(),
                    perfetto::AnnotationValue::UInt(imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    perfetto::AnnotationValue::UInt(isr as u64),
                );
                payload.insert(
                    "s".to_string(),
                    perfetto::AnnotationValue::Pointer(self.state.get_reg(RegName::S) as u64),
                );
                payload.insert(
                    "src".to_string(),
                    perfetto::AnnotationValue::Str(src_name.to_string()),
                );
                tracer.record_irq_event("KeyDeliver", payload);
            }
        }
        self.state.set_pc(vec & ADDRESS_MASK);
        self.state.set_halted(false);
        // Track interrupt entry in call-depth metrics for parity with Python trace counters.
        self.state.call_depth_inc();

        // Track interrupt metadata similar to Python snapshot fields.
        self.timer.in_interrupt = true;
        self.timer.irq_pending = false;
        self.timer.irq_source = Some(src_name.to_string());
        // Track interrupt metadata similar to Python snapshot fields.
        let irq_id = self.timer.next_interrupt_id;
        self.timer.interrupt_stack.push(irq_id);
        self.timer.delivered_masks.push(mask);
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
        // Track last IRQ metadata with the resolved vector and increment counters.
        self.timer.last_irq_src = Some(src_name.to_string());
        self.timer.last_irq_pc = Some(pc);
        self.timer.last_irq_vector = Some(vec & ADDRESS_MASK);
        self.timer.irq_total = self.timer.irq_total.saturating_add(1);
        match src_name {
            "KEY" => self.timer.irq_key = self.timer.irq_key.saturating_add(1),
            "MTI" => self.timer.irq_mti = self.timer.irq_mti.saturating_add(1),
            "STI" => self.timer.irq_sti = self.timer.irq_sti.saturating_add(1),
            _ => {}
        }

        // Emit an IRQ entry marker for perfetto parity with Python.
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "pc".to_string(),
                    perfetto::AnnotationValue::Pointer(pc as u64),
                );
                payload.insert(
                    "from".to_string(),
                    perfetto::AnnotationValue::Pointer(pc as u64),
                );
                payload.insert(
                    "vector".to_string(),
                    perfetto::AnnotationValue::Pointer(vec as u64),
                );
                payload.insert(
                    "imr_before".to_string(),
                    perfetto::AnnotationValue::UInt(imr as u64),
                );
                payload.insert(
                    "imr_after".to_string(),
                    perfetto::AnnotationValue::UInt(cleared_imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    perfetto::AnnotationValue::UInt(isr as u64),
                );
                payload.insert(
                    "s".to_string(),
                    perfetto::AnnotationValue::Pointer(self.state.get_reg(RegName::S) as u64),
                );
                payload.insert(
                    "y".to_string(),
                    perfetto::AnnotationValue::Pointer(self.state.get_reg(RegName::Y) as u64),
                );
                payload.insert(
                    "src".to_string(),
                    perfetto::AnnotationValue::Str(src_name.to_string()),
                );
                tracer.record_irq_event("IRQ_Enter", payload);
                let mut delivered = std::collections::HashMap::new();
                delivered.insert(
                    "from".to_string(),
                    perfetto::AnnotationValue::Pointer(pc as u64),
                );
                delivered.insert(
                    "vector".to_string(),
                    perfetto::AnnotationValue::Pointer(vec as u64),
                );
                delivered.insert(
                    "src".to_string(),
                    perfetto::AnnotationValue::Str(src_name.to_string()),
                );
                delivered.insert(
                    "imr".to_string(),
                    perfetto::AnnotationValue::UInt(imr as u64),
                );
                delivered.insert(
                    "isr".to_string(),
                    perfetto::AnnotationValue::UInt(isr as u64),
                );
                delivered.insert(
                    "s".to_string(),
                    perfetto::AnnotationValue::Pointer(self.state.get_reg(RegName::S) as u64),
                );
                tracer.record_irq_event("IRQ_Delivered", delivered);
            }
        Ok(())
    }
}

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

fn src_mask_for_name(name: &str) -> Option<u8> {
    match name {
        "KEY" => Some(ISR_KEYI),
        "ONK" => Some(ISR_ONKI),
        "MTI" => Some(ISR_MTI),
        "STI" => Some(ISR_STI),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;
    use std::fs;
    use std::sync::{MutexGuard, OnceLock};

    static PERFETTO_TEST_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();

    fn perfetto_test_guard() -> MutexGuard<'static, ()> {
        PERFETTO_TEST_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

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
            true, // pending
            0xAA, // imr
            0x55, // isr
            200,  // next_mti
            300,  // next_sti
            Some("MTI".to_string()),
            true,               // in_interrupt
            Some(vec![0xDEAD]), // interrupt_stack (flow IDs)
            5,                  // next_interrupt_id
            None,               // irq_bit_watch
        );
        rt.timer.delivered_masks = vec![ISR_MTI];
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
        assert_eq!(rt2.timer.delivered_masks, vec![ISR_MTI]);
        assert_eq!(rt2.timer.next_interrupt_id, 5);
        assert_eq!(rt2.timer.last_fired, None);
    }

    #[test]
    fn snapshot_roundtrip_keeps_irq_counters_and_last() {
        let tmp = std::env::temp_dir().join("core_snapshot_irq_counters.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        rt.timer.irq_total = 7;
        rt.timer.irq_key = 3;
        rt.timer.irq_mti = 2;
        rt.timer.irq_sti = 1;
        rt.timer.last_irq_src = Some("KEY".to_string());
        rt.timer.last_irq_pc = Some(0x012345);
        rt.timer.last_irq_vector = Some(0x0ABCDE);
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert_eq!(rt2.timer.irq_total, 7);
        assert_eq!(rt2.timer.irq_key, 3);
        assert_eq!(rt2.timer.irq_mti, 2);
        assert_eq!(rt2.timer.irq_sti, 1);
        assert_eq!(rt2.timer.last_irq_src, Some("KEY".to_string()));
        assert_eq!(rt2.timer.last_irq_pc, Some(0x012345));
        assert_eq!(rt2.timer.last_irq_vector, Some(0x0ABCDE));
    }

    #[test]
    fn imr_isr_hook_updates_mirrors_and_bit_watch() {
        // CoreRuntime installs the IMR/ISR hook on construction.
        let mut rt = CoreRuntime::new();
        // Write IMR and ensure mirror + bit-watch capture the change.
        rt.memory
            .write_internal_byte(crate::memory::IMEM_IMR_OFFSET, 0xAA);
        assert_eq!(rt.timer.irq_imr, 0xAA, "IMR mirror should update via hook");
        let watch = rt
            .timer
            .irq_bit_watch
            .as_ref()
            .and_then(|map| map.get("IMR"))
            .and_then(|v| v.as_object())
            .expect("IMR bit watch table populated");
        let bit7 = watch
            .get("7")
            .and_then(|v| v.as_object())
            .expect("bit 7 entry exists");
        let set_entries = bit7
            .get("set")
            .and_then(|v| v.as_array())
            .expect("set array exists");
        assert!(
            !set_entries.is_empty(),
            "IMR bit 7 set should be recorded in bit watch"
        );

        // ISR write should also refresh mirror and bit-watch.
        rt.memory
            .write_internal_byte(crate::memory::IMEM_ISR_OFFSET, 0x04);
        assert_eq!(rt.timer.irq_isr, 0x04, "ISR mirror should update via hook");
        let isr_watch = rt
            .timer
            .irq_bit_watch
            .as_ref()
            .and_then(|map| map.get("ISR"))
            .and_then(|v| v.as_object())
            .expect("ISR bit watch table populated");
        let bit2 = isr_watch
            .get("2")
            .and_then(|v| v.as_object())
            .expect("bit 2 entry exists");
        let isr_set = bit2
            .get("set")
            .and_then(|v| v.as_array())
            .expect("set array exists");
        assert!(
            !isr_set.is_empty(),
            "ISR bit 2 set should be recorded in bit watch"
        );
    }

    #[test]
    fn onk_press_sets_isr_and_irq_pending() {
        let mut rt = CoreRuntime::new();
        rt.press_on_key();

        let isr = rt
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .expect("isr present");
        assert!(isr & ISR_ONKI != 0, "ONKI should be asserted in ISR");
        assert!(rt.timer.irq_pending, "irq_pending should be set after ONK");
        assert_eq!(rt.timer.irq_source.as_deref(), Some("ONK"));

        rt.release_on_key();
        let cleared = rt
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .expect("isr present");
        assert_eq!(cleared & ISR_ONKI, 0, "ONKI should clear on release");
    }

    #[test]
    fn onk_press_sets_ssr_bit_on_read() {
        let mut rt = CoreRuntime::new();
        // Ensure SSR is clear before ONK.
        let ssr_before = rt
            .memory
            .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(ssr_before & 0x08, 0, "SSR ONK bit should start clear");

        rt.press_on_key();
        // Read SSR through the runtime bus path (simulating CPU load).
        let ssr_after = {
            // Minimal runtime bus mirroring CoreRuntime::step wiring.
            struct TestBus<'a> {
                mem: &'a mut crate::memory::MemoryImage,
                onk_level: bool,
            }
            impl<'a> crate::llama::eval::LlamaBus for TestBus<'a> {
                fn load(&mut self, addr: u32, bits: u8) -> u32 {
                    if crate::memory::MemoryImage::is_internal(addr) {
                        let offset =
                            (addr - crate::memory::INTERNAL_MEMORY_START) & crate::memory::INTERNAL_ADDR_MASK as u32;
                        if offset == crate::memory::IMEM_SSR_OFFSET {
                            let mut val = self.mem.read_internal_byte(offset).unwrap_or(0);
                            if self.onk_level {
                                val |= 0x08;
                            }
                            return val as u32;
                        }
                    }
                    self.mem.load(addr, bits).unwrap_or(0)
                }
            }
            let mut rt_bus = TestBus {
                mem: &mut rt.memory,
                onk_level: rt.onk_level,
            };
            rt_bus.load(
                crate::memory::INTERNAL_MEMORY_START + crate::memory::IMEM_SSR_OFFSET,
                8,
            ) as u8
        };
        assert_ne!(
            ssr_after & 0x08,
            0,
            "SSR should reflect ONK level when pressed"
        );

        rt.release_on_key();
        let ssr_clear = rt
            .memory
            .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(ssr_clear & 0x08, 0, "SSR ONK bit should clear after release");
    }

    #[test]
    fn keyboard_reads_increment_memory_counters() {
        use crate::llama::eval::LlamaBus;
        let mut mem = crate::memory::MemoryImage::new();
        let mut kb = KeyboardMatrix::new();
        // Seed KOL to make the read non-zero but predictable.
        kb.handle_write(0xF0, 0xFF, &mut mem);
        assert_eq!(mem.memory_read_count(), 0);

        struct TestBus<'a> {
            mem: &'a mut crate::memory::MemoryImage,
            kb: &'a mut KeyboardMatrix,
        }
        impl<'a> LlamaBus for TestBus<'a> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                if crate::memory::MemoryImage::is_internal(addr)
                    && (addr - crate::memory::INTERNAL_MEMORY_START)
                        <= crate::memory::INTERNAL_ADDR_MASK as u32
                {
                    let offset = (addr - crate::memory::INTERNAL_MEMORY_START)
                        & crate::memory::INTERNAL_ADDR_MASK as u32;
                    if let Some(val) = self.kb.handle_read(offset, self.mem) {
                        self.mem.bump_read_count();
                        self.mem.log_kio_read(offset, val);
                        return val as u32;
                    }
                }
                self.mem.load(addr, bits).unwrap_or(0)
            }
        }

        let mut bus = TestBus {
            mem: &mut mem,
            kb: &mut kb,
        };
        let _ = bus.load(crate::memory::INTERNAL_MEMORY_START + 0xF0, 8);
        assert_eq!(
            bus.mem.memory_read_count(),
            1,
            "KIO reads through bus should increment memory read count"
        );
    }

    #[test]
    fn lcd_overlay_write_counts_as_memory_write() {
        let mut rt = CoreRuntime::new();
        // Program: MV IMem8, imm8 targeting LCD overlay offset 0x00.
        rt.memory.write_external_slice(0, &[0xCC, 0x00, 0xC0]); // ON instruction byte
        rt.state.set_pc(0);

        assert_eq!(rt.memory.memory_write_count(), 0);
        rt.step(1).expect("execute overlay write");
        assert!(
            rt.memory.memory_write_count() >= 1,
            "overlay write should increment memory_write_count"
        );
    }

    #[test]
    fn lcd_overlay_read_counts_as_memory_read() {
        // Establish baseline read overhead (IRQ probes, opcode fetch) using a NOP.
        let mut baseline = CoreRuntime::new();
        baseline.memory.write_external_byte(0x0000, 0x00); // NOP
        baseline.state.set_pc(0);
        baseline.step(1).expect("execute NOP");
        let base_reads = baseline.memory.memory_read_count();

        let mut rt = CoreRuntime::new();
        // Program: MV A, IMem8 targeting LCD overlay offset 0x00.
        rt.memory.write_external_slice(0, &[0x80, 0x00]);
        rt.state.set_pc(0);

        rt.step(1).expect("execute overlay read");
        let overlay_reads = rt.memory.memory_read_count();
        assert!(
            overlay_reads >= base_reads + 2,
            "overlay path should add operand+overlay reads (base={base_reads}, got {overlay_reads})",
        );
    }

    #[test]
    fn e_port_inputs_are_written_into_imem() {
        let mut rt = CoreRuntime::new();
        rt.set_e_port_inputs(0xAA, 0x55);
        let eil = rt.memory.read_internal_byte(0xF5).expect("EIL readable");
        let eih = rt.memory.read_internal_byte(0xF6).expect("EIH readable");
        assert_eq!(eil, 0xAA);
        assert_eq!(eih, 0x55);
    }

    #[test]
    fn arm_pending_irq_from_isr_handles_masked_keyi() {
        let mut rt = CoreRuntime::new();
        // IMR master off, KEYI asserted: pending should stay clear until IMR enables it.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.timer.irq_pending = false;
        rt.arm_pending_irq_from_isr();
        assert!(
            !rt.timer.irq_pending,
            "KEYI should not arm pending while IMR master is 0"
        );
        assert!(rt.timer.irq_source.is_none());
        // Pure timer bit with master off should not arm.
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_MTI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.arm_pending_irq_from_isr();
        assert!(
            !rt.timer.irq_pending,
            "MTI with IMR master 0 should stay masked until IMR enables it"
        );
    }

    #[test]
    fn arm_pending_irq_prefers_keyboard_over_existing_timer_source() {
        let mut rt = CoreRuntime::new();
        // KEYI and MTI both set; IMR enables both.
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI | ISR_MTI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY | IMR_MTI);
        // Seed an existing timer source; KEY should override.
        rt.timer.irq_source = Some("MTI".to_string());
        rt.timer.irq_pending = false;

        rt.arm_pending_irq_from_isr();

        assert!(
            rt.timer.irq_pending,
            "pending should be armed when IMR+ISR allow delivery"
        );
        assert_eq!(
            rt.timer.irq_source,
            Some("KEY".to_string()),
            "keyboard should override existing timer source"
        );
    }

    #[test]
    fn arm_pending_irq_respects_imr_master_and_source_bits() {
        let mut rt = CoreRuntime::new();
        // IMR master off should not arm pending even when KEYI asserted externally.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.timer.irq_pending = false;
        rt.arm_pending_irq_from_isr();
        assert!(
            !rt.timer.irq_pending,
            "pending should stay clear when IMR master is 0"
        );
        assert!(
            rt.timer.irq_source.is_none(),
            "irq_source should not be latched when IMR master is 0"
        );

        // Enabling IMR master+KEY should arm pending.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.arm_pending_irq_from_isr();
        assert!(rt.timer.irq_pending, "pending should arm when IMR allows it");
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));
    }

    #[test]
    fn arm_pending_irq_ignored_during_interrupt() {
        let mut rt = CoreRuntime::new();
        rt.timer.in_interrupt = true;
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI | ISR_MTI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY | IMR_MTI);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;

        rt.arm_pending_irq_from_isr();

        assert!(
            !rt.timer.irq_pending,
            "pending should not arm while already in an interrupt"
        );
        assert!(
            rt.timer.irq_source.is_none(),
            "irq_source should remain unset while in interrupt"
        );
    }

    #[test]
    fn refresh_key_irq_latch_reasserts_keyi() {
        let mut rt = CoreRuntime::new();
        let kb = rt.keyboard.as_mut().unwrap();
        let mut events = 0;
        // Press a key and strobe columns so scan_tick can debounce.
        kb.press_matrix_code(0x10, &mut rt.memory);
        kb.handle_write(0xF0, 0xFF, &mut rt.memory);
        kb.handle_write(0xF1, 0x07, &mut rt.memory);
        for _ in 0..8 {
            events += kb.scan_tick(rt.timer.kb_irq_enabled);
            if events > 0 {
                break;
            }
        }
        kb.write_fifo_to_memory(&mut rt.memory, true);
        assert!(kb.fifo_len() > 0, "fifo should have data after scan");
        // Simulate firmware clearing ISR and dropping the latch.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);
        rt.timer.key_irq_latched = false;
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.refresh_key_irq_latch();
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(
            isr & ISR_KEYI,
            0,
            "KEYI should be reasserted from FIFO latch"
        );
        assert!(
            rt.timer.key_irq_latched,
            "latch should stay set when FIFO non-empty"
        );
        assert!(
            rt.timer.irq_pending,
            "pending IRQ should be armed from latch"
        );
        assert_eq!(rt.timer.irq_source, Some("KEY".to_string()));
    }

    #[test]
    fn refresh_key_irq_latch_respects_kb_irq_disable() {
        let mut rt = CoreRuntime::new();
        rt.timer.set_keyboard_irq_enabled(false);
        let kb = rt.keyboard.as_mut().unwrap();
        // Inject a matrix event to populate FIFO without relying on IRQ enable.
        kb.inject_matrix_event(0x10, false, &mut rt.memory, rt.timer.kb_irq_enabled);
        rt.refresh_key_irq_latch();
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "KEYI should stay masked when keyboard IRQs are disabled"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should not arm when kb IRQs are disabled"
        );
    }

    #[test]
    fn refresh_key_irq_latch_respects_kb_irq_disable_with_fifo_data() {
        let mut rt = CoreRuntime::new();
        rt.timer.set_keyboard_irq_enabled(false);
        let kb = rt.keyboard.as_mut().unwrap();
        kb.press_matrix_code(0x10, &mut rt.memory);
        kb.handle_write(0xF0, 0xFF, &mut rt.memory);
        kb.handle_write(0xF1, 0x0F, &mut rt.memory);
        let mut events = 0;
        for _ in 0..8 {
            events += kb.scan_tick(rt.timer.kb_irq_enabled);
            if events > 0 {
                break;
            }
        }
        assert!(events > 0, "expected a debounced key event");
        kb.write_fifo_to_memory(&mut rt.memory, rt.timer.kb_irq_enabled);
        assert!(kb.fifo_len() > 0, "fifo should hold the event");
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;

        rt.refresh_key_irq_latch();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "KEYI should remain masked when kb IRQs are disabled even with FIFO data"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending should not arm from FIFO latch when kb IRQs are disabled"
        );
        assert!(
            !rt.timer.key_irq_latched,
            "latch should clear while kb IRQs are disabled"
        );
    }

    #[test]
    fn refresh_key_irq_latch_skips_when_in_interrupt() {
        let mut rt = CoreRuntime::new();
        rt.timer.in_interrupt = true;
        rt.timer.key_irq_latched = true;
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;

        rt.refresh_key_irq_latch();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & ISR_KEYI, 0, "ISR should not change while in interrupt");
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should remain clear while in interrupt"
        );
    }

    #[test]
    fn force_strobe_llama_sets_keyi_and_pending() {
        let _guard = perfetto_test_guard();
        std::env::set_var("FORCE_STROBE_LLAMA", "1");
        let mut rt = CoreRuntime::new();
        // Seed a pressed key so the forced strobe surfaces an event.
        if let Some(kb) = rt.keyboard.as_mut() {
            kb.press_matrix_code(0x10, &mut rt.memory);
        }
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0);
        // Call repeatedly to satisfy debounce threshold.
        for _ in 0..6 {
            rt.maybe_force_keyboard_activity(true);
        }
        std::env::remove_var("FORCE_STROBE_LLAMA");
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "FORCE_STROBE_LLAMA should assert KEYI");
        assert!(rt.timer.irq_pending, "force strobe should mark pending IRQ");
    }

    #[test]
    fn force_keyi_llama_sets_keyi_without_scan() {
        let _guard = perfetto_test_guard();
        std::env::set_var("FORCE_KEYI_LLAMA", "1");
        let mut rt = CoreRuntime::new();
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0);
        rt.maybe_force_keyboard_activity(true);
        std::env::remove_var("FORCE_KEYI_LLAMA");
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "FORCE_KEYI_LLAMA should assert KEYI");
        assert!(rt.timer.irq_pending);
    }

    #[test]
    fn force_strobe_llama_respects_timer_gate() {
        let _guard = perfetto_test_guard();
        std::env::set_var("FORCE_STROBE_LLAMA", "1");
        let mut rt = CoreRuntime::new();
        // Seed a key so strobe can surface an event.
        if let Some(kb) = rt.keyboard.as_mut() {
            kb.press_matrix_code(0x10, &mut rt.memory);
        }
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;

        // Without a timer fire, forced strobe should be ignored.
        rt.maybe_force_keyboard_activity(false);
        let isr_no_fire = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr_no_fire & ISR_KEYI,
            0,
            "FORCE_STROBE_LLAMA should do nothing when no timer fired"
        );
        assert!(!rt.timer.irq_pending);

        // With a timer fire, forced strobe should assert KEYI.
        for _ in 0..6 {
            rt.maybe_force_keyboard_activity(true);
        }
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(
            isr & ISR_KEYI,
            0,
            "FORCE_STROBE_LLAMA should assert KEYI when timer fired"
        );
        assert!(rt.timer.irq_pending);
        std::env::remove_var("FORCE_STROBE_LLAMA");
    }

    #[test]
    fn halt_wakes_on_isr_even_when_imr_masked() {
        let mut rt = CoreRuntime::new();
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        let _ = rt.step(1);
        assert!(
            !rt.state.is_halted(),
            "HALT should clear when ISR is set regardless of IMR"
        );
        assert!(rt.timer.irq_pending, "pending IRQ should be armed on wake");
    }

    #[test]
    fn halt_ignores_keyi_when_kb_irq_disabled() {
        let mut rt = CoreRuntime::new();
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.timer.set_keyboard_irq_enabled(false);
        // Assert KEYI in ISR with IMR master off to mimic a host write.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);

        let _ = rt.step(1);

        // HALT should remain and no pending IRQ when kb IRQs are disabled.
        assert!(rt.state.is_halted(), "HALT should not wake on KEYI when kb IRQs disabled");
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should not arm for KEYI when kb IRQs disabled"
        );
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "ISR bit should remain set but ignored");
    }

    #[test]
    fn irq_delivery_waits_for_imr_master_then_delivers() {
        let mut rt = CoreRuntime::new();
        // Seed PC/SP and vector.
        rt.state.set_reg(RegName::PC, 0x0010);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_byte(0x0FFFFA, 0x34);
        rt.memory.write_external_byte(0x0FFFFB, 0x12);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);

        // Assert ISR but keep IMR master off.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        let _ = rt.step(1);
        // With IMR master off, pending should not arm and PC should not jump yet.
        assert!(
            !rt.timer.irq_pending,
            "pending should stay clear while IMR master=0"
        );
        assert_ne!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            0x001234,
            "PC should not jump while IMR master=0"
        );

        // Enable IMR master and KEY bits; next step should deliver.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        let _ = rt.step(1);
        assert!(
            !rt.timer.irq_pending,
            "pending should clear after delivery when IMR master enabled"
        );
        assert_eq!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            0x001234,
            "PC should jump to interrupt vector when IMR master enabled"
        );
    }

    #[test]
    fn onk_pending_survives_imr_mask_and_delivers_when_enabled() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x0010);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_byte(0x0FFFFA, 0x78);
        rt.memory.write_external_byte(0x0FFFFB, 0x56);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);

        // ONK latched, IMR master off.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        let _ = rt.step(1);
        assert!(
            !rt.timer.irq_pending,
            "pending should stay clear while IMR master is off"
        );
        // Enable master + ONK, then deliver.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_ONK);
        let _ = rt.step(1);
        assert!(
            !rt.timer.irq_pending,
            "ONK pending should clear after delivery once IMR enables it"
        );
        assert_eq!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            0x005678,
            "PC should jump to vector on ONK delivery"
        );
    }

    #[test]
    fn reti_clears_interrupt_state_and_isr_bit() {
        let mut rt = CoreRuntime::new();
        // Prepare stack and vector to a RETI instruction.
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.set_reg(RegName::PC, 0x0000);
        rt.memory.write_external_byte(0x0FFFFA, 0x10); // vector low
        rt.memory.write_external_byte(0x0FFFFB, 0x00);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.memory.write_external_byte(0x0010, 0x01); // RETI opcode
                                                     // Seed IMR/ISR and pending IRQ.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        // First step: deliver IRQ and jump to vector.
        rt.step(1).expect("deliver irq");
        assert!(rt.timer.in_interrupt, "interrupt flag should set on entry");
        assert_eq!(rt.state.get_reg(RegName::PC) & ADDRESS_MASK, 0x0010);

        // Second step: execute RETI, clear bookkeeping and ISR bit.
        rt.step(1).expect("execute reti");
        assert!(!rt.timer.in_interrupt, "RETI should clear in_interrupt");
        assert!(
            rt.timer.interrupt_stack.is_empty(),
            "interrupt stack should clear"
        );
        assert!(
            rt.timer.irq_source.is_none(),
            "irq source should clear after RETI"
        );
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & ISR_KEYI, 0, "RETI should clear delivered ISR bit");
    }

    #[test]
    fn reti_uses_interrupt_stack_mask_when_source_missing() {
        let mut rt = CoreRuntime::new();
        // Fake a pending interrupt state with a delivered mask stored separately.
        rt.state.set_reg(RegName::S, 0x0030);
        rt.state.set_reg(RegName::PC, 0x0000);
        // RETI opcode at PC.
        rt.memory.write_external_byte(0x0000, 0x01);
        // Stack frame IMR,F,PC
        rt.memory.write_external_byte(0x0030, 0xFF); // IMR
        rt.memory.write_external_byte(0x0031, 0x00); // F
        rt.memory.write_external_byte(0x0032, 0x34);
        rt.memory.write_external_byte(0x0033, 0x12);
        rt.memory.write_external_byte(0x0034, 0x00);
        // ISR has ONK bit set; irq_source unknown but delivered_masks carries mask.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        rt.timer.in_interrupt = true;
        rt.timer.interrupt_stack = vec![1]; // flow id placeholder
        rt.timer.delivered_masks = vec![ISR_ONKI];
        rt.timer.irq_source = None;

        rt.step(1).expect("execute reti without source");

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_ONKI,
            0,
            "RETI should clear ISR using delivered mask when source is None"
        );
        assert!(rt.timer.interrupt_stack.is_empty());
        assert!(rt.timer.delivered_masks.is_empty());
    }

    #[test]
    fn timers_do_not_tick_during_interrupts() {
        let mut rt = CoreRuntime::new();
        rt.timer.enabled = true;
        rt.timer.mti_period = 1;
        rt.timer.reset(0);
        rt.timer.in_interrupt = true;
        rt.state.set_reg(RegName::PC, 0x0000); // opcode 0x00 = NOP by default

        rt.step(1).expect("step while in interrupt");
        assert!(
            !rt.timer.irq_pending,
            "timer should not pend IRQs while in_interrupt"
        );
        assert_eq!(
            rt.timer.next_mti, 1,
            "next_mti should stay unchanged when ticking is gated"
        );
    }

    #[test]
    fn perfetto_irq_entry_exit_smoke() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_irq_smoke.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        let mut rt = CoreRuntime::new();
        // Place RETI at vector 0x0000.
        rt.memory.write_external_byte(0x0000, 0x01);
        rt.memory.write_external_byte(0x0FFFFA, 0x00);
        rt.memory.write_external_byte(0x0FFFFB, 0x00);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.state.set_reg(RegName::PC, 0x0100);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        rt.step(1).expect("deliver irq and jump to vector");
        rt.step(1).expect("execute RETI");

        // Flush perfetto trace to disk before reading.
        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }

        let size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        assert!(size > 0, "perfetto trace should be written");
        // Trace should contain IRQ markers.
        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("IRQ_Enter"),
            "trace should contain IRQ_Enter marker"
        );
        assert!(
            text.contains("KeyDeliver"),
            "trace should contain KeyDeliver marker"
        );
        assert!(
            text.contains("IRQ_Return"),
            "trace should contain IRQ_Return marker"
        );
        assert!(
            text.contains("src"),
            "trace should encode src annotation for IRQ"
        );
        let _ = std::mem::take(&mut *PERFETTO_TRACER.enter());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_forced_key_events_smoke() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_key_force.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        std::env::set_var("FORCE_KEYI_LLAMA", "1");
        let mut rt = CoreRuntime::new();
        // Seed a key press to give forced scan some activity.
        if let Some(kb) = rt.keyboard.as_mut() {
            kb.press_matrix_code(0x10, &mut rt.memory);
        }
        rt.maybe_force_keyboard_activity(true);
        std::env::remove_var("FORCE_KEYI_LLAMA");

        // Flush perfetto trace to disk before reading.
        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }

        let size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        assert!(size > 0, "perfetto trace should be written");
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_handle_reentrant_allows_nested() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_reentrant.perfetto-trace");
        let _ = fs::remove_file(&tmp);

        {
            let mut root = PERFETTO_TRACER.enter();
            *root = Some(PerfettoTracer::new(tmp.clone()));
            {
                let mut nested = PERFETTO_TRACER.enter();
                assert!(nested.is_some(), "nested guard should see tracer");
                if let Some(tracer) = nested.tracer_mut() {
                    tracer.record_call_flow("NESTED", 0x10, 0x20, 1);
                }
            }
            if let Some(tracer) = root.take() {
                let _ = tracer.finish();
            }
        }

        let size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        assert!(size > 0, "reentrant perfetto trace should be written");
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_timer_irq_smoke() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_timer_irq.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        let mut rt = CoreRuntime::new();
        // Seed IMR to allow MTI delivery and place a NOP at PC=0.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        rt.memory.write_external_byte(0x0000, 0x00); // NOP
        rt.state.set_reg(RegName::PC, 0x0000);
        rt.state.set_reg(RegName::S, 0x0200);
        // Configure timer to fire immediately.
        rt.timer.enabled = true;
        rt.timer.mti_period = 1;
        rt.timer.next_mti = 0;

        rt.step(1).expect("tick timer and deliver MTI");

        // Flush perfetto trace to disk before reading.
        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }

        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("TimerFired"),
            "trace should contain TimerFired marker"
        );
        assert!(
            text.contains("IRQ_Enter"),
            "trace should contain IRQ_Enter for MTI"
        );
        assert!(
            text.contains("src"),
            "trace should encode src annotation for MTI"
        );
        let _ = std::mem::take(&mut *PERFETTO_TRACER.enter());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_lcd_events_match_python_shape() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_lcd.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        let mut lcd = LcdController::new();
        // Emit an instruction (SetPage) and a data write so both paths are traced.
        lcd.write(0x02000, 0x81); // SetPage page=1, CS=both, write
        lcd.write(0x02002, 0xAA); // Data write to both chips

        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }

        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("Display"),
            "Display track should be present for LCD parity"
        );
        assert!(
            text.contains("LCD_SET_PAGE"),
            "LCD_SET_PAGE instruction should be traced"
        );
        assert!(
            text.contains("VRAM_Write"),
            "VRAM_Write data events should be traced"
        );
        let _ = std::mem::take(&mut *PERFETTO_TRACER.enter());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn host_overlay_write_counts_and_traces() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_host_overlay.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let mut rt = CoreRuntime::new();
        rt.set_host_write(move |_addr, _val| {
            flag.store(true, Ordering::Relaxed);
        });
        // Program: MV IMem8, imm8 targeting offset 0xF5 (python_required E-port input).
        rt.memory.write_external_slice(0, &[0xCC, 0xF5, 0xAA]);
        rt.state.set_pc(0);
        let before_writes = rt.memory.memory_write_count();

        rt.step(1).expect("execute host overlay write");

        assert!(called.load(Ordering::Relaxed), "host_write callback should fire");
        assert!(
            rt.memory.memory_write_count() > before_writes,
            "host overlay writes should bump memory_write_count"
        );

        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }
        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.to_ascii_lowercase().contains("python_overlay"),
            "perfetto trace should tag overlay writes with python_overlay space"
        );
        let _ = std::mem::take(&mut *PERFETTO_TRACER.enter());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_overlay_writes_tag_overlay_name() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_overlay_name.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        let mut mem = MemoryImage::new();
        mem.add_ram_overlay(0x8000, 1, "ram_overlay");
        let _ = mem.store_with_pc(0x8000, 8, 0xAA, Some(0x0123));

        if let Some(tracer) = std::mem::take(&mut *PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }
        let buf = fs::read(&tmp).expect("read perfetto overlay trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("ram_overlay"),
            "overlay name should be present in perfetto output"
        );
        let _ = std::mem::take(&mut *PERFETTO_TRACER.enter());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn reset_intrinsic_preserves_timer_state() {
        let mut rt = CoreRuntime::new();
        // Seed timer bookkeeping to verify RESET does not wipe it.
        rt.timer.irq_total = 5;
        rt.timer.irq_key = 2;
        rt.timer.irq_mti = 1;
        rt.timer.irq_sti = 1;
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());
        rt.timer.record_bit_watch_transition("IMR", 0x00, 0x80, 0x0100);
        rt.timer.record_bit_watch_transition("ISR", 0x00, 0x04, 0x0100);

        // Program RESET at PC=0.
        rt.memory.write_external_byte(0x0000, 0xFF);
        rt.state.set_reg(RegName::PC, 0x0000);

        rt.step(1).expect("execute RESET");

        // Timer counters should remain intact.
        assert_eq!(rt.timer.irq_total, 5);
        assert_eq!(rt.timer.irq_key, 2);
        assert_eq!(rt.timer.irq_mti, 1);
        assert_eq!(rt.timer.irq_sti, 1);
        assert!(rt.timer.irq_pending, "pending flag should not be cleared by RESET");
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));
        // Bit-watch tables should remain populated.
        let imr_watch = rt
            .timer
            .irq_bit_watch
            .as_ref()
            .and_then(|m| m.get("IMR"))
            .and_then(|v| v.as_object())
            .expect("IMR bit-watch should persist across RESET");
        let imr_bit7 = imr_watch.get("7").and_then(|v| v.as_object()).unwrap();
        assert!(
            imr_bit7
                .get("set")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| !arr.is_empty()),
            "IMR bit-watch set entries should remain after RESET"
        );
    }

    #[test]
    fn perfetto_irq_check_and_call_flow_smoke() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_irq_check_call.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            *guard = Some(PerfettoTracer::new(tmp.clone()));
        }

        // Emit diagnostics without running the runtime loop.
        if let Some(mut tracer) = PERFETTO_TRACER.enter().take() {
            tracer.record_irq_check(
                "IRQ_Check",
                0x0100,
                0x80,
                0x04,
                true,
                false,
                Some("KEY"),
                None,
                None,
            );
            tracer.record_call_flow("CALL", 0x012345, 0x020000, 2);
            let _ = tracer.finish();
        }

        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("IRQ_Check"),
            "trace should include IRQ_Check diagnostic"
        );
        assert!(
            text.contains("CALL"),
            "trace should include CALL flow marker"
        );
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn irq_counters_increment_on_delivery_only() {
        let mut rt = CoreRuntime::new();
        // Place RETI at vector 0x0000.
        rt.memory.write_external_byte(0x0000, 0x01);
        rt.memory.write_external_byte(0x0FFFFA, 0x00);
        rt.memory.write_external_byte(0x0FFFFB, 0x00);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.state.set_reg(RegName::PC, 0x0100);
        rt.state.set_reg(RegName::S, 0x0200);
        // Enable IMR master and KEY bit.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        // Seed ISR with KEYI pending; do not tick timers to avoid pre-delivery increments.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        // Before delivery, counters should be zero.
        assert_eq!(rt.timer.irq_total, 0);
        assert_eq!(rt.timer.irq_key, 0);

        // Deliver the IRQ (one step runs IR intrinsic, second executes RETI).
        rt.step(1).expect("deliver irq");
        assert_eq!(rt.timer.irq_total, 1);
        assert_eq!(rt.timer.irq_key, 1);
        rt.step(1).expect("execute RETI");
        // Counters should remain at one after RETI.
        assert_eq!(rt.timer.irq_total, 1);
        assert_eq!(rt.timer.irq_key, 1);
    }

    #[test]
    fn hardware_irq_updates_call_depth() {
        let mut rt = CoreRuntime::new();
        // Vector points to RETI at 0x0000.
        rt.memory.write_external_byte(0x0000, 0x01);
        rt.memory.write_external_byte(0x0FFFFA, 0x00);
        rt.memory.write_external_byte(0x0FFFFB, 0x00);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.state.set_reg(RegName::PC, 0x0100);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        rt.step(1).expect("deliver irq to vector");
        assert_eq!(rt.state.call_depth(), 1, "interrupt should raise call depth");
        rt.step(1).expect("execute RETI");
        assert_eq!(rt.state.call_depth(), 0, "RETI should restore call depth");
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
        *rt.timer = TimerContext::new(true, 1, 0);
        let res = rt.step(1);
        assert!(res.is_ok(), "step should execute without error");
        let isr = rt.memory.read_internal_byte(0xFC).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01, "MTI should set ISR bit after first step");
        assert_eq!(rt.metadata.cycle_count, 1);
    }

    #[test]
    fn wait_does_not_spin_timers() {
        // Python fast-path WAIT clears I/flags and still burns I cycles, ticking timers each loop.
        let mut rt = CoreRuntime::new();
        // Place WAIT at PC=0.
        rt.memory.write_external_slice(0, &[0xEF]);
        // Enable timers that would normally fire on the first cycle.
        *rt.timer = TimerContext::new(true, 1, 1);
        rt.timer.next_mti = 1;
        rt.timer.next_sti = 1;
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::I, 5);

        rt.step(1).expect("WAIT step");

        // Timers should fire across the idle cycles and pend IRQs.
        assert!(rt.timer.irq_pending, "WAIT idle loop should pend IRQs");
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & (ISR_MTI | ISR_STI), 0, "ISR should reflect timer fire");
        // Cycle counter should advance for opcode + I loops.
        assert_eq!(rt.metadata.cycle_count, 6);
    }

    #[test]
    fn requires_python_without_host_errors() {
        let mut rt = CoreRuntime::new();
        // Mark an external range as Python-only and point PC at it.
        rt.memory.set_python_ranges(vec![(0x0000_2000, 0x0000_2000)]);
        rt.state.set_reg(RegName::PC, 0x0000_2000);
        // Seed a NOP opcode so the fetch path is taken.
        rt.memory.write_external_byte(0x0000_2000, 0x00);
        let res = rt.step(1);
        assert!(
            matches!(res, Err(CoreError::Other(ref msg)) if msg.contains("python overlay")),
            "step should fail when Python-required overlays are missing: {res:?}"
        );
    }

    #[test]
    fn python_overlay_required_imem_access_fails_without_host() {
        let mut rt = CoreRuntime::new();
        // Program: MV IMem8, imm8 targeting offset 0xF5 (python_required E-port input).
        rt.memory.write_external_slice(0, &[0xCC, 0xF5, 0xAA]);
        rt.state.set_pc(0);

        let res = rt.step(1);
        assert!(
            matches!(res, Err(CoreError::Other(ref msg)) if msg.contains("python overlay")),
            "python-required IMEM access should fail without host overlay: {res:?}"
        );
    }

    #[test]
    fn runtime_overlay_helpers_route_through_memory_image() {
        let mut rt = CoreRuntime::new();
        rt.add_ram_overlay(0x8000, 2, "runtime_ram");
        rt.clear_overlay_logs();
        let _ = rt
            .memory
            .store_with_pc(0x8000, 16, 0xBEEF, Some(0x0100));
        let writes = rt.overlay_write_log();
        assert_eq!(writes.len(), 2, "should log 2 overlay byte writes");
        assert!(writes.iter().all(|entry| entry.overlay == "runtime_ram"));
        assert!(writes.iter().any(|entry| entry.pc == Some(0x0100)));
        let val = rt.memory.load_with_pc(0x8000, 16, Some(0x0200)).unwrap();
        assert_eq!(val, 0xBEEF);
    }
}
