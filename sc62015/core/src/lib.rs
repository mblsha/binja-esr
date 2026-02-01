// PY_SOURCE: sc62015/pysc62015/emulator.py:RegisterName
// PY_SOURCE: sc62015/pysc62015/emulator.py:Registers

pub mod device;
pub mod iq7000;
pub mod keyboard;
pub mod lcd;
pub mod lcd_text;
pub mod llama;
pub mod loop_detector;
pub mod memory;
pub mod pce500;
pub mod perfetto;
pub mod sio;
pub mod snapshot;
pub mod timer;
pub mod async_driver;
pub mod async_cpu;
pub mod async_devices;
pub mod async_runtime;

use crate::llama::state::PowerState;
use crate::llama::{opcodes::RegName, state::LlamaState};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use serde_json::json;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
use thiserror::Error;

pub use device::{DeviceModel, DeviceTextDecoder};
pub use async_driver::{
    current_cycle, emit_event, sleep_cycles, AsyncDriver, CycleSleep, DriverEvent, DriverRunResult,
};
pub use async_cpu::{AsyncCpuHandle, AsyncCpuStats, CpuTraceEvent};
pub use async_devices::{AsyncDisplayTask, AsyncTimerKeyboardTask};
pub use async_runtime::AsyncRuntimeRunner;
pub use keyboard::KeyboardMatrix;
pub use lcd::{
    create_lcd, LcdController, LcdHal, LcdKind, UnknownLcdController, LCD_CHIP_COLS, LCD_CHIP_ROWS,
    LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS,
};
pub use llama::state::LlamaState as CpuState;
pub use loop_detector::{
    LoopBranchInfo, LoopBranchKind, LoopCandidate, LoopDetector, LoopDetectorConfig, LoopIrqSource,
    LoopReport, LoopStep, LoopSummary, LoopTraceEntry,
};
pub use memory::{
    AccessKind, MemoryAccessLog, MemoryImage, MemoryOverlay, ADDRESS_MASK, EXTERNAL_SPACE,
    INTERNAL_ADDR_MASK, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START,
    INTERNAL_SPACE,
};
pub use perfetto::PerfettoTracer;
pub use sio::SioStub;
#[cfg(feature = "perfetto")]
pub type PerfettoHandle = retrobus_perfetto::ReentrantHandle<Option<PerfettoTracer>>;
#[cfg(feature = "perfetto")]
pub type PerfettoGuard<'a> = retrobus_perfetto::ReentrantGuard<'a, Option<PerfettoTracer>>;

#[cfg(not(feature = "perfetto"))]
pub struct PerfettoHandle;

#[cfg(not(feature = "perfetto"))]
pub struct PerfettoGuard<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "perfetto"))]
impl PerfettoHandle {
    pub const fn new() -> Self {
        Self
    }

    pub fn enter(&self) -> PerfettoGuard<'_> {
        PerfettoGuard {
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(not(feature = "perfetto"))]
impl<'a> PerfettoGuard<'a> {
    pub fn with_some<F, R>(&mut self, _f: F) -> Option<R>
    where
        F: FnOnce(&mut PerfettoTracer) -> R,
    {
        None
    }

    pub fn take(&mut self) -> Option<PerfettoTracer> {
        None
    }

    pub fn replace(&mut self, _value: Option<PerfettoTracer>) -> Option<PerfettoTracer> {
        None
    }
}

#[cfg(feature = "perfetto")]
pub static PERFETTO_TRACER: PerfettoHandle = PerfettoHandle::new(None);

#[cfg(not(feature = "perfetto"))]
pub static PERFETTO_TRACER: PerfettoHandle = PerfettoHandle::new();

#[cfg(test)]
pub(crate) fn perfetto_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static PERFETTO_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
        std::sync::OnceLock::new();
    PERFETTO_TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
pub use snapshot::{load_snapshot, save_snapshot};
pub use snapshot::{
    pack_registers, unpack_registers, SnapshotLoad, SNAPSHOT_MAGIC, SNAPSHOT_REGISTER_LAYOUT,
    SNAPSHOT_VERSION,
};
pub use timer::TimerContext;

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use crate::keyboard::KeyboardSnapshot;
use crate::llama::eval::{perfetto_last_pc, LlamaBus};
use crate::llama::state::mask_for;
use crate::memory::{
    IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_KIL_OFFSET, IMEM_RXD_OFFSET, IMEM_TXD_OFFSET,
    IMEM_UCR_OFFSET, IMEM_USR_OFFSET,
};

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "snapshot")]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_model: Option<DeviceModel>,
    pub created: String,
    pub instruction_count: u64,
    pub cycle_count: u64,
    #[serde(default)]
    pub memory_reads: u64,
    #[serde(default)]
    pub memory_writes: u64,
    pub pc: u32,
    #[serde(default)]
    pub power_state: PowerState,
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
            device_model: None,
            created: now_timestamp(),
            instruction_count: 0,
            cycle_count: 0,
            memory_reads: 0,
            memory_writes: 0,
            pc: 0,
            power_state: PowerState::Running,
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
    #[cfg(target_arch = "wasm32")]
    {
        // `std::time::SystemTime::now` is not supported on wasm32-unknown-unknown.
        return "0Z".to_string();
    }

    #[cfg(not(target_arch = "wasm32"))]
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
    loop_detector: Option<LoopDetector>,
    executor: crate::llama::eval::LlamaExecutor,
    pub keyboard: Option<KeyboardMatrix>,
    pub lcd: Option<Box<dyn LcdHal>>,
    pub sio: Option<SioStub>,
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
            loop_detector: None,
            executor: crate::llama::eval::LlamaExecutor::new(),
            keyboard: Some(KeyboardMatrix::new()),
            lcd: Some(Box::new(LcdController::new())),
            sio: None,
            timer: Box::new(TimerContext::new(false, 0, 0)),
            host_read: None,
            host_write: None,
            onk_level: false,
        };
        rt.install_imr_isr_hook();
        rt
    }

    pub fn device_model(&self) -> DeviceModel {
        self.metadata.device_model.unwrap_or(DeviceModel::PcE500)
    }

    pub fn set_device_model(&mut self, model: DeviceModel) {
        self.metadata.device_model = Some(model);
    }

    pub fn instruction_count(&self) -> u64 {
        self.metadata.instruction_count
    }

    pub fn cycle_count(&self) -> u64 {
        self.metadata.cycle_count
    }

    pub fn power_on_reset(&mut self) {
        struct ResetBus<'a> {
            mem: &'a mut MemoryImage,
        }

        impl LlamaBus for ResetBus<'_> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                self.mem.load(addr, bits).unwrap_or(0)
            }

            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                let _ = self.mem.store(addr, bits, value);
            }
        }

        let mut bus = ResetBus {
            mem: &mut self.memory,
        };
        crate::llama::eval::power_on_reset(&mut bus, &mut self.state);
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

    pub fn enable_sio_stub(&mut self) {
        if self.sio.is_none() {
            let mut stub = SioStub::new();
            stub.init(&mut self.memory);
            self.sio = Some(stub);
        }
    }

    pub fn enable_loop_detector(&mut self, config: LoopDetectorConfig) {
        self.loop_detector = Some(LoopDetector::new(config));
    }

    pub fn disable_loop_detector(&mut self) {
        self.loop_detector = None;
    }

    pub fn loop_detector(&self) -> Option<&LoopDetector> {
        self.loop_detector.as_ref()
    }

    pub fn loop_detector_mut(&mut self) -> Option<&mut LoopDetector> {
        self.loop_detector.as_mut()
    }

    fn install_imr_isr_hook(&mut self) {
        let timer_ptr: *mut TimerContext = self.timer.as_mut() as *mut TimerContext;
        self.memory.set_imr_isr_hook(Some(move |offset, prev, new| {
            let pc = crate::llama::eval::perfetto_instr_context()
                .map(|(_, pc)| pc)
                .unwrap_or_else(crate::llama::eval::perfetto_last_pc);
            unsafe {
                let timer = &mut *timer_ptr;
                let reg_name = if offset == IMEM_IMR_OFFSET {
                    "IMR"
                } else {
                    "ISR"
                };
                timer.record_bit_watch_transition(reg_name, prev, new, pc);
                if offset == IMEM_IMR_OFFSET {
                    timer.irq_imr = new;
                } else if offset == IMEM_ISR_OFFSET {
                    timer.irq_isr = new;
                }
                let mut guard = PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert(
                        "pc".to_string(),
                        perfetto::AnnotationValue::Pointer(pc as u64),
                    );
                    payload.insert(
                        "prev".to_string(),
                        perfetto::AnnotationValue::UInt(prev as u64),
                    );
                    payload.insert(
                        "value".to_string(),
                        perfetto::AnnotationValue::UInt(new as u64),
                    );
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
                });
            }
        }));
    }

    /// Set the ON key level high and assert ISR.ONKI/IRQ pending to mirror Python KEY_ON handling.
    pub fn press_on_key(&mut self) {
        self.onk_level = true;
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if (isr & ISR_ONKI) == 0 {
            let new_isr = isr | ISR_ONKI;
            self.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
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
        guard.with_some(|tracer| {
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
        });
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

    pub fn remove_overlay(&mut self, name: &str) {
        self.memory.remove_overlay(name);
    }

    pub fn overlays(&self) -> &[MemoryOverlay] {
        self.memory.overlays()
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

    fn refresh_key_irq_latch(&mut self) {
        if self.timer.in_interrupt {
            // Parity: do not reassert KEYI while already in an interrupt handler.
            return;
        }
        if self.keyboard.is_some() {
            // Only reassert when a latch is already active; do not recreate one purely from FIFO
            // contents. New latches are set at event time by the timer/keyboard path.
            if !self.timer.key_irq_latched {
                return;
            }
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
        }
    }

    pub(crate) fn tick_timers_and_keyboard(&mut self, cycle: u64) {
        if self.timer.in_interrupt {
            return;
        }
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let _ = self.timer.tick_timers_with_keyboard(
            &mut self.memory,
            cycle,
            |mem| {
                if let Some(kb) = self.keyboard.as_mut() {
                    // Parity: always count/key-latch events even when IRQs are masked.
                    let events = kb.scan_tick(mem, true);
                    let fifo_pending = kb.fifo_len() > 0;
                    if events > 0 || (kb_irq_enabled && fifo_pending) {
                        kb.write_fifo_to_memory(mem, kb_irq_enabled);
                    }
                    (
                        events,
                        events > 0 || (kb_irq_enabled && fifo_pending),
                        Some(kb.telemetry()),
                    )
                } else {
                    (0, false, None)
                }
            },
            Some(self.state.get_reg(RegName::Y)),
            Some(self.state.get_reg(RegName::PC)),
        );
        if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
            self.timer.irq_isr = isr;
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
        // Parity: Python marks irq_pending as soon as ISR bits are asserted, even if IMR master is 0
        // or the source mask is currently disabled. Delivery is still gated later.
        let src = if (isr_effective & ISR_KEYI) != 0 {
            Some("KEY")
        } else if (isr_effective & ISR_ONKI) != 0 {
            Some("ONK")
        } else if (isr_effective & ISR_MTI) != 0 {
            Some("MTI")
        } else if (isr_effective & ISR_STI) != 0 {
            Some("STI")
        } else {
            None
        };
        if src.is_none() {
            return;
        }
        self.timer.irq_pending = true;
        self.timer.irq_isr = isr_effective;
        self.timer.irq_imr = imr;
        // Allow a newly latched KEY/ONK to override earlier timer sources to match Python priority.
        match self.timer.irq_source.as_deref() {
            None => self.timer.irq_source = src.map(str::to_string),
            Some(cur) => {
                if let Some(src_name) = src {
                    if (src_name == "KEY" || src_name == "ONK") && cur != "KEY" && cur != "ONK" {
                        self.timer.irq_source = Some(src_name.to_string());
                    }
                }
            }
        }
        self.timer.last_fired = self.timer.irq_source.clone();
        let kil = self
            .memory
            .read_internal_byte_silent(IMEM_KIL_OFFSET)
            .unwrap_or(0);
        let imr_reg = self.state.get_reg(RegName::IMR) as u8;
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
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
        });
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
            lcd_ptr: Option<*mut dyn LcdHal>,
            sio_ptr: *mut SioStub,
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
        }
        impl<'a> LlamaBus for RuntimeBus<'a> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                // Route keyboard/LCD accesses to their devices for parity with Python overlays.
                unsafe {
                    // The SC62015 exposes keyboard registers as byte-wide ports (KOL/KOH/KIL),
                    // but firmware frequently uses word-sized access via KOL.w (touching 0xF0/0xF1).
                    // Split multi-byte accesses so the keyboard handler sees both bytes.
                    if bits > 8
                        && !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let bytes = bits.div_ceil(8).max(1) as u32;
                        let start = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        let end = start.saturating_add(bytes.saturating_sub(1));
                        if start <= 0xF2 && end >= 0xF0 {
                            let mut out = 0u32;
                            for byte_offset in 0..bytes {
                                let byte = self.load(addr.wrapping_add(byte_offset), 8) & 0xFF;
                                out |= byte << (byte_offset * 8);
                            }
                            return out;
                        }
                    }
                    let python_required = (*self.mem).requires_python(addr);
                    // Keyboard: internal IMEM offsets 0xF0-0xF2.
                    if !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if let Some(val) = (*self.keyboard_ptr).handle_read(offset, &mut *self.mem)
                        {
                            (*self.mem).bump_read_count();
                            (*self.mem).log_kio_read(offset, val);
                            return val as u32;
                        }
                    }
                    // LCD controller mirrored at 0x2000/0xA000.
                    if let Some(lcd_ptr) = self.lcd_ptr {
                        let lcd = &mut *lcd_ptr;
                        if lcd.handles(addr) {
                            if let Some(val) = lcd.read(addr) {
                                (*self.mem).bump_read_count();
                                return val as u32;
                            }
                        }
                    }
                    if !self.sio_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if matches!(
                            offset,
                            IMEM_UCR_OFFSET | IMEM_USR_OFFSET | IMEM_RXD_OFFSET | IMEM_TXD_OFFSET
                        ) {
                            if let Some(val) = (*self.sio_ptr).handle_read(offset, &mut *self.mem) {
                                return val as u32;
                            }
                        }
                    }
                    // Host overlay: delegate addresses flagged for external handling.
                    if python_required {
                        if let Some(cb) = self.host_read {
                            if let Some(val) = (*cb)(addr) {
                                (*self.mem).bump_read_count();
                                return val as u32;
                            }
                        }
                    }
                    // SSR (0xFF) must reflect ONK level even without host overlays to match Python/Perfetto.
                    if MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
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
                    // See `load`: split word-sized KOL.w writes so KOH is updated too.
                    if bits > 8
                        && !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let bytes = bits.div_ceil(8).max(1) as u32;
                        let start = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        let end = start.saturating_add(bytes.saturating_sub(1));
                        if start <= 0xF2 && end >= 0xF0 {
                            for byte_offset in 0..bytes {
                                let byte = (value >> (byte_offset * 8)) & 0xFF;
                                self.store(addr.wrapping_add(byte_offset), 8, byte);
                            }
                            return;
                        }
                    }
                    let python_required = (*self.mem).requires_python(addr);
                    // Keyboard KOL/KOH/KIL writes.
                    if !self.keyboard_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if (0xF0..=0xF2).contains(&offset)
                            && (*self.keyboard_ptr).handle_write(
                                offset,
                                value as u8,
                                &mut *self.mem,
                            )
                        {
                            // Mirror writes into IMEM except when the handler already wrote KIL.
                            if offset != 0xF2 {
                                let _ = (*self.mem).store(addr, bits, value);
                            }
                            return;
                        }
                    }
                    // LCD writes.
                    if let Some(lcd_ptr) = self.lcd_ptr {
                        let lcd = &mut *lcd_ptr;
                        if lcd.handles(addr) {
                            lcd.write(addr, value as u8);
                            let _ = (*self.mem).store(addr, bits, value);
                            return;
                        }
                    }
                    if !self.sio_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if matches!(
                            offset,
                            IMEM_UCR_OFFSET | IMEM_USR_OFFSET | IMEM_RXD_OFFSET | IMEM_TXD_OFFSET
                        ) && (*self.sio_ptr).handle_write(offset, value as u8, &mut *self.mem)
                        {
                            return;
                        }
                    }
                    if python_required {
                        if let Some(cb) = self.host_write {
                            (*cb)(addr, value as u8);
                            // Parity: overlay writes should still count as memory writes and emit Perfetto traces.
                            (*self.mem).bump_write_count();
                            let mut guard = PERFETTO_TRACER.enter();
                            guard.with_some(|tracer| {
                                if let Some((op_idx, pc_ctx)) =
                                    crate::llama::eval::perfetto_instr_context()
                                {
                                    let substep = crate::llama::eval::perfetto_next_substep();
                                    tracer.record_mem_write_with_substep(
                                        op_idx,
                                        pc_ctx,
                                        addr,
                                        value,
                                        "python_overlay",
                                        bits,
                                        substep,
                                    );
                                } else {
                                    tracer.record_mem_write_at_cycle(
                                        self.cycle,
                                        Some(self.pc),
                                        addr,
                                        value,
                                        "python_overlay",
                                        bits,
                                    );
                                }
                            });
                            return;
                        }
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
            if self.state.is_off() {
                if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                    // Assumption: OFF clears non-ONK IRQ state; verify on real hardware.
                    let onk_only = isr & ISR_ONKI;
                    if onk_only != isr {
                        self.memory.write_internal_byte(IMEM_ISR_OFFSET, onk_only);
                    }
                    self.timer.irq_pending = false;
                    self.timer.irq_source = None;
                    self.timer.last_fired = None;
                    self.timer.irq_isr = onk_only;
                    if (isr & ISR_KEYI) != 0 {
                        self.timer.key_irq_latched = false;
                    }
                    if onk_only != 0 {
                        self.state.set_power_state(PowerState::Running);
                        self.timer.irq_pending = true;
                        self.timer.irq_isr = onk_only;
                        self.timer.irq_imr = self
                            .memory
                            .read_internal_byte(IMEM_IMR_OFFSET)
                            .unwrap_or(self.timer.irq_imr);
                        self.timer.irq_source = Some("ONK".to_string());
                        self.timer.last_fired = self.timer.irq_source.clone();
                    } else {
                        return Ok(());
                    }
                } else {
                    return Ok(());
                }
            }
            if let Some(sio) = self.sio.as_mut() {
                if sio.maybe_short_circuit(self.state.pc(), &mut self.state, &mut self.memory) {
                    self.metadata.instruction_count =
                        self.metadata.instruction_count.saturating_add(1);
                    self.metadata.cycle_count = self.metadata.cycle_count.saturating_add(1);
                    continue;
                }
            }
            let step_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Emit a diagnostic IRQ_Check parity marker mirroring Python’s early pending probe.
                let imr = self
                    .memory
                    .read_internal_byte_silent(IMEM_IMR_OFFSET)
                    .unwrap_or(0);
                let isr = self
                    .memory
                    .read_internal_byte_silent(IMEM_ISR_OFFSET)
                    .unwrap_or(0);
                let kil = self
                    .memory
                    .read_internal_byte_silent(IMEM_KIL_OFFSET)
                    .unwrap_or(0);
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
                guard.with_some(|tracer| {
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
                });
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
                        let _ = self.timer.tick_timers_with_keyboard(
                            &mut self.memory,
                            new_cycle,
                            |mem| {
                                if let Some(kb) = self.keyboard.as_mut() {
                                    // Parity: always count/key-latch events even when IRQs are masked.
                                    let events = kb.scan_tick(mem, true);
                                    if events > 0 || (kb_irq_enabled && kb.fifo_len() > 0) {
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
                        // Reassert KEYI latch only when enabled, mirroring Python HALT wake behavior.
                        self.refresh_key_irq_latch();
                    }
                    self.deliver_pending_irq()?;
                    let mut guard = PERFETTO_TRACER.enter();
                    guard.with_some(|tracer| {
                        tracer.update_counters(
                            self.metadata.instruction_count,
                            self.state.call_depth(),
                            self.memory.memory_read_count(),
                            self.memory.memory_write_count(),
                        );
                    });
                    return Ok::<(), CoreError>(());
                }

                let in_interrupt_before = self.timer.in_interrupt;
                let irq_source_before = self
                    .timer
                    .irq_source
                    .as_deref()
                    .map(LoopIrqSource::from_name);

                let pc_before = self.state.get_reg(RegName::PC) & ADDRESS_MASK;
                let (opcode, instr_len, pc_after, wait_loops) = {
                    let keyboard_ptr = self
                        .keyboard
                        .as_mut()
                        .map(|kb| kb as *mut KeyboardMatrix)
                        .unwrap_or(std::ptr::null_mut());
                    let lcd_ptr = self.lcd.as_mut().map(|lcd| lcd.as_mut() as *mut dyn LcdHal);
                    let host_read = self
                        .host_read
                        .as_mut()
                        .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
                    let host_write = self
                        .host_write
                        .as_mut()
                        .map(|f| &mut **f as *mut (dyn FnMut(u32, u8) + Send));
                    let sio_ptr = self
                        .sio
                        .as_mut()
                        .map_or(std::ptr::null_mut(), |sio| sio as *mut SioStub);
                    let mut bus = RuntimeBus {
                        mem: &mut self.memory,
                        keyboard_ptr,
                        lcd_ptr,
                        sio_ptr,
                        host_read,
                        host_write,
                        onk_level: self.onk_level,
                        cycle: self.metadata.cycle_count,
                        pc: pc_before,
                        meta_ptr: &self.metadata as *const SnapshotMetadata,
                        state_ptr: &self.state as *const LlamaState,
                    };
                    let opcode = bus.load(pc_before, 8) as u8;
                    // Capture WAIT loop count before execution (executor clears I).
                    let wait_loops = if opcode == 0xEF {
                        self.state.get_reg(RegName::I) & mask_for(RegName::I)
                    } else {
                        0
                    };
                    let instr_len = match self.executor.execute(opcode, &mut self.state, &mut bus) {
                        Ok(len) => len,
                        Err(e) => {
                            return Err(CoreError::Other(format!(
                                "execute opcode 0x{opcode:02X}: {e}"
                            )))
                        }
                    };
                    let pc_after = self.state.get_reg(RegName::PC) & ADDRESS_MASK;
                    (opcode, instr_len, pc_after, wait_loops)
                };
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
                    // Align IRQ bookkeeping with the cleared IMEM registers so pending/latched state
                    // does not survive a soft RESET.
                    self.timer.clear_pending_for_reset();
                    self.state.reset_call_metrics();
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
                    self.timer.last_irq_pc = Some(pc_before & ADDRESS_MASK);
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
                        if !self.timer.in_interrupt {
                            let kb_irq_enabled = self.timer.kb_irq_enabled;
                            let _ = self.timer.tick_timers_with_keyboard(
                                &mut self.memory,
                                cyc,
                                |mem| {
                                    if let Some(kb) = self.keyboard.as_mut() {
                                        // Parity: always count/key-latch events even when IRQs are masked.
                                        let events = kb.scan_tick(mem, true);
                                        if events > 0 || (kb_irq_enabled && kb.fifo_len() > 0) {
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
                            // KEYI delivery is handled inside tick_timers_with_keyboard and respects kb_irq_enabled.
                            if let Some(isr) = self.memory.read_internal_byte(0xFC) {
                                self.timer.irq_isr = isr;
                            }
                        }
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
                            self.timer
                                .record_bit_watch_transition("ISR", prev, cleared, pc_before);
                        }
                    }
                    // Drop any stale interrupt-stack frames (used only for bookkeeping).
                    let _ = self.timer.interrupt_stack.pop();
                    let mut guard = PERFETTO_TRACER.enter();
                    guard.with_some(|tracer| {
                        let mut payload = std::collections::HashMap::new();
                        payload.insert(
                            "pc".to_string(),
                            perfetto::AnnotationValue::Pointer(pc_before as u64),
                        );
                        payload.insert(
                            "ret".to_string(),
                            perfetto::AnnotationValue::Pointer(self.state.pc() as u64),
                        );
                        payload.insert(
                            "src".to_string(),
                            perfetto::AnnotationValue::Str(irq_src.unwrap_or_default()),
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
                    });
                }
                if let Some(detector) = self.loop_detector.as_mut() {
                    detector.record_step(LoopStep {
                        pc_before,
                        pc_after,
                        opcode,
                        instr_len,
                        in_interrupt: in_interrupt_before,
                        irq_source: irq_source_before,
                    });
                }
                self.deliver_pending_irq()?;
                let mut guard = PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    tracer.update_counters(
                        self.metadata.instruction_count,
                        self.state.call_depth(),
                        self.memory.memory_read_count(),
                        self.memory.memory_write_count(),
                    );
                });
                Ok(())
            }));

            match step_result {
                Ok(inner) => inner?,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    pub fn save_snapshot(&self, path: &std::path::Path) -> Result<()> {
        let mut metadata = self.metadata.clone();
        metadata.instruction_count = self.metadata.instruction_count;
        metadata.cycle_count = self.metadata.cycle_count;
        metadata.pc = self.get_reg("PC");
        metadata.memory_reads = self.memory.memory_read_count();
        metadata.memory_writes = self.memory.memory_write_count();
        metadata.call_depth = self.state.call_depth();
        metadata.call_sub_level = self.state.call_sub_level();
        metadata.power_state = self.state.power_state();
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

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    pub fn load_snapshot(&mut self, path: &std::path::Path) -> Result<()> {
        let loaded = snapshot::load_snapshot(path, &mut self.memory)?;
        self.metadata = loaded.metadata;
        apply_registers(&mut self.state, &loaded.registers);
        self.state.set_power_state(self.metadata.power_state);
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
            self.timer.irq_bit_watch = watch.as_object().cloned();
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
        if let Some(lcd_meta) = self.metadata.lcd.as_ref() {
            let kind = crate::lcd::lcd_kind_from_snapshot_meta(lcd_meta, LcdKind::Hd61202);
            if self.lcd.as_ref().map(|lcd| lcd.kind()) != Some(kind) {
                self.lcd = Some(create_lcd(kind));
            }
            let model = self.metadata.device_model.unwrap_or(match kind {
                LcdKind::Iq7000Vram => DeviceModel::Iq7000,
                _ => DeviceModel::PcE500,
            });
            let rom = self.memory.external_slice();
            if let Some(lcd) = self.lcd.as_deref_mut() {
                crate::device::configure_lcd_char_tracing(lcd, model, rom);
            }
            if let Some(lcd) = self.lcd.as_mut() {
                let payload = loaded.lcd_payload.as_deref();
                let should_load = payload.is_some() || kind == LcdKind::Unknown;
                if should_load {
                    let _ = lcd.load_snapshot(lcd_meta, payload.unwrap_or(&[]));
                }
            }
        } else if self.lcd.is_none() {
            self.lcd = Some(create_lcd(LcdKind::Hd61202));
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
        let irm_enabled = (imr & IMR_MASTER) != 0;
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
        guard.with_some(|tracer| {
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
        });
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
        let (op_idx, pc_trace, tag) = match crate::llama::eval::perfetto_instr_context() {
            Some((idx, ctx_pc)) => (idx, ctx_pc, None),
            None => (
                crate::llama::eval::perfetto_last_instr_index(),
                pc,
                Some("irq_delivery_out_of_exec"),
            ),
        };

        let record_stack_write = |addr: u32, bits: u8, value: u32| {
            let mut guard = PERFETTO_TRACER.enter();
            guard.with_some(|tracer| {
                let space = if MemoryImage::is_internal(addr) {
                    "internal"
                } else {
                    "external"
                };
                let substep = crate::llama::eval::perfetto_next_substep();
                tracer.record_mem_write_with_substep(
                    op_idx, pc_trace, addr, value, space, bits, substep,
                );
            });
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
            guard.with_some(|tracer| {
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
            });
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
        guard.with_some(|tracer| {
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
            if let Some(tag) = tag {
                payload.insert(
                    "tag".to_string(),
                    perfetto::AnnotationValue::Str(tag.to_string()),
                );
            }
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
            if let Some(tag) = tag {
                delivered.insert(
                    "tag".to_string(),
                    perfetto::AnnotationValue::Str(tag.to_string()),
                );
            }
            tracer.record_irq_event("IRQ_Delivered", delivered);
        });
        Ok(())
    }

    pub async fn step_async(&mut self, instructions: usize) -> Result<()> {
        for _ in 0..instructions {
            crate::async_driver::sleep_cycles(1).await;
            self.step(1)?;
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
    use super::perfetto_test_guard;
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
        rt.state.set_reg(RegName::Temp(0), 0x0000_AABB);
        rt.state.set_reg(RegName::Temp(5), 0x123456);
        rt.state.set_reg(RegName::PC, 0x12345);
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert_eq!(rt2.state.call_depth(), 2);
        assert_eq!(rt2.state.call_sub_level(), 3);
        assert_eq!(rt2.state.get_reg(RegName::Temp(0)) & 0xFFFFFF, 0x0000_AABB);
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
    fn snapshot_roundtrip_preserves_power_state() {
        let tmp_off = std::env::temp_dir().join("core_snapshot_power_off.pcsnap");
        let tmp_halt = std::env::temp_dir().join("core_snapshot_power_halt.pcsnap");
        let _ = fs::remove_file(&tmp_off);
        let _ = fs::remove_file(&tmp_halt);

        let mut rt = CoreRuntime::new();
        rt.state.power_off();
        rt.save_snapshot(&tmp_off).expect("save off snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp_off).expect("load off snapshot");
        assert!(rt2.state.is_off(), "OFF state should round-trip");

        let mut rt3 = CoreRuntime::new();
        rt3.state.set_halted(true);
        rt3.save_snapshot(&tmp_halt).expect("save halt snapshot");

        let mut rt4 = CoreRuntime::new();
        rt4.load_snapshot(&tmp_halt).expect("load halt snapshot");
        assert!(rt4.state.is_halted(), "HALT state should round-trip");
        assert!(!rt4.state.is_off(), "HALT should not restore as OFF");
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
                        let offset = (addr - crate::memory::INTERNAL_MEMORY_START)
                            & crate::memory::INTERNAL_ADDR_MASK;
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
        assert_eq!(
            ssr_clear & 0x08,
            0,
            "SSR ONK bit should clear after release"
        );
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
                        <= crate::memory::INTERNAL_ADDR_MASK
                {
                    let offset = (addr - crate::memory::INTERNAL_MEMORY_START)
                        & crate::memory::INTERNAL_ADDR_MASK;
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
    fn lcd_mapped_write_counts_as_memory_write() {
        let mut rt = CoreRuntime::new();
        // Program: MV A, 0xC0 ; MV [0x2000], A (LCD instruction write, CS=both).
        rt.memory
            .write_external_slice(0, &[0x08, 0xC0, 0xA8, 0x00, 0x20, 0x00]);
        rt.state.set_pc(0);

        assert_eq!(rt.memory.memory_write_count(), 0);
        rt.step(2).expect("execute LCD write");
        assert!(
            rt.memory.memory_write_count() >= 1,
            "overlay write should increment memory_write_count"
        );
    }

    #[test]
    fn lcd_mapped_read_counts_as_memory_read() {
        // Establish baseline read overhead (IRQ probes, opcode fetch) using a NOP.
        let mut baseline = CoreRuntime::new();
        baseline.memory.write_external_byte(0x0000, 0x00); // NOP
        baseline.state.set_pc(0);
        baseline.step(1).expect("execute NOP");
        let base_reads = baseline.memory.memory_read_count();

        let mut rt = CoreRuntime::new();
        // Program: MV A, [0x2001] (LCD read, RW=1).
        rt.memory.write_external_slice(0, &[0x88, 0x01, 0x20, 0x00]);
        rt.state.set_pc(0);

        rt.step(1).expect("execute LCD read");
        let overlay_reads = rt.memory.memory_read_count();
        assert!(
            overlay_reads >= base_reads + 2,
            "overlay path should add operand+overlay reads (base={base_reads}, got {overlay_reads})",
        );
    }

    #[test]
    fn imem_low_offsets_are_plain_ram_not_lcd_overlay() {
        let mut rt = CoreRuntime::new();
        // Program:
        //   MV (IMEM 0x00), 0x12
        //   MV A, (IMEM 0x00)
        //   MV [0x000010], A
        //
        // Regression: CoreRuntime previously aliased IMEM 0x00..0x0F to the LCD overlay,
        // but PC-E500 ROM uses those bytes as scratch RAM.
        rt.memory.write_external_slice(
            0,
            &[
                0xCC, 0x00, 0x12, // MV IMem8, imm8
                0x80, 0x00, // MV A, IMem8
                0xA8, 0x10, 0x00, 0x00, // MV [abs20], A
            ],
        );
        rt.state.set_pc(0);

        let lcd_before = rt.lcd.as_ref().expect("lcd present").stats();
        rt.step(3).expect("execute IMEM scratch program");

        let stored = rt.memory.load(0x000010, 8).unwrap_or(0) as u8;
        assert_eq!(
            stored, 0x12,
            "IMEM low offsets must behave like RAM (expected scratch value to roundtrip)"
        );

        let lcd_after = rt.lcd.as_ref().expect("lcd present").stats();
        assert_eq!(
            lcd_after.instruction_counts, lcd_before.instruction_counts,
            "IMEM scratch writes must not become LCD instructions"
        );
        assert_eq!(
            lcd_after.data_write_counts, lcd_before.data_write_counts,
            "IMEM scratch writes must not become LCD data writes"
        );
        assert_eq!(
            lcd_after.cs_both_count, lcd_before.cs_both_count,
            "IMEM scratch writes must not select LCD chips"
        );
        assert_eq!(
            lcd_after.cs_left_count, lcd_before.cs_left_count,
            "IMEM scratch writes must not select LCD chips"
        );
        assert_eq!(
            lcd_after.cs_right_count, lcd_before.cs_right_count,
            "IMEM scratch writes must not select LCD chips"
        );
    }

    #[test]
    fn call_stack_tracks_call_targets() {
        use crate::llama::opcodes::RegName;
        let mut rt = CoreRuntime::new();
        // Program:
        //   CALL 0x0005
        //   NOP
        //   NOP
        //   HALT
        rt.memory
            .write_external_slice(0, &[0x04, 0x05, 0x00, 0x00, 0x00, 0xDE]);
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x001000);

        rt.step(1).expect("execute CALL");
        assert_eq!(rt.state.call_stack(), &[0x0005]);

        rt.step(1).expect("execute HALT");
        assert!(rt.state.is_halted());
        assert_eq!(
            rt.state.call_stack(),
            &[0x0005],
            "HALT should not unwind call stack"
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
        // IMR master off, KEYI asserted: pending should still latch to mirror Python.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.timer.irq_pending = false;
        rt.arm_pending_irq_from_isr();
        assert!(
            rt.timer.irq_pending,
            "KEYI should arm pending even while IMR master is 0"
        );
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));
        // Pure timer bit with master off should still arm pending.
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_MTI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.arm_pending_irq_from_isr();
        assert!(
            rt.timer.irq_pending,
            "MTI with IMR master 0 should still latch pending like Python"
        );
    }

    #[test]
    fn arm_pending_irq_ignores_keyi_when_kb_irq_disabled() {
        let mut rt = CoreRuntime::new();
        rt.timer.set_keyboard_irq_enabled(false);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;

        rt.arm_pending_irq_from_isr();

        assert!(
            !rt.timer.irq_pending,
            "KEYI should be masked when kb IRQs are disabled"
        );
        assert!(rt.timer.irq_source.is_none());
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
            rt.timer.irq_pending,
            "pending should arm even when IMR master is 0 to match Python latch semantics"
        );
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));

        // Enabling IMR master+KEY should keep pending set and ready for delivery.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.arm_pending_irq_from_isr();
        assert!(
            rt.timer.irq_pending,
            "pending should arm when IMR allows it"
        );
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
    fn refresh_key_irq_latch_requires_existing_latch() {
        let mut rt = CoreRuntime::new();
        let kb = rt.keyboard.as_mut().unwrap();
        // Press a key and strobe columns so scan_tick can debounce and leave FIFO populated.
        kb.press_matrix_code(0x10, &mut rt.memory);
        kb.handle_write(0xF0, 0xFF, &mut rt.memory);
        kb.handle_write(0xF1, 0x07, &mut rt.memory);
        let mut events = 0;
        for _ in 0..8 {
            events += kb.scan_tick(&mut rt.memory, true);
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
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "KEYI should not reassert once the latch was cleared"
        );
        assert!(
            !rt.timer.key_irq_latched,
            "latch should remain cleared without new events"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should stay cleared without a latch"
        );
        assert!(rt.timer.irq_source.is_none());
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
            events += kb.scan_tick(&mut rt.memory, true);
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
    fn refresh_key_irq_latch_preserves_latch_when_irq_disabled() {
        let mut rt = CoreRuntime::new();
        let kb = rt.keyboard.as_mut().unwrap();
        // Create a latched KEYI while IRQs are enabled.
        kb.press_matrix_code(0x10, &mut rt.memory);
        kb.handle_write(0xF0, 0xFF, &mut rt.memory);
        kb.handle_write(0xF1, 0x0F, &mut rt.memory);
        for _ in 0..8 {
            if kb.scan_tick(&mut rt.memory, true) > 0 {
                break;
            }
        }
        kb.write_fifo_to_memory(&mut rt.memory, rt.timer.kb_irq_enabled);
        rt.timer.key_irq_latched = true;
        rt.refresh_key_irq_latch();
        assert!(
            rt.timer.key_irq_latched,
            "latch should be set while enabled"
        );
        // Disable IRQs and clear ISR, then ensure refresh keeps the latch active.
        rt.timer.set_keyboard_irq_enabled(false);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;
        rt.refresh_key_irq_latch();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "KEYI should stay asserted while latched");
        assert!(
            rt.timer.key_irq_latched,
            "latch should persist across gating"
        );
        assert!(
            rt.timer.irq_pending,
            "pending IRQ should remain set while latched"
        );
        assert_eq!(rt.timer.irq_source, Some("KEY".to_string()));
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
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "ISR should not change while in interrupt"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should remain clear while in interrupt"
        );
    }

    #[test]
    fn refresh_key_irq_latch_reasserts_when_latched() {
        let mut rt = CoreRuntime::new();
        let kb = rt.keyboard.as_mut().unwrap();
        // Generate a debounced key event and mark the latch as active (set at event time).
        kb.press_matrix_code(0x10, &mut rt.memory);
        kb.handle_write(0xF0, 0xFF, &mut rt.memory);
        kb.handle_write(0xF1, 0x07, &mut rt.memory);
        for _ in 0..8 {
            if kb.scan_tick(&mut rt.memory, true) > 0 {
                break;
            }
        }
        kb.write_fifo_to_memory(&mut rt.memory, rt.timer.kb_irq_enabled);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.key_irq_latched = true;
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;

        rt.refresh_key_irq_latch();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "KEYI should reassert when latched");
        assert!(rt.timer.key_irq_latched, "latch should remain set");
        assert!(rt.timer.irq_pending, "pending IRQ should arm when latched");
        assert_eq!(rt.timer.irq_source, Some("KEY".to_string()));
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
    fn halt_does_not_execute_instructions() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_slice(0, &[0x00, 0x00]); // NOPs.
        rt.state.set_pc(0);
        rt.step(1).expect("execute NOP");
        let pc_before = rt.state.pc();
        let instr_before = rt.instruction_count();
        let cycle_before = rt.cycle_count();

        rt.state.set_halted(true);
        rt.step(3).expect("halt idle ticks");

        assert_eq!(rt.state.pc(), pc_before, "HALT should not advance PC");
        assert_eq!(
            rt.instruction_count(),
            instr_before,
            "HALT should not execute instructions"
        );
        assert!(
            rt.cycle_count() > cycle_before,
            "HALT should still advance cycles"
        );
    }

    #[test]
    fn halt_wakes_on_key_inject() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_slice(0, &[0x00]); // NOP after HALT.
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.set_halted(true);

        let kb_irq_enabled = rt.timer.kb_irq_enabled;
        let kb = rt.keyboard.as_mut().expect("keyboard present");
        let events = kb.inject_matrix_event(0x56, false, &mut rt.memory, kb_irq_enabled);
        assert!(events > 0, "key injection should enqueue an event");

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "KEYI should be asserted after injection");

        rt.step(1).expect("halt wake step");

        assert!(
            !rt.state.is_halted(),
            "HALT should clear on injected key event"
        );
    }

    #[test]
    fn halt_reenters_when_next_opcode_is_halt() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_slice(0, &[0xDE, 0x00]); // HALT, NOP.
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.set_halted(true);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);

        let kb_irq_enabled = rt.timer.kb_irq_enabled;
        let kb = rt.keyboard.as_mut().expect("keyboard present");
        let events = kb.inject_matrix_event(0x56, false, &mut rt.memory, kb_irq_enabled);
        assert!(events > 0, "key injection should enqueue an event");

        rt.step(1).expect("halt wake step");

        assert!(rt.state.is_halted(), "HALT should re-enter halt state");
        assert_eq!(rt.state.pc(), 1, "HALT should advance PC");
        assert_eq!(
            rt.instruction_count(),
            1,
            "HALT instruction should still be executed"
        );
    }

    #[test]
    fn halt_updates_perfetto_counters_on_idle_tick() {
        use std::fs;

        let _lock = perfetto_test_guard();
        let mut rt = CoreRuntime::new();
        // Enable timer so HALT idle loop produces memory traffic (ISR write).
        rt.timer.enabled = true;
        rt.timer.mti_period = 1;
        rt.timer.next_mti = 1;
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        rt.memory.write_external_byte(0x0000, 0x00); // NOP placeholder
        rt.state.set_pc(0);

        let tmp = std::env::temp_dir().join("halt_perfetto_counters.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
        }

        rt.step(1).expect("halt idle tick");

        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
            let counters = tracer.test_counters.borrow().clone();
            assert!(
                !counters.is_empty(),
                "halt idle loop should publish perfetto counters"
            );
            let (_idx, _cd, reads, writes) = counters.last().copied().unwrap();
            assert!(
                reads > 0 || writes > 0,
                "halt tick should reflect memory activity in counters"
            );
            let _ = tracer.finish();
        }
        let _ = fs::remove_file(&tmp);
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
        assert!(
            rt.state.is_halted(),
            "HALT should not wake on KEYI when kb IRQs disabled"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should not arm for KEYI when kb IRQs disabled"
        );
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_ne!(isr & ISR_KEYI, 0, "ISR bit should remain set but ignored");
    }

    #[test]
    fn off_only_wakes_on_onk() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.power_off();

        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_MTI);
        let _ = rt.step(1);
        assert!(rt.state.is_off(), "OFF should ignore MTI");

        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        let _ = rt.step(1);
        assert!(rt.state.is_off(), "OFF should ignore KEYI");

        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        let _ = rt.step(1);
        assert!(!rt.state.is_off(), "OFF should wake on ONKI");
        assert!(rt.timer.irq_pending, "ONKI wake should arm pending IRQ");
    }

    #[test]
    fn off_stops_timers() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.power_off();
        rt.timer.enabled = true;
        rt.timer.mti_period = 1;
        rt.timer.next_mti = 1;
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);

        let cycles_before = rt.cycle_count();
        let _ = rt.step(5);
        let cycles_after = rt.cycle_count();
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);

        assert_eq!(cycles_after, cycles_before, "OFF should not advance cycles");
        assert_eq!(isr & ISR_MTI, 0, "OFF should not tick MTI");
    }

    #[test]
    fn off_clears_non_onk_isr_and_pending() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.power_off();
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("MTI".to_string());
        rt.timer.last_fired = Some("MTI".to_string());
        rt.timer.key_irq_latched = true;
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI | ISR_MTI);

        let _ = rt.step(1);

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert!(rt.state.is_off(), "OFF should remain until ONK");
        assert_eq!(isr & (ISR_KEYI | ISR_MTI | ISR_STI), 0);
        assert!(!rt.timer.irq_pending, "OFF should clear pending IRQs");
        assert!(rt.timer.irq_source.is_none());
        assert!(rt.timer.last_fired.is_none());
        assert!(
            !rt.timer.key_irq_latched,
            "OFF should clear key latch state"
        );
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
        // With IMR master off, delivery should be gated and PC should not jump yet.
        assert_ne!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            0x001234,
            "PC should not jump while IMR master=0"
        );

        // Enable IMR master and KEY bits; next step should deliver.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        let _ = rt.step(1);
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
        // Enable master + ONK, then deliver.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_ONK);
        let _ = rt.step(1);
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
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
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
        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
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
        let _ = PERFETTO_TRACER.enter().take();
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
            root.replace(Some(PerfettoTracer::new(tmp.clone())));
            {
                let mut nested = PERFETTO_TRACER.enter();
                assert!(
                    nested.with_ref(|opt| opt.is_some()),
                    "nested guard should see tracer"
                );
                let _ = nested.with_some(|tracer| tracer.record_call_flow("NESTED", 0x10, 0x20, 1));
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
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
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
        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
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
        let _ = PERFETTO_TRACER.enter().take();
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
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
        }

        let mut lcd = LcdController::new();
        // Emit an instruction (SetPage) and a data write so both paths are traced.
        lcd.write(0x02000, 0x81); // SetPage page=1, CS=both, write
        lcd.write(0x02002, 0xAA); // Data write to both chips

        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
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
        let _ = PERFETTO_TRACER.enter().take();
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn host_overlay_write_counts_and_traces() {
        use std::fs;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_host_overlay.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        let mut guard = PERFETTO_TRACER.enter();
        guard.replace(Some(PerfettoTracer::new(tmp.clone())));

        let called = Arc::new(AtomicBool::new(false));
        let flag = called.clone();
        let mut rt = CoreRuntime::new();
        rt.set_host_write(move |_addr, _val| {
            flag.store(true, Ordering::Relaxed);
        });
        // Program: MV IMem8, imm8 targeting offset 0xF5 (E-port input, now locally emulated).
        rt.memory.write_external_slice(0, &[0xCC, 0xF5, 0xAA]);
        rt.state.set_pc(0);
        let before_writes = rt.memory.memory_write_count();

        rt.step(1).expect("execute host overlay write");

        assert!(
            !called.load(Ordering::Relaxed),
            "E-port writes are locally emulated; host_write should not be required"
        );
        assert!(
            rt.memory.memory_write_count() > before_writes,
            "E-port writes should bump memory_write_count"
        );

        if let Some(tracer) = guard.take() {
            let _ = tracer.finish();
        }
        let _ = PERFETTO_TRACER.enter().take();
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn irq_delivery_out_of_exec_tags_perfetto() {
        use std::fs;
        let _lock = perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_irq_out_of_exec.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = PERFETTO_TRACER.enter();
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
        }

        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0x0100);
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0100);
        // Assert KEYI while halted with master enabled.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        rt.deliver_pending_irq().expect("deliver pending irq");

        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
            let _ = tracer.finish();
        }
        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf).to_ascii_lowercase();
        assert!(
            text.contains("irq_delivery_out_of_exec"),
            "perfetto trace should tag out-of-executor IRQ delivery"
        );
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
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
        }

        let mut mem = MemoryImage::new();
        mem.add_ram_overlay(0x8000, 1, "ram_overlay");
        let _ = mem.store_with_pc(0x8000, 8, 0xAA, Some(0x0123));

        if let Some(tracer) = PERFETTO_TRACER.enter().take() {
            let _ = tracer.finish();
        }
        let buf = fs::read(&tmp).expect("read perfetto overlay trace");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("ram_overlay"),
            "overlay name should be present in perfetto output"
        );
        let _ = PERFETTO_TRACER.enter().take();
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
        rt.timer
            .record_bit_watch_transition("IMR", 0x00, 0x80, 0x0100);
        rt.timer
            .record_bit_watch_transition("ISR", 0x00, 0x04, 0x0100);

        // Program RESET at PC=0.
        rt.memory.write_external_byte(0x0000, 0xFF);
        rt.state.set_reg(RegName::PC, 0x0000);

        rt.step(1).expect("execute RESET");

        // Timer counters should remain intact but pending/latch state should clear.
        assert_eq!(rt.timer.irq_total, 5);
        assert_eq!(rt.timer.irq_key, 2);
        assert_eq!(rt.timer.irq_mti, 1);
        assert_eq!(rt.timer.irq_sti, 1);
        assert!(
            !rt.timer.irq_pending,
            "pending flag should be cleared by RESET to mirror Python"
        );
        assert!(
            rt.timer.irq_source.is_none(),
            "irq_source should clear on RESET"
        );
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
            guard.replace(Some(PerfettoTracer::new(tmp.clone())));
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
            text.contains("sub_020000"),
            "trace should include call-flow slice for destination"
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
        assert_eq!(
            rt.state.call_depth(),
            1,
            "interrupt should raise call depth"
        );
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
    fn snapshot_roundtrip_restores_unknown_lcd_kind() {
        let tmp = std::env::temp_dir().join("core_snapshot_unknown_lcd.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        rt.lcd = Some(create_lcd(LcdKind::Unknown));
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert_eq!(
            rt2.lcd.as_ref().expect("lcd restored").kind(),
            LcdKind::Unknown
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
        assert_ne!(
            isr & (ISR_MTI | ISR_STI),
            0,
            "ISR should reflect timer fire"
        );
        // Cycle counter should advance for opcode + I loops.
        assert_eq!(rt.metadata.cycle_count, 6);
    }

    #[test]
    fn requires_python_without_host_falls_back() {
        let mut rt = CoreRuntime::new();
        // Mark an external range as Python-only and point PC at it.
        rt.memory
            .set_python_ranges(vec![(0x0000_2000, 0x0000_2000)]);
        rt.state.set_reg(RegName::PC, 0x0000_2000);
        // Seed a NOP opcode so the fetch path is taken.
        rt.memory.write_external_byte(0x0000_2000, 0x00);
        let res = rt.step(1);
        assert!(
            res.is_ok(),
            "step should execute without host overlays: {res:?}"
        );
        assert_eq!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            0x0000_2001,
            "PC should advance on NOP even without overlays"
        );
        assert_eq!(
            rt.metadata.instruction_count, 1,
            "instruction counter should increment on NOP"
        );
    }

    #[test]
    fn imem_access_handles_e_port_without_host() {
        let mut rt = CoreRuntime::new();
        // Program: MV IMem8, imm8 targeting offset 0xF5 (E-port input, locally emulated).
        rt.memory.write_external_slice(0, &[0xCC, 0xF5, 0xAA]);
        rt.state.set_pc(0);

        let res = rt.step(1);
        assert!(
            res.is_ok(),
            "E-port IMEM accesses should be handled locally without a Python overlay: {res:?}"
        );
    }

    #[test]
    fn reset_intrinsic_clears_irq_and_call_metadata() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_byte(0x0000, 0xFF); // RESET opcode
        rt.state.set_pc(0);
        // Seed IRQ metadata that should be cleared by RESET.
        rt.timer.irq_pending = true;
        rt.timer.in_interrupt = true;
        rt.timer.irq_source = Some("KEY".to_string());
        rt.timer.key_irq_latched = true;
        rt.timer.delivered_masks = vec![0x04];
        rt.timer.interrupt_stack = vec![1, 2];
        rt.timer.next_interrupt_id = 3;
        rt.timer.last_irq_src = Some("KEY".to_string());
        rt.timer.last_irq_pc = Some(0x012345);
        rt.timer.last_irq_vector = Some(0x00ABCD);
        rt.state.call_depth_inc();
        rt.state.call_depth_inc();
        rt.state.push_call_page(0x0F0000);

        rt.step(1).expect("execute RESET");

        assert!(!rt.timer.irq_pending, "reset should clear pending IRQ");
        assert!(
            !rt.timer.in_interrupt,
            "reset should exit interrupt context"
        );
        assert!(rt.timer.irq_source.is_none(), "irq_source should clear");
        assert!(!rt.timer.key_irq_latched, "KEY latch should clear");
        assert!(
            rt.timer.delivered_masks.is_empty(),
            "delivered masks cleared"
        );
        assert!(
            rt.timer.interrupt_stack.is_empty(),
            "interrupt stack cleared"
        );
        assert_eq!(rt.timer.next_interrupt_id, 0, "interrupt id reset");
        assert!(rt.timer.last_irq_src.is_none(), "last_irq_src cleared");
        assert!(rt.timer.last_irq_pc.is_none(), "last_irq_pc cleared");
        assert!(
            rt.timer.last_irq_vector.is_none(),
            "last_irq_vector cleared"
        );
        assert_eq!(rt.state.call_depth(), 0, "call depth reset");
        assert_eq!(rt.state.call_sub_level(), 0, "call sub-level reset");
        assert!(
            rt.state.peek_call_page().is_none(),
            "call page stack cleared"
        );
    }

    #[test]
    fn runtime_overlay_helpers_route_through_memory_image() {
        let mut rt = CoreRuntime::new();
        rt.add_ram_overlay(0x8000, 2, "runtime_ram");
        rt.clear_overlay_logs();
        let _ = rt.memory.store_with_pc(0x8000, 16, 0xBEEF, Some(0x0100));
        let writes = rt.overlay_write_log();
        assert_eq!(writes.len(), 2, "should log 2 overlay byte writes");
        assert!(writes.iter().all(|entry| entry.overlay == "runtime_ram"));
        assert!(writes.iter().any(|entry| entry.pc == Some(0x0100)));
        let val = rt.memory.load_with_pc(0x8000, 16, Some(0x0200)).unwrap();
        assert_eq!(val, 0xBEEF);
    }
}
