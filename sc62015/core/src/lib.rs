// PY_SOURCE: sc62015/pysc62015/emulator.py:RegisterName
// PY_SOURCE: sc62015/pysc62015/emulator.py:Registers

pub mod async_driver;
pub mod device;
pub mod generated_key_input;
pub mod iq7000;
pub mod keyboard;
pub mod lcd;
pub mod lcd_text;
pub mod llama;
pub mod loop_detector;
pub mod memory;
pub mod pce500;
pub mod pce500_peripherals;
pub mod perfetto;
pub mod sio;
pub mod snapshot;
pub mod timer;

use crate::llama::state::{validate_f_image, PowerState};
use crate::llama::{opcodes::RegName, state::LlamaState};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use serde_json::json;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
use thiserror::Error;

pub use async_driver::{
    current_cycle, emit_event, sleep_cycles, AsyncDriver, CycleSleep, DriverEvent, DriverRunResult,
};
pub use device::{
    DeviceKeyboardProfile, DeviceMemoryCardProfile, DeviceModel, DeviceTextDecoder,
    DeviceTimerProfile, TimerProfileProvenance,
};
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
pub use pce500_peripherals::{
    CassetteBlock, CassetteBlockKind, CassetteError, CassettePulse, CassettePulseError,
    CassettePulseStream, CassettePulseTiming, CassetteRetryPolicy, CassetteTapeImage,
    MemoryCardBlock, MemoryCardImage, Pce500PeripheralBridge, RamDiskImage, StorageError,
};
pub use perfetto::PerfettoTracer;
pub use sio::{
    SioInputLines, SioQueuedByte, SioSnapshot, SioStub, SioTimedEvent, SioTimingConfig,
    SioTimingSnapshot,
};
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
use crate::llama::state::{mask_for, CallMetricsSnapshot};
use crate::memory::{
    IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_KIL_OFFSET, IMEM_KOH_OFFSET, IMEM_KOL_OFFSET,
    IMEM_LCC_OFFSET, IMEM_RXD_OFFSET, IMEM_SCR_OFFSET, IMEM_SSR_OFFSET, IMEM_TXD_OFFSET,
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
#[serde(deny_unknown_fields)]
pub struct TimerInfo {
    pub enabled: bool,
    pub mti_period: u64,
    pub sti_period: u64,
    pub next_mti: u64,
    pub next_sti: u64,
    pub kb_irq_enabled: bool,
    pub instruction_start_cycle: u64,
    pub last_mti_fire_cycle: Option<u64>,
    pub last_sti_fire_cycle: Option<u64>,
    pub fired_mti_since_boundary: bool,
    pub fired_sti_since_boundary: bool,
    pub preserve_phase: bool,
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
            instruction_start_cycle: 0,
            last_mti_fire_cycle: None,
            last_sti_fire_cycle: None,
            fired_mti_since_boundary: false,
            fired_sti_since_boundary: false,
            preserve_phase: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct InterruptInfo {
    pub pending: bool,
    pub in_interrupt: bool,
    pub key_irq_latched: bool,
    pub source: Option<String>,
    pub last_fired: Option<String>,
    pub stack: Vec<u32>,
    pub next_id: u32,
    pub imr: u8,
    pub isr: u8,
    pub irq_counts: Option<serde_json::Value>,
    pub last_irq: Option<serde_json::Value>,
    pub irq_bit_watch: Option<serde_json::Value>,
    pub delivered_masks: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    pub external_interrupt_level: bool,
    #[serde(default)]
    pub onk_level: bool,
    #[serde(default)]
    pub call_depth: u32,
    #[serde(default)]
    pub call_sub_level: u32,
    pub call_stack: Vec<u32>,
    pub call_page_stack: Vec<u32>,
    pub call_return_widths: Vec<u8>,
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
    /// Legacy host-runner optimization hint retained in snapshot JSON for
    /// schema compatibility. It is not architectural machine state.
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
            external_interrupt_level: false,
            onk_level: false,
            call_depth: 0,
            call_sub_level: 0,
            call_stack: Vec::new(),
            call_page_stack: Vec::new(),
            call_return_widths: Vec::new(),
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

pub const DEFAULT_REG_WIDTH: u8 = 24;
/// Number of SC62015 temporary registers mirrored by the Python emulator.
pub const NUM_TEMP_REGISTERS: u8 = 16;

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
    // TEMP0..TEMP15 are persisted in SnapshotMetadata::temps. Keeping the
    // fixed binary register file to these eight keys prevents the serializer
    // from silently inventing, dropping, or truncating fields.
    regs
}

pub fn apply_registers(state: &mut LlamaState, regs: &HashMap<String, u32>) -> Result<()> {
    // Validate quarantined raw fields before changing any state so a rejected
    // snapshot cannot be partially applied.
    if let Some(value) = regs.get("F") {
        validate_f_image(*value).map_err(|message| CoreError::InvalidSnapshot(message.into()))?;
    }
    for (name, _) in snapshot::SNAPSHOT_REGISTER_LAYOUT.iter() {
        let value = *regs.get(*name).unwrap_or(&0);
        if let Some(reg) = reg_from_name(name) {
            state.set_reg(reg, value & mask_for_width(register_width(name)));
        }
    }
    for idx in 0..NUM_TEMP_REGISTERS {
        let key = format!("TEMP{idx}");
        if let Some(value) = regs.get(&key) {
            state.set_reg(
                RegName::Temp(idx),
                *value & mask_for_width(DEFAULT_REG_WIDTH),
            );
        }
    }
    Ok(())
}

/// Extremely small placeholder runtime for LLAMA-only execution.
pub struct CoreRuntime {
    metadata: SnapshotMetadata,
    pub memory: MemoryImage,
    pub state: LlamaState,
    loop_detector: Option<LoopDetector>,
    executor: crate::llama::eval::LlamaExecutor,
    pub keyboard: Option<KeyboardMatrix>,
    pub lcd: Option<Box<dyn LcdHal>>,
    pub sio: Option<SioStub>,
    pub pce500_peripherals: Option<Pce500PeripheralBridge>,
    pub timer: Box<TimerContext>,
    host_read: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
    host_peek: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
    host_write: Option<Box<dyn FnMut(u32, u8) + Send>>,
    iq7000_clock_seed: Option<iq7000::Iq7000ClockSeed>,
    iq7000_rtc: Option<iq7000::Iq7000RtcPeripheral>,
    onk_level: bool,
    external_interrupt_level: bool,
    poisoned: Option<String>,
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
            loop_detector: None,
            executor: crate::llama::eval::LlamaExecutor::new(),
            keyboard: Some(KeyboardMatrix::new()),
            lcd: Some(Box::new(LcdController::new())),
            sio: None,
            pce500_peripherals: None,
            timer: Box::new(TimerContext::new(false, 0, 0)),
            host_read: None,
            host_peek: None,
            host_write: None,
            iq7000_clock_seed: None,
            iq7000_rtc: None,
            onk_level: false,
            external_interrupt_level: false,
            poisoned: None,
        };
        rt.set_device_model(DeviceModel::PcE500)
            .expect("device model settings missing");
        rt.install_imr_isr_hook();
        rt
    }

    /// Construct a complete machine runtime from one authoritative model profile.
    pub fn for_model(model: DeviceModel, rom: &[u8]) -> Result<Self> {
        let mut runtime = Self::new();
        model.configure_fresh_runtime(&mut runtime, rom)?;
        Ok(runtime)
    }

    pub fn device_model(&self) -> DeviceModel {
        self.metadata.device_model.unwrap_or(DeviceModel::PcE500)
    }

    pub fn set_device_model(&mut self, model: DeviceModel) -> Result<()> {
        self.metadata.device_model = Some(model);
        self.memory
            .set_internal_ram_mirror(model.spec().internal_ram_mirror);
        Ok(())
    }

    pub fn instruction_count(&self) -> u64 {
        self.metadata.instruction_count
    }

    pub fn cycle_count(&self) -> u64 {
        self.metadata.cycle_count
    }

    pub fn power_on_reset(&mut self) -> Result<()> {
        struct ResetBus<'a> {
            mem: &'a mut MemoryImage,
            host_read: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            host_peek: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            lcd_ptr: Option<*mut dyn LcdHal>,
            keyboard_active: bool,
            sio_active: bool,
            rtc_active: bool,
            pc: u32,
        }

        impl LlamaBus for ResetBus<'_> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                let addr = addr & ADDRESS_MASK;
                if self.mem.requires_python(addr) {
                    if let Some(read) = self.host_read {
                        // SAFETY: the reset bus exclusively owns the callback
                        // pointer for the duration of this synchronous call.
                        if let Some(value) = unsafe { (*read)(addr) } {
                            self.mem.bump_read_count();
                            return u32::from(value);
                        }
                    }
                }
                self.mem
                    .load_with_pc(addr, bits, Some(self.pc))
                    .unwrap_or(0)
            }

            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                let _ = self.mem.store(addr, bits, value);
            }

            fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                self.peek_byte_silent_at(addr, self.pc)
            }

            fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
                let addr = addr & ADDRESS_MASK;
                if let Some(offset) = MemoryImage::internal_offset(addr) {
                    if (self.keyboard_active
                        && matches!(offset, IMEM_KOL_OFFSET | IMEM_KOH_OFFSET | IMEM_KIL_OFFSET))
                        || (self.rtc_active && offset == iq7000::IMEM_EIL_OFFSET)
                        || (self.sio_active
                            && matches!(
                                offset,
                                IMEM_UCR_OFFSET
                                    | IMEM_USR_OFFSET
                                    | IMEM_RXD_OFFSET
                                    | IMEM_TXD_OFFSET
                            ))
                    {
                        return None;
                    }
                }
                if let Some(lcd_ptr) = self.lcd_ptr {
                    // SAFETY: read-only address classification; ResetBus owns
                    // the machine borrow while this pointer is live.
                    if unsafe { (&*lcd_ptr).handles(addr) } {
                        return None;
                    }
                }
                if self.mem.requires_python(addr) {
                    return self.host_peek.and_then(|peek| {
                        // SAFETY: see the corresponding architectural reader.
                        unsafe { (*peek)(addr) }
                    });
                }
                self.mem.read_byte_for_preflight(addr, Some(context_pc))
            }

            fn vector_transfer_provenance(&self) -> (usize, u64) {
                self.mem.vector_transfer_provenance()
            }

            fn instruction_byte_is_stable(&self, addr: u32) -> bool {
                self.mem.instruction_byte_is_stable(addr)
            }
        }

        let host_read = self
            .host_read
            .as_mut()
            .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
        let host_peek = self
            .host_peek
            .as_mut()
            .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
        let lcd_ptr = self.lcd.as_mut().map(|lcd| lcd.as_mut() as *mut dyn LcdHal);
        let mut bus = ResetBus {
            mem: &mut self.memory,
            host_read,
            host_peek,
            lcd_ptr,
            keyboard_active: self.keyboard.is_some(),
            sio_active: self.sio.is_some(),
            rtc_active: self.iq7000_rtc.is_some(),
            pc: self.state.pc() & ADDRESS_MASK,
        };
        let result = crate::llama::eval::power_on_reset(&mut bus, &mut self.state)
            .map_err(|error| CoreError::Other(format!("power-on reset: {error}")));
        if result.is_ok() {
            self.poisoned = None;
            let scr = self
                .memory
                .read_internal_byte_silent(IMEM_SCR_OFFSET)
                .unwrap_or(0);
            self.timer
                .sync_scr_selection(scr, self.metadata.cycle_count);
        }
        result
    }

    /// Provide an optional host overlay reader for IMEM regions that require Python/device handling
    /// (e.g., E-port inputs, ONK). Called only when `MemoryImage::requires_python` flags an address.
    pub fn set_host_read<F>(&mut self, f: F)
    where
        F: FnMut(u32) -> Option<u8> + Send + 'static,
    {
        self.host_read = Some(Box::new(f));
    }

    /// Provide the explicitly side-effect-free counterpart to `set_host_read`.
    /// Decode/vector preflight refuses host-routed addresses when this is not
    /// installed; it never falls back to the observable reader.
    pub fn set_host_peek<F>(&mut self, f: F)
    where
        F: FnMut(u32) -> Option<u8> + Send + 'static,
    {
        self.host_peek = Some(Box::new(f));
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
        self.host_peek = None;
        self.host_write = None;
    }

    pub fn set_iq7000_clock_seed_yyyymmddhhmm(&mut self, raw: &str) -> Result<()> {
        let seed = iq7000::Iq7000ClockSeed::from_yyyymmddhhmm(raw).map_err(CoreError::Other)?;
        seed.apply_to_memory(&mut self.memory);
        if let Some(rtc) = self.iq7000_rtc.as_mut() {
            rtc.set_seed(seed.clone());
        } else {
            self.iq7000_rtc = Some(iq7000::Iq7000RtcPeripheral::new(seed.clone()));
        }
        self.iq7000_clock_seed = Some(seed);
        Ok(())
    }

    pub fn clear_iq7000_clock_seed(&mut self) {
        self.iq7000_clock_seed = None;
        self.iq7000_rtc = None;
        for idx in 0..iq7000::CLOCK_WORKSPACE_LEN {
            self.memory
                .write_external_byte(iq7000::CLOCK_WORKSPACE_START + idx as u32, 0);
        }
        self.memory
            .write_external_byte(iq7000::CLOCK_INITIALIZED_FLAG, 0);
    }

    /// Change one physical keyboard-matrix contact. The selected electrical
    /// KIL level is sampled at the next scheduler boundary; this path does not
    /// inject a translated FIFO byte or directly assert `ISR.KEYI`.
    pub fn set_physical_matrix_key(&mut self, code: u8, pressed: bool) -> bool {
        let Some(keyboard) = self.keyboard.as_mut() else {
            return false;
        };
        if usize::from(code) >= keyboard.matrix_code_capacity() {
            return false;
        }
        if pressed {
            keyboard.press_matrix_code(code, &mut self.memory);
        } else {
            keyboard.release_matrix_code(code, &mut self.memory);
        }
        true
    }

    /// Queue an already-translated host input byte without changing the
    /// physical matrix or manufacturing a silicon KEYI level. This is used for
    /// non-matrix inputs such as digitizer samples.
    pub fn queue_translated_key_event(&mut self, code: u8) -> usize {
        let Some(keyboard) = self.keyboard.as_mut() else {
            return 0;
        };
        let events = keyboard.inject_input_event(code, &mut self.memory, self.timer.kb_irq_enabled);
        if events > 0 {
            self.timer.key_irq_latched = true;
        }
        events
    }

    /// Compatibility helper for tests that require an immediate debounced
    /// matrix FIFO event as well as a physical contact transition. Normal UI
    /// input should use `set_physical_matrix_key`.
    pub fn inject_immediate_matrix_event_for_diagnostics(
        &mut self,
        code: u8,
        release: bool,
    ) -> usize {
        let Some(keyboard) = self.keyboard.as_mut() else {
            return 0;
        };
        let events = keyboard.inject_matrix_event(
            code,
            release,
            &mut self.memory,
            self.timer.kb_irq_enabled,
        );
        if events > 0 {
            self.timer.key_irq_latched = true;
        }
        events
    }

    pub fn enable_sio_stub(&mut self) {
        if self.sio.is_none() {
            let mut stub = SioStub::new();
            stub.init(&mut self.memory);
            self.sio = Some(stub);
        }
    }

    pub fn queue_sio_receive_byte(&mut self, value: u8) {
        self.enable_sio_stub();
        if let Some(sio) = self.sio.as_mut() {
            sio.queue_receive_byte(value, &mut self.memory);
        }
        self.refresh_sio_interrupts();
    }

    fn advance_sio(&mut self, timing_units: u64) {
        let events = self
            .sio
            .as_mut()
            .map(|sio| sio.tick_cycles(timing_units, &mut self.memory));
        let Some(events) = events else {
            return;
        };
        let tx_completed = events
            .iter()
            .any(|event| matches!(event, SioTimedEvent::TxComplete(_)));
        self.refresh_sio_interrupts();
        if tx_completed {
            self.assert_sio_transmit_ready();
        }
    }

    fn refresh_sio_interrupts(&mut self) {
        if self.sio.is_none() {
            return;
        }
        let usr = self
            .memory
            .read_internal_byte_silent(IMEM_USR_OFFSET)
            .unwrap_or(0);
        let imr = self
            .memory
            .read_internal_byte_silent(IMEM_IMR_OFFSET)
            .unwrap_or(0);
        let isr = self
            .memory
            .read_internal_byte_silent(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        let mut new_isr = isr;
        let mut source = None;

        if (usr & USR_RX_READY) != 0 {
            new_isr |= ISR_RXI;
            source = Some("RX");
        }

        if new_isr == isr {
            return;
        }
        self.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
        self.timer.irq_isr = new_isr;
        self.timer.irq_imr = imr;
        if !self.timer.in_interrupt {
            self.timer.irq_pending = true;
            if let Some(src) = source {
                match self.timer.irq_source.as_deref() {
                    None => self.timer.irq_source = Some(src.to_string()),
                    Some(cur) if irq_source_priority(src) < irq_source_priority(cur) => {
                        self.timer.irq_source = Some(src.to_string());
                    }
                    _ => {}
                }
            }
        }
        self.timer.last_fired = self.timer.irq_source.clone();
    }

    pub fn assert_sio_transmit_ready(&mut self) {
        self.assert_irq_source(ISR_TXI, "TX");
    }

    /// Set a neutral external interrupt input level.
    ///
    /// While asserted, hardware re-latches `ISR.EXI` after firmware clears
    /// it. No device, connector, or physical meaning is coupled to this input.
    pub fn set_external_interrupt_level(&mut self, asserted: bool) {
        self.external_interrupt_level = asserted;
        if asserted {
            self.refresh_external_interrupt_level();
            return;
        }

        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let new_isr = isr & !ISR_EXI;
        if new_isr != isr {
            self.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
        }
        self.timer.irq_isr = new_isr;
        if !self.timer.in_interrupt && self.timer.irq_source.as_deref() == Some("EX") {
            let next_source = if (new_isr & ISR_RXI) != 0 {
                Some("RX")
            } else if (new_isr & ISR_TXI) != 0 {
                Some("TX")
            } else if (new_isr & ISR_ONKI) != 0 {
                Some("ONK")
            } else if (new_isr & ISR_KEYI) != 0 {
                Some("KEY")
            } else if (new_isr & ISR_STI) != 0 {
                Some("STI")
            } else if (new_isr & ISR_MTI) != 0 {
                Some("MTI")
            } else {
                None
            };
            self.timer.irq_source = next_source.map(str::to_string);
            self.timer.irq_pending = next_source.is_some();
        }
    }

    pub fn external_interrupt_level(&self) -> bool {
        self.external_interrupt_level
    }

    fn refresh_external_interrupt_level(&mut self) {
        if !self.external_interrupt_level {
            return;
        }
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if (isr & ISR_EXI) == 0 {
            self.memory
                .write_internal_byte(IMEM_ISR_OFFSET, isr | ISR_EXI);
        }
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        if !self.timer.in_interrupt {
            self.timer.irq_pending = true;
            match self.timer.irq_source.as_deref() {
                None => self.timer.irq_source = Some("EX".to_string()),
                Some(current) if irq_source_priority("EX") < irq_source_priority(current) => {
                    self.timer.irq_source = Some("EX".to_string());
                }
                _ => {}
            }
            self.timer.last_fired = self.timer.irq_source.clone();
        }
    }

    fn assert_irq_source(&mut self, mask: u8, source: &str) {
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if (isr & mask) == 0 {
            self.memory.write_internal_byte(IMEM_ISR_OFFSET, isr | mask);
        }
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        if !self.timer.in_interrupt {
            self.timer.irq_pending = true;
            self.timer.irq_source = Some(source.to_string());
            self.timer.last_fired = self.timer.irq_source.clone();
        }
    }

    pub fn enable_pce500_peripheral_bridge(&mut self, card_capacity: usize) {
        if self.pce500_peripherals.is_none() {
            self.pce500_peripherals = Some(Pce500PeripheralBridge::new(card_capacity));
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
                    if (new & ISR_KNOWN_MASK) == 0 {
                        timer.irq_pending = false;
                        timer.irq_source = None;
                    }
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

    /// Clear the physical ON-key level without acknowledging latched ONKI.
    ///
    /// Both stock-ROM dispatchers explicitly clear the selected ISR bit after
    /// service. Treating key release as that acknowledgement can lose an
    /// interrupt before firmware observes it.
    pub fn release_on_key(&mut self) {
        self.onk_level = false;
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        self.timer.irq_isr = isr;
        if !self.timer.in_interrupt
            && self.timer.irq_source.as_deref() == Some("ONK")
            && (isr & ISR_ONKI) == 0
        {
            let next_source = if (isr & ISR_RXI) != 0 {
                Some("RX")
            } else if (isr & ISR_EXI) != 0 {
                Some("EX")
            } else if (isr & ISR_TXI) != 0 {
                Some("TX")
            } else if (isr & ISR_KEYI) != 0 {
                Some("KEY")
            } else if (isr & ISR_STI) != 0 {
                Some("STI")
            } else if (isr & ISR_MTI) != 0 {
                Some("MTI")
            } else {
                None
            };
            self.timer.irq_source = next_source.map(str::to_string);
            self.timer.irq_pending = next_source.is_some();
        }
    }

    fn refresh_on_key_interrupt_level(&mut self) {
        if !self.onk_level {
            return;
        }
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if (isr & ISR_ONKI) == 0 {
            self.memory
                .write_internal_byte(IMEM_ISR_OFFSET, isr | ISR_ONKI);
        }
        self.timer.irq_isr = self
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(self.timer.irq_isr);
        self.timer.irq_imr = self
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(self.timer.irq_imr);
        if !self.timer.in_interrupt {
            self.timer.irq_pending = true;
            match self.timer.irq_source.as_deref() {
                None => self.timer.irq_source = Some("ONK".to_string()),
                Some(current) if irq_source_priority("ONK") < irq_source_priority(current) => {
                    self.timer.irq_source = Some("ONK".to_string());
                }
                _ => {}
            }
            self.timer.last_fired = self.timer.irq_source.clone();
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

    fn raw_selected_kil(&self) -> u8 {
        // LCC.KSD disconnects keyboard scanning. Otherwise the physical level
        // is independent of debounce/FIFO policy and IMR.
        if self
            .memory
            .read_internal_byte_silent(IMEM_LCC_OFFSET)
            .unwrap_or(0)
            & 0x04
            != 0
        {
            return 0;
        }
        self.keyboard
            .as_ref()
            .map_or(0, KeyboardMatrix::compute_physical_kil)
    }

    fn refresh_raw_key_irq_level(&mut self) {
        if self.raw_selected_kil() != 0 {
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
            if !self.timer.in_interrupt && self.timer.irq_source.is_none() {
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
        self.tick_timers_and_keyboard_selected(cycle, true, true);
    }

    fn tick_timers_and_keyboard_selected(&mut self, cycle: u64, run_mti: bool, run_sti: bool) {
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let mirror_pce500_fifo = matches!(
            self.device_model(),
            DeviceModel::PcE500 | DeviceModel::PcE500Jp
        );
        let _ = self.timer.tick_timers_with_keyboard_selected(
            &mut self.memory,
            cycle,
            |mem| {
                if let Some(kb) = self.keyboard.as_mut() {
                    // Parity: always count/key-latch events even when IRQs are masked.
                    let events = kb.scan_tick(mem, true);
                    let fifo_pending = kb.fifo_len() > 0;
                    if events > 0 || (kb_irq_enabled && fifo_pending) {
                        let drained = if mirror_pce500_fifo {
                            kb.drain_fifo_to_pce500_iocs_workspace(mem, kb_irq_enabled)
                        } else {
                            0
                        };
                        if drained == 0 {
                            kb.write_fifo_to_memory(mem, kb_irq_enabled);
                        }
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
            run_mti,
            run_sti,
        );
        if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
            self.timer.irq_isr = isr;
        }
    }

    fn arm_pending_irq_from_isr(&mut self) {
        if self.timer.irq_pending {
            return;
        }
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        if isr == 0 {
            return;
        }
        let isr_effective = isr;
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        // Parity: Python marks irq_pending as soon as ISR bits are asserted, even if IMR master is 0
        // or the source mask is currently disabled. Delivery is still gated later.
        let src = if (isr_effective & ISR_RXI) != 0 {
            Some("RX")
        } else if (isr_effective & ISR_EXI) != 0 {
            Some("EX")
        } else if (isr_effective & ISR_TXI) != 0 {
            Some("TX")
        } else if (isr_effective & ISR_ONKI) != 0 {
            Some("ONK")
        } else if (isr_effective & ISR_KEYI) != 0 {
            Some("KEY")
        } else if (isr_effective & ISR_STI) != 0 {
            Some("STI")
        } else if (isr_effective & ISR_MTI) != 0 {
            Some("MTI")
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
        if !self.timer.in_interrupt {
            match self.timer.irq_source.as_deref() {
                None => self.timer.irq_source = src.map(str::to_string),
                Some(cur) => {
                    if let Some(src_name) = src {
                        if irq_source_priority(src_name) < irq_source_priority(cur) {
                            self.timer.irq_source = Some(src_name.to_string());
                        }
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

    /// Decide whether the IRQ transfer is already selected at this scheduling
    /// boundary using only side-effect-free state. Sources asserted later by a
    /// timer or device remain pending for the next boundary.
    fn irq_transfer_selected_at_step_entry(&self) -> bool {
        if self.state.is_off() || self.state.is_halted() || self.timer.in_interrupt {
            return false;
        }

        let imr = self
            .memory
            .read_internal_byte_silent(IMEM_IMR_OFFSET)
            .unwrap_or(0);
        if (imr & IMR_MASTER) == 0 {
            return false;
        }

        let asserted_isr = self
            .memory
            .read_internal_byte_silent(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        let key_will_reassert = self.raw_selected_kil() != 0;
        let sio_rx_will_assert = self.sio.is_some()
            && (imr & IMR_RX) != 0
            && (self
                .memory
                .read_internal_byte_silent(IMEM_USR_OFFSET)
                .unwrap_or(0)
                & USR_RX_READY)
                != 0;

        let mut predicted_isr = asserted_isr;
        if key_will_reassert {
            predicted_isr |= ISR_KEYI;
        }
        if self.onk_level {
            predicted_isr |= ISR_ONKI;
        }
        if self.external_interrupt_level {
            predicted_isr |= ISR_EXI;
        }
        if sio_rx_will_assert {
            predicted_isr |= ISR_RXI;
        }

        let pending_or_will_reassert = self.timer.irq_pending
            || key_will_reassert
            || self.onk_level
            || self.external_interrupt_level
            || sio_rx_will_assert
            || (asserted_isr & ISR_KNOWN_MASK) != 0;

        pending_or_will_reassert && (predicted_isr & imr & ISR_KNOWN_MASK) != 0
    }

    pub fn load_rom(&mut self, blob: &[u8], start: usize) {
        let end = (start + blob.len()).min(self.memory.external_len());
        if start < end {
            self.memory
                .write_external_slice(start, &blob[..(end - start)]);
        }
    }

    pub fn step(&mut self, instructions: usize) -> Result<()> {
        if let Some(reason) = self.poisoned.as_deref() {
            return Err(CoreError::Other(format!(
                "SC62015 CoreRuntime is poisoned after a failed side-effecting operation; \
                 power-on reset required: {reason}"
            )));
        }
        // Execute real instructions through the LLAMA evaluator instead of bumping PC.
        struct RuntimeBus<'a> {
            mem: &'a mut MemoryImage,
            keyboard_ptr: *mut KeyboardMatrix,
            lcd_ptr: Option<*mut dyn LcdHal>,
            sio_ptr: *mut SioStub,
            host_read: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            host_peek: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            host_write: Option<*mut (dyn FnMut(u32, u8) + Send)>,
            iq7000_clock_seed: Option<*const iq7000::Iq7000ClockSeed>,
            iq7000_rtc: *mut iq7000::Iq7000RtcPeripheral,
            timer_ptr: *mut TimerContext,
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
                    if let Some(seed_ptr) = self.iq7000_clock_seed {
                        if let Some(val) = (*seed_ptr).read(addr, bits) {
                            (*self.mem).bump_read_count();
                            return val;
                        }
                    }
                    if bits == 8
                        && !self.iq7000_rtc.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if offset == iq7000::IMEM_EIL_OFFSET {
                            let val = (*self.iq7000_rtc).handle_eil_read();
                            let _ = (*self.mem).store(addr, bits, val as u32);
                            return val as u32;
                        }
                    }
                    if bits > 8
                        && !self.sio_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let bytes = bits.div_ceil(8).max(1) as u32;
                        let start = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        let end = start.saturating_add(bytes.saturating_sub(1));
                        if start <= IMEM_TXD_OFFSET && end >= IMEM_UCR_OFFSET {
                            let mut out = 0u32;
                            for byte_offset in 0..bytes {
                                let byte = self.load(addr.wrapping_add(byte_offset), 8) & 0xFF;
                                out |= byte << (byte_offset * 8);
                            }
                            return out;
                        }
                    }
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
                                val |= SSR_ONK;
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
                    if bits == 8
                        && !self.iq7000_rtc.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let offset = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        if offset == iq7000::IMEM_EOL_OFFSET {
                            (*self.iq7000_rtc).handle_eol_write(value as u8);
                            let _ = (*self.mem).store(addr, bits, value);
                            return;
                        }
                    }
                    if bits > 8
                        && !self.sio_ptr.is_null()
                        && MemoryImage::is_internal(addr)
                        && (addr - INTERNAL_MEMORY_START) <= INTERNAL_ADDR_MASK
                    {
                        let bytes = bits.div_ceil(8).max(1) as u32;
                        let start = (addr - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
                        let end = start.saturating_add(bytes.saturating_sub(1));
                        if start <= IMEM_TXD_OFFSET && end >= IMEM_UCR_OFFSET {
                            for byte_offset in 0..bytes {
                                let byte = (value >> (byte_offset * 8)) & 0xFF;
                                self.store(addr.wrapping_add(byte_offset), 8, byte);
                            }
                            return;
                        }
                    }
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
                    let bytes = bits.div_ceil(8).max(1) as u32;
                    let wrote_scr = (0..bytes).any(|byte_offset| {
                        MemoryImage::internal_offset(addr.wrapping_add(byte_offset))
                            == Some(IMEM_SCR_OFFSET)
                    });
                    if wrote_scr {
                        if let Some(timer) = self.timer_ptr.as_mut() {
                            let scr = (*self.mem)
                                .read_internal_byte_silent(IMEM_SCR_OFFSET)
                                .unwrap_or(0);
                            timer.sync_scr_selection(scr, self.cycle);
                        }
                    }
                }
            }
            fn resolve_emem(&mut self, base: u32) -> u32 {
                base
            }
            fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                self.peek_byte_silent_at(addr, self.pc)
            }
            fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
                let addr = addr & ADDRESS_MASK;
                unsafe {
                    if let Some(seed_ptr) = self.iq7000_clock_seed {
                        if let Some(value) = (*seed_ptr).read(addr, 8) {
                            return Some(value as u8);
                        }
                    }
                    if MemoryImage::is_internal(addr) {
                        let offset = MemoryImage::internal_offset(addr)?;
                        if offset == iq7000::IMEM_EIL_OFFSET && !self.iq7000_rtc.is_null() {
                            return None;
                        }
                        if matches!(offset, IMEM_KOL_OFFSET | IMEM_KOH_OFFSET | IMEM_KIL_OFFSET)
                            && !self.keyboard_ptr.is_null()
                        {
                            return None;
                        }
                        if matches!(
                            offset,
                            IMEM_UCR_OFFSET | IMEM_USR_OFFSET | IMEM_RXD_OFFSET | IMEM_TXD_OFFSET
                        ) && !self.sio_ptr.is_null()
                        {
                            return None;
                        }
                        if offset == IMEM_SSR_OFFSET {
                            let mut value = (*self.mem).read_internal_byte_silent(offset)?;
                            if self.onk_level {
                                value |= SSR_ONK;
                            }
                            return Some(value);
                        }
                    }
                    if let Some(lcd_ptr) = self.lcd_ptr {
                        if (&*lcd_ptr).handles(addr) {
                            return None;
                        }
                    }
                    if (*self.mem).requires_python(addr) {
                        return self.host_peek.and_then(|peek| (*peek)(addr));
                    }
                    (*self.mem).read_byte_for_preflight(addr, Some(context_pc))
                }
            }
            fn vector_transfer_provenance(&self) -> (usize, u64) {
                self.mem.vector_transfer_provenance()
            }

            fn instruction_byte_is_stable(&self, addr: u32) -> bool {
                self.mem.instruction_byte_is_stable(addr)
            }
            fn peek_imem_silent(&mut self, offset: u32) -> u8 {
                self.mem.read_internal_byte_silent(offset).unwrap_or(0)
            }
            fn supports_wait_cycles(&self) -> bool {
                true
            }
            fn wait_cycles(&mut self, _cycles: u32) {
                // CoreRuntime applies WAIT timing after execution so timer,
                // keyboard, and metadata updates share one outer-step path.
            }
            fn supports_timer_phase_clear(&self) -> bool {
                !self.timer_ptr.is_null()
            }
            fn clear_timer_phases(&mut self, clear_sti: bool, clear_mti: bool) {
                unsafe {
                    if let Some(timer) = self.timer_ptr.as_mut() {
                        if clear_mti {
                            timer.next_mti = self.cycle.wrapping_add(timer.mti_period);
                        }
                        if clear_sti {
                            timer.next_sti = self.cycle.wrapping_add(timer.sti_period);
                        }
                    }
                }
            }
        }

        for _ in 0..instructions {
            let halted_at_step_entry = self.state.is_halted();
            let irq_transfer_selected = self.irq_transfer_selected_at_step_entry();
            // Reject quarantined/invalid encodings before level re-latching,
            // SIO/keyboard/device callbacks, IRQ wake/delivery, or timer
            // advancement. A halted core with no current wake source does not
            // fetch an instruction at all.
            let asserted_isr = self
                .memory
                .read_internal_byte_silent(IMEM_ISR_OFFSET)
                .unwrap_or(0);
            let should_preflight = if self.state.is_off() {
                // OFF ignores every status source except ONKI.  In
                // particular, do not perform an architectural opcode fetch
                // merely because an unrelated ISR bit is set.
                (asserted_isr & ISR_ONKI) != 0
            } else {
                !self.state.is_halted()
                    || asserted_isr != 0
                    || self.onk_level
                    || self.external_interrupt_level
            };
            let prepared_instruction = {
                let pc = self.state.pc() & ADDRESS_MASK;
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
                let host_peek = self
                    .host_peek
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
                let iq7000_rtc = self
                    .iq7000_rtc
                    .as_mut()
                    .map_or(std::ptr::null_mut(), |rtc| {
                        rtc as *mut iq7000::Iq7000RtcPeripheral
                    });
                let mut bus = RuntimeBus {
                    mem: &mut self.memory,
                    keyboard_ptr,
                    lcd_ptr,
                    sio_ptr,
                    host_read,
                    host_peek,
                    host_write,
                    iq7000_clock_seed: self
                        .iq7000_clock_seed
                        .as_ref()
                        .map(|seed| seed as *const iq7000::Iq7000ClockSeed),
                    iq7000_rtc,
                    timer_ptr: self.timer.as_mut() as *mut TimerContext,
                    onk_level: self.onk_level,
                    cycle: self.metadata.cycle_count,
                    pc,
                    meta_ptr: &self.metadata as *const SnapshotMetadata,
                    state_ptr: &self.state as *const LlamaState,
                };
                // Fully validate the current instruction through the silent
                // bus first. A malformed current encoding takes precedence
                // even when the IRQ destination aliases it, and must consume
                // zero architectural reads.
                let silent_prepared_opcode = if should_preflight && !irq_transfer_selected {
                    let silent_opcode = bus.peek_byte_silent(pc).ok_or_else(|| {
                        CoreError::Other(format!(
                        "preflight opcode at 0x{pc:05X}: side-effect-free memory is unavailable"
                    ))
                    })?;
                    let instruction_len = self
                        .executor
                        .validate_before_scheduling_with_length(
                            silent_opcode,
                            &self.state,
                            &mut bus,
                        )
                        .map_err(|error| {
                            CoreError::Other(format!(
                                "preflight opcode 0x{silent_opcode:02X} at 0x{pc:05X}: {error}"
                            ))
                        })?;
                    if (0..u32::from(instruction_len)).any(|offset| {
                        !bus.instruction_byte_is_stable(pc.wrapping_add(offset) & ADDRESS_MASK)
                    }) {
                        return Err(CoreError::Other(format!(
                            "preflight opcode 0x{silent_opcode:02X} at 0x{pc:05X}: \
                         callback-backed instruction bytes cannot cross scheduler tick"
                        )));
                    }
                    if silent_opcode == 0xFF {
                        let (reset_target, reset_target_len) =
                            crate::llama::eval::validate_vector_transfer_with_length(
                                crate::pce500::ROM_RESET_VECTOR_ADDR,
                                &self.state,
                                &mut bus,
                            )
                            .map_err(|error| {
                                CoreError::Other(format!("RESET vector preflight: {error}"))
                            })?;
                        if (0..3).any(|offset| {
                            !bus.instruction_byte_is_stable(
                                crate::pce500::ROM_RESET_VECTOR_ADDR.wrapping_add(offset)
                                    & ADDRESS_MASK,
                            )
                        }) || (0..u32::from(reset_target_len)).any(|offset| {
                            !bus.instruction_byte_is_stable(
                                reset_target.wrapping_add(offset) & ADDRESS_MASK,
                            )
                        }) {
                            return Err(CoreError::Other(
                                "RESET vector preflight: callback-backed vector/target".to_string(),
                            ));
                        }
                    }
                    let timing = crate::llama::timing::PreparedInstructionTiming::prepare(
                        silent_opcode,
                        pc,
                        crate::llama::dispatch::lookup,
                        |address| bus.peek_byte_silent_at(address, pc),
                    )
                    .map_err(|error| {
                        CoreError::Other(format!(
                            "preflight timing for opcode 0x{silent_opcode:02X} at 0x{pc:05X}: {error}"
                        ))
                    })?;
                    Some((silent_opcode, timing))
                } else {
                    None
                };

                if irq_transfer_selected {
                    // Prove the asynchronous IRQ vector and destination only
                    // when this scheduling boundary can actually deliver it.
                    // The final transfer performs its own one-shot validation.
                    let (irq_target, irq_target_len) =
                        crate::llama::eval::validate_vector_transfer_with_length(
                            INTERRUPT_VECTOR_ADDR,
                            &self.state,
                            &mut bus,
                        )
                        .map_err(|error| {
                            CoreError::Other(format!("IRQ vector preflight: {error}"))
                        })?;
                    if (0..3).any(|offset| {
                        !bus.instruction_byte_is_stable(
                            INTERRUPT_VECTOR_ADDR.wrapping_add(offset) & ADDRESS_MASK,
                        )
                    }) || (0..u32::from(irq_target_len)).any(|offset| {
                        !bus.instruction_byte_is_stable(
                            irq_target.wrapping_add(offset) & ADDRESS_MASK,
                        )
                    }) {
                        return Err(CoreError::Other(
                            "IRQ vector preflight: callback-backed vector/target".to_string(),
                        ));
                    }
                    // A deliverable asynchronous IRQ replaces the current
                    // instruction after exactly one opcode-byte fetch. Do not
                    // decode or read operands from the discarded instruction.
                    let _discarded_opcode = bus.load(pc, 8);
                }

                if let Some((silent_opcode, timing)) = silent_prepared_opcode {
                    let opcode = bus.load(pc, 8) as u8;
                    if opcode != silent_opcode {
                        return Err(CoreError::Other(format!(
                            "architectural opcode fetch at 0x{pc:05X} disagrees with preflight: \
                         fetched 0x{opcode:02X}, preflight 0x{silent_opcode:02X}"
                        )));
                    }
                    let transfer = match opcode {
                        0xFE => Some(
                            crate::llama::eval::prepare_validated_vector(
                                INTERRUPT_VECTOR_ADDR,
                                &self.state,
                                &mut bus,
                            )
                            .map_err(|error| {
                                CoreError::Other(format!("IR vector transfer: {error}"))
                            })?,
                        ),
                        0xFF => Some(
                            crate::llama::eval::fetch_validated_vector(
                                crate::pce500::ROM_RESET_VECTOR_ADDR,
                                &self.state,
                                &mut bus,
                            )
                            .map_err(|error| {
                                CoreError::Other(format!("RESET vector transfer: {error}"))
                            })?,
                        ),
                        _ => None,
                    };
                    Some((opcode, transfer, timing))
                } else {
                    None
                }
            };
            if !self.state.is_off() {
                let scr = self
                    .memory
                    .read_internal_byte_silent(IMEM_SCR_OFFSET)
                    .unwrap_or(0);
                self.timer
                    .sync_scr_selection(scr, self.metadata.cycle_count);
            }
            if irq_transfer_selected {
                // Materialize level-sensitive sources only after the silent
                // vector proof and discarded-opcode fetch above. Delivery
                // writes the frame, then performs the architectural vector
                // reads. The recursive one-instruction step executes the
                // selected handler within this caller's instruction budget.
                self.refresh_on_key_interrupt_level();
                self.refresh_external_interrupt_level();
                self.refresh_sio_interrupts();
                self.refresh_raw_key_irq_level();
                self.arm_pending_irq_from_isr();
                self.deliver_pending_irq()?;
                if !self.timer.in_interrupt {
                    return Err(CoreError::Other(
                        "selected IRQ boundary did not enter its handler".to_string(),
                    ));
                }
                self.step(1)?;
                continue;
            }
            if self.state.is_off() {
                if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                    // Hardware wake filtering is not evidence that ignored
                    // status bits are destroyed. Preserve the complete ISR
                    // image and only gate the provisional OFF wake decision.
                    self.timer.irq_isr = isr;
                    if (isr & ISR_ONKI) != 0 {
                        self.state.set_power_state(PowerState::Running);
                        self.timer.irq_pending = true;
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
            // ONK and external interrupts are level-sensitive: firmware
            // clears each ISR bit, waits one instruction, then retests the
            // input. Re-latch once per runtime step while the corresponding
            // neutral host level remains high, including before host-side
            // instruction short-circuits.
            self.refresh_on_key_interrupt_level();
            self.refresh_external_interrupt_level();
            if let Some(sio) = self.sio.as_mut() {
                if sio.maybe_short_circuit(self.state.pc(), &mut self.state, &mut self.memory) {
                    self.metadata.instruction_count =
                        self.metadata.instruction_count.saturating_add(1);
                    self.metadata.cycle_count = self.metadata.cycle_count.saturating_add(1);
                    continue;
                }
            }
            if let Some(bridge) = self.pce500_peripherals.as_mut() {
                if bridge.maybe_short_circuit(self.state.pc(), &mut self.state, &mut self.memory) {
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
                        if (isr & ISR_RXI) != 0 {
                            Some("RX".to_string())
                        } else if (isr & ISR_EXI) != 0 {
                            Some("EX".to_string())
                        } else if (isr & ISR_TXI) != 0 {
                            Some("TX".to_string())
                        } else if (isr & ISR_ONKI) != 0 {
                            Some("ONK".to_string())
                        } else if (isr & ISR_KEYI) != 0 {
                            Some("KEY".to_string())
                        } else if (isr & ISR_STI) != 0 {
                            Some("STI".to_string())
                        } else if (isr & ISR_MTI) != 0 {
                            Some("MTI".to_string())
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
                // Mirror SIO hardware status bits into ISR before foreground/IRQ polling.
                self.refresh_sio_interrupts();
                // Sample the selected, undebounced matrix level. Host FIFO and
                // debounce bookkeeping are not silicon KEYI sources.
                self.refresh_raw_key_irq_level();
                // If ISR already has pending bits (e.g., host write) arm a pending IRQ so delivery can occur once IMR allows it.
                self.arm_pending_irq_from_isr();

                // HALT wake-up: exit low-power state when any ISR bit is set, even if IMR is masked.
                if self.state.is_halted() {
                    if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                        // kb_irq_enabled controls host generation, not the
                        // meaning of an already asserted silicon status bit.
                        if isr != 0 {
                            self.state.set_halted(false);
                            self.timer.irq_pending = true;
                            self.timer.irq_isr = isr;
                            self.timer.irq_imr = self
                                .memory
                                .read_internal_byte(IMEM_IMR_OFFSET)
                                .unwrap_or(self.timer.irq_imr);
                            if self.timer.irq_source.is_none() {
                                let src = if (isr & ISR_RXI) != 0 {
                                    "RX"
                                } else if (isr & ISR_EXI) != 0 {
                                    "EX"
                                } else if (isr & ISR_TXI) != 0 {
                                    "TX"
                                } else if (isr & ISR_ONKI) != 0 {
                                    "ONK"
                                } else if (isr & ISR_KEYI) != 0 {
                                    "KEY"
                                } else if (isr & ISR_STI) != 0 {
                                    "STI"
                                } else if (isr & ISR_MTI) != 0 {
                                    "MTI"
                                } else {
                                    "IRQ"
                                };
                                self.timer.irq_source = Some(src.to_string());
                            }
                            self.timer.last_fired = self.timer.irq_source.clone();
                        }
                    }
                }

                // A HALT boundary remains an idle boundary even when a status
                // level wakes the core. Execute the first foreground opcode
                // on the next scheduler step, after it receives its own
                // silent validation and architectural fetch.
                if halted_at_step_entry {
                    let prev_cycle = self.metadata.cycle_count;
                    let new_cycle = prev_cycle.wrapping_add(1);
                    // HALT stops the SC62015 system clock, so the main timer
                    // and its keyboard scan retain phase.  The 32 kHz subclock
                    // continues and may wake the core through STI.
                    self.timer.defer_mti(1);
                    self.metadata.cycle_count = new_cycle;
                    self.tick_timers_and_keyboard_selected(new_cycle, false, true);
                    if self
                        .memory
                        .read_internal_byte(IMEM_ISR_OFFSET)
                        .is_some_and(|isr| isr != 0)
                    {
                        self.state.set_halted(false);
                        self.arm_pending_irq_from_isr();
                    }
                    self.refresh_raw_key_irq_level();
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
                let (prepared_opcode, prepared_transfer, prepared_timing) = prepared_instruction
                    .ok_or_else(|| {
                        CoreError::Other(
                            "running CPU reached execution without a prepared opcode".to_string(),
                        )
                    })?;
                let initial_i = (self.state.get_reg(RegName::I) & mask_for(RegName::I)) as u16;
                let (opcode, instr_len, pc_after) = {
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
                    let host_peek = self
                        .host_peek
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
                    let iq7000_rtc = self
                        .iq7000_rtc
                        .as_mut()
                        .map_or(std::ptr::null_mut(), |rtc| {
                            rtc as *mut iq7000::Iq7000RtcPeripheral
                        });
                    let mut bus = RuntimeBus {
                        mem: &mut self.memory,
                        keyboard_ptr,
                        lcd_ptr,
                        sio_ptr,
                        host_read,
                        host_peek,
                        host_write,
                        iq7000_clock_seed: self
                            .iq7000_clock_seed
                            .as_ref()
                            .map(|seed| seed as *const iq7000::Iq7000ClockSeed),
                        iq7000_rtc,
                        timer_ptr: self.timer.as_mut() as *mut TimerContext,
                        onk_level: self.onk_level,
                        cycle: self.metadata.cycle_count,
                        pc: pc_before,
                        meta_ptr: &self.metadata as *const SnapshotMetadata,
                        state_ptr: &self.state as *const LlamaState,
                    };
                    let opcode = prepared_opcode;
                    let instr_len = match self.executor.execute_with_vector_transfer(
                        opcode,
                        &mut self.state,
                        &mut bus,
                        prepared_transfer,
                    ) {
                        Ok(len) => len,
                        Err(e) => {
                            return Err(CoreError::Other(format!(
                                "execute opcode 0x{opcode:02X}: {e}"
                            )))
                        }
                    };
                    let pc_after = self.state.get_reg(RegName::PC) & ADDRESS_MASK;
                    (opcode, instr_len, pc_after)
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
                    // The executor consumed the one prepared architectural
                    // vector fetch and set PC from that exact proof. Re-reading
                    // a volatile vector here could make metadata disagree with
                    // the actual destination after the frame already committed.
                    self.timer.last_irq_vector = Some(pc_after & ADDRESS_MASK);
                }
                self.metadata.instruction_count = self.metadata.instruction_count.wrapping_add(1);

                // Advance in documented SC62015 relative timing units.  The
                // evaluator has already performed every fail-closed check, so
                // devices only observe time for an instruction that retired.
                let run_timer_cycles = true;
                let sequential_pc = pc_before.wrapping_add(u32::from(instr_len)) & ADDRESS_MASK;
                let branch_taken = matches!(prepared_timing.resolved_opcode(), 0x14..=0x1f)
                    && pc_after != sequential_pc;
                let cycle_increment = prepared_timing.timing_units(initial_i, branch_taken);
                let prev_cycle = self.metadata.cycle_count;
                let new_cycle = prev_cycle.wrapping_add(cycle_increment);
                if run_timer_cycles {
                    let mut timer_cycle = prev_cycle;
                    while let Some(fire_cycle) =
                        self.timer.next_fire_cycle_in_span(timer_cycle, new_cycle)
                    {
                        self.tick_timers_and_keyboard(fire_cycle);
                        timer_cycle = fire_cycle;
                    }
                }
                self.advance_sio(cycle_increment);
                self.metadata.cycle_count = new_cycle;
                if opcode == 0x01 {
                    let irq_src = self.timer.irq_source.clone();
                    // ROM-consistent model: RETI restores the interrupt frame without an
                    // implicit ISR acknowledgement. Both stock dispatchers explicitly clear
                    // the selected ISR bit first, so direct unacknowledged silicon behavior
                    // remains a hardware-trace question.
                    let delivered_mask = self.timer.delivered_masks.pop();
                    self.timer.in_interrupt = false;
                    if irq_src.as_deref().is_some_and(|source| source == "KEY") {
                        // Retire the synthetic host-key delivery latch. This is emulator bridge
                        // bookkeeping, not an architectural ISR acknowledgement.
                        self.timer.key_irq_latched = false;
                    }
                    self.timer.irq_source = None;
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
                        if let Some(mask) = delivered_mask {
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
    fn reject_unrepresented_snapshot_runtime(&self) -> Result<()> {
        // Perfetto is process-global. Serialize snapshot checks with tests
        // that temporarily install a tracer so an unrelated parallel test is
        // not mistaken for state owned by this CoreRuntime instance.
        #[cfg(test)]
        let _perfetto_test_lock = crate::perfetto::perfetto_test_guard();

        self.memory.validate_snapshot_overlay_contract()?;
        let mut active = Vec::new();
        if self.poisoned.is_some() {
            active.push("poisoned fail-stop runtime state");
        }
        if self.sio.is_some() {
            active.push("SIO queues/line/timing state");
        }
        if self.pce500_peripherals.is_some() {
            active.push("PCE peripheral/card/cassette state");
        }
        if self.host_read.is_some() || self.host_peek.is_some() || self.host_write.is_some() {
            active.push("host overlay callbacks and external state");
        }
        if self.iq7000_clock_seed.is_some() || self.iq7000_rtc.is_some() {
            active.push("IQ-7000 RTC protocol state");
        }
        if PERFETTO_TRACER.enter().with_some(|_tracer| ()).is_some() {
            active.push("active Perfetto trace state");
        }
        if active.is_empty() {
            Ok(())
        } else {
            Err(CoreError::InvalidSnapshot(format!(
                "snapshot v4 cannot exactly represent active {}",
                active.join(", ")
            )))
        }
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    pub fn save_snapshot(&self, path: &std::path::Path) -> Result<()> {
        self.reject_unrepresented_snapshot_runtime()?;
        let mut metadata = self.metadata.clone();
        metadata.instruction_count = self.metadata.instruction_count;
        metadata.cycle_count = self.metadata.cycle_count;
        metadata.pc = self.get_reg("PC");
        metadata.memory_reads = self.memory.memory_read_count();
        metadata.memory_writes = self.memory.memory_write_count();
        // CoreRuntime has one execution path. Preserve the v4 JSON member for
        // compatibility, but never claim an inert tuning mode is active.
        metadata.fast_mode = false;
        metadata.fallback_ranges = self.memory.python_ranges().to_vec();
        metadata.readonly_ranges = self.memory.readonly_ranges().to_vec();
        metadata.call_depth = self.state.call_depth();
        metadata.call_sub_level = self.state.call_sub_level();
        let call_metrics = self.state.snapshot_call_metrics();
        metadata.call_stack = call_metrics.call_stack;
        metadata.call_page_stack = call_metrics.call_page_stack;
        metadata.call_return_widths = call_metrics.call_return_widths;
        metadata.power_state = self.state.power_state();
        metadata.external_interrupt_level = self.external_interrupt_level;
        metadata.onk_level = self.onk_level;
        metadata.temps = (0..NUM_TEMP_REGISTERS)
            .map(|index| {
                (
                    index.to_string(),
                    self.state.get_reg(RegName::Temp(index)) & 0xFF_FFFF,
                )
            })
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
        self.reject_unrepresented_snapshot_runtime()?;
        let loaded = snapshot::load_snapshot(path)?;
        let metadata = loaded.metadata.clone();
        let model = metadata.device_model.unwrap_or(DeviceModel::PcE500);
        if model != self.device_model() {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot device model {model:?} does not match active machine {:?}",
                self.device_model()
            )));
        }

        let active_fallback_ranges =
            snapshot::canonical_snapshot_ranges("active fallback", self.memory.python_ranges())?;
        if active_fallback_ranges != metadata.fallback_ranges {
            return Err(CoreError::InvalidSnapshot(
                "snapshot fallback ranges do not match the active machine".to_string(),
            ));
        }
        let active_readonly_ranges =
            snapshot::canonical_snapshot_ranges("active readonly", self.memory.readonly_ranges())?;
        if active_readonly_ranges != metadata.readonly_ranges {
            return Err(CoreError::InvalidSnapshot(
                "snapshot read-only ranges do not match the active machine".to_string(),
            ));
        }
        for (start, end) in &active_readonly_ranges {
            let unchanged = if *end < crate::memory::EXTERNAL_SPACE as u32 {
                let start = *start as usize;
                let end = *end as usize + 1;
                self.memory.external_slice()[start..end] == loaded.external_memory[start..end]
            } else if *start >= crate::memory::INTERNAL_MEMORY_START
                && *end
                    < crate::memory::INTERNAL_MEMORY_START + crate::memory::INTERNAL_SPACE as u32
            {
                let start = (*start - crate::memory::INTERNAL_MEMORY_START) as usize;
                let end = (*end - crate::memory::INTERNAL_MEMORY_START) as usize + 1;
                self.memory.internal_slice()[start..end] == loaded.imem[start..end]
            } else {
                false
            };
            if !unchanged {
                return Err(CoreError::InvalidSnapshot(
                    "snapshot attempts to replace active read-only memory".to_string(),
                ));
            }
        }

        // Validate the typed card candidate against the complete current
        // overlay attestation before constructing any other replacement state.
        // The epoch-bound plan is committed first at the mutation boundary.
        let memory_card_candidate = self
            .memory
            .prepare_memory_card_restore(loaded.memory_card.clone())?;

        let mut state_candidate = LlamaState::new();
        apply_registers(&mut state_candidate, &loaded.registers)?;
        for index in 0..NUM_TEMP_REGISTERS {
            let value = metadata
                .temps
                .get(&index.to_string())
                .copied()
                .ok_or_else(|| {
                    CoreError::InvalidSnapshot(format!(
                        "snapshot is missing temporary register TEMP{index}"
                    ))
                })?;
            state_candidate.set_reg(RegName::Temp(index), value);
        }
        state_candidate.restore_call_metrics(CallMetricsSnapshot {
            call_stack: metadata.call_stack.clone(),
            call_depth: metadata.call_depth,
            call_sub_level: metadata.call_sub_level,
            call_page_stack: metadata.call_page_stack.clone(),
            call_return_widths: metadata.call_return_widths.clone(),
        });
        state_candidate.set_power_state(metadata.power_state);

        let mut timer_candidate = (*self.timer).clone();
        timer_candidate.apply_snapshot_info(
            &metadata.timer,
            &metadata.interrupts,
            metadata.cycle_count,
        );
        timer_candidate.restore_scr_selector(loaded.imem[IMEM_SCR_OFFSET as usize]);

        let keyboard_candidate = match metadata.keyboard.as_ref() {
            Some(value) => {
                let snapshot: KeyboardSnapshot =
                    serde_json::from_value(value.clone()).map_err(|error| {
                        CoreError::InvalidSnapshot(format!(
                            "invalid keyboard snapshot metadata: {error}"
                        ))
                    })?;
                let mut keyboard = KeyboardMatrix::new();
                keyboard.load_snapshot_state(&snapshot).map_err(|error| {
                    CoreError::InvalidSnapshot(format!("invalid keyboard snapshot: {error}"))
                })?;
                let restored = serde_json::to_value(keyboard.snapshot_state())?;
                if restored != *value {
                    return Err(CoreError::InvalidSnapshot(
                        "keyboard snapshot is not exactly representable".to_string(),
                    ));
                }
                Some(keyboard)
            }
            None if self.keyboard.is_none() => None,
            None => {
                return Err(CoreError::InvalidSnapshot(
                    "snapshot is missing keyboard state".to_string(),
                ));
            }
        };

        let lcd_candidate = if let Some(lcd_meta) = metadata.lcd.as_ref() {
            let kind = crate::lcd::lcd_kind_from_snapshot_meta(lcd_meta, LcdKind::Hd61202);
            let lcd_model = metadata.device_model.unwrap_or(match kind {
                LcdKind::Iq7000Vram => DeviceModel::Iq7000,
                _ => DeviceModel::PcE500,
            });
            let mut lcd = create_lcd(kind);
            crate::device::configure_lcd_char_tracing(
                lcd.as_mut(),
                lcd_model,
                &loaded.external_memory,
            );
            lcd.load_snapshot(lcd_meta, loaded.lcd_payload.as_deref().unwrap_or(&[]))
                .map_err(|error| {
                    CoreError::InvalidSnapshot(format!("invalid LCD snapshot: {error}"))
                })?;
            let (restored_meta, restored_payload) = lcd.export_snapshot();
            if restored_meta != *lcd_meta
                || restored_payload.as_slice() != loaded.lcd_payload.as_deref().unwrap_or(&[])
            {
                return Err(CoreError::InvalidSnapshot(
                    "LCD snapshot is not exactly representable".to_string(),
                ));
            }
            Some(lcd)
        } else {
            None
        };

        // Commit only after every archive member and candidate subsystem has
        // been parsed and validated. No fallible operation follows this point.
        self.memory
            .commit_memory_card_restore(memory_card_candidate)?;
        self.memory
            .copy_external_from(&loaded.external_memory)
            .expect("validated exact external memory length");
        self.memory.write_imem(&loaded.imem);
        self.memory
            .set_python_ranges(metadata.fallback_ranges.clone());
        self.memory
            .set_readonly_ranges(metadata.readonly_ranges.clone());
        self.memory
            .set_internal_ram_mirror(model.spec().internal_ram_mirror);
        self.memory.clear_dirty();
        self.memory
            .set_memory_counts(metadata.memory_reads, metadata.memory_writes);
        self.state = state_candidate;
        *self.timer = timer_candidate;
        self.keyboard = keyboard_candidate;
        self.lcd = lcd_candidate;
        self.external_interrupt_level = metadata.external_interrupt_level;
        self.onk_level = metadata.onk_level;
        self.metadata = metadata;
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
        let mut sp = self.state.get_reg(reg) & mask;
        for i in (0..bytes).rev() {
            sp = sp.wrapping_sub(1) & mask;
            let byte = (value >> (8 * i)) & 0xFF;
            let _ = self.memory.store(sp, 8, byte);
        }
        self.state.set_reg(reg, sp);
    }

    fn deliver_pending_irq(&mut self) -> Result<()> {
        if !self.timer.irq_pending {
            return Ok(());
        }
        // Nested hardware IRQ delivery is not established by ROM or hardware
        // traces. Match the device runtimes and retain the pending source for
        // delivery after RETI instead of manufacturing a nested frame.
        if self.timer.in_interrupt {
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
                if (isr & ISR_RXI) != 0 {
                    Some("RX".to_string())
                } else if (isr & ISR_EXI) != 0 {
                    Some("EX".to_string())
                } else if (isr & ISR_TXI) != 0 {
                    Some("TX".to_string())
                } else if (isr & ISR_ONKI) != 0 {
                    Some("ONK".to_string())
                } else if (isr & ISR_KEYI) != 0 {
                    Some("KEY".to_string())
                } else if (isr & ISR_STI) != 0 {
                    Some("STI".to_string())
                } else if (isr & ISR_MTI) != 0 {
                    Some("MTI".to_string())
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
        // Match the IQ-7000 ROM interrupt dispatcher priority.
        let src = if (isr & ISR_RXI != 0) && (imr & IMR_RX != 0) {
            Some((ISR_RXI, "RX"))
        } else if (isr & ISR_EXI != 0) && (imr & IMR_EX != 0) {
            Some((ISR_EXI, "EX"))
        } else if (isr & ISR_TXI != 0) && (imr & IMR_TX != 0) {
            Some((ISR_TXI, "TX"))
        } else if (isr & ISR_ONKI != 0) && (imr & IMR_ONK != 0) {
            Some((ISR_ONKI, "ONK"))
        } else if (isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0) {
            Some((ISR_KEYI, "KEY"))
        } else if (isr & ISR_STI != 0) && (imr & IMR_STI != 0) {
            Some((ISR_STI, "STI"))
        } else if (isr & ISR_MTI != 0) && (imr & IMR_MTI != 0) {
            Some((ISR_MTI, "MTI"))
        } else {
            None
        };
        let Some((mask, src_name)) = src else {
            return Ok(());
        };

        // Resolve the vector and statically validate its destination before
        // constructing the interrupt frame or changing IMR/runtime metadata.
        // The proof remains silent until the complete frame is present.
        struct VectorBus<'a> {
            mem: &'a mut MemoryImage,
            host_read: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            host_peek: Option<*mut (dyn FnMut(u32) -> Option<u8> + Send)>,
            lcd_ptr: Option<*mut dyn LcdHal>,
            keyboard_active: bool,
            sio_active: bool,
            rtc_active: bool,
            pc: u32,
        }

        impl LlamaBus for VectorBus<'_> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                let addr = addr & ADDRESS_MASK;
                if self.mem.requires_python(addr) {
                    if let Some(read) = self.host_read {
                        // SAFETY: this synchronous bus exclusively owns
                        // the callback pointer while delivering the IRQ.
                        if let Some(value) = unsafe { (*read)(addr) } {
                            self.mem.bump_read_count();
                            return u32::from(value);
                        }
                    }
                }
                self.mem
                    .load_with_pc(addr, bits, Some(self.pc))
                    .unwrap_or(0)
            }

            fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {
                unreachable!("IRQ vector validation must never write memory")
            }

            fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                self.peek_byte_silent_at(addr, self.pc)
            }

            fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
                let addr = addr & ADDRESS_MASK;
                if let Some(offset) = MemoryImage::internal_offset(addr) {
                    if (self.keyboard_active
                        && matches!(offset, IMEM_KOL_OFFSET | IMEM_KOH_OFFSET | IMEM_KIL_OFFSET))
                        || (self.rtc_active && offset == iq7000::IMEM_EIL_OFFSET)
                        || (self.sio_active
                            && matches!(
                                offset,
                                IMEM_UCR_OFFSET
                                    | IMEM_USR_OFFSET
                                    | IMEM_RXD_OFFSET
                                    | IMEM_TXD_OFFSET
                            ))
                    {
                        return None;
                    }
                }
                if let Some(lcd_ptr) = self.lcd_ptr {
                    // SAFETY: address classification is read-only.
                    if unsafe { (&*lcd_ptr).handles(addr) } {
                        return None;
                    }
                }
                if self.mem.requires_python(addr) {
                    return self.host_peek.and_then(|peek| {
                        // SAFETY: this is the explicit safe callback.
                        unsafe { (*peek)(addr) }
                    });
                }
                self.mem.read_byte_for_preflight(addr, Some(context_pc))
            }

            fn vector_transfer_provenance(&self) -> (usize, u64) {
                self.mem.vector_transfer_provenance()
            }

            fn instruction_byte_is_stable(&self, addr: u32) -> bool {
                self.mem.instruction_byte_is_stable(addr)
            }

            fn supports_wait_cycles(&self) -> bool {
                true
            }
        }

        let vector_transfer = {
            let host_read = self
                .host_read
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
            let host_peek = self
                .host_peek
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
            let lcd_ptr = self.lcd.as_mut().map(|lcd| lcd.as_mut() as *mut dyn LcdHal);
            let mut bus = VectorBus {
                mem: &mut self.memory,
                host_read,
                host_peek,
                lcd_ptr,
                keyboard_active: self.keyboard.is_some(),
                sio_active: self.sio.is_some(),
                rtc_active: self.iq7000_rtc.is_some(),
                pc: self.state.pc() & ADDRESS_MASK,
            };
            crate::llama::eval::prepare_validated_vector(
                INTERRUPT_VECTOR_ADDR,
                &self.state,
                &mut bus,
            )
            .map_err(|error| CoreError::Other(format!("IRQ vector transfer: {error}")))?
        };

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

        let vector_result = {
            let host_read = self
                .host_read
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
            let host_peek = self
                .host_peek
                .as_mut()
                .map(|f| &mut **f as *mut (dyn FnMut(u32) -> Option<u8> + Send));
            let lcd_ptr = self.lcd.as_mut().map(|lcd| lcd.as_mut() as *mut dyn LcdHal);
            let mut bus = VectorBus {
                mem: &mut self.memory,
                host_read,
                host_peek,
                lcd_ptr,
                keyboard_active: self.keyboard.is_some(),
                sio_active: self.sio.is_some(),
                rtc_active: self.iq7000_rtc.is_some(),
                pc,
            };
            vector_transfer
                .consume_after_architectural_fetch(INTERRUPT_VECTOR_ADDR, &self.state, &mut bus)
                .map_err(|error| CoreError::Other(format!("IRQ vector transfer: {error}")))
        };
        let vec = match vector_result {
            Ok(vec) => vec,
            Err(error) => {
                // HW-014 establishes that the architectural vector read occurs
                // after the complete frame and IMR.IRM clear. Those writes are
                // already observable and cannot be rolled back safely, so a
                // failed transfer must prevent a second frame from being
                // manufactured on a later step.
                if self.poisoned.is_none() {
                    self.poisoned = Some(error.to_string());
                }
                return Err(error);
            }
        };

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
        self.state.set_pc(vec);
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
        self.timer.last_irq_vector = Some(vec);
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
}

const IMR_MASTER: u8 = 0x80;
const IMR_MTI: u8 = 0x01;
const IMR_STI: u8 = 0x02;
const IMR_KEY: u8 = 0x04;
const IMR_ONK: u8 = 0x08;
const IMR_TX: u8 = 0x10;
const IMR_RX: u8 = 0x20;
const IMR_EX: u8 = 0x40;
const ISR_MTI: u8 = 0x01;
const ISR_STI: u8 = 0x02;
const ISR_KEYI: u8 = 0x04;
const ISR_ONKI: u8 = 0x08;
const ISR_TXI: u8 = 0x10;
const ISR_RXI: u8 = 0x20;
const ISR_EXI: u8 = 0x40;
const ISR_KNOWN_MASK: u8 = ISR_MTI | ISR_STI | ISR_KEYI | ISR_ONKI | ISR_TXI | ISR_RXI | ISR_EXI;
const USR_RX_READY: u8 = 0x20;
#[cfg(test)]
const SSR_CI: u8 = 0x02;
const SSR_ONK: u8 = 0x08;
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;

fn irq_source_priority(name: &str) -> u8 {
    match name {
        "RX" => 0,
        "EX" => 1,
        "TX" => 2,
        "ONK" => 3,
        "KEY" => 4,
        "STI" => 5,
        "MTI" => 6,
        _ => u8::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;
    use crate::memory::IMEM_LCC_OFFSET;
    use crate::perfetto::perfetto_test_guard;
    use std::fs;

    fn install_noncanonical_irq_vector(rt: &mut CoreRuntime) {
        rt.memory.write_external_byte(INTERRUPT_VECTOR_ADDR, 0x78);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 1, 0x56);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 2, 0xF4);
    }

    #[test]
    fn binary_snapshot_registers_are_exactly_the_eight_core_fields() {
        let mut source = LlamaState::new();
        source.set_pc(0x12345);
        source.set_reg(RegName::Temp(15), 0x12_3456);

        let registers = collect_registers(&source);
        assert_eq!(registers.len(), snapshot::SNAPSHOT_REGISTER_LAYOUT.len());
        assert_eq!(registers.get("PC"), Some(&0x12345));
        assert!(!registers.contains_key("TEMP15"));
        assert!(snapshot::SNAPSHOT_REGISTER_LAYOUT
            .iter()
            .all(|(name, _)| registers.contains_key(*name)));
    }

    #[test]
    fn register_snapshot_rejects_unverified_f_atomically() {
        let mut state = LlamaState::new();
        state.set_pc(0x12345);
        state.set_reg(RegName::BA, 0x5678);
        state.set_reg(RegName::F, 0x01);
        let registers = HashMap::from([
            ("PC".to_string(), 0xABCDE),
            ("BA".to_string(), 0x9ABC),
            ("F".to_string(), 0xA4),
        ]);

        let error = apply_registers(&mut state, &registers).unwrap_err();

        assert!(error.to_string().contains("bits 2-7"));
        assert_eq!(state.pc(), 0x12345);
        assert_eq!(state.get_reg(RegName::BA), 0x5678);
        assert_eq!(state.get_reg(RegName::F), 0x01);
    }

    #[test]
    fn iq7000_runtime_model_can_be_configured() {
        let mut rt = CoreRuntime::new();
        let rom = vec![0u8; crate::iq7000::ROM_WINDOW_LEN];

        DeviceModel::Iq7000
            .configure_runtime(&mut rt, &rom)
            .expect("iq-7000 runtime should configure");

        assert_eq!(rt.device_model(), DeviceModel::Iq7000);
        assert_eq!(
            rt.lcd.as_ref().expect("lcd present").kind(),
            LcdKind::Iq7000Vram
        );
        assert!(
            rt.keyboard.is_some(),
            "iq-7000 keeps keyboard bridge available for matrix scanning"
        );
    }

    #[test]
    fn iq7000_clock_seed_populates_and_clears_workspace() {
        let mut rt = CoreRuntime::new();
        rt.set_iq7000_clock_seed_yyyymmddhhmm("202604252119")
            .expect("clock seed");

        let mut bytes = Vec::new();
        for idx in 0..12 {
            bytes.push(
                rt.memory
                    .load(crate::iq7000::CLOCK_WORKSPACE_START + idx, 8)
                    .unwrap_or(0) as u8,
            );
        }
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "202604252119");
        assert_eq!(
            rt.memory
                .load(crate::iq7000::CLOCK_INITIALIZED_FLAG, 8)
                .unwrap_or(0),
            1
        );

        rt.clear_iq7000_clock_seed();
        assert_eq!(
            rt.memory
                .load(crate::iq7000::CLOCK_WORKSPACE_START, 8)
                .unwrap_or(1),
            0
        );
        assert_eq!(
            rt.memory
                .load(crate::iq7000::CLOCK_INITIALIZED_FLAG, 8)
                .unwrap_or(1),
            0
        );
    }

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
    fn snapshot_roundtrip_keeps_timer_state_without_inert_runtime_modes() {
        let tmp = std::env::temp_dir().join("core_snapshot_timer.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        // Simulate loading historical v4 metadata that claimed the old hint
        // was enabled. A new CoreRuntime save must normalize the inert field.
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
            true,          // in_interrupt
            Some(vec![3]), // interrupt_stack (flow IDs)
            5,             // next_interrupt_id
            None,          // irq_bit_watch
        );
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0xAA);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x55);
        rt.timer.delivered_masks = vec![ISR_MTI];
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut rt2 = CoreRuntime::new();
        rt2.load_snapshot(&tmp).expect("load snapshot");
        assert!(!rt2.metadata.fast_mode);
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
        assert_eq!(rt2.timer.interrupt_stack, vec![3]);
        assert_eq!(rt2.timer.delivered_masks, vec![ISR_MTI]);
        assert_eq!(rt2.timer.next_interrupt_id, 5);
        assert_eq!(rt2.timer.last_fired, None);
    }

    #[test]
    fn snapshot_uses_live_memory_ranges_and_rejects_a_different_machine() {
        let tmp = std::env::temp_dir().join("core_snapshot_live_ranges.pcsnap");
        let _ = fs::remove_file(&tmp);

        let fallback = vec![(0x3000, 0x300F), (0x2000, 0x200F)];
        let readonly = vec![(0xD0000, 0xFFFFF), (0xC0000, 0xCFFFF)];
        let canonical_fallback =
            snapshot::canonical_snapshot_ranges("test fallback", &fallback).unwrap();
        let canonical_readonly =
            snapshot::canonical_snapshot_ranges("test readonly", &readonly).unwrap();
        let mut rt = CoreRuntime::new();
        rt.memory.set_python_ranges(fallback.clone());
        rt.memory.set_readonly_ranges(readonly.clone());
        rt.metadata.fallback_ranges = vec![(0x1111, 0x1111)];
        rt.metadata.readonly_ranges = vec![(0x2222, 0x2222)];
        rt.save_snapshot(&tmp).expect("save snapshot");

        let loaded = snapshot::load_snapshot(&tmp).expect("load raw snapshot");
        assert_eq!(loaded.metadata.fallback_ranges, canonical_fallback);
        assert_eq!(loaded.metadata.readonly_ranges, canonical_readonly);

        let mut equivalent = CoreRuntime::new();
        equivalent.memory.set_python_ranges(fallback);
        equivalent.memory.set_readonly_ranges(readonly);
        equivalent
            .load_snapshot(&tmp)
            .expect("unsorted but equivalent active ranges must match");
        assert_eq!(
            equivalent.memory.python_ranges(),
            canonical_fallback.as_slice()
        );
        assert_eq!(
            equivalent.memory.readonly_ranges(),
            canonical_readonly.as_slice()
        );

        let mut incompatible = CoreRuntime::new();
        let error = incompatible
            .load_snapshot(&tmp)
            .expect_err("different active range configuration must be rejected");
        assert!(error.to_string().contains("fallback ranges"));
        assert!(incompatible.memory.python_ranges().is_empty());
        assert!(incompatible.memory.readonly_ranges().is_empty());

        let _ = fs::remove_file(tmp);
    }

    #[test]
    fn snapshot_v4_card_state_overrides_the_active_machine() {
        use crate::memory::{MemoryCardMode, MemoryCardSnapshot};

        let base = std::env::temp_dir();
        let present_path = base.join("core_snapshot_v4_card_present.pcsnap");
        let absent_path = base.join("core_snapshot_v4_card_absent.pcsnap");
        let none_path = base.join("core_snapshot_v4_card_none.pcsnap");
        for path in [&present_path, &absent_path, &none_path] {
            let _ = fs::remove_file(path);
        }

        let present_payload = vec![0xA5; 8192];
        let mut present_source = CoreRuntime::new();
        present_source
            .memory
            .load_memory_card_with_writable(&present_payload, false)
            .unwrap();
        present_source.save_snapshot(&present_path).unwrap();

        let mut present_target = CoreRuntime::new();
        present_target
            .memory
            .load_memory_card_with_writable(&vec![0x3C; 65536], true)
            .unwrap();
        present_target.load_snapshot(&present_path).unwrap();
        assert_eq!(
            present_target.memory.memory_card_snapshot().unwrap(),
            Some(MemoryCardSnapshot {
                mode: MemoryCardMode::Present,
                capacity: 8192,
                writable: false,
                payload: present_payload,
            })
        );

        let absent_payload = vec![0x5A; 16384];
        let mut absent_source = CoreRuntime::new();
        absent_source
            .memory
            .load_memory_card_with_writable(&absent_payload, true)
            .unwrap();
        absent_source.memory.set_memory_card_slot_present(false);
        absent_source.save_snapshot(&absent_path).unwrap();

        let mut absent_target = CoreRuntime::new();
        absent_target
            .memory
            .load_memory_card_with_writable(&vec![0xC3; 32768], false)
            .unwrap();
        absent_target.load_snapshot(&absent_path).unwrap();
        assert_eq!(
            absent_target.memory.memory_card_snapshot().unwrap(),
            Some(MemoryCardSnapshot {
                mode: MemoryCardMode::Absent,
                capacity: 16384,
                writable: true,
                payload: absent_payload,
            })
        );

        CoreRuntime::new().save_snapshot(&none_path).unwrap();
        let mut none_target = CoreRuntime::new();
        none_target
            .memory
            .load_memory_card_with_writable(&vec![0x7E; 8192], true)
            .unwrap();
        none_target.load_snapshot(&none_path).unwrap();
        assert_eq!(none_target.memory.memory_card_snapshot().unwrap(), None);

        for path in [present_path, absent_path, none_path] {
            let _ = fs::remove_file(path);
        }
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
    fn snapshot_roundtrip_preserves_external_interrupt_level() {
        let tmp = std::env::temp_dir().join("core_snapshot_external_level.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        rt.set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        rt.set_external_interrupt_level(true);
        rt.save_snapshot(&tmp).expect("save snapshot");

        let mut restored = CoreRuntime::new();
        restored
            .set_device_model(DeviceModel::Iq7000)
            .expect("set matching IQ-7000 model");
        restored.load_snapshot(&tmp).expect("load snapshot");
        assert_eq!(restored.device_model(), DeviceModel::Iq7000);
        assert!(restored.external_interrupt_level());
        assert_ne!(
            restored
                .memory
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & ISR_EXI,
            0,
            "restored asserted input must re-latch ISR.EXI"
        );

        restored.set_external_interrupt_level(false);
        assert!(!restored.external_interrupt_level());
        assert_eq!(
            restored
                .memory
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & ISR_EXI,
            0
        );
        let _ = fs::remove_file(tmp);
    }

    #[test]
    fn snapshot_roundtrip_preserves_held_on_level_outside_ssr_storage() {
        let tmp = std::env::temp_dir().join("core_snapshot_held_on_level.pcsnap");
        let _ = fs::remove_file(&tmp);

        let mut rt = CoreRuntime::new();
        // AND (ISR),F7; NOP mirrors the clear/wait sequence at F5247.
        rt.load_rom(&[0x30, 0x71, 0xFC, 0xF7, 0x00], 0);
        rt.state.set_reg(RegName::PC, 0);
        rt.timer.in_interrupt = true;
        rt.memory
            .write_internal_byte(crate::memory::IMEM_SSR_OFFSET, SSR_CI);
        rt.press_on_key();
        assert_eq!(
            rt.memory
                .read_internal_byte_silent(crate::memory::IMEM_SSR_OFFSET)
                .unwrap_or(0),
            SSR_CI,
            "the physical ON-key level must not be stored in the SSR image"
        );
        rt.save_snapshot(&tmp).expect("save held-ON snapshot");

        let mut restored = CoreRuntime::new();
        restored.load_snapshot(&tmp).expect("load held-ON snapshot");
        assert!(restored.onk_level, "held ON-key level must round-trip");
        assert_eq!(
            restored
                .memory
                .read_internal_byte_silent(crate::memory::IMEM_SSR_OFFSET)
                .unwrap_or(0),
            SSR_CI,
            "restoring the physical level must leave raw SSR storage unchanged"
        );

        restored.step(1).expect("execute ISR.ONKI clear");
        assert_eq!(
            restored
                .memory
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & ISR_ONKI,
            0,
            "firmware clear must take effect for the current instruction"
        );
        restored.step(1).expect("execute wait NOP");
        assert_ne!(
            restored
                .memory
                .read_internal_byte(IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & ISR_ONKI,
            0,
            "restored held level must re-latch ISR.ONKI"
        );

        let _ = fs::remove_file(tmp);
    }

    #[test]
    fn snapshot_rejects_unrepresented_runtime_extensions() {
        fn assert_rejected(rt: &CoreRuntime, path: &std::path::Path, label: &str) {
            let error = rt
                .save_snapshot(path)
                .expect_err("unrepresented runtime state must fail closed");
            assert!(
                error.to_string().contains(label),
                "unexpected error for {label}: {error}"
            );
            assert!(!path.exists(), "rejected save must not create a checkpoint");
        }

        let base = std::env::temp_dir();

        let mut sio = CoreRuntime::new();
        sio.enable_sio_stub();
        let path = base.join("core_snapshot_reject_sio.pcsnap");
        let _ = fs::remove_file(&path);
        assert_rejected(&sio, &path, "SIO");

        let mut peripherals = CoreRuntime::new();
        peripherals.enable_pce500_peripheral_bridge(0x10000);
        let path = base.join("core_snapshot_reject_peripherals.pcsnap");
        let _ = fs::remove_file(&path);
        assert_rejected(&peripherals, &path, "PCE peripheral");

        let mut callbacks = CoreRuntime::new();
        callbacks.set_host_read(|_| None);
        let path = base.join("core_snapshot_reject_callbacks.pcsnap");
        let _ = fs::remove_file(&path);
        assert_rejected(&callbacks, &path, "host overlay callbacks");

        let mut rtc = CoreRuntime::new();
        rtc.set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        rtc.set_iq7000_clock_seed_yyyymmddhhmm("202604252119")
            .expect("install RTC seed");
        let path = base.join("core_snapshot_reject_rtc.pcsnap");
        let _ = fs::remove_file(&path);
        assert_rejected(&rtc, &path, "RTC protocol");

        let mut overlay = CoreRuntime::new();
        overlay.add_ram_overlay(0x8000, 16, "snapshot-test");
        let path = base.join("core_snapshot_reject_overlay.pcsnap");
        let _ = fs::remove_file(&path);
        assert_rejected(&overlay, &path, "attestation");
    }

    #[test]
    fn snapshot_load_rejects_active_extensions_and_model_mismatch() {
        let pc_path = std::env::temp_dir().join("core_snapshot_model_pc.pcsnap");
        let iq_path = std::env::temp_dir().join("core_snapshot_model_iq.pcsnap");
        let _ = fs::remove_file(&pc_path);
        let _ = fs::remove_file(&iq_path);

        CoreRuntime::new()
            .save_snapshot(&pc_path)
            .expect("save clean PC snapshot");
        let mut active = CoreRuntime::new();
        active.enable_sio_stub();
        let error = active
            .load_snapshot(&pc_path)
            .expect_err("active unrepresented state must reject load");
        assert!(error.to_string().contains("SIO"));

        let mut iq = CoreRuntime::new();
        iq.set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        iq.save_snapshot(&iq_path).expect("save IQ snapshot");
        let mut pc = CoreRuntime::new();
        let error = pc
            .load_snapshot(&iq_path)
            .expect_err("cross-model load must fail closed");
        assert!(error.to_string().contains("does not match active machine"));
        assert_eq!(pc.device_model(), DeviceModel::PcE500);

        let _ = fs::remove_file(pc_path);
        let _ = fs::remove_file(iq_path);
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
        assert_ne!(
            cleared & ISR_ONKI,
            0,
            "ONKI must remain latched until firmware acknowledges it"
        );
    }

    #[test]
    fn held_external_level_relatched_after_firmware_clear() {
        let mut rt = CoreRuntime::new();
        rt.set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        // AND (ISR),BF; NOP mirrors the clear/wait sequence at F527F.
        rt.load_rom(&[0x30, 0x71, 0xFC, 0xBF, 0x00], 0);
        rt.state.set_reg(RegName::PC, 0);
        rt.timer.in_interrupt = true;
        rt.set_external_interrupt_level(true);

        rt.step(1).expect("execute ISR.EXI clear");
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_EXI,
            0,
            "firmware clear must take effect for the current instruction"
        );
        rt.step(1).expect("execute wait NOP");
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_EXI,
            0,
            "held external level must re-latch ISR.EXI before the retest"
        );

        rt.set_external_interrupt_level(false);
        assert!(!rt.external_interrupt_level());
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_EXI,
            0
        );
    }

    #[test]
    fn held_on_level_relatched_after_firmware_clear_without_changing_ci() {
        let mut rt = CoreRuntime::new();
        rt.set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        // AND (ISR),F7; NOP mirrors the clear/wait sequence at F5247.
        rt.load_rom(&[0x30, 0x71, 0xFC, 0xF7, 0x00], 0);
        rt.state.set_reg(RegName::PC, 0);
        rt.timer.in_interrupt = true;
        rt.memory
            .write_internal_byte(crate::memory::IMEM_SSR_OFFSET, SSR_CI);
        rt.press_on_key();

        rt.step(1).expect("execute ISR.ONKI clear");
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_ONKI,
            0,
            "firmware clear must take effect for the current instruction"
        );
        rt.step(1).expect("execute wait NOP");
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_ONKI,
            0,
            "held ON level must re-latch ISR.ONKI before the retest"
        );
        assert_eq!(
            rt.memory
                .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
                .unwrap_or(0),
            SSR_CI,
            "ON-key level handling must not synthesize or clear SSR.CI"
        );

        rt.release_on_key();
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_ONKI,
            0,
            "release must not acknowledge the re-latched ONKI"
        );
        assert_eq!(
            rt.memory
                .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
                .unwrap_or(0),
            SSR_CI
        );
    }

    #[test]
    fn onk_press_sets_ssr_bit_on_read() {
        let mut rt = CoreRuntime::new();
        // CI is an independent external input and must survive ON-key changes.
        rt.memory
            .write_internal_byte(crate::memory::IMEM_SSR_OFFSET, SSR_CI);
        let ssr_before = rt
            .memory
            .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(
            ssr_before & (SSR_CI | SSR_ONK),
            SSR_CI,
            "SSR.CI should be set independently while SSR.ONK starts clear"
        );

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
                                val |= SSR_ONK;
                            }
                            return val as u32;
                        }
                    }
                    self.mem.load(addr, bits).unwrap_or(0)
                }

                fn store(&mut self, addr: u32, bits: u8, value: u32) {
                    let _ = self.mem.store(addr, bits, value);
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
        assert_eq!(
            ssr_after & SSR_CI,
            SSR_CI,
            "SSR.CI should retain its independently supplied level"
        );
        assert_eq!(
            ssr_after & SSR_ONK,
            SSR_ONK,
            "SSR.ONK should reflect the pressed ON-key level"
        );

        rt.release_on_key();
        let ssr_clear = rt
            .memory
            .read_internal_byte(crate::memory::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(
            ssr_clear & (SSR_CI | SSR_ONK),
            SSR_CI,
            "releasing ON must not clear the independent SSR.CI input"
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

            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                let _ = self.mem.store(addr, bits, value);
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
    fn sio_receive_byte_asserts_serial_rx_irq() {
        let mut rt = CoreRuntime::new();
        rt.queue_sio_receive_byte(0xA5);

        assert_eq!(
            rt.sio
                .as_ref()
                .expect("SIO stub should be enabled")
                .pending_receive()
                .iter()
                .map(|entry| entry.value)
                .collect::<Vec<_>>(),
            vec![0xA5]
        );
        assert_eq!(rt.memory.read_internal_byte(IMEM_RXD_OFFSET), Some(0xA5));
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0) & 0x20,
            0,
            "USR.RX_READY should be visible after host byte injection"
        );
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_RXI,
            0,
            "serial RX ISR bit should be asserted"
        );
        assert!(rt.timer.irq_pending);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("RX"));
    }

    #[test]
    fn serial_rx_irq_delivers_when_imr_enables_rx() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x0010);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_byte(0x0FFFFA, 0x78);
        rt.memory.write_external_byte(0x0FFFFB, 0x56);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_RX);

        rt.queue_sio_receive_byte(0xA5);
        rt.deliver_pending_irq().expect("deliver serial RX IRQ");

        assert_eq!(rt.state.get_reg(RegName::PC) & ADDRESS_MASK, 0x005678);
        assert!(rt.timer.in_interrupt);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("RX"));
        assert_eq!(rt.timer.delivered_masks, vec![ISR_RXI]);
    }

    #[test]
    fn async_irq_rejects_noncanonical_vector_before_frame_or_metadata_mutation() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x012345);
        rt.state.set_reg(RegName::S, 0x000240);
        rt.state.set_reg(RegName::F, 0x03);
        rt.state
            .set_reg(RegName::IMR, u32::from(IMR_MASTER | IMR_ONK));
        rt.memory.write_external_byte(0x0FFFFA, 0x78);
        rt.memory.write_external_byte(0x0FFFFB, 0x56);
        rt.memory.write_external_byte(0x0FFFFC, 0xF4);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_ONK);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("ONK".to_string());
        rt.timer.last_fired = Some("ONK".to_string());

        let external_before = rt.memory.external_slice().to_vec();
        let internal_before = rt.memory.internal_slice().to_vec();
        let writes_before = rt.memory.memory_write_count();
        let err = rt
            .deliver_pending_irq()
            .expect_err("noncanonical IRQ vector must be quarantined");

        assert!(err
            .to_string()
            .contains(crate::llama::eval::VECTOR_UPPER_NIBBLE_ERROR));
        assert_eq!(rt.state.get_reg(RegName::PC), 0x012345);
        assert_eq!(rt.state.get_reg(RegName::S), 0x000240);
        assert_eq!(rt.state.get_reg(RegName::F), 0x03);
        assert_eq!(
            rt.state.get_reg(RegName::IMR),
            u32::from(IMR_MASTER | IMR_ONK)
        );
        assert_eq!(rt.state.call_depth(), 0);
        assert_eq!(rt.memory.external_slice(), external_before);
        assert_eq!(rt.memory.internal_slice(), internal_before);
        assert_eq!(rt.memory.memory_write_count(), writes_before);
        assert!(rt.timer.irq_pending);
        assert!(!rt.timer.in_interrupt);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("ONK"));
        assert_eq!(rt.timer.last_fired.as_deref(), Some("ONK"));
        assert!(rt.timer.delivered_masks.is_empty());
        assert_eq!(rt.timer.irq_total, 0);
        assert!(rt.poisoned.is_none());
    }

    #[test]
    fn post_frame_irq_vector_failure_poisons_until_power_on_reset() {
        let mut rt = CoreRuntime::new();
        let source_pc = 0x012345;
        rt.state.set_reg(RegName::PC, source_pc);
        // The three PC frame writes land on the IRQ vector itself. The
        // architectural post-frame fetch therefore observes source_pc rather
        // than the silently validated handler target.
        rt.state.set_reg(RegName::S, 0x0FFFFD);
        rt.state.set_reg(RegName::F, 0x03);
        rt.memory.write_external_byte(source_pc, 0x00); // valid NOP target
        rt.memory.write_external_byte(INTERRUPT_VECTOR_ADDR, 0x00);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 1, 0x02);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 2, 0x00);
        rt.memory.write_external_byte(0x00200, 0x00);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        let error = rt
            .step(1)
            .expect_err("post-frame vector mismatch must fail closed");

        assert!(error
            .to_string()
            .contains(crate::llama::eval::VECTOR_CHANGED_DURING_PREFLIGHT_ERROR));
        assert_eq!(rt.state.get_reg(RegName::PC), source_pc);
        assert_eq!(rt.state.get_reg(RegName::S), 0x0FFFF8);
        assert_eq!(rt.memory.read_internal_byte(IMEM_IMR_OFFSET), Some(IMR_KEY));
        assert_eq!(
            [
                rt.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap() as u8,
                rt.memory.load(INTERRUPT_VECTOR_ADDR + 1, 8).unwrap() as u8,
                rt.memory.load(INTERRUPT_VECTOR_ADDR + 2, 8).unwrap() as u8,
            ],
            [0x45, 0x23, 0x01]
        );
        assert!(rt.poisoned.is_some());

        let registers_after_failure = collect_registers(&rt.state);
        let writes_after_failure = rt.memory.memory_write_count();
        let retry = rt.step(1).expect_err("poison must block another frame");
        assert!(retry.to_string().contains("poisoned"));
        assert_eq!(collect_registers(&rt.state), registers_after_failure);
        assert_eq!(rt.memory.memory_write_count(), writes_after_failure);

        // External reset is the recovery boundary. The untouched reset vector
        // resolves to zero in this fixture, where a NOP is available.
        rt.power_on_reset().expect("power-on reset clears poison");
        assert!(rt.poisoned.is_none());
        rt.step(1).expect("execution resumes after reset");
    }

    #[test]
    fn core_power_on_reset_rejects_bad_vector_without_mutation() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x012345);
        rt.state.set_reg(RegName::S, 0x000240);
        rt.state.set_reg(RegName::F, 0x03);
        rt.state.halt();
        rt.memory.write_external_byte(0x0FFFFD, 0x78);
        rt.memory.write_external_byte(0x0FFFFE, 0x56);
        rt.memory.write_external_byte(0x0FFFFF, 0xF4);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0xA5);

        let external_before = rt.memory.external_slice().to_vec();
        let internal_before = rt.memory.internal_slice().to_vec();
        let writes_before = rt.memory.memory_write_count();
        let err = rt
            .power_on_reset()
            .expect_err("noncanonical reset vector must be quarantined");

        assert!(err
            .to_string()
            .contains(crate::llama::eval::VECTOR_UPPER_NIBBLE_ERROR));
        assert_eq!(rt.state.get_reg(RegName::PC), 0x012345);
        assert_eq!(rt.state.get_reg(RegName::S), 0x000240);
        assert_eq!(rt.state.get_reg(RegName::F), 0x03);
        assert!(rt.state.is_halted());
        assert_eq!(rt.memory.external_slice(), external_before);
        assert_eq!(rt.memory.internal_slice(), internal_before);
        assert_eq!(rt.memory.memory_write_count(), writes_before);
    }

    #[test]
    fn irq_frame_wraps_each_byte_on_the_20_bit_external_bus() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x034567);
        rt.state.set_reg(RegName::S, 0x000002);
        rt.state.set_reg(RegName::F, 0x03);
        rt.memory.write_external_byte(0x0FFFFA, 0x78);
        rt.memory.write_external_byte(0x0FFFFB, 0x56);
        rt.memory.write_external_byte(0x0FFFFC, 0x02);
        let imr = IMR_MASTER | IMR_ONK;
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, imr);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("ONK".to_string());

        rt.deliver_pending_irq().expect("deliver wrapped IRQ frame");

        assert_eq!(rt.state.get_reg(RegName::S), 0x0FFFFD);
        assert_eq!(rt.state.get_reg(RegName::PC), 0x025678);
        for (address, expected) in [
            (0x0FFFFF, 0x67),
            (0x000000, 0x45),
            (0x000001, 0x03),
            (0x0FFFFE, 0x03),
            (0x0FFFFD, imr),
        ] {
            assert_eq!(
                rt.memory.load(address, 8),
                Some(u32::from(expected)),
                "wrapped frame byte at 0x{address:05X}"
            );
        }
    }

    #[test]
    fn sio_transmit_complete_asserts_serial_tx_irq() {
        let mut rt = CoreRuntime::new();
        rt.enable_sio_stub();
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_TX);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);

        rt.assert_sio_transmit_ready();

        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_TXI,
            0,
            "completed TXD transmission should assert serial TXI"
        );
        assert!(rt.timer.irq_pending);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("TX"));
    }

    #[test]
    fn retired_instruction_timing_drives_sio_completion() {
        let mut rt = CoreRuntime::new();
        rt.enable_sio_stub();
        let sio = rt.sio.as_mut().expect("SIO stub");
        sio.disable_auto_response();
        sio.set_timing_config(SioTimingConfig {
            tx_complete_cycles: 4,
            ..SioTimingConfig::default()
        });
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);
        // MV (TXD),55h takes three documented timing units, followed by NOP.
        rt.memory
            .write_external_slice(0, &[0xCC, IMEM_TXD_OFFSET as u8, 0x55, 0x00]);
        rt.state.set_pc(0);

        rt.step(1).expect("write TXD");
        assert_eq!(rt.sio.as_ref().unwrap().pending_transmit(), vec![0x55]);
        assert_eq!(rt.sio.as_ref().unwrap().completed_transmit_len(), 0);
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_TXI,
            0
        );

        rt.step(1).expect("advance SIO by NOP timing");
        assert!(rt.sio.as_ref().unwrap().pending_transmit().is_empty());
        assert_eq!(rt.sio.as_ref().unwrap().completed_transmit_len(), 1);
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_TXI,
            0,
            "TX completion must latch independently of IMR"
        );
        assert_eq!(
            rt.sio.as_mut().unwrap().complete_transmit(&mut rt.memory),
            Some(0x55)
        );
    }

    #[test]
    fn sio_transmit_complete_reasserts_inside_irq_without_nesting() {
        let mut rt = CoreRuntime::new();
        rt.enable_sio_stub();
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_TX);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);
        rt.timer.in_interrupt = true;
        rt.timer.irq_pending = false;

        rt.assert_sio_transmit_ready();

        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_TXI,
            0,
            "TXI should be visible to the ROM TX handler after it clears ISR.0x10"
        );
        assert!(
            !rt.timer.irq_pending,
            "refreshing TXI while already in an interrupt must not request nested delivery"
        );
    }

    #[test]
    fn interrupt_layout_and_priority_match_rom_dispatcher() {
        assert_eq!((ISR_TXI, ISR_RXI, ISR_EXI), (0x10, 0x20, 0x40));
        assert_eq!((IMR_TX, IMR_RX, IMR_EX), (0x10, 0x20, 0x40));

        let mut rt = CoreRuntime::new();
        rt.memory.write_internal_byte(
            IMEM_ISR_OFFSET,
            ISR_RXI | ISR_EXI | ISR_TXI | ISR_STI | ISR_MTI,
        );
        rt.arm_pending_irq_from_isr();
        assert_eq!(rt.timer.irq_source.as_deref(), Some("RX"));

        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_EXI | ISR_TXI | ISR_STI | ISR_MTI);
        rt.arm_pending_irq_from_isr();
        assert_eq!(rt.timer.irq_source.as_deref(), Some("EX"));

        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_STI | ISR_MTI);
        rt.arm_pending_irq_from_isr();
        assert_eq!(rt.timer.irq_source.as_deref(), Some("STI"));
    }

    #[test]
    fn arm_pending_irq_honors_asserted_keyi_when_host_event_generation_is_disabled() {
        let mut rt = CoreRuntime::new();
        rt.timer.set_keyboard_irq_enabled(false);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.timer.irq_pending = false;
        rt.timer.irq_source = None;

        rt.arm_pending_irq_from_isr();

        assert!(rt.timer.irq_pending);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));
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
    fn arm_pending_irq_records_status_during_interrupt_without_selecting_nested_source() {
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
            rt.timer.irq_pending,
            "status remains pending while already in an interrupt"
        );
        assert!(
            rt.timer.irq_source.is_none(),
            "irq_source should remain unset while in interrupt"
        );
    }

    #[test]
    fn raw_selected_key_reasserts_without_host_event_latch() {
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
        rt.refresh_raw_key_irq_level();
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & ISR_KEYI, ISR_KEYI);
        assert!(
            !rt.timer.key_irq_latched,
            "latch should remain cleared without new events"
        );
        assert!(rt.timer.irq_pending);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("KEY"));
    }

    #[test]
    fn unselected_host_key_event_does_not_assert_raw_keyi() {
        let mut rt = CoreRuntime::new();
        rt.timer.set_keyboard_irq_enabled(false);
        let kb = rt.keyboard.as_mut().unwrap();
        // Inject a matrix event to populate FIFO without relying on IRQ enable.
        kb.inject_matrix_event(0x10, false, &mut rt.memory, rt.timer.kb_irq_enabled);
        rt.refresh_raw_key_irq_level();
        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "an unselected key and host FIFO event are not raw KEYI sources"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should not arm when kb IRQs are disabled"
        );
    }

    #[test]
    fn translated_input_does_not_change_physical_matrix_or_assert_keyi() {
        let mut rt = CoreRuntime::for_model(DeviceModel::Iq7000, &[]).unwrap();
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.memory.write_internal_byte(IMEM_KOL_OFFSET, 0xFF);
        rt.memory.write_internal_byte(IMEM_KOH_OFFSET, 0xFF);

        assert_eq!(rt.queue_translated_key_event(0xA3), 1);
        assert_eq!(rt.keyboard.as_ref().unwrap().compute_physical_kil(), 0);
        assert_eq!(rt.keyboard.as_ref().unwrap().fifo_snapshot(), vec![0xA3]);
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            0
        );
        assert!(!rt.timer.irq_pending);
        assert!(rt.timer.key_irq_latched, "host FIFO state remains visible");
    }

    #[test]
    fn physical_input_reaches_keyi_only_when_selected() {
        let mut rt = CoreRuntime::for_model(DeviceModel::PcE500, &[]).unwrap();
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        assert!(rt.set_physical_matrix_key(0x10, true));
        assert!(rt.keyboard.as_ref().unwrap().fifo_snapshot().is_empty());

        rt.refresh_raw_key_irq_level();
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            0,
            "an unselected physical contact is not a KEYI source"
        );

        rt.keyboard
            .as_mut()
            .unwrap()
            .handle_write(IMEM_KOL_OFFSET, 0x04, &mut rt.memory);
        rt.refresh_raw_key_irq_level();
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            0
        );
    }

    #[test]
    fn raw_selected_key_ignores_host_event_irq_disable() {
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

        rt.refresh_raw_key_irq_level();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & ISR_KEYI, ISR_KEYI);
        assert!(rt.timer.irq_pending);
        assert!(
            !rt.timer.key_irq_latched,
            "latch should clear while kb IRQs are disabled"
        );
    }

    #[test]
    fn selected_raw_key_ignores_host_event_disable_and_preserves_bookkeeping() {
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
        rt.refresh_raw_key_irq_level();
        assert!(
            rt.timer.key_irq_latched,
            "latch should be set while enabled"
        );
        // Disable IRQs and clear ISR, then ensure refresh keeps the latch active.
        rt.timer.set_keyboard_irq_enabled(false);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;
        rt.refresh_raw_key_irq_level();

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
    fn host_event_latch_alone_does_not_assert_keyi_during_interrupt() {
        let mut rt = CoreRuntime::new();
        rt.timer.in_interrupt = true;
        rt.timer.key_irq_latched = true;
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;

        rt.refresh_raw_key_irq_level();

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_KEYI,
            0,
            "host event bookkeeping alone must not assert raw KEYI"
        );
        assert!(
            !rt.timer.irq_pending,
            "pending IRQ should remain clear while in interrupt"
        );
    }

    #[test]
    fn raw_selected_key_reasserts_during_handler_until_release() {
        let mut rt = CoreRuntime::new();
        rt.timer.in_interrupt = true;
        {
            let kb = rt.keyboard.as_mut().unwrap();
            kb.press_matrix_code(0x10, &mut rt.memory);
            kb.handle_write(IMEM_KOL_OFFSET, 0xFF, &mut rt.memory);
            kb.handle_write(IMEM_KOH_OFFSET, 0x0F, &mut rt.memory);
        }
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;

        rt.refresh_raw_key_irq_level();

        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            ISR_KEYI
        );
        assert!(rt.timer.irq_pending);
        assert!(
            rt.timer.irq_source.is_none(),
            "no nested source is selected"
        );

        rt.keyboard
            .as_mut()
            .unwrap()
            .release_matrix_code(0x10, &mut rt.memory);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.timer.irq_pending = false;
        rt.refresh_raw_key_irq_level();
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            0
        );
        assert!(!rt.timer.irq_pending);
    }

    #[test]
    fn selected_raw_key_reasserts_independently_of_host_latch() {
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

        rt.refresh_raw_key_irq_level();

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
    fn halt_wake_does_not_preflight_irq_vector() {
        let mut rt = CoreRuntime::new();
        rt.state.set_halted(true);
        rt.state.set_pc(0);
        rt.memory.write_external_byte(0, 0x00);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());
        install_noncanonical_irq_vector(&mut rt);

        rt.step(1)
            .expect("HALT wake must not inspect a vector it cannot deliver on entry");

        assert!(!rt.state.is_halted());
        assert!(rt.timer.irq_pending);
    }

    #[test]
    fn ordinary_step_does_not_preflight_irq_vector() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.memory.write_external_byte(0, 0x00);
        install_noncanonical_irq_vector(&mut rt);

        rt.step(1)
            .expect("ordinary execution must not inspect an unselected IRQ vector");

        assert_eq!(rt.state.pc(), 1);
        assert_eq!(rt.metadata.instruction_count, 1);
    }

    #[test]
    fn masked_pending_irq_defers_vector_preflight_until_unmasked() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_slice(0, &[0x00, 0x00]);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, IMR_KEY);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());
        install_noncanonical_irq_vector(&mut rt);

        rt.step(1).expect("masked IRQ must not inspect its vector");
        assert_eq!(rt.state.pc(), 1);
        assert!(rt.timer.irq_pending);

        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        let pc_before = rt.state.pc();
        let sp_before = rt.state.get_reg(RegName::S);
        let cycle_before = rt.metadata.cycle_count;
        let instruction_before = rt.metadata.instruction_count;

        let error = rt
            .step(1)
            .expect_err("unmasked deliverable IRQ must validate its vector");

        assert!(error
            .to_string()
            .contains(crate::llama::eval::VECTOR_UPPER_NIBBLE_ERROR));
        assert_eq!(rt.state.pc(), pc_before);
        assert_eq!(rt.state.get_reg(RegName::S), sp_before);
        assert_eq!(rt.metadata.cycle_count, cycle_before);
        assert_eq!(rt.metadata.instruction_count, instruction_before);
        assert!(!rt.timer.in_interrupt);
        assert!(rt.timer.irq_pending);
    }

    #[test]
    fn active_interrupt_handler_does_not_preflight_irq_vector() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.memory.write_external_byte(0, 0x00);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_MTI);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        rt.timer.irq_pending = true;
        rt.timer.in_interrupt = true;
        rt.timer.irq_source = Some("MTI".to_string());
        install_noncanonical_irq_vector(&mut rt);

        rt.step(1)
            .expect("active handler must not inspect a nested IRQ vector");

        assert_eq!(rt.state.pc(), 1);
        assert!(rt.timer.in_interrupt);
        assert!(rt.timer.irq_pending);
    }

    #[test]
    fn timer_armed_irq_defers_vector_preflight_until_next_step() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_slice(0, &[0x00, 0x00]);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        *rt.timer = TimerContext::new(true, 1, 0);
        install_noncanonical_irq_vector(&mut rt);

        rt.step(1)
            .expect("timer armed during this step must not use an unproved vector");
        assert_eq!(rt.state.pc(), 1);
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_MTI,
            0
        );
        assert!(rt.timer.irq_pending);

        let pc_before = rt.state.pc();
        let cycle_before = rt.metadata.cycle_count;
        let instruction_before = rt.metadata.instruction_count;
        let error = rt
            .step(1)
            .expect_err("next boundary must validate the now-selected IRQ vector");

        assert!(error
            .to_string()
            .contains(crate::llama::eval::VECTOR_UPPER_NIBBLE_ERROR));
        assert_eq!(rt.state.pc(), pc_before);
        assert_eq!(rt.metadata.cycle_count, cycle_before);
        assert_eq!(rt.metadata.instruction_count, instruction_before);
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
    fn halt_freezes_main_timer_phase_while_sub_timer_keeps_running() {
        let mut rt = CoreRuntime::new();
        rt.timer.enabled = true;
        rt.timer.configure_scr_periods(1, 1, 1, 1, 0, 0);
        rt.timer.next_sti = 100;
        rt.state.set_halted(true);

        rt.step(3).expect("halt idle boundaries");

        assert_eq!(rt.cycle_count(), 3);
        assert_eq!(rt.timer.next_mti, 4, "HALT must retain MTI phase");
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_MTI,
            0,
            "the stopped system clock must not assert MTI"
        );

        rt.timer.next_sti = 4;
        rt.step(1).expect("subclock wake boundary");
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_STI,
            0,
            "the subclock must continue during HALT"
        );
        assert!(!rt.state.is_halted(), "STI must wake HALT");
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
        kb.handle_write(IMEM_KOL_OFFSET, 0xFF, &mut rt.memory);
        kb.handle_write(IMEM_KOH_OFFSET, 0x0F, &mut rt.memory);
        let events = kb.inject_matrix_event(0x56, false, &mut rt.memory, kb_irq_enabled);
        assert!(events > 0, "key injection should enqueue an event");

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & ISR_KEYI, 0, "FIFO injection alone is not raw KEYI");

        rt.step(1).expect("halt wake step");

        assert!(
            !rt.state.is_halted(),
            "HALT should clear on injected key event"
        );
        assert_ne!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_KEYI,
            0
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
        kb.handle_write(IMEM_KOL_OFFSET, 0xFF, &mut rt.memory);
        kb.handle_write(IMEM_KOH_OFFSET, 0x0F, &mut rt.memory);
        let events = kb.inject_matrix_event(0x56, false, &mut rt.memory, kb_irq_enabled);
        assert!(events > 0, "key injection should enqueue an event");

        rt.step(1).expect("halt wake step");

        assert!(!rt.state.is_halted(), "wake is an idle boundary");
        assert_eq!(rt.state.pc(), 0);
        assert_eq!(rt.instruction_count(), 0);

        rt.step(1).expect("execute HALT after wake");

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
        // Enable the sub timer so the HALT idle loop produces an ISR write.
        rt.timer.enabled = true;
        rt.timer.mti_period = 0;
        rt.timer.next_mti = 0;
        rt.timer.sti_period = 1;
        rt.timer.next_sti = 1;
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_STI);
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
    fn halt_wakes_on_asserted_keyi_even_when_host_generation_is_disabled() {
        let mut rt = CoreRuntime::new();
        rt.state.set_halted(true);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.timer.set_keyboard_irq_enabled(false);
        // Assert KEYI in ISR with IMR master off to mimic a host write.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);

        let _ = rt.step(1);

        // The host knob suppresses generation; it must not reinterpret an
        // already asserted hardware status bit.
        assert!(!rt.state.is_halted(), "asserted KEYI should wake HALT");
        assert!(
            rt.timer.irq_pending,
            "asserted KEYI wake should arm a pending IRQ"
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
    fn off_with_non_onk_isr_does_not_fetch_opcode() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut rt = CoreRuntime::new();
        let host_reads = Arc::new(AtomicUsize::new(0));
        let host_peeks = Arc::new(AtomicUsize::new(0));
        let read_count = Arc::clone(&host_reads);
        let peek_count = Arc::clone(&host_peeks);
        rt.set_host_read(move |_addr| {
            read_count.fetch_add(1, Ordering::Relaxed);
            Some(0x00)
        });
        rt.set_host_peek(move |_addr| {
            peek_count.fetch_add(1, Ordering::Relaxed);
            Some(0x00)
        });
        rt.memory.set_python_ranges(vec![(0x02000, 0x02000)]);
        rt.state.set_pc(0x02000);
        rt.state.power_off();
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);

        rt.step(1).expect("OFF ignores non-ONKI status");

        assert!(rt.state.is_off());
        assert_eq!(rt.state.pc(), 0x02000);
        assert_eq!(host_reads.load(Ordering::Relaxed), 0);
        assert_eq!(host_peeks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn off_preserves_ignored_isr_bits_before_and_during_onk_wake() {
        let mut rt = CoreRuntime::new();
        rt.state.power_off();
        rt.memory
            .write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI | ISR_MTI);

        rt.step(1).expect("ignored OFF status step");

        assert!(rt.state.is_off());
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET),
            Some(ISR_KEYI | ISR_MTI)
        );

        let combined = ISR_KEYI | ISR_MTI | ISR_ONKI;
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, combined);
        rt.step(1).expect("ONKI OFF wake step");

        assert!(!rt.state.is_off());
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET),
            Some(combined)
        );
        assert_eq!(rt.timer.irq_isr, combined);
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
    fn off_preserves_non_onk_status_and_bookkeeping() {
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
        assert_eq!(isr, ISR_KEYI | ISR_MTI);
        assert!(rt.timer.irq_pending, "OFF must not erase pending IRQ state");
        assert_eq!(rt.timer.irq_source.as_deref(), Some("MTI"));
        assert_eq!(rt.timer.last_fired.as_deref(), Some("MTI"));
        assert!(
            rt.timer.key_irq_latched,
            "OFF must not clear key latch state"
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
            0x001235,
            "the selected handler's first NOP executes on the delivery step"
        );
    }

    #[test]
    fn pending_irq_is_retained_without_nested_delivery() {
        let mut rt = CoreRuntime::new();
        rt.state.set_reg(RegName::PC, 0x012345);
        rt.state.set_reg(RegName::S, 0x000200);
        rt.timer.irq_pending = true;
        rt.timer.in_interrupt = true;
        rt.timer.irq_source = Some("MTI".to_string());
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_MTI);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_MTI);

        rt.deliver_pending_irq()
            .expect("nested delivery should be deferred");

        assert_eq!(rt.state.get_reg(RegName::PC), 0x012345);
        assert_eq!(rt.state.get_reg(RegName::S), 0x000200);
        assert!(rt.timer.in_interrupt);
        assert!(rt.timer.irq_pending);
        assert_eq!(rt.timer.irq_source.as_deref(), Some("MTI"));
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
            0x005679,
            "the ONK handler's first NOP executes on the delivery step"
        );
    }

    #[test]
    fn reti_clears_bookkeeping_but_preserves_isr_bit() {
        let mut rt = CoreRuntime::new();
        // Prepare stack and vector to a RETI instruction.
        rt.state.set_reg(RegName::S, 0x0200);
        rt.state.set_reg(RegName::PC, 0x0000);
        rt.memory.write_external_byte(0x0FFFFA, 0x10); // vector low
        rt.memory.write_external_byte(0x0FFFFB, 0x00);
        rt.memory.write_external_byte(0x0FFFFC, 0x00);
        rt.memory.write_external_byte(0x0010, 0x00); // first handler NOP
        rt.memory.write_external_byte(0x0011, 0x01); // RETI opcode
                                                     // Seed IMR/ISR and pending IRQ.
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());

        // First step: deliver IRQ and jump to vector.
        rt.step(1).expect("deliver irq");
        assert!(rt.timer.in_interrupt, "interrupt flag should set on entry");
        assert_eq!(rt.state.get_reg(RegName::PC) & ADDRESS_MASK, 0x0011);

        // Second step: RETI restores state and clears emulator bookkeeping only.
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
        assert_eq!(
            isr & ISR_KEYI,
            ISR_KEYI,
            "ROM-consistent model leaves ISR acknowledgement to firmware"
        );
    }

    #[test]
    fn reti_preserves_isr_when_interrupt_source_bookkeeping_is_missing() {
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
        // ISR has ONK set; irq_source is unknown but bookkeeping retains the mask.
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_ONKI);
        rt.timer.in_interrupt = true;
        rt.timer.interrupt_stack = vec![1]; // flow id placeholder
        rt.timer.delivered_masks = vec![ISR_ONKI];
        rt.timer.irq_source = None;

        rt.step(1).expect("execute reti without source");

        let isr = rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & ISR_ONKI,
            ISR_ONKI,
            "ROM-consistent model does not implicitly acknowledge ISR"
        );
        assert!(rt.timer.interrupt_stack.is_empty());
        assert!(rt.timer.delivered_masks.is_empty());
    }

    #[test]
    fn timers_tick_and_latch_status_during_interrupts_without_nesting() {
        let mut rt = CoreRuntime::new();
        rt.timer.enabled = true;
        rt.timer.mti_period = 1;
        rt.timer.reset(0);
        rt.timer.in_interrupt = true;
        rt.state.set_reg(RegName::PC, 0x0000); // opcode 0x00 = NOP by default

        rt.step(1).expect("step while in interrupt");
        assert!(
            rt.timer.irq_pending,
            "timer status should pend while the handler is active"
        );
        assert_eq!(
            rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_MTI,
            ISR_MTI,
            "MTI should latch in ISR while the handler is active"
        );
        assert_eq!(rt.timer.next_mti, 2, "MTI phase should keep advancing");
        assert_eq!(
            rt.state.pc(),
            1,
            "the handler executes without a nested frame"
        );
        assert!(rt.timer.in_interrupt, "handler service remains active");
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

        rt.step(1).expect("tick timer and arm MTI");
        rt.step(1)
            .expect("validate and deliver MTI at the next scheduling boundary");

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

        // Delivery and the selected handler's first instruction share this
        // runtime step; the handler contains RETI.
        rt.step(1).expect("deliver irq");
        assert_eq!(rt.timer.irq_total, 1);
        assert_eq!(rt.timer.irq_key, 1);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        rt.step(1)
            .expect("execute foreground instruction after acknowledgement");
        assert_eq!(rt.timer.irq_total, 1);
        assert_eq!(rt.timer.irq_key, 1);
    }

    #[test]
    fn hardware_irq_updates_call_depth() {
        let mut rt = CoreRuntime::new();
        // Vector points to NOP at 0x0000, followed by RETI.
        rt.memory.write_external_byte(0x0000, 0x00);
        rt.memory.write_external_byte(0x0001, 0x01);
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
    fn runtime_cycle_counter_uses_documented_instruction_units() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_slice(
            0,
            &[
                0x08, 0x55, // MV A,55H: 2
                0x32, 0xc8, 0x20, 0x21, // PRE + MV (20H),(21H): 1 + 6
            ],
        );
        rt.memory.write_internal_byte(0x21, 0xa5);
        rt.state.set_pc(0);

        rt.step(1).expect("execute immediate move");
        assert_eq!(rt.cycle_count(), 2);
        rt.step(1).expect("execute fused PRE move");
        assert_eq!(rt.cycle_count(), 9);
        assert_eq!(rt.memory.read_internal_byte(0x20), Some(0xa5));
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
    fn hw002_wait_with_zero_i_advances_full_wrap_and_preserves_flags() {
        let mut rt = CoreRuntime::new();
        rt.memory.write_external_slice(0, &[0xEF]);
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::I, 0);
        rt.state.set_reg(RegName::FC, 1);
        rt.state.set_reg(RegName::FZ, 1);

        rt.step(1).expect("HW-002 WAIT I=0 must execute");

        assert_eq!(rt.metadata.cycle_count, 0x1_0001);
        assert_eq!(rt.metadata.instruction_count, 1);
        assert_eq!(rt.state.get_reg(RegName::I), 0);
        assert_eq!(rt.state.get_reg(RegName::FC), 1);
        assert_eq!(rt.state.get_reg(RegName::FZ), 1);
        assert_eq!(rt.state.pc(), 1);
    }

    #[test]
    fn quarantined_opcodes_fail_before_scheduler_or_level_mutation() {
        for opcode in [0x20, 0xBF] {
            let mut rt = CoreRuntime::new();
            rt.memory.write_external_byte(0, opcode);
            rt.state.set_pc(0);
            rt.state.set_reg(RegName::I, 0);
            *rt.timer = TimerContext::new(true, 1, 1);
            rt.timer.next_mti = 0;
            rt.timer.next_sti = 0;
            rt.onk_level = true;
            rt.external_interrupt_level = true;
            let reads_before = rt.memory.memory_read_count();
            let writes_before = rt.memory.memory_write_count();

            rt.step(1)
                .expect_err("quarantined opcode must fail in preflight");

            assert_eq!(rt.metadata.cycle_count, 0, "opcode 0x{opcode:02X}");
            assert_eq!(rt.metadata.instruction_count, 0, "opcode 0x{opcode:02X}");
            assert_eq!(rt.state.pc(), 0, "opcode 0x{opcode:02X}");
            assert_eq!(
                rt.memory.read_internal_byte_silent(IMEM_ISR_OFFSET),
                Some(0),
                "opcode 0x{opcode:02X} must not re-latch IRQ levels"
            );
            assert_eq!(rt.timer.next_mti, 0, "opcode 0x{opcode:02X}");
            assert_eq!(rt.timer.next_sti, 0, "opcode 0x{opcode:02X}");
            assert_eq!(rt.memory.memory_read_count(), reads_before);
            assert_eq!(rt.memory.memory_write_count(), writes_before);
        }
    }

    #[test]
    fn hw001_tcl_restarts_selected_timer_phases_without_clearing_status() {
        for (lcc, clear_mti, clear_sti) in [
            (0x00_u8, false, false),
            (0x01, true, false),
            (0x02, false, true),
            (0x03, true, true),
        ] {
            let mut rt = CoreRuntime::new();
            rt.memory.write_external_byte(0, 0xCE);
            rt.memory.write_internal_byte(IMEM_LCC_OFFSET, lcc);
            rt.memory.write_internal_byte(IMEM_ISR_OFFSET, 0xA5);
            rt.state.set_pc(0);
            *rt.timer = TimerContext::new(true, 10, 20);
            rt.timer.next_mti = 77;
            rt.timer.next_sti = 88;

            rt.step(1).expect("silicon-verified TCL must execute");

            assert_eq!(rt.state.pc(), 1);
            assert_eq!(rt.metadata.instruction_count, 1);
            assert_eq!(rt.metadata.cycle_count, 1);
            assert_eq!(rt.timer.next_mti, if clear_mti { 10 } else { 77 });
            assert_eq!(rt.timer.next_sti, if clear_sti { 20 } else { 88 });
            assert_eq!(rt.memory.read_internal_byte(IMEM_LCC_OFFSET), Some(lcc));
            assert_eq!(rt.memory.read_internal_byte(IMEM_ISR_OFFSET), Some(0xA5));
        }
    }

    #[test]
    fn deliverable_irq_fetches_but_does_not_decode_reserved_fallthrough() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::S, 0x0200);
        rt.memory.write_external_byte(0, 0x20);
        rt.memory.write_external_byte(0x0100, 0x00);
        rt.memory.write_external_byte(INTERRUPT_VECTOR_ADDR, 0x00);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 1, 0x01);
        rt.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 2, 0x00);
        rt.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        rt.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        rt.timer.irq_pending = true;
        rt.timer.irq_source = Some("KEY".to_string());
        rt.memory.set_python_ranges(vec![(0, 0)]);
        let fallthrough_reads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let callback_reads = std::sync::Arc::clone(&fallthrough_reads);
        rt.set_host_read(move |address| {
            assert_eq!(address, 0);
            callback_reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(0x20)
        });

        rt.step(1)
            .expect("reserved fall-through byte is fetched but not decoded");

        assert_eq!(rt.state.pc(), 0x0101);
        assert_eq!(rt.metadata.instruction_count, 1);
        assert_eq!(
            fallthrough_reads.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn malformed_operands_fail_before_scheduler_or_level_mutation() {
        let cases: &[(&str, &[u8])] = &[
            ("JP narrow selector", &[0x11, 0x00]),
            ("E3 invalid mode", &[0xE3, 0x10, 0x00]),
            ("EB invalid mode", &[0xEB, 0x10, 0x00]),
            ("unverified BP+PY selector", &[0x21, 0xC8, 0x00, 0x01]),
        ];

        for &(name, bytes) in cases {
            let mut rt = CoreRuntime::new();
            rt.memory.write_external_slice(0, bytes);
            rt.state.set_pc(0);
            rt.state.set_reg(RegName::I, 1);
            rt.state.set_reg(RegName::S, 0x23456);
            rt.state.set_reg(RegName::FC, 1);
            rt.state.set_reg(RegName::FZ, 1);
            *rt.timer = TimerContext::new(true, 1, 1);
            rt.timer.next_mti = 0;
            rt.timer.next_sti = 0;
            rt.onk_level = true;
            rt.external_interrupt_level = true;
            let reads_before = rt.memory.memory_read_count();
            let writes_before = rt.memory.memory_write_count();

            let error = rt.step(1).expect_err(name);

            assert!(
                error.to_string().contains("preflight opcode"),
                "{name}: {error}"
            );
            assert_eq!(rt.metadata.cycle_count, 0, "{name}");
            assert_eq!(rt.metadata.instruction_count, 0, "{name}");
            assert_eq!(rt.state.pc(), 0, "{name}");
            assert_eq!(rt.state.get_reg(RegName::I), 1, "{name}");
            assert_eq!(rt.state.get_reg(RegName::S), 0x23456, "{name}");
            assert_eq!(rt.state.get_reg(RegName::FC), 1, "{name}");
            assert_eq!(rt.state.get_reg(RegName::FZ), 1, "{name}");
            assert_eq!(
                rt.memory.read_internal_byte_silent(IMEM_ISR_OFFSET),
                Some(0),
                "{name} must not re-latch IRQ levels"
            );
            assert_eq!(rt.timer.next_mti, 0, "{name}");
            assert_eq!(rt.timer.next_sti, 0, "{name}");
            assert!(!rt.timer.irq_pending, "{name}");
            assert_eq!(rt.memory.memory_read_count(), reads_before, "{name}");
            assert_eq!(rt.memory.memory_write_count(), writes_before, "{name}");
        }
    }

    #[test]
    fn software_reset_rejects_dynamic_vector_or_target_before_runtime_mutation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let _perfetto_lock = perfetto_test_guard();
        for dynamic_vector in [true, false] {
            let mut rt = CoreRuntime::new();
            let target = 0x00200;
            rt.state.set_pc(0);
            rt.state.set_reg(RegName::S, 0x00300);
            rt.memory.write_external_byte(0, 0xFF);
            rt.memory
                .write_external_byte(crate::pce500::ROM_RESET_VECTOR_ADDR, (target & 0xFF) as u8);
            rt.memory.write_external_byte(
                crate::pce500::ROM_RESET_VECTOR_ADDR + 1,
                ((target >> 8) & 0xFF) as u8,
            );
            rt.memory.write_external_byte(
                crate::pce500::ROM_RESET_VECTOR_ADDR + 2,
                ((target >> 16) & 0x0F) as u8,
            );
            rt.memory.write_external_byte(target, 0x00);
            if dynamic_vector {
                rt.memory.set_python_ranges(vec![(
                    crate::pce500::ROM_RESET_VECTOR_ADDR,
                    crate::pce500::ROM_RESET_VECTOR_ADDR + 2,
                )]);
            } else {
                rt.memory.set_python_ranges(vec![(target, target)]);
            }
            rt.set_host_peek(move |addr| {
                Some(match addr & ADDRESS_MASK {
                    addr if addr == crate::pce500::ROM_RESET_VECTOR_ADDR => (target & 0xFF) as u8,
                    addr if addr == crate::pce500::ROM_RESET_VECTOR_ADDR + 1 => {
                        ((target >> 8) & 0xFF) as u8
                    }
                    addr if addr == crate::pce500::ROM_RESET_VECTOR_ADDR + 2 => {
                        ((target >> 16) & 0x0F) as u8
                    }
                    addr if addr == target => 0x00,
                    _ => 0x00,
                })
            });
            let architectural_reads = Arc::new(AtomicUsize::new(0));
            let read_count = Arc::clone(&architectural_reads);
            rt.set_host_read(move |_addr| {
                read_count.fetch_add(1, Ordering::Relaxed);
                Some(0)
            });
            *rt.timer = TimerContext::new(true, 1, 1);
            rt.timer.next_mti = 0;
            rt.timer.next_sti = 0;
            rt.onk_level = true;
            rt.external_interrupt_level = true;
            let reads_before = rt.memory.memory_read_count();
            let writes_before = rt.memory.memory_write_count();
            let register_names = [
                RegName::PC,
                RegName::BA,
                RegName::I,
                RegName::X,
                RegName::Y,
                RegName::U,
                RegName::S,
                RegName::F,
            ];
            let registers_before = register_names.map(|name| rt.state.get_reg(name));
            let power_before = rt.state.power_state();
            let calls_before = rt.state.snapshot_call_metrics();

            let error = rt
                .step(1)
                .expect_err("dynamic RESET vector state must fail before runtime mutation");

            assert!(
                error
                    .to_string()
                    .contains("RESET vector preflight: callback-backed vector/target"),
                "dynamic_vector={dynamic_vector}: {error}"
            );
            assert_eq!(
                register_names.map(|name| rt.state.get_reg(name)),
                registers_before,
                "dynamic_vector={dynamic_vector}"
            );
            assert_eq!(rt.state.power_state(), power_before);
            let calls_after = rt.state.snapshot_call_metrics();
            assert_eq!(calls_after.call_stack, calls_before.call_stack);
            assert_eq!(calls_after.call_depth, calls_before.call_depth);
            assert_eq!(calls_after.call_sub_level, calls_before.call_sub_level);
            assert_eq!(calls_after.call_page_stack, calls_before.call_page_stack);
            assert_eq!(
                calls_after.call_return_widths,
                calls_before.call_return_widths
            );
            assert_eq!(rt.metadata.cycle_count, 0);
            assert_eq!(rt.metadata.instruction_count, 0);
            assert_eq!(rt.timer.next_mti, 0);
            assert_eq!(rt.timer.next_sti, 0);
            assert!(!rt.timer.irq_pending);
            assert_eq!(
                rt.memory.read_internal_byte_silent(IMEM_ISR_OFFSET),
                Some(0)
            );
            assert_eq!(architectural_reads.load(Ordering::Relaxed), 0);
            assert_eq!(rt.memory.memory_read_count(), reads_before);
            assert_eq!(rt.memory.memory_write_count(), writes_before);
        }
    }

    #[test]
    fn reti_stacked_f_upper_bits_normalize_through_scheduler() {
        let mut rt = CoreRuntime::new();
        rt.state.set_pc(0);
        rt.state.set_reg(RegName::U, 0x00180);
        rt.state.set_reg(RegName::S, 0x00190);
        rt.state.set_reg(RegName::FC, 1);
        rt.state.set_reg(RegName::FZ, 1);
        rt.state.set_reg(RegName::IMR, 0x55);
        rt.memory.write_internal_byte(IMEM_IMR_OFFSET, 0x55);
        rt.memory.write_external_byte(0, 0x01);
        rt.memory.write_external_byte(0x00190, 0xA5);
        rt.memory.write_external_byte(0x00191, 0xFC);
        rt.memory.write_external_byte(0x00192, 0x12);
        rt.memory.write_external_byte(0x00193, 0x34);
        rt.memory.write_external_byte(0x00194, 0x05);
        *rt.timer = TimerContext::new(true, 1, 1);
        rt.timer.next_mti = 100;
        rt.timer.next_sti = 100;

        rt.step(1).expect("RETI accepts and normalizes stacked F");

        assert_eq!(rt.metadata.instruction_count, 1);
        assert_eq!(rt.state.pc(), 0x53412);
        assert_eq!(rt.state.get_reg(RegName::S), 0x00195);
        assert_eq!(rt.state.get_reg(RegName::F), 0);
        assert_eq!(rt.state.get_reg(RegName::IMR), 0xA5);
        assert_eq!(
            rt.memory.read_internal_byte_silent(IMEM_IMR_OFFSET),
            Some(0xA5)
        );
    }

    #[test]
    fn hardware_resolved_data_paths_execute_through_scheduler() {
        let mut popu = CoreRuntime::new();
        popu.state.set_pc(0);
        popu.state.set_reg(RegName::U, 0x00180);
        popu.state.set_reg(RegName::F, 0x03);
        popu.memory.write_external_byte(0, 0x3E);
        popu.memory.write_external_byte(0x00180, 0xFC);

        popu.step(1).expect("POPU F upper bits normalize");

        assert_eq!(popu.metadata.instruction_count, 1);
        assert_eq!(popu.state.pc(), 1);
        assert_eq!(popu.state.get_reg(RegName::U), 0x00181);
        assert_eq!(popu.state.get_reg(RegName::F), 0);

        let mut pops = CoreRuntime::new();
        pops.state.set_pc(0);
        pops.state.set_reg(RegName::S, 0x00180);
        pops.state.set_reg(RegName::F, 0x03);
        pops.memory.write_external_byte(0, 0x5F);
        pops.memory.write_external_byte(0x00180, 0xA5);

        pops.step(1).expect("POPS F upper bits normalize");

        assert_eq!(pops.metadata.instruction_count, 1);
        assert_eq!(pops.state.pc(), 1);
        assert_eq!(pops.state.get_reg(RegName::S), 0x00181);
        assert_eq!(pops.state.get_reg(RegName::F), 0x01);

        let mut exp = CoreRuntime::new();
        exp.state.set_pc(0);
        exp.state.set_reg(RegName::F, 0x03);
        exp.memory.write_external_slice(0, &[0xC2, 0x20, 0x30]);
        for (offset, value) in [0x11, 0x22, 0xF0].into_iter().enumerate() {
            exp.memory.write_internal_byte(0x20 + offset as u32, value);
        }
        for (offset, value) in [0x44, 0x55, 0x10].into_iter().enumerate() {
            exp.memory.write_internal_byte(0x30 + offset as u32, value);
        }

        exp.step(1).expect("EXP exchanges raw 24-bit triples");

        assert_eq!(exp.metadata.instruction_count, 1);
        assert_eq!(exp.state.pc(), 3);
        assert_eq!(exp.state.get_reg(RegName::F), 0x03);
        assert_eq!(
            (0..3)
                .map(|offset| exp.memory.read_internal_byte_silent(0x20 + offset))
                .collect::<Vec<_>>(),
            vec![Some(0x44), Some(0x55), Some(0x10)]
        );
        assert_eq!(
            (0..3)
                .map(|offset| exp.memory.read_internal_byte_silent(0x30 + offset))
                .collect::<Vec<_>>(),
            vec![Some(0x11), Some(0x22), Some(0xF0)]
        );
    }

    #[test]
    fn requires_python_without_safe_host_peek_rejects_before_execution() {
        let mut rt = CoreRuntime::new();
        // Mark an external range as Python-only and point PC at it.
        rt.memory
            .set_python_ranges(vec![(0x0000_2000, 0x0000_2000)]);
        rt.state.set_reg(RegName::PC, 0x0000_2000);
        rt.memory.write_external_byte(0x0000_2000, 0x00);
        let pc_before = rt.state.pc();
        let instructions_before = rt.metadata.instruction_count;
        let cycles_before = rt.metadata.cycle_count;

        let error = rt
            .step(1)
            .expect_err("host-routed code needs an explicit side-effect-free peek");

        assert!(error
            .to_string()
            .contains("side-effect-free memory is unavailable"));
        assert_eq!(
            rt.state.get_reg(RegName::PC) & ADDRESS_MASK,
            pc_before,
            "PC must not advance on rejected preflight"
        );
        assert_eq!(rt.metadata.instruction_count, instructions_before);
        assert_eq!(rt.metadata.cycle_count, cycles_before);
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
