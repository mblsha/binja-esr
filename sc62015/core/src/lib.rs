//! Placeholder pure-Rust SC62015 core crate.
//!
//! This scaffolding exists to start the extraction away from the PyO3-specific
//! runtime. The goal is to grow this crate into the real runtime/bus/peripheral
//! layer, leaving the PyO3 crate as a thin wrapper.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use thiserror::Error;
use rust_scil::state::State as RsState;

pub mod memory;
pub mod keyboard;
pub mod lcd;
pub mod snapshot;
pub mod timer;
pub mod eval;
pub mod exec;
pub mod perfetto;

pub use keyboard::KeyboardMatrix;
pub use lcd::{LcdController, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
pub use memory::{
    MemoryImage, ADDRESS_MASK, EXTERNAL_SPACE, INTERNAL_ADDR_MASK, INTERNAL_MEMORY_START,
    INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE,
};
pub use eval::{LayoutEntryView, ManifestEntryView, BoundInstrView, eval_manifest_entry};
pub use exec::{
    BoundInstrBuilder, BusProfiler, DecodedInstr, ExecManifestEntry, HostMemory, HybridBus,
    InstructionStream, NullProfiler, OpcodeIndexView, OpcodeLookup, OpcodeLookupKey,
    decode_bound_from_memory, execute_step, resolve_bus_address, PRE_PREFIXES,
};
pub use perfetto::PerfettoTracer;
pub use timer::TimerContext;
pub use snapshot::{
    load_snapshot, pack_registers, save_snapshot, unpack_registers, SnapshotLoad, SNAPSHOT_MAGIC,
    SNAPSHOT_REGISTER_LAYOUT, SNAPSHOT_VERSION,
};
pub use rust_scil::state::State as CpuState;

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub magic: String,
    pub version: u32,
    pub backend: String,
    pub created: String,
    pub instruction_count: u64,
    pub cycle_count: u64,
    pub pc: u32,
    pub timer: TimerInfo,
    pub interrupts: InterruptInfo,
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
            pc: 0,
            timer: TimerInfo::default(),
            interrupts: InterruptInfo::default(),
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

pub const DEFAULT_REG_WIDTH: u8 = 24;

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

pub fn collect_registers(state: &RsState) -> HashMap<String, u32> {
    let mut regs = HashMap::new();
    for (name, width_bytes) in snapshot::SNAPSHOT_REGISTER_LAYOUT.iter() {
        let bits = (width_bytes * 8) as u8;
        regs.insert((*name).to_string(), state.get_reg(name, bits));
    }
    regs
}

pub fn apply_registers(state: &mut RsState, regs: &HashMap<String, u32>) {
    for (name, _) in snapshot::SNAPSHOT_REGISTER_LAYOUT.iter() {
        let value = *regs.get(*name).unwrap_or(&0);
        let width = register_width(name);
        state.set_reg(name, value, width);
    }
}

/// Extremely small placeholder runtime. This will be replaced with the full
/// SC62015 execution/bus once extracted.
pub struct CoreRuntime {
    metadata: SnapshotMetadata,
    pub memory: MemoryImage,
    pub state: RsState,
    pub fast_mode: bool,
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
            state: RsState::default(),
            fast_mode: false,
        }
    }

    pub fn load_rom(&mut self, blob: &[u8], start: usize) {
        let end = (start + blob.len()).min(self.memory.external_len());
        if start < end {
            self.memory
                .write_external_slice(start, &blob[..(end - start)]);
        }
    }

    pub fn step(&mut self, _instructions: usize) -> Result<()> {
        // Placeholder: real execution will live here after extraction.
        let pc = self.state.get_reg("PC", register_width("PC"));
        self.state
            .set_reg("PC", pc.wrapping_add(1), register_width("PC"));
        self.metadata.instruction_count = self.metadata.instruction_count.wrapping_add(1);
        self.metadata.cycle_count = self.metadata.cycle_count.wrapping_add(1);
        Ok(())
    }

    pub fn save_snapshot(&self, path: &std::path::Path) -> Result<()> {
        let mut metadata = self.metadata.clone();
        metadata.instruction_count = self.metadata.instruction_count;
        metadata.cycle_count = self.metadata.cycle_count;
        metadata.pc = self.get_reg("PC");
        metadata.memory_dump_pc = 0;
        let regs = collect_registers(&self.state);
        snapshot::save_snapshot(path, &metadata, &regs, &self.memory, None)
    }

    pub fn load_snapshot(&mut self, path: &std::path::Path) -> Result<()> {
        let loaded = snapshot::load_snapshot(path, &mut self.memory)?;
        self.metadata = loaded.metadata;
        apply_registers(&mut self.state, &loaded.registers);
        self.fast_mode = self.metadata.fast_mode;
        Ok(())
    }

    pub fn set_reg(&mut self, name: &str, value: u32) {
        let width = register_width(name);
        self.state.set_reg(name, value, width);
    }

    pub fn get_reg(&self, name: &str) -> u32 {
        self.state.get_reg(name, register_width(name))
    }

    pub fn set_flag(&mut self, name: &str, value: u8) {
        self.state.set_flag(name, value as u32);
    }

    pub fn get_flag(&self, name: &str) -> u8 {
        self.state.get_flag(name) as u8
    }
}
