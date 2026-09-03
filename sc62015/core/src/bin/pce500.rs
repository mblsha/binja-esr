// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/cli.py

use chrono::{Datelike, Local, Timelike};
use clap::Parser;
use crc32fast::Hasher as Crc32Hasher;
use flate2::{write::ZlibEncoder, Compression};
use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
use sc62015_core::{
    apply_registers, collect_registers, create_lcd, emit_event,
    generated_key_input::{lookup_generated_key_input, GeneratedKeyInputKind},
    iq7000::{self, Iq7000ClockSeed, Iq7000RtcPeripheral},
    keyboard::{KeyboardMatrix, KeyboardSnapshot},
    lcd::{lcd_kind_from_snapshot_meta, LcdHal, LcdKind, LcdWriteTrace},
    llama::{
        eval::{
            fetch_validated_vector, perfetto_next_substep, power_on_reset,
            prepare_validated_vector, set_perf_instr_counter, validate_vector_transfer_with_length,
            LlamaBus, LlamaExecutor, TimerTrace, ValidatedVectorTransfer,
        },
        opcodes::RegName,
        state::{mask_for, validate_f_image, CallMetricsSnapshot, LlamaState, PowerState},
    },
    memory::{
        MemoryImage, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_KIL_OFFSET, IMEM_KOH_OFFSET,
        IMEM_KOL_OFFSET, IMEM_LCC_OFFSET, IMEM_SCR_OFFSET, IMEM_SSR_OFFSET, IMEM_UCR_OFFSET,
        IMEM_USR_OFFSET,
    },
    pce500::{
        load_pce500_rom_window_into_memory, load_pce500_system_image_into_memory,
        seed_pce500_bootstrap_imem, NO_RAM_WINDOW_END, NO_RAM_WINDOW_START, ROM_RESET_VECTOR_ADDR,
        ROM_WINDOW_LEN, ROM_WINDOW_START,
    },
    perfetto::set_call_ui_function_names,
    sleep_cycles, snapshot,
    timer::TimerContext,
    AsyncDriver, CoreRuntime, DeviceMemoryCardProfile, DeviceModel, DeviceTextDecoder, DriverEvent,
    PerfettoTracer, SnapshotMetadata, ADDRESS_MASK, INTERNAL_MEMORY_START, NUM_TEMP_REGISTERS,
    PERFETTO_TRACER,
};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::error::Error;
use std::fs;
use std::io::{BufWriter, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[cfg(test)]
use sc62015_core::memory::IMEM_EIH_OFFSET;

const PCE500_IOCS_WS_PTR_ADDR: u32 = 0x00BFD17;
const PCE500_KEY_FIFO_BASE_OFFSET: u32 = 0x02;
const PCE500_KEY_FIFO_TAIL_OFFSET: u32 = 0x04;
const PCE500_KEY_FIFO_HEAD_OFFSET: u32 = 0x05;
const PCE500_KEY_FIFO_CAPACITY: u32 = 0x10;
const VEC_RANGE_START: u32 = 0x00BFCC6;
const VEC_RANGE_END: u32 = 0x00BFCCC;
const ISR_KEYI: u8 = 0x04;
const ISR_ONKI: u8 = 0x08;
const ISR_TXI: u8 = 0x10;
const ISR_RXI: u8 = 0x20;
const ISR_EXI: u8 = 0x40;
const ISR_MTI: u8 = 0x01;
const ISR_STI: u8 = 0x02;
const IMR_MASTER: u8 = 0x80;
const IMR_KEY: u8 = 0x04;
const IMR_MTI: u8 = 0x01;
const IMR_STI: u8 = 0x02;
const IMR_ONK: u8 = 0x08;
const IMR_TX: u8 = 0x10;
const IMR_RX: u8 = 0x20;
const IMR_EX: u8 = 0x40;
#[cfg(test)]
const SSR_CI: u8 = 0x02;
const SSR_ONK: u8 = 0x08;
const PF1_CODE: u8 = 0x56; // col=10, row=6
const PF2_CODE: u8 = 0x55; // col=10, row=5
const KEY_SEQ_DEFAULT_HOLD: u64 = 1_000;
const DEFAULT_RUN_STEPS: u64 = 20_000;
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;
const CPU_DONE_EVENT: u32 = 1;
const LCD_CAPTURE_SCALE: usize = 3;
const IQ7000_ANNUNCIATOR_SHADOW_ADDR: u32 = 0x006160;
const IQ7000_KEY_STATE_ADDR: u32 = 0x001FDA3;
const IQ7000_SHIFT_ANNUNCIATOR: u8 = 0x10;
const IQ7000_CAPS_ANNUNCIATOR: u8 = 0x08;
const IQ7000_NAMED_ANNUNCIATOR_MASK: u8 = IQ7000_SHIFT_ANNUNCIATOR | IQ7000_CAPS_ANNUNCIATOR;
const PCLINK_SERIAL_RX_PACE_STEPS: u32 = 1_000;
const PCLINK_SERIAL_POST_CLIENT_SETTLE_STEPS: usize = 2_500_000;
const PCLINK_SERIAL_XON: u8 = 0x11;
const PCLINK_SERIAL_XOFF: u8 = 0x13;
#[cfg(test)]
const IQ7000_PACOM_EIH_DATA: u8 = 0x40;
#[cfg(test)]
const IQ7000_PACOM_RELEASE_STEPS: u8 = 8;

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum CardMode {
    Auto,
    Present,
    Absent,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeEngine {
    Core,
    Legacy,
}

impl CardMode {
    fn resolve(self, model: DeviceModel) -> DeviceMemoryCardProfile {
        match self {
            Self::Auto => model.default_memory_card_profile(),
            Self::Present => DeviceMemoryCardProfile::BlankWritable64KiB,
            Self::Absent => DeviceMemoryCardProfile::Absent,
        }
    }
}

struct IrqPerfetto {
    builder: PerfettoTraceBuilder,
    track_timer: TrackId,
    track_key: TrackId,
    track_misc: TrackId,
    path: PathBuf,
}

impl IrqPerfetto {
    fn new(path: PathBuf) -> Self {
        let mut builder = PerfettoTraceBuilder::new("pce500-llama");
        let track_timer = builder.add_thread("irq.timer");
        let track_key = builder.add_thread("irq.key");
        let track_misc = builder.add_thread("irq.misc");
        Self {
            builder,
            track_timer,
            track_key,
            track_misc,
            path,
        }
    }

    fn track_for(&self, src: Option<&str>) -> TrackId {
        match src {
            Some(s) if s.contains("MTI") || s.contains("STI") => self.track_timer,
            Some("KEY") | Some("ONK") => self.track_key,
            Some(s) if s.contains("KEY") => self.track_key,
            _ => self.track_misc,
        }
    }

    fn instant<'a>(
        &mut self,
        name: &str,
        src: Option<&str>,
        ts: u64,
        annotations: impl IntoIterator<Item = (&'a str, AnnotationValue)>,
    ) {
        let mut ev =
            self.builder
                .add_instant_event(self.track_for(src), name.to_string(), ts as i64);
        if let Some(s) = src {
            ev.add_annotation("src", s);
        }
        for (k, v) in annotations {
            ev.add_annotation(k, v);
        }
        ev.finish();
    }

    fn finish(self) -> Result<PathBuf, String> {
        self.builder
            .save(&self.path)
            .map_err(|e| format!("perfetto save: {e}"))?;
        Ok(self.path)
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "pce500-llama",
    about = "Standalone Rust LLAMA runner (ROM selectable; defaults to PC-E500)."
)]
struct Args {
    /// Maximum machine-scheduler boundaries to execute before exiting.
    #[arg(long, default_value_t = DEFAULT_RUN_STEPS)]
    steps: u64,

    /// ROM model/profile to run (sets defaults for --rom and --bnida).
    #[arg(long, value_enum, default_value_t = DeviceModel::DEFAULT)]
    model: DeviceModel,

    /// Machine scheduler to use. The shared core is the correctness path;
    /// legacy is retained only for specialized trace-replay diagnostics.
    #[arg(long, value_enum, default_value_t = RuntimeEngine::Core)]
    runtime: RuntimeEngine,

    /// ROM image to load (defaults to the repo-symlinked ROM for --model).
    #[arg(long, value_name = "PATH")]
    rom: Option<PathBuf>,

    /// Enable/disable memory card emulation (0x040000..0x04FFFF).
    #[arg(long, value_enum, default_value_t = CardMode::Auto)]
    card: CardMode,

    /// Scripted key sequence (comma/semicolon separated).
    #[arg(long, value_name = "SEQ")]
    key_seq: Option<String>,

    /// JSON scenario file with steps/key_seq/expect/capture/debug settings.
    #[arg(long, value_name = "PATH")]
    scenario: Option<PathBuf>,

    /// Log key-seq events (press/release/wait triggers).
    #[arg(long, default_value_t = false)]
    key_seq_log: bool,

    /// Decode LCD text and require this substring to appear (can repeat).
    #[arg(long, value_name = "TEXT")]
    expect_text: Vec<String>,

    /// Decode LCD text and require ROW:TEXT (row is zero-based, e.g., 0:S2(CARD)).
    #[arg(long, value_name = "ROW:TEXT")]
    expect_row: Vec<String>,

    /// Emit perf summary (instr/sec).
    #[arg(long, default_value_t = false)]
    perf: bool,

    /// Force LCD write logging (honours --lcd-log-limit).
    #[arg(long, default_value_t = false)]
    lcd_log: bool,

    /// Maximum LCD writes to log when tracing is enabled.
    #[arg(long, value_name = "N")]
    lcd_log_limit: Option<u32>,

    /// Stop execution when PC matches this address (hex or decimal).
    #[arg(long, value_name = "ADDR")]
    stop_pc: Option<String>,

    /// Trace specific PCs (hex or decimal); logs when hit.
    #[arg(long, value_name = "ADDR", num_args = 1.., value_delimiter = ',')]
    trace_pc: Vec<String>,

    /// After a traced PC hit, log the next N PCs (helpful to follow IRQ paths).
    #[arg(long, value_name = "N")]
    trace_pc_window: Option<u64>,

    /// When tracing PCs, also dump a small register snapshot (A,F,IMR,S,Y).
    #[arg(long, default_value_t = false)]
    trace_regs: bool,

    /// Disable timers (MTI/STI) to isolate keyboard IRQ behaviour.
    #[arg(long, default_value_t = false)]
    disable_timers: bool,

    /// Emit a Perfetto trace with IRQ/IMR/ISR events.
    #[arg(long, default_value_t = false)]
    perfetto: bool,

    /// Path to write the Perfetto trace.
    #[arg(long, value_name = "PATH", default_value = "pc-e500.perfetto-trace")]
    perfetto_path: PathBuf,

    /// Dump LCD write trace (PC + call stack per addressing unit) as JSON.
    #[arg(long, value_name = "PATH")]
    dump_lcd_trace: Option<PathBuf>,

    /// Dump external bus accesses as JSONL (one byte-level event per line).
    #[arg(long, value_name = "PATH")]
    dump_bus_trace: Option<PathBuf>,

    /// Load function names from a BNIDA export (rom-analysis/.../bnida.json) and use them to label
    /// the "Functions" track in Perfetto traces (replacing sub_XXXXXX fallbacks).
    #[arg(long, value_name = "PATH")]
    bnida: Option<PathBuf>,

    /// Load a snapshot (.pcsnap) before executing.
    #[arg(long, value_name = "PATH")]
    snapshot_in: Option<PathBuf>,

    /// Save a snapshot (.pcsnap) after executing.
    #[arg(long, value_name = "PATH")]
    snapshot_out: Option<PathBuf>,

    /// Save the final LCD pixels as a PNG render.
    #[arg(long, value_name = "PATH")]
    capture_png: Option<PathBuf>,

    /// Save final run/capture metadata as JSON.
    #[arg(long, value_name = "PATH")]
    capture_json: Option<PathBuf>,

    /// Save the IQ-7000 PC-Link-ready LCD pixels before serving PC-Link serial clients.
    #[arg(long, value_name = "PATH")]
    iq7p_ready_capture_png: Option<PathBuf>,

    /// Save IQ-7000 PC-Link-ready run/capture metadata before serving PC-Link serial clients.
    #[arg(long, value_name = "PATH")]
    iq7p_ready_capture_json: Option<PathBuf>,

    /// Save structured debug probes as JSON.
    #[arg(long, value_name = "PATH")]
    debug_probe_json: Option<PathBuf>,

    /// Add debug memory range probe as NAME@ADDR:LEN or ADDR:LEN (can repeat).
    #[arg(long, value_name = "NAME@ADDR:LEN")]
    debug_probe_range: Vec<String>,

    /// Resume the JP ROM from the trace-backed post-OFF turnon2 state.
    #[arg(long, default_value_t = false)]
    turnon2_resume: bool,

    /// Path to a trace-derived resume profile JSON (defaults to the JP turnon2 profile).
    #[arg(long, value_name = "PATH")]
    turnon_profile: Option<PathBuf>,

    /// Use reset-trace-backed CE1/CE6 values to reproduce the JP reset-trace boot path
    /// and reach MAIN MENU parity.
    #[arg(long, default_value_t = false)]
    reset_trace_card: bool,

    /// Resume from the 2_0-guided reset-trace2 state and force the JP MAIN MENU target display.
    #[arg(long, default_value_t = false)]
    reset_trace2_main_display: bool,

    /// Path to a reset-trace2 guided main-display profile JSON.
    #[arg(long, value_name = "PATH")]
    reset_trace2_profile: Option<PathBuf>,

    /// IQ-7000 clock seed: host, off, or YYYYMMDDHHMM.
    #[arg(long, value_name = "host|off|YYYYMMDDHHMM", default_value = "host")]
    iq7000_rtc: String,

    /// Listen for raw PC-Link serial bytes and bridge them to IQ-7000 SC62015 UART registers.
    #[arg(long, value_name = "HOST:PORT")]
    pclink_serial_listen: Option<String>,

    /// Number of raw PC-Link serial socket clients to serve before exiting.
    #[arg(long, default_value_t = 1)]
    pclink_serial_clients: usize,

    /// Drive the IQ-7000 ROM UI into PC-Link mode before serving PC-Link serial clients.
    #[arg(long, default_value_t = false)]
    iq7p_enter_pclink: bool,

    /// Type a MEMO through the IQ-7000 foreground ROM UI before entering PC-Link.
    #[arg(long, value_name = "TEXT")]
    iq7000_seed_memo: Vec<String>,
    // (legacy automation flags removed; use --key-seq instead)
}

#[derive(Serialize)]
struct LcdTraceDump {
    executed: u64,
    pc: u32,
    halted: bool,
    lcd_lines: Vec<String>,
    vram: Vec<Vec<u8>>,
    trace: Vec<Vec<LcdWriteTrace>>,
}

#[derive(serde::Deserialize)]
struct BnidaExport {
    #[serde(default)]
    names: HashMap<String, String>,
}

struct RunSummary {
    executed: u64,
    pc: u32,
    halted: bool,
    lcd_writes: u64,
    imr_mem: u8,
    isr_mem: u8,
    imr_reg: u8,
    lcd_stats: sc62015_core::lcd::LcdStats,
    lcd_lines: Vec<String>,
    lcd_pixels: Vec<Vec<u8>>,
    lcd_annunciators: Option<LcdAnnunciators>,
    lcd_trace: Option<LcdTraceDump>,
    debug_probe: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct LcdAnnunciators {
    state_raw: u8,
    shadow_raw: u8,
    raw_union: u8,
    unmapped_state: u8,
    unmapped_shadow: u8,
    unmapped_union: u8,
    shift: bool,
    caps: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ScenarioFile {
    #[serde(default)]
    model: Option<DeviceModel>,
    #[serde(default)]
    steps: Option<u64>,
    #[serde(default)]
    key_seq: Option<ScenarioKeySeq>,
    #[serde(default)]
    key_seq_log: Option<bool>,
    #[serde(default)]
    expect_text: Vec<String>,
    #[serde(default)]
    expect_row: Vec<String>,
    #[serde(default)]
    capture_png: Option<PathBuf>,
    #[serde(default)]
    capture_json: Option<PathBuf>,
    #[serde(default)]
    debug_probe_json: Option<PathBuf>,
    #[serde(default)]
    debug_probe_range: Vec<String>,
    #[serde(default)]
    iq7000_rtc: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ScenarioKeySeq {
    One(String),
    Many(Vec<String>),
}

impl ScenarioKeySeq {
    fn join(self) -> String {
        match self {
            Self::One(raw) => raw,
            Self::Many(parts) => parts.join(";"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DebugProbeRange {
    name: String,
    addr: u32,
    len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Iq7000RtcSeed {
    clock: Iq7000ClockSeed,
    source: String,
}

impl Iq7000RtcSeed {
    fn from_host_now() -> Result<Self, String> {
        let now = Local::now();
        let raw = format!(
            "{:04}{:02}{:02}{:02}{:02}",
            now.year(),
            now.month(),
            now.day(),
            now.hour(),
            now.minute()
        );
        Self::from_yyyymmddhhmm(&raw, "host")
    }

    fn from_yyyymmddhhmm(raw: &str, source: impl Into<String>) -> Result<Self, String> {
        Ok(Self {
            clock: Iq7000ClockSeed::from_yyyymmddhhmm(raw)?,
            source: source.into(),
        })
    }
}

fn parse_iq7000_rtc_arg(raw: &str) -> Result<Option<Iq7000RtcSeed>, String> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("off") || trimmed.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    if trimmed.eq_ignore_ascii_case("host") || trimmed.is_empty() {
        return Iq7000RtcSeed::from_host_now().map(Some);
    }
    Iq7000RtcSeed::from_yyyymmddhhmm(trimmed, "fixed").map(Some)
}

#[derive(Debug, Clone, Deserialize)]
struct TurnonResumeByte {
    addr: u32,
    value: u8,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct TurnonResumeProfile {
    resume_pc: u32,
    onk_release_cycle: u64,
    onk_release_instr: u64,
    #[serde(default)]
    target_instruction_count: Option<u64>,
    #[serde(default)]
    target_pc: Option<u32>,
    trace_read_anchor_pc: u32,
    user_stack: u32,
    system_stack: u32,
    usr: u8,
    ssr_base: u8,
    iocs_workspace: u32,
    machine_area: u32,
    #[serde(default)]
    target_rows: Vec<String>,
    #[serde(default)]
    target_lcd_meta: Option<Value>,
    #[serde(default)]
    target_lcd_payload: Vec<u8>,
    #[serde(default)]
    trace_reads: Vec<TurnonResumeByte>,
    visible_bytes: Vec<TurnonResumeByte>,
}

#[derive(Debug, Clone, Deserialize)]
struct TraceResumeRegisters {
    ba: u32,
    i: u32,
    x: u32,
    y: u32,
    u: u32,
    s: u32,
    f: u8,
    #[serde(default)]
    call_sub_level: u32,
    #[serde(default)]
    temps: HashMap<String, u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct TraceResumeImemByte {
    offset: u32,
    value: u8,
}

#[derive(Debug, Clone, Deserialize)]
struct ResetTrace2MainDisplayProfile {
    #[serde(default)]
    seed_instruction_count: u64,
    resume_pc: u32,
    resume_registers: TraceResumeRegisters,
    #[serde(default)]
    visible_bytes: Vec<TurnonResumeByte>,
    #[serde(default)]
    imem_bytes: Vec<TraceResumeImemByte>,
    #[serde(default)]
    resume_lcd_meta: Option<Value>,
    #[serde(default)]
    resume_lcd_payload: Vec<u8>,
    #[serde(default)]
    target_instruction_count: Option<u64>,
    #[serde(default)]
    target_lcd_meta: Option<Value>,
    #[serde(default)]
    target_lcd_payload: Vec<u8>,
}

fn default_rom_path(model: DeviceModel) -> PathBuf {
    let data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../data/{}", model.rom_basename()));
    if data_path.exists() {
        data_path
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../../roms/{}", model.rom_basename()))
    }
}

fn resolve_scenario_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn apply_scenario(args: &mut Args) -> Result<(), Box<dyn Error>> {
    let Some(path) = args.scenario.clone() else {
        return Ok(());
    };
    let raw = fs::read_to_string(&path)?;
    let scenario: ScenarioFile = serde_json::from_str(&raw)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    if let Some(model) = scenario.model {
        args.model = model;
    }
    if let Some(steps) = scenario.steps {
        args.steps = steps;
    }
    if let Some(key_seq) = scenario.key_seq {
        args.key_seq = Some(key_seq.join());
    }
    if let Some(enabled) = scenario.key_seq_log {
        args.key_seq_log = enabled;
    }
    if !scenario.expect_text.is_empty() {
        args.expect_text = scenario.expect_text;
    }
    if !scenario.expect_row.is_empty() {
        args.expect_row = scenario.expect_row;
    }
    if let Some(capture_png) = scenario.capture_png {
        args.capture_png = Some(resolve_scenario_path(base, capture_png));
    }
    if let Some(capture_json) = scenario.capture_json {
        args.capture_json = Some(resolve_scenario_path(base, capture_json));
    }
    if let Some(debug_probe_json) = scenario.debug_probe_json {
        args.debug_probe_json = Some(resolve_scenario_path(base, debug_probe_json));
    }
    if !scenario.debug_probe_range.is_empty() {
        args.debug_probe_range = scenario.debug_probe_range;
    }
    if let Some(iq7000_rtc) = scenario.iq7000_rtc {
        args.iq7000_rtc = iq7000_rtc;
    }

    Ok(())
}

fn default_bnida_path(model: DeviceModel) -> PathBuf {
    match model {
        // When running via binja-esr-tests/scripts/run_rom_tests.sh, CWD is public-src.
        // Use CARGO_MANIFEST_DIR so `cargo run --manifest-path ...` works from any directory.
        DeviceModel::Iq7000 => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../rom-analysis/iq-7000/bnida.json"),
        DeviceModel::PcE500 => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../rom-analysis/pc-e500/en/bnida.json"),
        DeviceModel::PcE500Jp => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../rom-analysis/pc-e500/jp/bnida.json"),
    }
}

fn default_turnon_resume_profile_path(model: DeviceModel) -> PathBuf {
    match model {
        DeviceModel::PcE500Jp => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../../rom-analysis/pc-e500/jp-turnon2/pc-e500-jp.turnon2.resume_profile.json",
        ),
        _ => PathBuf::new(),
    }
}

fn default_reset_trace2_main_display_profile_path(model: DeviceModel) -> PathBuf {
    match model {
        DeviceModel::PcE500Jp => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../../rom-analysis/pc-e500/jp-reset-trace2/pc-e500-jp.reset-trace2.main_display_profile.json",
        ),
        _ => PathBuf::new(),
    }
}

fn load_turnon_resume_profile(
    model: DeviceModel,
    path: Option<PathBuf>,
) -> Result<Option<TurnonResumeProfile>, Box<dyn Error>> {
    let Some(candidate) = path.or_else(|| {
        let default = default_turnon_resume_profile_path(model);
        if default.as_os_str().is_empty() {
            None
        } else {
            Some(default)
        }
    }) else {
        return Ok(None);
    };
    if !candidate.exists() {
        return Err(format!("turnon resume profile not found: {}", candidate.display()).into());
    }
    let raw = fs::read_to_string(&candidate)?;
    let profile: TurnonResumeProfile = serde_json::from_str(&raw)?;
    Ok(Some(profile))
}

fn load_reset_trace2_main_display_profile(
    model: DeviceModel,
    path: Option<PathBuf>,
) -> Result<Option<ResetTrace2MainDisplayProfile>, Box<dyn Error>> {
    let Some(candidate) = path.or_else(|| {
        let default = default_reset_trace2_main_display_profile_path(model);
        if default.as_os_str().is_empty() {
            None
        } else {
            Some(default)
        }
    }) else {
        return Ok(None);
    };
    if !candidate.exists() {
        return Err(format!(
            "reset-trace2 main-display profile not found: {}",
            candidate.display()
        )
        .into());
    }
    let raw = fs::read_to_string(&candidate)?;
    let profile: ResetTrace2MainDisplayProfile = serde_json::from_str(&raw)?;
    Ok(Some(profile))
}

#[derive(Serialize)]
struct BusTraceEvent {
    index: u64,
    kind: &'static str,
    region: &'static str,
    addr: u32,
    value: u8,
    bits: u8,
    byte_offset: u8,
    pc: u32,
    instr_index: u64,
    cycle: u64,
}

fn load_bnida_names(
    model: DeviceModel,
    path: Option<PathBuf>,
) -> Result<HashMap<u32, String>, Box<dyn Error>> {
    let candidate = path.unwrap_or_else(|| default_bnida_path(model));
    if !candidate.exists() {
        return Ok(HashMap::new());
    }

    let raw = fs::read_to_string(&candidate)?;
    let bnida: BnidaExport = serde_json::from_str(&raw)?;
    if bnida.names.is_empty() {
        return Ok(HashMap::new());
    }

    let mut out: HashMap<u32, String> = HashMap::with_capacity(bnida.names.len());
    for (addr_str, name) in bnida.names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            continue;
        }
        let addr: u32 = match addr_str.trim().parse::<u32>() {
            Ok(v) => v & 0x000f_ffff,
            Err(_) => continue,
        };
        out.insert(addr, trimmed.to_string());
    }
    Ok(out)
}

struct StandaloneBus {
    memory: MemoryImage,
    lcd: Box<dyn LcdHal>,
    timer: TimerContext,
    cycle_count: u64,
    timer_finalize_clamp: bool,
    keyboard: KeyboardMatrix,
    lcd_writes: u64,
    log_lcd: bool,
    log_lcd_count: u32,
    log_lcd_limit: u32,
    irq_pending: bool,
    in_interrupt: bool,
    delivered_irq_count: u32,
    pending_kil: bool,
    pending_onk: bool,
    deferred_key_irq: bool,
    deferred_pending_kil: bool,
    last_kbd_access: Option<String>,
    kil_reads: u32,
    rom_koh_reads: u32,
    rom_kol_reads: u32,
    trace_kbd: bool,
    scan_on_timer: bool,
    last_pc: u32,
    instr_index: u64,
    vec_patched: bool,
    perfetto: Option<IrqPerfetto>,
    last_irq_src: Option<String>,
    active_irq_mask: u8,
    #[allow(dead_code)]
    perfetto_enabled: bool,
    host_read: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
    host_peek: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
    host_write: Option<Box<dyn FnMut(u32, u8) + Send>>,
    bus_trace: Option<BufWriter<fs::File>>,
    bus_trace_index: u64,
    trace_resume_ssr_onk: bool,
    trace_resume_onk_release_cycle: Option<u64>,
    trace_resume_onk_release_instr: Option<u64>,
    trace_resume_read_anchor_pc: Option<u32>,
    trace_resume_read_enabled: bool,
    trace_resume_read_index: usize,
    trace_resume_reads: Vec<TurnonResumeByte>,
    trace_resume_ce1_shadow_enabled: bool,
    trace_resume_ce1_shadow: Vec<u8>,
    trace_reset_ce1_readonly: bool,
    trace_reset_ce6_shadow_enabled: bool,
    trace_reset_ce6_shadow: Vec<u8>,
    trace_reset_ce6_readonly: bool,
    iq7000_clock_seed: Option<Iq7000RtcSeed>,
    iq7000_rtc: Option<Iq7000RtcPeripheral>,
    poisoned: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoKeyKind {
    Matrix(u8),
    Chord { modifier: u8, code: u8 },
    Event(u8),
    InputEvent(u8),
    OnKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeySeqKind {
    Press,
    KeyDown,
    KeyUp,
    WaitOp,
    WaitText,
    WaitPower,
    WaitScreenChange,
    WaitScreenEmpty,
    WaitScreenDraw,
}

#[derive(Clone, Debug)]
struct KeySeqAction {
    kind: KeySeqKind,
    key: Option<AutoKeyKind>,
    label: String,
    hold: u64,
    op_target: u64,
    op_target_set: bool,
    text: String,
    power_on: bool,
    screen_baseline_set: bool,
    screen_baseline_hash: u64,
}

impl KeySeqAction {
    fn new(kind: KeySeqKind) -> Self {
        Self {
            kind,
            key: None,
            label: String::new(),
            hold: 0,
            op_target: 0,
            op_target_set: false,
            text: String::new(),
            power_on: false,
            screen_baseline_set: false,
            screen_baseline_hash: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ScreenState {
    valid: bool,
    is_blank: bool,
    signature: u64,
    text_valid: bool,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeySeqEventKind {
    Press,
    Release,
    Log,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct KeySeqEvent {
    kind: KeySeqEventKind,
    key: Option<AutoKeyKind>,
    label: String,
    op_index: u64,
    hold: u64,
    message: String,
}

struct KeySeqRunner {
    actions: Vec<KeySeqAction>,
    log_enabled: bool,
    active_key: Option<AutoKeyKind>,
    active_label: String,
    active_release_at: u64,
    action_index: usize,
}

impl KeySeqRunner {
    fn new(actions: Vec<KeySeqAction>) -> Self {
        let mut runner = Self {
            actions,
            log_enabled: false,
            active_key: None,
            active_label: String::new(),
            active_release_at: 0,
            action_index: 0,
        };
        runner.reset_state();
        runner
    }

    fn reset(&mut self, actions: Vec<KeySeqAction>) {
        self.actions = actions;
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.active_key = None;
        self.active_label.clear();
        self.active_release_at = 0;
        self.action_index = 0;
        for action in &mut self.actions {
            action.op_target_set = false;
            action.screen_baseline_set = false;
            action.screen_baseline_hash = 0;
        }
    }

    fn set_log_enabled(&mut self, enabled: bool) {
        self.log_enabled = enabled;
    }

    fn is_complete(&self) -> bool {
        self.action_index >= self.actions.len() && self.active_key.is_none()
    }

    fn push_log(log_enabled: bool, events: &mut Vec<KeySeqEvent>, message: String) {
        if !log_enabled {
            return;
        }
        events.push(KeySeqEvent {
            kind: KeySeqEventKind::Log,
            key: None,
            label: String::new(),
            op_index: 0,
            hold: 0,
            message,
        });
    }

    fn step(&mut self, op_index: u64, power_on: bool, screen: &ScreenState) -> Vec<KeySeqEvent> {
        let mut events = Vec::new();
        let log_enabled = self.log_enabled;
        if let Some(active_key) = self.active_key {
            if op_index >= self.active_release_at {
                events.push(KeySeqEvent {
                    kind: KeySeqEventKind::Release,
                    key: Some(active_key),
                    label: self.active_label.clone(),
                    op_index,
                    hold: 0,
                    message: String::new(),
                });
                Self::push_log(
                    log_enabled,
                    &mut events,
                    format!("key-seq: release {} at {}", self.active_label, op_index),
                );
                self.active_key = None;
                self.active_label.clear();
            }
        }

        if self.active_key.is_none() && self.action_index < self.actions.len() {
            let action = &mut self.actions[self.action_index];
            match action.kind {
                KeySeqKind::WaitOp => {
                    if !action.op_target_set {
                        action.op_target = action.op_target.saturating_add(op_index);
                        action.op_target_set = true;
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-op until {}", action.op_target),
                        );
                    }
                    if op_index >= action.op_target {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-op done at {}", op_index),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::WaitText => {
                    if screen.text_valid && screen.text.contains(&action.text) {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-text '{}' at {}", action.text, op_index),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::WaitPower => {
                    if power_on == action.power_on {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!(
                                "key-seq: wait-power {} at {}",
                                if action.power_on { "on" } else { "off" },
                                op_index
                            ),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::WaitScreenChange => {
                    if !screen.valid {
                        return events;
                    }
                    if !action.screen_baseline_set {
                        action.screen_baseline_set = true;
                        action.screen_baseline_hash = screen.signature;
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-screen-change baseline {}", screen.signature),
                        );
                    } else if screen.signature != action.screen_baseline_hash {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-screen-change detected at {}", op_index),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::WaitScreenEmpty => {
                    if screen.valid && screen.is_blank {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-screen-empty at {}", op_index),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::WaitScreenDraw => {
                    if screen.valid && !screen.is_blank {
                        Self::push_log(
                            log_enabled,
                            &mut events,
                            format!("key-seq: wait-screen-draw at {}", op_index),
                        );
                        self.action_index += 1;
                    }
                }
                KeySeqKind::Press => {
                    let key = action.key;
                    events.push(KeySeqEvent {
                        kind: KeySeqEventKind::Press,
                        key,
                        label: action.label.clone(),
                        op_index,
                        hold: action.hold,
                        message: String::new(),
                    });
                    if let Some(key) = key {
                        self.active_key = Some(key);
                        self.active_label = action.label.clone();
                        self.active_release_at = op_index.saturating_add(action.hold);
                    }
                    Self::push_log(
                        log_enabled,
                        &mut events,
                        format!(
                            "key-seq: press {} at {} hold {}",
                            action.label, op_index, action.hold
                        ),
                    );
                    self.action_index += 1;
                }
                KeySeqKind::KeyDown => {
                    let key = action.key;
                    events.push(KeySeqEvent {
                        kind: KeySeqEventKind::Press,
                        key,
                        label: action.label.clone(),
                        op_index,
                        hold: 0,
                        message: String::new(),
                    });
                    Self::push_log(
                        log_enabled,
                        &mut events,
                        format!("key-seq: down {} at {}", action.label, op_index),
                    );
                    self.action_index += 1;
                }
                KeySeqKind::KeyUp => {
                    let key = action.key;
                    events.push(KeySeqEvent {
                        kind: KeySeqEventKind::Release,
                        key,
                        label: action.label.clone(),
                        op_index,
                        hold: 0,
                        message: String::new(),
                    });
                    Self::push_log(
                        log_enabled,
                        &mut events,
                        format!("key-seq: up {} at {}", action.label, op_index),
                    );
                    self.action_index += 1;
                }
            }
        }

        events
    }
}

impl StandaloneBus {
    fn log_perfetto(&self, msg: &str) {
        let _ = msg;
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        memory: MemoryImage,
        lcd: Box<dyn LcdHal>,
        timer: TimerContext,
        log_lcd: bool,
        log_lcd_limit: u32,
        trace_kbd: bool,
        perfetto: Option<IrqPerfetto>,
        host_read: Option<Box<dyn FnMut(u32) -> Option<u8> + Send>>,
        host_write: Option<Box<dyn FnMut(u32, u8) + Send>>,
    ) -> Self {
        Self {
            memory,
            lcd,
            timer,
            cycle_count: 0,
            timer_finalize_clamp: false,
            keyboard: KeyboardMatrix::new(),
            lcd_writes: 0,
            log_lcd,
            log_lcd_count: 0,
            log_lcd_limit,
            irq_pending: false,
            in_interrupt: false,
            delivered_irq_count: 0,
            pending_kil: false,
            pending_onk: false,
            deferred_key_irq: false,
            deferred_pending_kil: false,
            last_kbd_access: None,
            kil_reads: 0,
            rom_koh_reads: 0,
            rom_kol_reads: 0,
            trace_kbd,
            scan_on_timer: true,
            last_pc: 0,
            instr_index: 0,
            vec_patched: false,
            perfetto,
            last_irq_src: None,
            active_irq_mask: 0,
            perfetto_enabled: false,
            host_read,
            host_peek: None,
            host_write,
            bus_trace: None,
            bus_trace_index: 0,
            trace_resume_ssr_onk: false,
            trace_resume_onk_release_cycle: None,
            trace_resume_onk_release_instr: None,
            trace_resume_read_anchor_pc: None,
            trace_resume_read_enabled: false,
            trace_resume_read_index: 0,
            trace_resume_reads: Vec::new(),
            trace_resume_ce1_shadow_enabled: false,
            trace_resume_ce1_shadow: vec![0u8; 0x1_0000],
            trace_reset_ce1_readonly: false,
            trace_reset_ce6_shadow_enabled: false,
            trace_reset_ce6_shadow: vec![0u8; 0x1_0000],
            trace_reset_ce6_readonly: false,
            iq7000_clock_seed: None,
            iq7000_rtc: None,
            poisoned: None,
        }
    }

    fn lcd(&self) -> &dyn LcdHal {
        self.lcd.as_ref()
    }

    fn set_pc(&mut self, pc: u32) {
        self.last_pc = pc & ADDRESS_MASK;
    }

    fn set_instr_index(&mut self, idx: u64) {
        self.instr_index = idx;
    }

    fn set_bus_trace(&mut self, writer: Option<BufWriter<fs::File>>) {
        self.bus_trace = writer;
        self.bus_trace_index = 0;
    }

    fn install_iq7000_clock_seed(&mut self, seed: Iq7000RtcSeed) {
        if let Some(rtc) = self.iq7000_rtc.as_mut() {
            rtc.set_seed(seed.clock.clone());
        } else {
            self.iq7000_rtc = Some(Iq7000RtcPeripheral::new(seed.clock.clone()));
        }
        self.iq7000_clock_seed = Some(seed);
        self.reapply_iq7000_clock_seed();
    }

    fn reapply_iq7000_clock_seed(&mut self) {
        let Some(seed) = self.iq7000_clock_seed.as_ref() else {
            return;
        };
        seed.clock.apply_to_memory(&mut self.memory);
    }

    fn iq7000_clock_workspace_read(&self, addr: u32, bits: u8) -> Option<u32> {
        self.iq7000_clock_seed
            .as_ref()
            .and_then(|seed| seed.clock.read(addr, bits))
    }

    fn finish_bus_trace(&mut self) {
        if let Some(writer) = self.bus_trace.as_mut() {
            if let Err(err) = writer.flush() {
                eprintln!("warning: failed to flush bus trace: {err}");
            }
        }
    }

    fn ssr_onk_visible(&self) -> bool {
        self.pending_onk || self.trace_resume_ssr_onk
    }

    fn enable_trace_resume_onk(&mut self, release_cycle: u64, release_instr: u64) {
        self.trace_resume_ssr_onk = true;
        self.trace_resume_onk_release_cycle = Some(release_cycle);
        self.trace_resume_onk_release_instr = Some(release_instr);
    }

    fn install_trace_resume_reads(&mut self, anchor_pc: u32, reads: Vec<TurnonResumeByte>) {
        self.trace_resume_read_anchor_pc = Some(anchor_pc & ADDRESS_MASK);
        self.trace_resume_read_enabled = false;
        self.trace_resume_read_index = 0;
        self.trace_resume_reads = reads;
    }

    fn enable_trace_resume_ce1_shadow(&mut self) {
        self.trace_resume_ce1_shadow_enabled = true;
        self.trace_resume_ce1_shadow.fill(0);
        self.trace_reset_ce1_readonly = false;
    }

    fn enable_reset_trace_card(&mut self) {
        self.trace_resume_ce1_shadow_enabled = true;
        self.trace_resume_ce1_shadow.fill(0);
        self.trace_reset_ce1_readonly = false;
        self.trace_reset_ce6_shadow_enabled = true;
        self.trace_reset_ce6_shadow.fill(0);
        self.trace_reset_ce6_readonly = true;
    }

    fn maybe_enable_trace_resume_reads(&mut self) {
        if self.trace_resume_read_enabled {
            return;
        }
        if let Some(anchor_pc) = self.trace_resume_read_anchor_pc {
            if self.last_pc == anchor_pc {
                self.trace_resume_read_enabled = true;
            }
        }
    }

    fn maybe_trace_resume_read(&mut self, addr: u32, bits: u8) -> Option<u32> {
        self.maybe_enable_trace_resume_reads();
        if !self.trace_resume_read_enabled {
            return None;
        }
        let width_bytes = usize::from(bits.div_ceil(8));
        if width_bytes == 0 {
            return None;
        }
        let search_end = self
            .trace_resume_reads
            .len()
            .min(self.trace_resume_read_index.saturating_add(128));
        for start in self.trace_resume_read_index..search_end {
            if start + width_bytes > self.trace_resume_reads.len() {
                break;
            }
            let mut value = 0u32;
            let mut matched = true;
            for idx in 0..width_bytes {
                let byte = &self.trace_resume_reads[start + idx];
                if byte.addr != (addr.wrapping_add(idx as u32) & ADDRESS_MASK) {
                    matched = false;
                    break;
                }
                value |= (byte.value as u32) << (idx * 8);
            }
            if !matched {
                continue;
            }
            for idx in 0..width_bytes {
                let byte = &self.trace_resume_reads[start + idx];
                self.memory.write_external_byte(byte.addr, byte.value);
            }
            self.trace_resume_read_index = start + width_bytes;
            return Some(value & mask_bits(bits));
        }
        None
    }

    fn update_trace_resume_state(&mut self) {
        if !self.trace_resume_ssr_onk {
            return;
        }
        let cycle_ready = self
            .trace_resume_onk_release_cycle
            .is_some_and(|release_cycle| self.cycle_count >= release_cycle);
        let instr_ready = self
            .trace_resume_onk_release_instr
            .is_some_and(|release_instr| self.instr_index >= release_instr);
        if !cycle_ready && !instr_ready {
            return;
        }
        self.trace_resume_ssr_onk = false;
        self.trace_resume_onk_release_cycle = None;
        self.trace_resume_onk_release_instr = None;
    }

    /// Classify one instruction byte using the same PC-sensitive replay
    /// context as the preceding silent peek. A vector destination is decoded
    /// in its own context, which may differ from `last_pc` at the source.
    fn instruction_byte_is_stable_for_context(&self, addr: u32, context_pc: u32) -> bool {
        let addr = addr & ADDRESS_MASK;
        let context_pc = context_pc & ADDRESS_MASK;
        let trace_resume_active =
            self.trace_resume_read_enabled || self.trace_resume_read_anchor_pc == Some(context_pc);
        let trace_resume_overrides_addr = trace_resume_active
            && self
                .trace_resume_reads
                .get(
                    self.trace_resume_read_index
                        ..self
                            .trace_resume_reads
                            .len()
                            .min(self.trace_resume_read_index.saturating_add(128)),
                )
                .is_some_and(|bytes| bytes.iter().any(|byte| byte.addr == addr));
        if self.iq7000_clock_workspace_read(addr, 8).is_some()
            || self.lcd.handles(addr)
            || (self.trace_reset_ce6_shadow_enabled && (0x010000..=0x01FFFF).contains(&addr))
            || (self.trace_resume_ce1_shadow_enabled && (0x040000..=0x04FFFF).contains(&addr))
            || trace_resume_overrides_addr
        {
            return false;
        }
        self.memory.instruction_byte_is_stable(addr)
    }

    fn bus_trace_region(addr: u32) -> Option<&'static str> {
        let addr = addr & ADDRESS_MASK;
        if (0x2000..=0x200F).contains(&addr) {
            return Some("lcd_primary");
        }
        if (0xA000..=0xA00F).contains(&addr) {
            return Some("lcd_mirror");
        }
        if (0x010000..=0x01FFFF).contains(&addr) {
            return Some("ce6_rom");
        }
        if (0x040000..=0x07FFFF).contains(&addr) {
            return Some("ce1_slot");
        }
        if (0x080000..=0x0BFFFF).contains(&addr) {
            return Some("system_ram");
        }
        if (0x0C0000..=0x0FFFFF).contains(&addr) {
            return Some("main_rom");
        }
        if (0x000000..=0x03FFFF).contains(&addr) {
            return Some("low_rom");
        }
        None
    }

    fn trace_bus_access(&mut self, kind: &'static str, addr: u32, bits: u8, value: u32) {
        if bits == 0 || MemoryImage::is_internal(addr) {
            return;
        }
        let Some(_) = self.bus_trace.as_ref() else {
            return;
        };

        let byte_count = bits.div_ceil(8).max(1);
        let mut write_error: Option<String> = None;
        for offset in 0..byte_count {
            let byte_addr = addr.wrapping_add(offset as u32) & ADDRESS_MASK;
            let Some(region) = Self::bus_trace_region(byte_addr) else {
                continue;
            };
            let event = BusTraceEvent {
                index: self.bus_trace_index,
                kind,
                region,
                addr: byte_addr,
                value: ((value >> (offset * 8)) & 0xFF) as u8,
                bits,
                byte_offset: offset,
                pc: self.last_pc,
                instr_index: self.instr_index,
                cycle: self.cycle_count,
            };
            let line = match serde_json::to_string(&event) {
                Ok(line) => line,
                Err(err) => {
                    write_error = Some(err.to_string());
                    break;
                }
            };
            if let Some(writer) = self.bus_trace.as_mut() {
                if let Err(err) = writer
                    .write_all(line.as_bytes())
                    .and_then(|_| writer.write_all(b"\n"))
                {
                    write_error = Some(err.to_string());
                    break;
                }
            }
            self.bus_trace_index = self.bus_trace_index.saturating_add(1);
        }
        if let Some(err) = write_error {
            eprintln!("warning: disabling bus trace after write failure: {err}");
            self.bus_trace = None;
        }
    }

    fn trace_kbd_access(&self, kind: &str, addr: u32, offset: u32, bits: u8, value: u32) {
        if !self.trace_kbd {
            return;
        }
        println!(
            "[kbd-trace-{kind}] pc=0x{pc:05X} addr=0x{addr:05X} offset=0x{offset:02X} bits={bits} value=0x{val:08X}",
            pc = self.last_pc,
            addr = addr,
            offset = offset,
            bits = bits,
            val = value & mask_bits(bits),
        );
    }

    fn trace_imem_access(&self, kind: &str, addr: u32, bits: u8, value: u32) {
        if !self.trace_kbd {
            return;
        }
        if let Some(offset) = MemoryImage::internal_offset(addr) {
            println!(
                "[imem-trace-{kind}] pc=0x{pc:05X} addr=0x{addr:05X} offset=0x{offset:02X} bits={bits} value=0x{val:08X}",
                pc = self.last_pc,
                addr = addr,
                offset = offset,
                bits = bits,
                val = value & mask_bits(bits),
            );
        }
    }

    fn keyboard_fifo_addresses(&self) -> Option<(u32, u32, u32)> {
        let read = |address| self.memory.read_byte_silent(address).map(u32::from);
        let workspace_base = read(PCE500_IOCS_WS_PTR_ADDR)?
            | (read(PCE500_IOCS_WS_PTR_ADDR + 1)? << 8)
            | (read(PCE500_IOCS_WS_PTR_ADDR + 2)? << 16);
        if workspace_base == 0 {
            return None;
        }
        let fifo_offset = read(workspace_base + PCE500_KEY_FIFO_BASE_OFFSET)?
            | (read(workspace_base + PCE500_KEY_FIFO_BASE_OFFSET + 1)? << 8);
        Some((
            (workspace_base + fifo_offset) & ADDRESS_MASK,
            (workspace_base + PCE500_KEY_FIFO_TAIL_OFFSET) & ADDRESS_MASK,
            (workspace_base + PCE500_KEY_FIFO_HEAD_OFFSET) & ADDRESS_MASK,
        ))
    }

    fn is_keyboard_fifo_address(&self, addr: u32) -> bool {
        let Some((fifo_base, fifo_tail, fifo_head)) = self.keyboard_fifo_addresses() else {
            return false;
        };
        addr == fifo_tail
            || addr == fifo_head
            || (fifo_base..fifo_base + PCE500_KEY_FIFO_CAPACITY).contains(&addr)
    }

    fn trace_fifo_access(&self, kind: &str, addr: u32, bits: u8, value: u32) {
        if !self.trace_kbd {
            return;
        }
        if !self.is_keyboard_fifo_address(addr) {
            return;
        }
        println!(
            "[fifo-trace-{kind}] pc=0x{pc:05X} addr=0x{addr:06X} bits={bits} value=0x{val:08X}",
            pc = self.last_pc,
            addr = addr,
            bits = bits,
            val = value & mask_bits(bits)
        );
    }

    fn trace_mem_write(&self, addr: u32, bits: u8, value: u32) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let space = if MemoryImage::is_internal(addr) {
                "internal"
            } else {
                "external"
            };
            let substep = perfetto_next_substep();
            tracer.record_mem_write_with_substep(
                self.instr_index,
                self.last_pc,
                addr & ADDRESS_MASK,
                value & mask_bits(bits),
                space,
                bits,
                substep,
            );
        });
    }

    /// Parity: leave vectors to the ROM; no patching.
    fn maybe_patch_vectors(&mut self) {
        self.vec_patched = true;
    }

    fn tick_keyboard(&mut self) {
        // Scan only on timer cadence. Events feed host/ROM FIFO bookkeeping;
        // raw KEYI is sampled separately from selected physical KIL.
        let events = self.keyboard.scan_tick(&mut self.memory, true);
        let fifo_pending = self.keyboard.fifo_len() > 0;
        let pending = events > 0 || fifo_pending;
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        if events > 0 || (kb_irq_enabled && fifo_pending) {
            self.deferred_key_irq = true;
            self.deferred_pending_kil = pending;
            self.last_kbd_access = Some("scan".to_string());
            self.log_irq_event(
                "KeyScan",
                Some("KEY"),
                [
                    (
                        "isr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                    (
                        "imr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                    ("pc", AnnotationValue::Pointer(self.last_pc as u64)),
                ],
            );
        }
    }

    fn apply_deferred_key_irq(&mut self) {
        if !self.deferred_key_irq {
            return;
        }
        self.deferred_key_irq = false;
        let pending = self.deferred_pending_kil;
        self.deferred_pending_kil = false;
        if !pending {
            return;
        }
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let drained = self
            .keyboard
            .drain_fifo_to_pce500_iocs_workspace(&mut self.memory, kb_irq_enabled);
        if drained == 0 {
            self.keyboard
                .write_fifo_to_memory(&mut self.memory, kb_irq_enabled);
        }
        self.pending_kil = true;
        self.timer.key_irq_latched = true;
    }

    fn press_key(&mut self, code: u8) {
        // Auto-key presses update physical matrix state. Timer-driven scans
        // handle host FIFO timing; CPU boundaries sample raw KIL/KEYI.
        self.keyboard.press_matrix_code(code, &mut self.memory);
    }

    fn release_key(&mut self, code: u8) {
        // Parity: release updates the matrix state; scan_tick determines when to emit FIFO events.
        self.keyboard.release_matrix_code(code, &mut self.memory);
    }

    fn inject_input_event(&mut self, code: u8) {
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let events = self
            .keyboard
            .inject_input_event(code, &mut self.memory, kb_irq_enabled);
        if events > 0 {
            self.pending_kil = true;
            self.timer.key_irq_latched = true;
        }
    }

    fn inject_matrix_event(&mut self, code: u8, release: bool) {
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let events =
            self.keyboard
                .inject_matrix_event(code, release, &mut self.memory, kb_irq_enabled);
        if events > 0 {
            self.pending_kil = true;
            self.timer.key_irq_latched = true;
        }
    }

    fn press_on_key(&mut self) {
        // ON key is not part of the matrix; assert ONK input and pending IRQ.
        let ssr = self.memory.read_internal_byte(0xFF).unwrap_or(0);
        let new_ssr = ssr | SSR_ONK;
        self.memory.write_internal_byte(0xFF, new_ssr);
        if let Some(isr) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
            if (isr & ISR_ONKI) == 0 {
                self.memory
                    .write_internal_byte(IMEM_ISR_OFFSET, isr | ISR_ONKI);
            }
        }
        self.pending_onk = true;
        self.irq_pending = true;
        if !self.in_interrupt {
            self.last_irq_src = Some("ONK".to_string());
        }
    }

    fn clear_on_key(&mut self) {
        let ssr = self.memory.read_internal_byte(0xFF).unwrap_or(0);
        let new_ssr = ssr & !SSR_ONK;
        self.memory.write_internal_byte(0xFF, new_ssr);
        self.pending_onk = false;
    }

    fn log_irq_event<'a>(
        &mut self,
        name: &str,
        src: Option<&str>,
        annotations: impl IntoIterator<Item = (&'a str, AnnotationValue)>,
    ) {
        if let Some(tracer) = self.perfetto.as_mut() {
            tracer.instant(name, src, self.cycle_count, annotations);
        }
    }

    fn log_imem_write(&mut self, offset: u32, prev: u8, new: u8) {
        if !matches!(offset, IMEM_IMR_OFFSET | IMEM_ISR_OFFSET) {
            return;
        }
        let reg = if offset == IMEM_IMR_OFFSET {
            "IMR"
        } else {
            "ISR"
        };
        let mut src_hint: Option<&str> = None;
        if reg == "ISR" && (new & ISR_KEYI) != 0 {
            src_hint = Some("KEY");
        }
        self.log_irq_event(
            "IMEM_Write",
            src_hint,
            [
                ("reg", AnnotationValue::Str(reg.to_string())),
                ("prev", AnnotationValue::UInt(prev as u64)),
                ("value", AnnotationValue::UInt(new as u64)),
                ("pc", AnnotationValue::Pointer(self.last_pc as u64)),
            ],
        );
        if reg == "ISR" && (new & ISR_KEYI) != 0 {
            self.log_irq_event(
                "KEYI_Set",
                Some("KEY"),
                [
                    ("pc", AnnotationValue::Pointer(self.last_pc as u64)),
                    ("prev", AnnotationValue::UInt(prev as u64)),
                    ("value", AnnotationValue::UInt(new as u64)),
                    (
                        "imr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                ],
            );
        }
    }

    fn irq_pending(&mut self) -> bool {
        let mut isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        let in_interrupt = self.in_interrupt;
        let raw_kil = if self
            .memory
            .read_internal_byte_silent(IMEM_LCC_OFFSET)
            .unwrap_or(0)
            & 0x04
            == 0
        {
            self.keyboard.compute_physical_kil()
        } else {
            0
        };
        // KEYI follows the selected, undebounced physical KIL level. This
        // remains active while servicing an interrupt; only nested delivery
        // is deferred below.
        if raw_kil != 0 && (isr & ISR_KEYI) == 0 {
            self.memory
                .write_internal_byte(IMEM_ISR_OFFSET, isr | ISR_KEYI);
            isr |= ISR_KEYI;
            self.irq_pending = true;
            if !self.in_interrupt && !matches!(self.last_irq_src.as_deref(), Some("KEY" | "ONK")) {
                self.last_irq_src = Some("KEY".to_string());
            }
        }
        // ONK is level-triggered like KEYI; if latched and cleared while masked, reassert.
        if self.pending_onk && (isr & ISR_ONKI) == 0 {
            self.memory
                .write_internal_byte(IMEM_ISR_OFFSET, isr | ISR_ONKI);
            isr |= ISR_ONKI;
            self.irq_pending = true;
            if !self.in_interrupt {
                self.last_irq_src = Some("ONK".to_string());
            }
        }
        // Track the highest-priority pending low source in the ROM's order.
        let pending_src = if (isr & ISR_RXI) != 0 {
            Some("RX")
        } else if (isr & ISR_EXI) != 0 {
            Some("EX")
        } else if (isr & ISR_TXI) != 0 {
            Some("TX")
        } else if (isr & ISR_ONKI) != 0 {
            Some("ONK")
        } else if (isr & ISR_KEYI) != 0 {
            Some("KEY")
        } else if (isr & ISR_STI) != 0 {
            Some("STI")
        } else if (isr & ISR_MTI) != 0 {
            Some("MTI")
        } else {
            None
        };
        if let Some(src) = pending_src {
            self.last_irq_src = Some(src.to_string());
        }
        if isr != 0 {
            self.irq_pending = true;
        }
        if in_interrupt {
            // Avoid nested IRQs; RETI/RETF clears the in_interrupt latch.
            return false;
        }
        // Gate delivery on IMR master + source masks.
        let irm_enabled = (imr & IMR_MASTER) != 0;
        // Match Python gating: attempt delivery only when a pending IRQ is latched and
        // IRM is enabled and (IMR & ISR) != 0.
        self.irq_pending && irm_enabled && (imr & isr) != 0
    }

    #[cfg(feature = "llama-tests")]
    #[allow(dead_code)]
    fn trace_kio(&self, pc: u32, offset: u8, value: u8) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            tracer.record_kio_read(Some(pc), offset, value, None);
        });
    }

    fn log_irq_delivery(&mut self, _src: Option<&str>, _vec: u32, _imr: u8, _isr: u8, _pc: u32) {}

    fn deliver_irq(&mut self, state: &mut LlamaState) -> Result<(), &'static str> {
        if self.poisoned.is_some() {
            return Err("standalone runtime is poisoned; power-on reset required");
        }
        // Mirror the IR intrinsic: push PC, F, IMR, clear IRM, jump to vector.
        fn push_stack(
            memory: &mut MemoryImage,
            state: &mut LlamaState,
            reg: RegName,
            value: u32,
            bits: u8,
        ) {
            let bytes = bits.div_ceil(8);
            let mask = mask_for(reg);
            let mut sp = state.get_reg(reg) & mask;
            for i in (0..bytes).rev() {
                sp = sp.wrapping_sub(1) & mask;
                let byte = ((value >> (8 * i)) & 0xFF) as u8;
                let _ = memory.store(sp, 8, byte as u32);
            }
            state.set_reg(reg, sp);
        }

        let pc = state.pc() & ADDRESS_MASK;
        let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
        let imr = (self.memory.load(imr_addr, 8).unwrap_or(0) & 0xFF) as u8;
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        // Deliver highest-priority pending respecting masks.
        let (src, mask) = if (isr & ISR_RXI != 0) && (imr & IMR_RX) != 0 {
            (Some("RX"), ISR_RXI)
        } else if (isr & ISR_EXI != 0) && (imr & IMR_EX) != 0 {
            (Some("EX"), ISR_EXI)
        } else if (isr & ISR_TXI != 0) && (imr & IMR_TX) != 0 {
            (Some("TX"), ISR_TXI)
        } else if (isr & ISR_ONKI != 0) && (imr & IMR_ONK) != 0 {
            (Some("ONK"), ISR_ONKI)
        } else if (isr & ISR_KEYI != 0) && (imr & IMR_KEY) != 0 {
            (Some("KEY"), ISR_KEYI)
        } else if (isr & ISR_STI != 0) && (imr & IMR_STI) != 0 {
            (Some("STI"), ISR_STI)
        } else if (isr & ISR_MTI != 0) && (imr & IMR_MTI) != 0 {
            (Some("MTI"), ISR_MTI)
        } else {
            return Ok(());
        };

        let vector_transfer = prepare_validated_vector(INTERRUPT_VECTOR_ADDR, state, self)?;
        self.maybe_patch_vectors();

        // Hardware writes the complete frame before the one low-to-high
        // architectural vector fetch. Silent validation above is deliberately
        // unobservable and does not substitute for those reads.
        push_stack(&mut self.memory, state, RegName::S, pc, 24);
        let f = state.get_reg(RegName::F) & 0xFF;
        push_stack(&mut self.memory, state, RegName::S, f, 8);
        push_stack(&mut self.memory, state, RegName::S, imr as u32, 8);
        let cleared_imr = imr & 0x7F;
        let _ = self.memory.store(imr_addr, 8, cleared_imr as u32);
        state.set_reg(RegName::IMR, u32::from(cleared_imr));

        let vector_result =
            vector_transfer.consume_after_architectural_fetch(INTERRUPT_VECTOR_ADDR, state, self);
        let vec = match vector_result {
            Ok(vec) => vec,
            Err(error) => {
                self.poisoned = Some(error.to_string());
                return Err(error);
            }
        };

        state.set_pc(vec);
        state.set_halted(false);
        self.in_interrupt = true;
        self.irq_pending = false;
        self.last_irq_src = src.map(|s| s.to_string());
        self.active_irq_mask = mask;
        let src_clone = self.last_irq_src.clone();
        self.log_irq_delivery(src_clone.as_deref(), vec, imr, isr, pc);
        self.log_irq_event(
            "IRQ_Enter",
            src_clone.as_deref(),
            [
                ("from", AnnotationValue::Pointer(pc as u64)),
                (
                    "vector",
                    AnnotationValue::Pointer((vec & ADDRESS_MASK) as u64),
                ),
                ("imr_before", AnnotationValue::UInt(imr as u64)),
                ("imr_after", AnnotationValue::UInt(cleared_imr as u64)),
                ("isr", AnnotationValue::UInt(isr as u64)),
            ],
        );
        self.delivered_irq_count = self.delivered_irq_count.wrapping_add(1);
        if self.log_lcd && self.log_lcd_count < 50 {
            println!(
                "[irq] delivered: vec=0x{vec:05X} imr=0x{imr:02X} pc_prev=0x{pc:05X}",
                vec = vec & ADDRESS_MASK,
                imr = imr,
                pc = pc
            );
        }
        Ok(())
    }

    fn handle_irq_return(&mut self, opcode: u8, state: &LlamaState) {
        if opcode == 0x01 {
            // RETI completes interrupt service (RETF returns to the epilogue only).
            let last_src = self.last_irq_src.clone();
            self.log_irq_event(
                "IRQ_Return",
                last_src.as_deref(),
                [
                    ("pc", AnnotationValue::Pointer(state.pc() as u64)),
                    (
                        "imr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                    (
                        "isr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                ],
            );
            self.in_interrupt = false;
            self.active_irq_mask = 0;
            self.last_irq_src = None;
        }
    }

    fn strobe_all_columns(&mut self) {
        let _ = self
            .keyboard
            .handle_write(IMEM_KOL_OFFSET, 0xFF, &mut self.memory);
        let _ = self
            .keyboard
            .handle_write(IMEM_KOH_OFFSET, 0x07, &mut self.memory);
    }

    fn finish_perfetto(&mut self) {
        self.log_perfetto("finishing perfetto traces");
        if let Some(tracer) = self.perfetto.take() {
            match tracer.finish() {
                Ok(_path) => {}
                Err(err) => eprintln!("[perfetto] failed to save IRQ trace: {err}"),
            }
        }
        // Flush the global instruction trace if present.
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.take() {
            if let Err(err) = tracer.finish() {
                eprintln!("[perfetto] failed to save instruction trace: {err}");
            }
        }
    }

    fn tick_timers_only(&mut self, cycle: u64) {
        let kb_irq_enabled = self.timer.kb_irq_enabled;
        let scan_on_timer = self.scan_on_timer;
        let mut pending_kil = false;
        let (mti, sti, key_events, _kb_stats) = self.timer.tick_timers_with_keyboard(
            &mut self.memory,
            cycle,
            |mem| {
                if !scan_on_timer {
                    return (0, false, Some(self.keyboard.telemetry()));
                }
                // Parity: always count/key-latch events even when IRQs are masked.
                let events = self.keyboard.scan_tick(mem, true);
                let fifo_pending = self.keyboard.fifo_len() > 0;
                pending_kil = events > 0 || fifo_pending;
                if events > 0 || (kb_irq_enabled && fifo_pending) {
                    let drained = self
                        .keyboard
                        .drain_fifo_to_pce500_iocs_workspace(mem, kb_irq_enabled);
                    if drained == 0 {
                        self.keyboard.write_fifo_to_memory(mem, kb_irq_enabled);
                    }
                }
                (events, pending_kil, Some(self.keyboard.telemetry()))
            },
            None,
            Some(self.last_pc),
        );
        if mti {
            self.irq_pending = true;
            self.last_irq_src = Some("MTI".to_string());
        }
        if sti {
            self.irq_pending = true;
            self.last_irq_src = Some("STI".to_string());
        }
        if scan_on_timer && mti && key_events > 0 {
            self.pending_kil = pending_kil;
            if self.pending_kil {
                self.timer.key_irq_latched = true;
            }
            self.last_kbd_access = Some("scan".to_string());
            self.log_irq_event(
                "KeyScan",
                Some("KEY"),
                [
                    (
                        "isr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                    (
                        "imr",
                        AnnotationValue::UInt(
                            self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u64,
                        ),
                    ),
                    ("pc", AnnotationValue::Pointer(self.last_pc as u64)),
                ],
            );
        }
        // Host FIFO occupancy is not an electrical KEYI source. Raw selected
        // KIL is sampled by refresh_raw_key_irq().
    }

    fn advance_cycle(&mut self) {
        self.advance_cycles(1);
    }

    fn advance_cycles(&mut self, cycles: u64) {
        let end_cycle = self.cycle_count.wrapping_add(cycles);
        while let Some(fire_cycle) = self
            .timer
            .next_fire_cycle_in_span(self.cycle_count, end_cycle)
        {
            self.cycle_count = fire_cycle;
            self.tick_timers_only(fire_cycle);
        }
        self.cycle_count = end_cycle;
    }

    fn finalize_instruction(&mut self) {
        self.timer
            .finalize_instruction_with_clamp(self.cycle_count, self.timer_finalize_clamp);
        if !self.scan_on_timer {
            self.tick_keyboard();
        }
        self.update_trace_resume_state();
    }
}

fn mask_bits(bits: u8) -> u32 {
    if bits == 0 || bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn reject_unrepresented_snapshot_runtime(bus: &StandaloneBus) -> Result<(), Box<dyn Error>> {
    bus.memory.validate_snapshot_overlay_contract()?;
    let mut active = Vec::new();
    if bus.poisoned.is_some() {
        active.push("poisoned fail-stop runtime state");
    }
    if bus.host_read.is_some() || bus.host_peek.is_some() || bus.host_write.is_some() {
        active.push("host callbacks and their external state");
    }
    if bus.iq7000_clock_seed.is_some() || bus.iq7000_rtc.is_some() {
        active.push("IQ-7000 RTC protocol state");
    }
    if bus.deferred_key_irq || bus.deferred_pending_kil {
        active.push("deferred keyboard interrupt state");
    }
    if bus.trace_resume_ssr_onk
        || bus.trace_resume_onk_release_cycle.is_some()
        || bus.trace_resume_onk_release_instr.is_some()
        || bus.trace_resume_read_anchor_pc.is_some()
        || bus.trace_resume_read_enabled
        || !bus.trace_resume_reads.is_empty()
        || bus.trace_resume_ce1_shadow_enabled
        || bus.trace_reset_ce6_shadow_enabled
    {
        active.push("trace-resume device state");
    }
    if bus.perfetto.is_some() || bus.bus_trace.is_some() {
        active.push("active trace output state");
    }
    if active.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "snapshot v4 cannot exactly represent active {}",
            active.join(", ")
        )
        .into())
    }
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn load_snapshot_state(
    path: &Path,
    bus: &mut StandaloneBus,
    state: &mut LlamaState,
    model: DeviceModel,
    rom_bytes: &[u8],
) -> Result<SnapshotMetadata, Box<dyn Error>> {
    reject_unrepresented_snapshot_runtime(bus)?;
    let loaded = snapshot::load_snapshot(path)?;
    let metadata = loaded.metadata.clone();
    if metadata.device_model.is_some_and(|saved| saved != model) {
        return Err("snapshot device model does not match the requested machine".into());
    }

    let active_fallback_ranges =
        snapshot::canonical_snapshot_ranges("active fallback", bus.memory.python_ranges())?;
    if active_fallback_ranges != metadata.fallback_ranges {
        return Err("snapshot fallback ranges do not match the active machine".into());
    }
    let active_readonly_ranges =
        snapshot::canonical_snapshot_ranges("active readonly", bus.memory.readonly_ranges())?;
    if active_readonly_ranges != metadata.readonly_ranges {
        return Err("snapshot read-only ranges do not match the active machine".into());
    }
    for (start, end) in &active_readonly_ranges {
        let unchanged = if *end < sc62015_core::EXTERNAL_SPACE as u32 {
            let start = *start as usize;
            let end = *end as usize + 1;
            bus.memory.external_slice()[start..end] == loaded.external_memory[start..end]
        } else if *start >= INTERNAL_MEMORY_START
            && *end < INTERNAL_MEMORY_START + sc62015_core::INTERNAL_SPACE as u32
        {
            let start = (*start - INTERNAL_MEMORY_START) as usize;
            let end = (*end - INTERNAL_MEMORY_START) as usize + 1;
            bus.memory.internal_slice()[start..end] == loaded.imem[start..end]
        } else {
            false
        };
        if !unchanged {
            return Err("snapshot attempts to replace active read-only memory".into());
        }
    }

    let memory_card_candidate = bus
        .memory
        .prepare_memory_card_restore(loaded.memory_card.clone())?;

    let mut state_candidate = LlamaState::new();
    apply_registers(&mut state_candidate, &loaded.registers)?;
    for index in 0..NUM_TEMP_REGISTERS {
        let value = metadata
            .temps
            .get(&index.to_string())
            .copied()
            .ok_or_else(|| format!("snapshot is missing TEMP{index}"))?;
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

    let mut timer_candidate = bus.timer.clone();
    timer_candidate.apply_snapshot_info(
        &metadata.timer,
        &metadata.interrupts,
        metadata.cycle_count,
    );

    let kb_meta = metadata
        .keyboard
        .as_ref()
        .ok_or("snapshot is missing keyboard state")?;
    let kb_snapshot: KeyboardSnapshot = serde_json::from_value(kb_meta.clone())?;
    let mut keyboard_candidate = KeyboardMatrix::new();
    keyboard_candidate
        .load_snapshot_state(&kb_snapshot)
        .map_err(|error| format!("invalid keyboard snapshot: {error}"))?;
    if serde_json::to_value(keyboard_candidate.snapshot_state())? != *kb_meta {
        return Err("keyboard snapshot is not exactly representable".into());
    }

    let lcd_meta = metadata
        .lcd
        .as_ref()
        .ok_or("snapshot is missing LCD state")?;
    let kind = lcd_kind_from_snapshot_meta(lcd_meta, model.lcd_kind());
    let mut lcd_candidate = create_lcd(kind);
    sc62015_core::device::configure_lcd_char_tracing(lcd_candidate.as_mut(), model, rom_bytes);
    lcd_candidate
        .load_snapshot(lcd_meta, loaded.lcd_payload.as_deref().unwrap_or(&[]))
        .map_err(|error| format!("invalid LCD snapshot: {error}"))?;
    let (restored_lcd_meta, restored_lcd_payload) = lcd_candidate.export_snapshot();
    if restored_lcd_meta != *lcd_meta
        || restored_lcd_payload.as_slice() != loaded.lcd_payload.as_deref().unwrap_or(&[])
    {
        return Err("LCD snapshot is not exactly representable".into());
    }

    // All parsing and candidate construction is complete. Commit the exact
    // machine image without fallible best-effort restoration.
    bus.memory
        .commit_memory_card_restore(memory_card_candidate)?;
    bus.memory
        .copy_external_from(&loaded.external_memory)
        .expect("validated exact external memory length");
    bus.memory.write_imem(&loaded.imem);
    bus.memory
        .set_python_ranges(metadata.fallback_ranges.clone());
    bus.memory
        .set_readonly_ranges(metadata.readonly_ranges.clone());
    bus.memory.clear_dirty();
    bus.memory
        .set_memory_counts(metadata.memory_reads, metadata.memory_writes);
    *state = state_candidate;
    bus.timer = timer_candidate;
    bus.keyboard = keyboard_candidate;
    bus.lcd = lcd_candidate;
    bus.cycle_count = metadata.cycle_count;
    bus.instr_index = metadata.instruction_count;
    bus.last_pc = metadata.pc;
    bus.irq_pending = metadata.interrupts.pending;
    bus.in_interrupt = metadata.interrupts.in_interrupt;
    bus.last_irq_src = metadata.interrupts.source.clone();
    bus.active_irq_mask = metadata
        .interrupts
        .delivered_masks
        .last()
        .copied()
        .unwrap_or(0);
    bus.pending_onk = metadata.onk_level;
    bus.pending_kil = kb_snapshot.fifo_len > 0;
    bus.deferred_key_irq = false;
    bus.deferred_pending_kil = false;
    bus.lcd_writes = 0;
    bus.vec_patched = true;

    Ok(metadata)
}

fn apply_turnon_resume_profile(
    bus: &mut StandaloneBus,
    state: &mut LlamaState,
    profile: &TurnonResumeProfile,
) {
    seed_pce500_bootstrap_imem(&mut bus.memory);
    bus.memory.write_internal_byte(IMEM_UCR_OFFSET, 0x00);
    bus.memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);
    bus.memory.write_internal_byte(IMEM_SCR_OFFSET, 0x00);
    bus.memory.write_internal_byte(IMEM_USR_OFFSET, profile.usr);
    bus.memory
        .write_internal_byte(IMEM_SSR_OFFSET, profile.ssr_base);

    for byte in &profile.visible_bytes {
        bus.memory.write_external_byte(byte.addr, byte.value);
    }

    for (offset, value) in [
        (0xE6u32, (profile.iocs_workspace & 0xFF) as u8),
        (0xE7u32, ((profile.iocs_workspace >> 8) & 0xFF) as u8),
        (0xE8u32, ((profile.iocs_workspace >> 16) & 0xFF) as u8),
    ] {
        bus.memory.write_internal_byte(offset, value);
    }

    // The post-OFF trace enters the stage-05 global-init helper at 0x0E0359, which
    // uses the IOCS logic-register block at D4..D6 to form the CE1 probe pointer for
    // sub_e043e. The raw Saleae profile does not uniquely resolve the earlier read at
    // 0x0BE000, but the later CE1 accesses prove this scratch pointer must be 0x040000.
    bus.memory.write_external_byte(0x0BE000, 0x00);
    bus.memory.write_external_byte(0x0BE001, 0x00);
    bus.memory.write_internal_byte(0x00, 0x00);
    bus.memory.write_internal_byte(0x01, 0x00);
    bus.memory.write_internal_byte(0x02, 0x04);
    bus.memory.write_internal_byte(0xD6, 0x04);

    state.set_reg(RegName::A, 0);
    state.set_reg(RegName::B, 0);
    state.set_reg(RegName::X, 0);
    state.set_reg(RegName::Y, 0);
    state.set_reg(RegName::U, profile.user_stack & ADDRESS_MASK);
    state.set_reg(RegName::S, profile.system_stack & ADDRESS_MASK);
    state.set_reg(
        RegName::IMR,
        bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u32,
    );
    state.record_off_transition(profile.resume_pc.wrapping_sub(1) & ADDRESS_MASK);
    state.set_pc(profile.resume_pc & ADDRESS_MASK);
    state.set_power_state(PowerState::Running);
    state.clear_call_page_stack();

    bus.cycle_count = 0;
    bus.pending_onk = false;
    bus.irq_pending = false;
    bus.last_irq_src = None;
    bus.timer.irq_pending = false;
    bus.timer.irq_source = None;
    bus.timer.last_fired = None;
    bus.timer.irq_isr = 0;
    bus.enable_trace_resume_onk(profile.onk_release_cycle, profile.onk_release_instr);
    bus.install_trace_resume_reads(profile.trace_read_anchor_pc, profile.trace_reads.clone());
    bus.enable_trace_resume_ce1_shadow();
}

fn apply_turnon_resume_target_lcd(
    bus: &mut StandaloneBus,
    profile: &TurnonResumeProfile,
) -> Result<bool, String> {
    apply_lcd_snapshot(
        bus,
        profile.target_lcd_meta.as_ref(),
        &profile.target_lcd_payload,
    )
}

fn apply_lcd_snapshot(
    bus: &mut StandaloneBus,
    meta: Option<&Value>,
    payload: &[u8],
) -> Result<bool, String> {
    let Some(meta) = meta else {
        return Ok(false);
    };
    if payload.is_empty() {
        return Ok(false);
    }
    bus.lcd.load_snapshot(meta, payload).map(|_| true)
}

fn apply_reset_trace2_main_display_profile(
    bus: &mut StandaloneBus,
    state: &mut LlamaState,
    profile: &ResetTrace2MainDisplayProfile,
) -> Result<(), String> {
    validate_f_image(u32::from(profile.resume_registers.f)).map_err(str::to_string)?;
    seed_pce500_bootstrap_imem(&mut bus.memory);
    for byte in &profile.visible_bytes {
        bus.memory.write_external_byte(byte.addr, byte.value);
    }
    for byte in &profile.imem_bytes {
        bus.memory.write_internal_byte(byte.offset, byte.value);
    }

    let regs = &profile.resume_registers;
    state.set_reg(RegName::BA, regs.ba & 0xFFFF);
    state.set_reg(RegName::I, regs.i & 0xFFFF);
    state.set_reg(RegName::X, regs.x & ADDRESS_MASK);
    state.set_reg(RegName::Y, regs.y & ADDRESS_MASK);
    state.set_reg(RegName::U, regs.u & ADDRESS_MASK);
    state.set_reg(RegName::S, regs.s & ADDRESS_MASK);
    state.set_reg(RegName::F, regs.f as u32);
    for (name, value) in &regs.temps {
        if let Some(idx_str) = name.strip_prefix("TEMP") {
            if let Ok(idx) = idx_str.parse::<u8>() {
                state.set_reg(RegName::Temp(idx), *value & mask_for(RegName::Temp(idx)));
            }
        }
    }
    state.set_call_depth(0);
    state.set_call_sub_level(regs.call_sub_level);
    state.set_reg(
        RegName::IMR,
        bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) as u32,
    );
    state.set_pc(profile.resume_pc & ADDRESS_MASK);
    state.set_power_state(PowerState::Running);
    state.clear_call_page_stack();

    bus.cycle_count = 0;
    bus.pending_onk = (bus.memory.read_internal_byte(IMEM_SSR_OFFSET).unwrap_or(0) & SSR_ONK) != 0;
    bus.pending_kil = false;
    bus.deferred_key_irq = false;
    bus.deferred_pending_kil = false;
    bus.irq_pending = (bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & 0x0F) != 0;
    bus.in_interrupt = false;
    bus.last_irq_src = None;
    bus.timer.irq_pending = bus.irq_pending;
    bus.timer.irq_source = None;
    bus.timer.last_fired = None;
    bus.timer.irq_isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
    let _ = apply_lcd_snapshot(
        bus,
        profile.resume_lcd_meta.as_ref(),
        &profile.resume_lcd_payload,
    )?;
    Ok(())
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
#[allow(clippy::field_reassign_with_default)]
fn save_snapshot_state(
    path: &Path,
    bus: &StandaloneBus,
    state: &LlamaState,
    instruction_count: u64,
    model: DeviceModel,
) -> Result<(), Box<dyn Error>> {
    reject_unrepresented_snapshot_runtime(bus)?;
    let mut metadata = SnapshotMetadata::default();
    metadata.backend = "rust".to_string();
    metadata.device_model = Some(model);
    metadata.instruction_count = instruction_count;
    metadata.cycle_count = bus.cycle_count;
    metadata.pc = state.pc() & ADDRESS_MASK;
    metadata.memory_reads = bus.memory.memory_read_count();
    metadata.memory_writes = bus.memory.memory_write_count();
    metadata.call_depth = state.call_depth();
    metadata.call_sub_level = state.call_sub_level();
    let call_metrics = state.snapshot_call_metrics();
    metadata.call_stack = call_metrics.call_stack;
    metadata.call_page_stack = call_metrics.call_page_stack;
    metadata.call_return_widths = call_metrics.call_return_widths;
    metadata.power_state = state.power_state();
    metadata.onk_level = bus.pending_onk;
    metadata.temps = (0..NUM_TEMP_REGISTERS)
        .map(|index| {
            (
                index.to_string(),
                state.get_reg(RegName::Temp(index)) & 0xFF_FFFF,
            )
        })
        .collect();
    metadata.readonly_ranges = bus.memory.readonly_ranges().to_vec();
    metadata.memory_image_size = bus.memory.external_len();

    let (timer_info, mut interrupts) = bus.timer.snapshot_info();
    interrupts.pending = bus.irq_pending;
    interrupts.in_interrupt = bus.in_interrupt;
    interrupts.source = bus.last_irq_src.clone();
    interrupts.imr = bus
        .memory
        .read_internal_byte(IMEM_IMR_OFFSET)
        .unwrap_or(interrupts.imr);
    interrupts.isr = bus
        .memory
        .read_internal_byte(IMEM_ISR_OFFSET)
        .unwrap_or(interrupts.isr);
    // A snapshot is a checkpoint, not a scheduler reset. Preserve the exact
    // absolute next-fire targets, including disabled-timer state.
    metadata.timer = timer_info;
    metadata.interrupts = interrupts;

    let kb_state = bus.keyboard.snapshot_state();
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
            "kb_irq_enabled": bus.timer.kb_irq_enabled,
        }));
    }

    let (lcd_meta, payload) = bus.lcd.export_snapshot();
    metadata.lcd = Some(lcd_meta);
    metadata.lcd_payload_size = payload.len();
    let lcd_payload = Some(payload);

    let regs = collect_registers(state);
    snapshot::save_snapshot(path, &metadata, &regs, &bus.memory, lcd_payload.as_deref())?;
    Ok(())
}

impl LlamaBus for StandaloneBus {
    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        let addr = addr & ADDRESS_MASK;
        if bits == 0 {
            return 0;
        }
        if bits > 8 && MemoryImage::is_internal(addr) {
            if let Some(start) = MemoryImage::internal_offset(addr) {
                let bytes = u32::from(bits.div_ceil(8).max(1));
                let end = start.saturating_add(bytes.saturating_sub(1));
                if start <= IMEM_KIL_OFFSET && end >= IMEM_KOL_OFFSET {
                    let mut out = 0u32;
                    for byte_offset in 0..bytes {
                        let byte = self.load(addr.wrapping_add(byte_offset), 8) & 0xFF;
                        out |= byte << (byte_offset * 8);
                    }
                    return out & mask_bits(bits);
                }
            }
        }
        if let Some(value) = self.iq7000_clock_workspace_read(addr, bits) {
            self.trace_bus_access("read", addr, bits, value);
            return value;
        }
        let kbd_offset = MemoryImage::internal_offset(addr);
        if let Some(offset) = kbd_offset {
            if bits == 8 && offset == iq7000::IMEM_EIL_OFFSET {
                if let Some(rtc) = self.iq7000_rtc.as_mut() {
                    let byte = rtc.handle_eil_read();
                    let _ = self.memory.store(addr, bits, byte as u32);
                    self.trace_imem_access("read", addr, bits, byte as u32);
                    self.trace_bus_access("read", addr, bits, byte as u32);
                    return byte as u32;
                }
            }
            if offset == IMEM_KIL_OFFSET {
                // The ROM quiescence loops at 0xF1CEF and 0xF1EDF set LCC.KSD,
                // then poll KIL until it reads zero. Do not consume a queued
                // synthetic key event while satisfying that hardware contract.
                let lcc = self.memory.read_internal_byte(IMEM_LCC_OFFSET).unwrap_or(0);
                if lcc & 0x04 != 0 {
                    self.trace_kbd_access("read-ksd-masked", addr, offset, bits, 0);
                    return 0;
                }
            }
            let had_pending = offset == IMEM_KIL_OFFSET && self.keyboard.fifo_len() > 0;
            if let Some(byte) = self.keyboard.handle_read(offset, &mut self.memory) {
                match offset {
                    IMEM_KIL_OFFSET => self.kil_reads = self.kil_reads.saturating_add(1),
                    IMEM_KOH_OFFSET => self.rom_koh_reads = self.rom_koh_reads.saturating_add(1),
                    IMEM_KOL_OFFSET => self.rom_kol_reads = self.rom_kol_reads.saturating_add(1),
                    _ => {}
                }
                if offset == IMEM_KIL_OFFSET && (had_pending || self.keyboard.fifo_len() == 0) {
                    self.timer.key_irq_latched = false;
                    self.pending_kil = false;
                }
                if offset == IMEM_KIL_OFFSET {
                    // Emit perfetto event for KIL read with PC/value.
                    {
                        let mut guard = PERFETTO_TRACER.enter();
                        guard.with_some(|tracer| {
                            tracer.record_kio_read(
                                Some(self.last_pc),
                                offset as u8,
                                byte,
                                Some(self.instr_index),
                            );
                        });
                    }
                }
                if matches!(
                    offset,
                    IMEM_KIL_OFFSET
                        | IMEM_KOL_OFFSET
                        | IMEM_KOH_OFFSET
                        | IMEM_IMR_OFFSET
                        | IMEM_ISR_OFFSET
                ) {
                    self.trace_kbd_access("read", addr, offset, bits, byte as u32);
                }
                if false {
                    let val = self.memory.read_internal_byte(offset).unwrap_or(0);
                    println!(
                        "[kbd-read] pc=0x{pc:05X} addr=0x{addr:05X} offset=0x{offset:02X} value=0x{val:02X} last={last:?}",
                        pc = self.last_pc,
                        addr = addr,
                        offset = offset,
                        val = val,
                        last = self.last_kbd_access
                    );
                }
                return byte as u32;
            } else if matches!(
                offset,
                IMEM_KIL_OFFSET
                    | IMEM_KOL_OFFSET
                    | IMEM_KOH_OFFSET
                    | IMEM_IMR_OFFSET
                    | IMEM_ISR_OFFSET
                    | 0xF5
                    | 0xF6
            ) && self.trace_kbd
            {
                // Trace fallthrough reads (handled by memory, not keyboard).
                if let Some(val) = self.memory.read_internal_byte(offset) {
                    self.trace_kbd_access("read-fallthrough", addr, offset, bits, val as u32);
                }
            }
        }
        // Parity: IMEM offsets 0x00-0x0F are normal internal RAM for the PC-E500 ROM
        // (used for BP-relative locals). LCD is memory-mapped at 0x2000/0xA000.
        if self.trace_kbd && (VEC_RANGE_START..=VEC_RANGE_END).contains(&addr) {
            println!(
                "[vec-trace-read] pc=0x{pc:05X} addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                pc = self.last_pc,
                addr = addr,
                bits = bits,
                val = self.memory.load(addr, bits).unwrap_or(0) & mask_bits(bits)
            );
        }
        if self.memory.requires_python(addr) {
            if let Some(cb) = self.host_read.as_mut() {
                if let Some(val) = (cb)(addr) {
                    self.trace_bus_access("read", addr, bits, val as u32);
                    return val as u32;
                }
            }
        }
        if self.trace_reset_ce6_shadow_enabled && (0x010000..=0x01FFFF).contains(&addr) {
            let width_bytes = usize::from(bits.div_ceil(8));
            let mut value = 0u32;
            for idx in 0..width_bytes {
                let off = ((addr - 0x010000) as usize).saturating_add(idx);
                if off >= self.trace_reset_ce6_shadow.len() {
                    break;
                }
                value |= (self.trace_reset_ce6_shadow[off] as u32) << (idx * 8);
            }
            self.trace_bus_access("read", addr, bits, value);
            return value & mask_bits(bits);
        }
        if self.trace_resume_ce1_shadow_enabled && (0x040000..=0x04FFFF).contains(&addr) {
            let width_bytes = usize::from(bits.div_ceil(8));
            let mut value = 0u32;
            for idx in 0..width_bytes {
                let off = ((addr - 0x040000) as usize).saturating_add(idx);
                if off >= self.trace_resume_ce1_shadow.len() {
                    break;
                }
                value |= (self.trace_resume_ce1_shadow[off] as u32) << (idx * 8);
            }
            self.trace_bus_access("read", addr, bits, value);
            return value & mask_bits(bits);
        }
        if !MemoryImage::is_internal(addr) && !self.lcd.handles(addr) {
            if let Some(val) = self.maybe_trace_resume_read(addr, bits) {
                self.trace_bus_access("read", addr, bits, val);
                return val;
            }
        }
        if MemoryImage::is_internal(addr) {
            if let Some(offset) = MemoryImage::internal_offset(addr) {
                if offset == IMEM_SSR_OFFSET {
                    let mut val = self.memory.read_internal_byte(offset).unwrap_or(0);
                    if self.ssr_onk_visible() {
                        val |= SSR_ONK;
                    }
                    self.trace_imem_access("read", addr, bits, val as u32);
                    return (val as u32) & mask_bits(bits);
                }
            }
        }
        let value = self
            .memory
            .load(addr, bits)
            .map(|val| {
                if MemoryImage::is_internal(addr) {
                    self.trace_imem_access("read", addr, bits, val);
                } else if bits == 8 && self.lcd.handles(addr) {
                    if let Some(byte) = self.lcd.read(addr) {
                        return byte as u32;
                    }
                    return self.lcd.read_placeholder(addr);
                }
                self.trace_fifo_access("read", addr, bits, val);
                val & mask_bits(bits)
            })
            .unwrap_or(0);
        self.trace_bus_access("read", addr, bits, value);
        value
    }

    fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
        self.peek_byte_silent_at(addr, self.last_pc)
    }

    fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
        let addr = addr & ADDRESS_MASK;
        if let Some(value) = self.iq7000_clock_workspace_read(addr, 8) {
            return Some(value as u8);
        }
        if let Some(offset) = MemoryImage::internal_offset(addr) {
            if offset == iq7000::IMEM_EIL_OFFSET && self.iq7000_rtc.is_some() {
                return None;
            }
            if matches!(offset, IMEM_KOL_OFFSET | IMEM_KOH_OFFSET | IMEM_KIL_OFFSET) {
                return None;
            }
            if offset == IMEM_SSR_OFFSET {
                let mut value = self.memory.read_internal_byte_silent(offset)?;
                if self.ssr_onk_visible() {
                    value |= SSR_ONK;
                }
                return Some(value);
            }
        }
        if self.memory.requires_python(addr) {
            return self.host_peek.as_mut().and_then(|peek| peek(addr));
        }
        if self.trace_reset_ce6_shadow_enabled && (0x010000..=0x01FFFF).contains(&addr) {
            return self
                .trace_reset_ce6_shadow
                .get((addr - 0x010000) as usize)
                .copied();
        }
        if self.trace_resume_ce1_shadow_enabled && (0x040000..=0x04FFFF).contains(&addr) {
            return self
                .trace_resume_ce1_shadow
                .get((addr - 0x040000) as usize)
                .copied();
        }
        let trace_resume_active =
            self.trace_resume_read_enabled || self.trace_resume_read_anchor_pc == Some(context_pc);
        if trace_resume_active {
            let search_end = self
                .trace_resume_reads
                .len()
                .min(self.trace_resume_read_index.saturating_add(128));
            if let Some(byte) = self
                .trace_resume_reads
                .get(self.trace_resume_read_index..search_end)
                .and_then(|bytes| bytes.iter().find(|byte| byte.addr == addr))
            {
                return Some(byte.value);
            }
        }
        if self.lcd.handles(addr) {
            return None;
        }
        self.memory.read_byte_for_preflight(addr, Some(context_pc))
    }

    fn vector_transfer_provenance(&self) -> (usize, u64) {
        self.memory.vector_transfer_provenance()
    }

    fn instruction_byte_is_stable(&self, addr: u32) -> bool {
        self.instruction_byte_is_stable_for_context(addr, self.last_pc)
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        let addr = addr & ADDRESS_MASK;
        if bits > 8 && MemoryImage::is_internal(addr) {
            if let Some(start) = MemoryImage::internal_offset(addr) {
                let bytes = u32::from(bits.div_ceil(8).max(1));
                let end = start.saturating_add(bytes.saturating_sub(1));
                if start <= IMEM_KIL_OFFSET && end >= IMEM_KOL_OFFSET {
                    for byte_offset in 0..bytes {
                        let byte = (value >> (byte_offset * 8)) & 0xFF;
                        self.store(addr.wrapping_add(byte_offset), 8, byte);
                    }
                    return;
                }
            }
        }
        let kbd_offset = MemoryImage::internal_offset(addr);
        let mut imr_isr_prev: Option<(u32, u8)> = None;
        if MemoryImage::is_internal(addr) {
            if let Some(offset) = MemoryImage::internal_offset(addr) {
                if matches!(offset, IMEM_IMR_OFFSET | IMEM_ISR_OFFSET) {
                    if let Some(prev) = self.memory.read_internal_byte(offset) {
                        imr_isr_prev = Some((offset, prev));
                    }
                }
            }
        }
        if let Some(offset) = kbd_offset {
            if bits == 8 && offset == iq7000::IMEM_EOL_OFFSET {
                if let Some(rtc) = self.iq7000_rtc.as_mut() {
                    rtc.handle_eol_write(value as u8);
                    let _ = self.memory.store(addr, bits, value);
                    self.trace_imem_access("write", addr, bits, value);
                    self.trace_mem_write(addr, bits, value);
                    self.trace_bus_access("write", addr, bits, value);
                    return;
                }
            }
            if self
                .keyboard
                .handle_write(offset, value as u8, &mut self.memory)
            {
                if matches!(
                    offset,
                    IMEM_KIL_OFFSET
                        | IMEM_KOL_OFFSET
                        | IMEM_KOH_OFFSET
                        | IMEM_IMR_OFFSET
                        | IMEM_ISR_OFFSET
                ) {
                    self.trace_kbd_access("write", addr, offset, bits, value);
                }
                self.trace_mem_write(addr, bits, value);
                if false {
                    println!(
                        "[kbd-write] pc=0x{pc:05X} addr=0x{addr:05X} offset=0x{offset:02X} value=0x{val:02X} last={last:?}",
                        pc = self.last_pc,
                        addr = addr,
                        offset = offset,
                        val = value as u8,
                        last = self.last_kbd_access
                    );
                }
                if let Some((off, prev)) = imr_isr_prev {
                    if let Some(cur) = self.memory.read_internal_byte(off) {
                        if cur != prev {
                            self.log_imem_write(off, prev, cur);
                        }
                        if off == IMEM_ISR_OFFSET && (prev & ISR_KEYI) != 0 && (cur & ISR_KEYI) == 0
                        {
                            self.timer.key_irq_latched = false;
                            self.pending_kil = false;
                            self.keyboard.consume_pending_events();
                        }
                        if off == IMEM_ISR_OFFSET
                            && (cur & (ISR_KEYI | ISR_ONKI | ISR_MTI | ISR_STI)) == 0
                        {
                            self.timer.irq_pending = false;
                            self.timer.irq_source = None;
                        }
                    }
                }
                return;
            }
        }
        // Parity: do not alias IMEM 0x00-0x0F onto LCD; those bytes are used as scratch RAM.
        if self.lcd.handles(addr) {
            if bits == 8 {
                self.lcd.write(addr, value as u8);
                self.lcd_writes = self.lcd_writes.saturating_add(1);
                if self.log_lcd && self.log_lcd_count < self.log_lcd_limit {
                    println!(
                        "[lcd-write] addr=0x{addr:05X} value=0x{val:02X} count={cnt}",
                        addr = addr,
                        val = value as u8,
                        cnt = self.lcd_writes
                    );
                    self.log_lcd_count += 1;
                }
                self.trace_mem_write(addr, bits, value);
                self.trace_bus_access("write", addr, bits, value);
            }
            return;
        }
        if MemoryImage::is_internal(addr) {
            self.trace_imem_access("write", addr, bits, value);
        }
        if self.trace_reset_ce6_shadow_enabled && (0x010000..=0x01FFFF).contains(&addr) {
            self.trace_mem_write(addr, bits, value);
            self.trace_bus_access("write", addr, bits, value);
            if self.trace_reset_ce6_readonly {
                return;
            }
            let width_bytes = usize::from(bits.div_ceil(8));
            for idx in 0..width_bytes {
                let off = ((addr - 0x010000) as usize).saturating_add(idx);
                if off >= self.trace_reset_ce6_shadow.len() {
                    break;
                }
                self.trace_reset_ce6_shadow[off] = ((value >> (idx * 8)) & 0xFF) as u8;
            }
            return;
        }
        if self.trace_resume_ce1_shadow_enabled && (0x040000..=0x04FFFF).contains(&addr) {
            self.trace_mem_write(addr, bits, value);
            self.trace_bus_access("write", addr, bits, value);
            if self.trace_reset_ce1_readonly {
                return;
            }
            let width_bytes = usize::from(bits.div_ceil(8));
            for idx in 0..width_bytes {
                let off = ((addr - 0x040000) as usize).saturating_add(idx);
                if off >= self.trace_resume_ce1_shadow.len() {
                    break;
                }
                self.trace_resume_ce1_shadow[off] = ((value >> (idx * 8)) & 0xFF) as u8;
            }
            return;
        }
        self.trace_fifo_access("write", addr, bits, value);
        if self.memory.requires_python(addr) {
            if let Some(cb) = self.host_write.as_mut() {
                (cb)(addr, value as u8);
                self.trace_bus_access("write", addr, bits, value);
                return;
            }
        }
        if self.trace_kbd && (VEC_RANGE_START..=VEC_RANGE_END).contains(&addr) {
            println!(
                "[vec-trace-write] pc=0x{pc:05X} addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                pc = self.last_pc,
                addr = addr,
                bits = bits,
                val = value & mask_bits(bits)
            );
        }
        let _ = self.memory.store(addr, bits, value);
        self.trace_mem_write(addr, bits, value);
        self.trace_bus_access("write", addr, bits, value);
        if let Some((offset, prev)) = imr_isr_prev {
            if let Some(cur) = self.memory.read_internal_byte(offset) {
                if cur != prev {
                    self.log_imem_write(offset, prev, cur);
                }
                if offset == IMEM_ISR_OFFSET && (prev & ISR_KEYI) != 0 && (cur & ISR_KEYI) == 0 {
                    self.timer.key_irq_latched = false;
                    self.pending_kil = false;
                    self.keyboard.consume_pending_events();
                }
                if offset == IMEM_ISR_OFFSET
                    && (cur & (ISR_KEYI | ISR_ONKI | ISR_MTI | ISR_STI)) == 0
                {
                    self.timer.irq_pending = false;
                    self.timer.irq_source = None;
                }
            }
        }
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base & ADDRESS_MASK
    }

    fn peek_imem(&mut self, offset: u32) -> u8 {
        self.memory.read_internal_byte(offset).unwrap_or(0)
    }
    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        // Bypass perfetto IMR read logging when sampling for tracing metadata.
        self.memory
            .read_internal_byte_silent(offset)
            .or_else(|| self.memory.read_internal_byte(offset))
            .unwrap_or(0)
    }

    fn timer_trace(&mut self) -> Option<TimerTrace> {
        let (mti, sti) = self.timer.tick_counts(self.cycle_count);
        Some(TimerTrace {
            mti_ticks: mti,
            sti_ticks: sti,
        })
    }

    fn cycle_count(&mut self) -> Option<u64> {
        Some(self.cycle_count)
    }

    fn supports_wait_cycles(&self) -> bool {
        true
    }

    fn supports_timer_phase_clear(&self) -> bool {
        true
    }

    fn clear_timer_phases(&mut self, clear_sti: bool, clear_mti: bool) {
        if clear_mti {
            self.timer.next_mti = self.cycle_count.wrapping_add(self.timer.mti_period);
        }
        if clear_sti {
            self.timer.next_sti = self.cycle_count.wrapping_add(self.timer.sti_period);
        }
    }

    fn wait_cycles(&mut self, cycles: u32) {
        // Python WAIT burns one instruction cycle without ticking timers, then loops I times.
        let cycles = cycles.max(1);
        self.cycle_count = self.cycle_count.wrapping_add(1);
        for _ in 0..cycles {
            self.advance_cycle();
        }
    }
}

fn load_rom(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let data = fs::read(path)?;
    if data.len() < 0x100000 {
        eprintln!(
            "warning: ROM image is smaller than expected ({} bytes)",
            data.len()
        );
    }
    Ok(data)
}

fn configure_bus_for_model(bus: &mut StandaloneBus, model: DeviceModel) {
    if model.is_pce500_family() {
        // Baseline PC-E500 scans the key matrix each instruction (not just on MTI).
        bus.scan_on_timer = false;
    }
    model.configure_keyboard(&mut bus.keyboard);
    bus.memory.set_internal_ram_mirror(model.is_pce500_family());
}

fn parse_matrix_code(raw: &str) -> Result<Option<AutoKeyKind>, Box<dyn Error>> {
    let lowered = raw.trim().to_lowercase();
    if lowered == "pf1" {
        return Ok(Some(AutoKeyKind::Matrix(PF1_CODE)));
    }
    if lowered == "pf2" {
        return Ok(Some(AutoKeyKind::Matrix(PF2_CODE)));
    }
    if lowered == "on" || lowered == "key_on" || lowered == "onk" {
        return Ok(Some(AutoKeyKind::OnKey));
    }
    if let Some(hex) = lowered.strip_prefix("0x") {
        let value = u8::from_str_radix(hex, 16)?;
        return Ok(Some(AutoKeyKind::Matrix(value)));
    }
    if let Ok(value) = lowered.parse::<u8>() {
        return Ok(Some(AutoKeyKind::Matrix(value)));
    }
    Err(format!("could not parse matrix code '{raw}'").into())
}

fn parse_u64_value(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("missing numeric value".to_string());
    }
    let lowered = trimmed.to_ascii_lowercase();
    if let Some(hex) = lowered.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex value '{raw}'"));
    }
    trimmed
        .parse::<u64>()
        .map_err(|_| format!("invalid number '{raw}'"))
}

fn parse_u8_value(raw: &str) -> Result<u8, String> {
    let value = parse_u64_value(raw)?;
    u8::try_from(value).map_err(|_| format!("value out of u8 range: '{raw}'"))
}

fn iq7000_named_key(raw: &str) -> Option<AutoKeyKind> {
    let code = match raw {
        "calc" => 0x00,
        "shift" | "iq-shift" => 0x01,
        "search-up" | "search_up" | "search-prev" | "search_previous" => 0x03,
        "function" | "fn" => 0x04,
        "memo" => 0x08,
        "home" | "caps" | "caps-lock" | "caps-off" | "iq-caps" => 0x09,
        "search-down" | "search_down" | "search-next" | "search_next" => 0x0B,
        "tel" | "telephone" => 0x10,
        "calendar" => 0x18,
        "schedule" => 0x19,
        "card" | "card-samples" | "samples" => 0x1A,
        "world" => 0x1B,
        "option" | "opts" | "settings" => 0x1D,
        "line" | "newline" | "memo-line" | "memo-return" | "hooked-return" => 0x3D,
        "memo-enter" | "store" | "enter" | "return" | "ret" => 0x45,
        _ => return None,
    };
    Some(AutoKeyKind::Event(code))
}

fn generated_key_seq_key(model: DeviceModel, ch: char) -> Option<AutoKeyKind> {
    let input = lookup_generated_key_input(model.label(), ch)?;
    match input.kind {
        GeneratedKeyInputKind::Matrix => match input.modifier {
            Some(modifier) => Some(AutoKeyKind::Chord {
                modifier,
                code: input.code,
            }),
            None => Some(AutoKeyKind::Matrix(input.code)),
        },
        GeneratedKeyInputKind::InputEvent => Some(AutoKeyKind::Event(input.code)),
    }
}

fn resolve_key_seq_key(model: DeviceModel, raw: &str) -> Result<AutoKeyKind, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("empty key token".to_string());
    }
    let lowered = trimmed.to_ascii_lowercase();
    if model == DeviceModel::Iq7000 {
        if let Some(key) = iq7000_named_key(&lowered) {
            return Ok(key);
        }
    }
    for prefix in ["event"] {
        if let Some(value) = lowered.strip_prefix(prefix).and_then(|rest| {
            rest.strip_prefix(':')
                .or_else(|| rest.strip_prefix('='))
                .map(str::trim)
        }) {
            return Ok(AutoKeyKind::Event(parse_u8_value(value)?));
        }
    }
    for prefix in ["digitizer", "input", "raw-input", "raw_event", "raw-event"] {
        if let Some(value) = lowered.strip_prefix(prefix).and_then(|rest| {
            rest.strip_prefix(':')
                .or_else(|| rest.strip_prefix('='))
                .map(str::trim)
        }) {
            return Ok(AutoKeyKind::InputEvent(parse_u8_value(value)?));
        }
    }
    if matches!(lowered.as_str(), "enter" | "return" | "ret") {
        if let Some(code) = KeyboardMatrix::matrix_code_for_key_name("KEY_ENTER") {
            return Ok(AutoKeyKind::Matrix(code));
        }
        return Err("enter key is not mapped in the keyboard matrix".to_string());
    }
    if lowered == "space" {
        if let Some(code) = KeyboardMatrix::matrix_code_for_key_name("KEY_SPACE") {
            return Ok(AutoKeyKind::Matrix(code));
        }
        return Err("space key is not mapped in the keyboard matrix".to_string());
    }
    if trimmed.chars().count() == 1 {
        let ch = trimmed.chars().next().unwrap();
        if let Some(key) = generated_key_seq_key(model, ch) {
            return Ok(key);
        }
        if let Some(code) = KeyboardMatrix::matrix_code_for_char(ch) {
            return Ok(AutoKeyKind::Matrix(code));
        }
    }
    match parse_matrix_code(trimmed) {
        Ok(Some(kind)) => Ok(kind),
        Ok(None) => Err(format!("unknown key token '{raw}'")),
        Err(err) => Err(err.to_string()),
    }
}

fn parse_key_seq(
    raw: &str,
    default_hold: u64,
    model: DeviceModel,
) -> Result<Vec<KeySeqAction>, String> {
    let mut actions = Vec::new();
    'tokens: for token_raw in raw.split([',', ';']) {
        let token = token_raw.trim();
        if token.is_empty() {
            continue;
        }
        let lower = token.to_ascii_lowercase();
        if lower.starts_with("wait-op") {
            let sep = token.find(':').or_else(|| token.find('='));
            let Some(sep) = sep else {
                return Err(format!("wait-op missing value: '{token}'"));
            };
            let value = token[sep + 1..].trim();
            let count = parse_u64_value(value)?;
            let mut action = KeySeqAction::new(KeySeqKind::WaitOp);
            action.op_target = count;
            actions.push(action);
            continue;
        }
        if lower.starts_with("wait-text") {
            let sep = token.find(':').or_else(|| token.find('='));
            let Some(sep) = sep else {
                return Err(format!("wait-text missing value: '{token}'"));
            };
            let value = token[sep + 1..].trim();
            if value.is_empty() {
                return Err(format!("wait-text expects non-empty value: '{token}'"));
            }
            let mut action = KeySeqAction::new(KeySeqKind::WaitText);
            action.text = value.to_string();
            actions.push(action);
            continue;
        }
        if lower.starts_with("wait-screen-change") {
            if token.contains(':') || token.contains('=') {
                return Err(format!(
                    "wait-screen-change does not take a value: '{token}'"
                ));
            }
            actions.push(KeySeqAction::new(KeySeqKind::WaitScreenChange));
            continue;
        }
        if lower.starts_with("wait-screen-empty") {
            if token.contains(':') || token.contains('=') {
                return Err(format!(
                    "wait-screen-empty does not take a value: '{token}'"
                ));
            }
            actions.push(KeySeqAction::new(KeySeqKind::WaitScreenEmpty));
            continue;
        }
        if lower.starts_with("wait-screen-draw") {
            if token.contains(':') || token.contains('=') {
                return Err(format!("wait-screen-draw does not take a value: '{token}'"));
            }
            actions.push(KeySeqAction::new(KeySeqKind::WaitScreenDraw));
            continue;
        }
        if lower.starts_with("wait-power") {
            let sep = token.find(':').or_else(|| token.find('='));
            let Some(sep) = sep else {
                return Err(format!("wait-power missing value: '{token}'"));
            };
            let value = token[sep + 1..].trim().to_ascii_lowercase();
            if value != "on" && value != "off" {
                return Err(format!("wait-power expects on/off, got '{value}'"));
            }
            let mut action = KeySeqAction::new(KeySeqKind::WaitPower);
            action.power_on = value == "on";
            actions.push(action);
            continue;
        }
        for (prefix, kind) in [
            ("down", KeySeqKind::KeyDown),
            ("keydown", KeySeqKind::KeyDown),
            ("key-down", KeySeqKind::KeyDown),
            ("press", KeySeqKind::KeyDown),
            ("up", KeySeqKind::KeyUp),
            ("keyup", KeySeqKind::KeyUp),
            ("key-up", KeySeqKind::KeyUp),
            ("release", KeySeqKind::KeyUp),
        ] {
            if let Some(rest) = lower.strip_prefix(prefix).and_then(|tail| {
                tail.strip_prefix(':')
                    .or_else(|| tail.strip_prefix('='))
                    .map(str::trim)
            }) {
                if rest.is_empty() {
                    return Err(format!("{prefix} expects a key token: '{token}'"));
                }
                let sep = token.find(':').or_else(|| token.find('=')).unwrap();
                let raw_rest = token[sep + 1..].trim();
                let key = resolve_key_seq_key(model, raw_rest)?;
                let mut action = KeySeqAction::new(kind);
                action.key = Some(key);
                action.label = raw_rest.to_string();
                actions.push(action);
                continue 'tokens;
            }
        }
        if lower.starts_with("text:")
            || lower.starts_with("text=")
            || lower.starts_with("type:")
            || lower.starts_with("type=")
        {
            let sep = token.find(':').or_else(|| token.find('=')).unwrap();
            let value = &token[sep + 1..];
            push_text_key_seq_actions(&mut actions, value, default_hold, model)?;
            continue;
        }
        if lower.starts_with("digitizer:")
            || lower.starts_with("digitizer=")
            || lower.starts_with("input:")
            || lower.starts_with("input=")
            || lower.starts_with("event:")
            || lower.starts_with("event=")
        {
            let key = resolve_key_seq_key(model, token)?;
            let mut action = KeySeqAction::new(KeySeqKind::Press);
            action.key = Some(key);
            action.label = token.to_string();
            action.hold = default_hold;
            actions.push(action);
            continue;
        }

        let mut key_part = token;
        let mut hold = default_hold;
        if let Some(colon) = token.find(':') {
            key_part = token[..colon].trim();
            let hold_raw = token[colon + 1..].trim();
            if !hold_raw.is_empty() {
                hold = parse_u64_value(hold_raw)?;
            }
        }
        let key = resolve_key_seq_key(model, key_part)?;
        let mut action = KeySeqAction::new(KeySeqKind::Press);
        action.key = Some(key);
        action.label = key_part.to_string();
        action.hold = hold;
        actions.push(action);
    }
    Ok(actions)
}

fn push_text_key_seq_actions(
    actions: &mut Vec<KeySeqAction>,
    text: &str,
    default_hold: u64,
    model: DeviceModel,
) -> Result<(), String> {
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let key = if ch == '\\' && chars.peek().copied() == Some('n') {
            chars.next();
            resolve_key_seq_key(model, "newline")?
        } else {
            generated_key_seq_key(model, ch)
                .or_else(|| KeyboardMatrix::matrix_code_for_char(ch).map(AutoKeyKind::Matrix))
                .ok_or_else(|| {
                    format!(
                        "text input character '{ch}' is not mapped for {}",
                        model.label()
                    )
                })?
        };
        let mut action = KeySeqAction::new(KeySeqKind::Press);
        action.key = Some(key);
        action.label = ch.to_string();
        action.hold = default_hold;
        actions.push(action);
    }
    Ok(())
}

fn capture_screen_state(
    lcd: &dyn LcdHal,
    decoder: Option<&DeviceTextDecoder>,
    include_text: bool,
) -> ScreenState {
    let bytes = lcd.display_vram_bytes();
    let mut signature: u64 = 0xcbf29ce484222325;
    let mut blank = true;
    for row in bytes.iter() {
        for byte in row.iter() {
            if *byte != 0 {
                blank = false;
            }
            signature ^= u64::from(*byte);
            signature = signature.wrapping_mul(0x100000001b3);
        }
    }
    let mut text = String::new();
    let mut text_valid = false;
    if include_text {
        if let Some(decoder) = decoder {
            let lines = decoder.decode_display_text(lcd);
            if !lines.is_empty() {
                text = lines.join("\n");
            }
            text_valid = true;
        }
    }
    ScreenState {
        valid: true,
        is_blank: blank,
        signature,
        text_valid,
        text,
    }
}

fn lcd_pixels(lcd: &dyn LcdHal) -> Vec<Vec<u8>> {
    if lcd.kind() == LcdKind::Iq7000Vram {
        const IQ7000_COLS: usize = 96;
        const IQ7000_ROWS: usize = 64;
        const IQ7000_PAGES: usize = IQ7000_ROWS / 8;
        let bytes = lcd.display_vram_bytes();
        let mut out = vec![vec![0u8; IQ7000_COLS]; IQ7000_ROWS];
        for page in 0..IQ7000_PAGES {
            for col in 0..IQ7000_COLS {
                let byte = bytes[page][col];
                for dy in 0..8usize {
                    let bit = 7usize.saturating_sub(dy);
                    out[(page * 8) + dy][col] = (byte >> bit) & 1;
                }
            }
        }
        return out;
    }
    lcd.display_buffer()
        .iter()
        .map(|row| row.iter().map(|px| u8::from(*px != 0)).collect())
        .collect()
}

fn iq7000_lcd_annunciators(memory: &MemoryImage) -> LcdAnnunciators {
    let shadow = memory.load(IQ7000_ANNUNCIATOR_SHADOW_ADDR, 8).unwrap_or(0) as u8;
    let key_state = memory.load(IQ7000_KEY_STATE_ADDR, 8).unwrap_or(0) as u8;
    let raw_union = shadow | key_state;
    LcdAnnunciators {
        state_raw: key_state,
        shadow_raw: shadow,
        raw_union,
        unmapped_state: key_state & !IQ7000_NAMED_ANNUNCIATOR_MASK,
        unmapped_shadow: shadow & !IQ7000_NAMED_ANNUNCIATOR_MASK,
        unmapped_union: raw_union & !IQ7000_NAMED_ANNUNCIATOR_MASK,
        shift: raw_union & IQ7000_SHIFT_ANNUNCIATOR != 0,
        caps: raw_union & IQ7000_CAPS_ANNUNCIATOR != 0,
    }
}

fn lcd_annunciators(model: DeviceModel, memory: &MemoryImage) -> Option<LcdAnnunciators> {
    (model == DeviceModel::Iq7000).then(|| iq7000_lcd_annunciators(memory))
}

fn read_memory_bytes(memory: &MemoryImage, addr: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|idx| {
            memory
                .load(addr.wrapping_add(idx as u32) & ADDRESS_MASK, 8)
                .unwrap_or(0) as u8
        })
        .collect()
}

fn bytes_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn build_debug_probe(
    model: DeviceModel,
    bus: &StandaloneBus,
    state: &LlamaState,
    executed: u64,
    ranges: &[DebugProbeRange],
) -> Value {
    let range_json: Vec<Value> = ranges
        .iter()
        .map(|range| {
            let bytes = read_memory_bytes(&bus.memory, range.addr, range.len);
            json!({
                "name": range.name,
                "addr": format!("0x{:05X}", range.addr),
                "len": range.len,
                "bytes": bytes,
                "hex": bytes_hex(&bytes),
            })
        })
        .collect();

    let storage_workspace = read_memory_bytes(&bus.memory, 0x1FD00, 0x40);
    let iocs_workspace_ptr = read_memory_bytes(&bus.memory, 0x1FE36, 3);
    let annunciators = lcd_annunciators(model, &bus.memory);

    json!({
        "model": model.label(),
        "executed": executed,
        "pc": format!("0x{:05X}", state.pc() & ADDRESS_MASK),
        "halted": state.is_halted(),
        "keyboard": {
            "fifo": bus.keyboard.fifo_snapshot(),
            "fifo_len": bus.keyboard.fifo_len(),
            "pending_kil": bus.pending_kil,
            "irq_pending": bus.irq_pending,
            "last_access": bus.last_kbd_access,
            "kil_reads": bus.kil_reads,
            "rom_koh_reads": bus.rom_koh_reads,
            "rom_kol_reads": bus.rom_kol_reads,
        },
        "imem": {
            "imr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0)),
            "isr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0)),
            "kol": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_KOL_OFFSET).unwrap_or(0)),
            "koh": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_KOH_OFFSET).unwrap_or(0)),
            "kil": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_KIL_OFFSET).unwrap_or(0)),
            "scr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_SCR_OFFSET).unwrap_or(0)),
            "ssr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_SSR_OFFSET).unwrap_or(0)),
            "ucr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_UCR_OFFSET).unwrap_or(0)),
            "usr": format!("0x{:02X}", bus.memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0)),
        },
        "iq7000": {
            "storage_workspace_addr": "0x1FD00",
            "storage_workspace": storage_workspace,
            "storage_workspace_hex": bytes_hex(&storage_workspace),
            "iocs_workspace_ptr_addr": "0x1FE36",
            "iocs_workspace_ptr": iocs_workspace_ptr,
            "iocs_workspace_ptr_hex": bytes_hex(&iocs_workspace_ptr),
            "lcd_annunciators": annunciators,
        },
        "ranges": range_json,
    })
}

fn build_runtime_debug_probe(
    model: DeviceModel,
    runtime: &CoreRuntime,
    ranges: &[DebugProbeRange],
) -> Value {
    let range_json: Vec<Value> = ranges
        .iter()
        .map(|range| {
            let bytes = read_memory_bytes(&runtime.memory, range.addr, range.len);
            json!({
                "name": range.name,
                "addr": format!("0x{:05X}", range.addr),
                "len": range.len,
                "bytes": bytes,
                "hex": bytes_hex(&bytes),
            })
        })
        .collect();

    let keyboard = runtime.keyboard.as_ref().map(|keyboard| {
        let telemetry = keyboard.telemetry();
        json!({
            "fifo": keyboard.fifo_snapshot(),
            "fifo_len": keyboard.fifo_len(),
            "pressed_physical_keys": telemetry.pressed,
            "strobe_count": telemetry.strobe_count,
            "kol": format!("0x{:02X}", telemetry.kol),
            "koh": format!("0x{:02X}", telemetry.koh),
            "active_columns": telemetry.active_columns,
        })
    });
    let storage_workspace = read_memory_bytes(&runtime.memory, 0x1FD00, 0x40);
    let iocs_workspace_ptr = read_memory_bytes(&runtime.memory, 0x1FE36, 3);
    let annunciators = lcd_annunciators(model, &runtime.memory);

    json!({
        "schema": "sc62015-core-runtime-debug-probe-v1",
        "model": model.label(),
        "executed": runtime.instruction_count(),
        "timing_units": runtime.cycle_count(),
        "pc": format!("0x{:05X}", runtime.state.pc() & ADDRESS_MASK),
        "halted": runtime.state.is_halted(),
        "off": runtime.state.is_off(),
        "keyboard": keyboard,
        "interrupts": {
            "pending": runtime.timer.irq_pending,
            "source": runtime.timer.irq_source,
            "last_source": runtime.timer.last_irq_src,
            "last_pc": runtime.timer.last_irq_pc.map(|pc| format!("0x{pc:05X}")),
            "last_vector": runtime.timer.last_irq_vector.map(|pc| format!("0x{pc:05X}")),
        },
        "imem": {
            "imr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0)),
            "isr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0)),
            "kol": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_KOL_OFFSET).unwrap_or(0)),
            "koh": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_KOH_OFFSET).unwrap_or(0)),
            "kil": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_KIL_OFFSET).unwrap_or(0)),
            "scr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_SCR_OFFSET).unwrap_or(0)),
            "ssr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_SSR_OFFSET).unwrap_or(0)),
            "ucr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_UCR_OFFSET).unwrap_or(0)),
            "usr": format!("0x{:02X}", runtime.memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0)),
        },
        "iq7000": {
            "storage_workspace_addr": "0x1FD00",
            "storage_workspace": storage_workspace,
            "storage_workspace_hex": bytes_hex(&storage_workspace),
            "iocs_workspace_ptr_addr": "0x1FE36",
            "iocs_workspace_ptr": iocs_workspace_ptr,
            "iocs_workspace_ptr_hex": bytes_hex(&iocs_workspace_ptr),
            "lcd_annunciators": annunciators,
        },
        "ranges": range_json,
    })
}

fn write_runtime_diagnostic_artifacts(
    args: &Args,
    runtime: &CoreRuntime,
    lcd_lines: &[String],
) -> Result<(), Box<dyn Error>> {
    if let Some(path) = &args.dump_lcd_trace {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let lcd = runtime.lcd.as_deref().ok_or("missing LCD runtime")?;
        let dump = LcdTraceDump {
            executed: runtime.instruction_count(),
            pc: runtime.state.pc(),
            halted: runtime.state.is_halted(),
            lcd_lines: lcd_lines.to_vec(),
            vram: lcd.display_vram_bytes().map(|row| row.to_vec()).to_vec(),
            trace: lcd.display_trace_buffer().map(|row| row.to_vec()).to_vec(),
        };
        fs::write(path, serde_json::to_string_pretty(&dump)?)?;
        println!("Wrote LCD trace dump: {}", path.display());
    }
    if let Some(path) = &args.debug_probe_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let ranges = parse_debug_probe_ranges(&args.debug_probe_range)
            .map_err(|error| format!("--debug-probe-range: {error}"))?;
        let probe = build_runtime_debug_probe(args.model, runtime, &ranges);
        fs::write(path, serde_json::to_string_pretty(&probe)?)?;
        println!("Wrote debug probe JSON: {}", path.display());
    }
    Ok(())
}

fn append_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut hasher = Crc32Hasher::new();
    hasher.update(kind);
    hasher.update(data);
    out.extend_from_slice(&hasher.finalize().to_be_bytes());
}

fn iq7000_status_glyph(ch: char) -> Option<[u8; 5]> {
    match ch {
        'A' => Some([0b010, 0b101, 0b111, 0b101, 0b101]),
        'C' => Some([0b111, 0b100, 0b100, 0b100, 0b111]),
        'F' => Some([0b111, 0b100, 0b110, 0b100, 0b100]),
        'H' => Some([0b101, 0b101, 0b111, 0b101, 0b101]),
        'I' => Some([0b111, 0b010, 0b010, 0b010, 0b111]),
        'P' => Some([0b110, 0b101, 0b110, 0b100, 0b100]),
        'S' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
        'T' => Some([0b111, 0b010, 0b010, 0b010, 0b010]),
        _ => None,
    }
}

fn draw_iq7000_status_label(pixels: &mut [Vec<u8>], x: usize, y: usize, label: &str, active: bool) {
    let shade = if active { 1 } else { 2 };
    for (char_idx, ch) in label.chars().enumerate() {
        let Some(rows) = iq7000_status_glyph(ch) else {
            continue;
        };
        let glyph_x = x + char_idx * 4;
        for (row_idx, bits) in rows.into_iter().enumerate() {
            let py = y + row_idx;
            if py >= pixels.len() {
                continue;
            }
            for col in 0..3 {
                if bits & (1 << (2 - col)) == 0 {
                    continue;
                }
                let px = glyph_x + col;
                if px < pixels[py].len() {
                    pixels[py][px] = shade;
                }
            }
        }
    }
}

fn pixels_with_iq7000_annunciators(
    pixels: &[Vec<u8>],
    annunciators: Option<&LcdAnnunciators>,
) -> Vec<Vec<u8>> {
    let Some(annunciators) = annunciators else {
        return pixels.to_vec();
    };
    let width = pixels.first().map_or(0, Vec::len);
    if width == 0 {
        return pixels.to_vec();
    }

    let mut out = pixels.to_vec();
    out.push(vec![0; width]);
    let label_y = out.len();
    out.extend((0..6).map(|_| vec![0; width]));
    draw_iq7000_status_label(&mut out, 2, label_y, "SHIFT", annunciators.shift);
    draw_iq7000_status_label(&mut out, 32, label_y, "CAPS", annunciators.caps);
    out
}

fn write_lcd_png(
    path: &Path,
    pixels: &[Vec<u8>],
    scale: usize,
    annunciators: Option<&LcdAnnunciators>,
) -> Result<(), Box<dyn Error>> {
    let pixels = pixels_with_iq7000_annunciators(pixels, annunciators);
    let height = pixels.len();
    let width = pixels.first().map_or(0, Vec::len);
    if width == 0 || height == 0 || scale == 0 {
        return Err("cannot render an empty LCD capture".into());
    }

    let out_width = width * scale;
    let out_height = height * scale;
    let mut raw = Vec::with_capacity((out_width + 1) * out_height);
    for row in &pixels {
        for _ in 0..scale {
            raw.extend(std::iter::once(0)); // PNG filter type 0.
            for pixel in row {
                let shade = match *pixel {
                    0 => 0xC8,
                    1 => 0x18,
                    _ => 0x78,
                };
                raw.extend(std::iter::repeat_n(shade, scale));
            }
        }
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;

    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(out_width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(out_height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]); // 8-bit grayscale, deflate, no interlace.
    append_png_chunk(&mut png, b"IHDR", &ihdr);
    append_png_chunk(&mut png, b"IDAT", &compressed);
    append_png_chunk(&mut png, b"IEND", &[]);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, png)?;
    Ok(())
}

fn parse_address(raw: &str) -> Result<u32, Box<dyn Error>> {
    let trimmed = raw.trim();
    if let Some(hex) = trimmed.strip_prefix("0x") {
        let value = u32::from_str_radix(hex, 16)?;
        return Ok(value);
    }
    Ok(trimmed.parse::<u32>()?)
}

fn parse_expected_row(raw: &str) -> Result<(usize, String), String> {
    let (idx, text) = raw
        .split_once(':')
        .ok_or_else(|| format!("expect-row must be ROW:TEXT, got '{raw}'"))?;
    let row_idx = idx
        .parse::<usize>()
        .map_err(|_| format!("could not parse row index in '{raw}'"))?;
    Ok((row_idx, text.to_string()))
}

fn parse_debug_probe_range(raw: &str) -> Result<DebugProbeRange, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("empty debug probe range".to_string());
    }
    let (name, rest) = match trimmed.split_once('@') {
        Some((name, rest)) => (name.trim().to_string(), rest.trim()),
        None => ("range".to_string(), trimmed),
    };
    if name.is_empty() {
        return Err(format!("debug probe range has empty name: '{raw}'"));
    }
    let (addr_raw, len_raw) = rest.split_once(':').ok_or_else(|| {
        format!("debug probe range must be NAME@ADDR:LEN or ADDR:LEN, got '{raw}'")
    })?;
    let addr = parse_address(addr_raw.trim()).map_err(|err| err.to_string())? & ADDRESS_MASK;
    let len_u64 = parse_u64_value(len_raw.trim())?;
    let len =
        usize::try_from(len_u64).map_err(|_| format!("range length too large: '{len_raw}'"))?;
    if len == 0 {
        return Err(format!(
            "debug probe range length must be non-zero: '{raw}'"
        ));
    }
    Ok(DebugProbeRange { name, addr, len })
}

fn parse_debug_probe_ranges(raw: &[String]) -> Result<Vec<DebugProbeRange>, String> {
    raw.iter()
        .map(|item| parse_debug_probe_range(item))
        .collect()
}

fn encode_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for byte in data {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0F) as usize] as char);
    }
    out
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacomHostTxPhase {
    WaitRomReadyLow,
    WaitRomAckHigh,
    ReleaseHigh(u8),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PacomHostTxByte {
    byte: u8,
    bit_index: u8,
    phase: PacomHostTxPhase,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacomRomRxPhase {
    StartLow,
    Data,
    SpacingHigh,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PacomRomRxByte {
    byte: u8,
    bit_index: u8,
    phase: PacomRomRxPhase,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct Iq7000PclinkSerialPeer {
    host_to_rom: VecDeque<u8>,
    rom_to_host: VecDeque<u8>,
    host_tx: Option<PacomHostTxByte>,
    rom_rx: Option<PacomRomRxByte>,
    eih_high: bool,
}

#[cfg(test)]
impl Iq7000PclinkSerialPeer {
    fn attach(&mut self, memory: &mut MemoryImage) {
        self.eih_high = true;
        self.drive_eih(memory, true);
    }

    fn queue_host_bytes(&mut self, bytes: &[u8]) {
        self.host_to_rom.extend(bytes.iter().copied());
    }

    fn pop_rom_byte(&mut self) -> Option<u8> {
        self.rom_to_host.pop_front()
    }

    fn before_instruction(&mut self, memory: &mut MemoryImage) {
        self.advance_host_tx_release(memory);
        if self.host_tx.is_none() && self.rom_rx.is_none() {
            if let Some(byte) = self.host_to_rom.pop_front() {
                self.host_tx = Some(PacomHostTxByte {
                    byte,
                    bit_index: 0,
                    phase: PacomHostTxPhase::WaitRomReadyLow,
                });
                self.drive_eih(memory, false);
            }
        }
    }

    fn observe_eoh_write(&mut self, eoh_high: bool, memory: &mut MemoryImage) {
        if let Some(tx) = self.host_tx.as_mut() {
            let mut drive: Option<bool> = None;
            match tx.phase {
                PacomHostTxPhase::WaitRomReadyLow => {
                    if !eoh_high {
                        let bit = ((tx.byte >> (7 - tx.bit_index)) & 1) != 0;
                        tx.phase = PacomHostTxPhase::WaitRomAckHigh;
                        drive = Some(bit);
                    }
                }
                PacomHostTxPhase::WaitRomAckHigh => {
                    if eoh_high {
                        tx.phase = PacomHostTxPhase::ReleaseHigh(IQ7000_PACOM_RELEASE_STEPS);
                        drive = Some(true);
                    }
                }
                PacomHostTxPhase::ReleaseHigh(_) => {}
            }
            if let Some(high) = drive {
                self.drive_eih(memory, high);
            }
            return;
        }

        if let Some(rx) = self.rom_rx.as_mut() {
            let mut drive: Option<bool> = None;
            let mut completed_byte: Option<u8> = None;
            match rx.phase {
                PacomRomRxPhase::StartLow => {
                    if !eoh_high {
                        rx.phase = PacomRomRxPhase::Data;
                    }
                }
                PacomRomRxPhase::Data => {
                    rx.byte = (rx.byte << 1) | u8::from(eoh_high);
                    rx.bit_index = rx.bit_index.saturating_add(1);
                    rx.phase = PacomRomRxPhase::SpacingHigh;
                    drive = Some(true);
                }
                PacomRomRxPhase::SpacingHigh => {
                    if eoh_high {
                        if rx.bit_index >= 8 {
                            completed_byte = Some(rx.byte);
                            drive = Some(true);
                        } else {
                            rx.phase = PacomRomRxPhase::StartLow;
                            drive = Some(false);
                        }
                    }
                }
            }
            if let Some(byte) = completed_byte {
                self.rom_to_host.push_back(byte);
                self.rom_rx = None;
            }
            if let Some(high) = drive {
                self.drive_eih(memory, high);
            }
            return;
        }

        if !eoh_high {
            self.rom_rx = Some(PacomRomRxByte {
                byte: 0,
                bit_index: 0,
                phase: PacomRomRxPhase::Data,
            });
            self.drive_eih(memory, false);
        }
    }

    fn advance_host_tx_release(&mut self, memory: &mut MemoryImage) {
        let mut next_bit = false;
        let mut finished = false;
        if let Some(tx) = self.host_tx.as_mut() {
            if let PacomHostTxPhase::ReleaseHigh(remaining) = &mut tx.phase {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    tx.bit_index = tx.bit_index.saturating_add(1);
                    if tx.bit_index >= 8 {
                        finished = true;
                    } else {
                        tx.phase = PacomHostTxPhase::WaitRomReadyLow;
                        next_bit = true;
                    }
                }
            }
        }
        if finished {
            self.host_tx = None;
            self.drive_eih(memory, true);
        } else if next_bit {
            self.drive_eih(memory, false);
        }
    }

    fn drive_eih(&mut self, memory: &mut MemoryImage, high: bool) {
        self.eih_high = high;
        let current = memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0);
        let next = if high {
            current | IQ7000_PACOM_EIH_DATA
        } else {
            current & !IQ7000_PACOM_EIH_DATA
        };
        if next != current {
            memory.write_internal_byte(IMEM_EIH_OFFSET, next);
        }
    }
}

fn serve_iq7000_pclink_serial_clients(
    bind: &str,
    client_count: usize,
    runtime: &mut CoreRuntime,
) -> Result<(), Box<dyn Error>> {
    if client_count == 0 {
        return Err("--pclink-serial-clients must be greater than zero".into());
    }

    let listener = TcpListener::bind(bind)?;
    listener.set_nonblocking(true)?;
    let local_addr = listener.local_addr()?;
    println!("[pclink-serial] listening on tcp://{local_addr} for {client_count} client(s)");

    runtime.enable_sio_stub();
    if let Some(sio) = runtime.sio.as_mut() {
        sio.disable_auto_response();
    }
    runtime.memory.clear_dirty();

    let mut stream: Option<TcpStream> = None;
    let mut pending_read = VecDeque::<u8>::new();
    let mut pending_write = VecDeque::<u8>::new();
    let mut rx_pace_steps = 0u32;
    let mut rom_flow_paused = false;
    let mut served = 0usize;
    let mut accepted = 0usize;
    let mut host_rx_bytes = 0usize;
    let mut rom_tx_bytes = 0usize;

    while served < client_count {
        if stream.is_none() {
            match listener.accept() {
                Ok((accepted_stream, peer_addr)) => {
                    accepted_stream.set_nonblocking(true)?;
                    accepted += 1;
                    println!(
                        "[pclink-serial] accepted client {accepted}/{client_count} from {peer_addr}"
                    );
                    stream = Some(accepted_stream);
                    pending_read.clear();
                    pending_write.clear();
                    rx_pace_steps = 0;
                    rom_flow_paused = false;
                    host_rx_bytes = 0;
                    rom_tx_bytes = 0;
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => {}
                Err(err) => return Err(err.into()),
            }
        }

        let mut close_stream = false;
        if let Some(client) = stream.as_mut() {
            let mut read_buf = [0u8; 256];
            loop {
                match client.read(&mut read_buf) {
                    Ok(0) => {
                        close_stream = true;
                        break;
                    }
                    Ok(n) => {
                        host_rx_bytes = host_rx_bytes.saturating_add(n);
                        println!(
                            "[pclink-serial] host->rom {} byte(s): {}",
                            n,
                            encode_hex(&read_buf[..n])
                        );
                        for byte in &read_buf[..n] {
                            pending_read.push_back(*byte);
                        }
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                    Err(err)
                        if matches!(
                            err.kind(),
                            ErrorKind::ConnectionReset | ErrorKind::BrokenPipe
                        ) =>
                    {
                        close_stream = true;
                        break;
                    }
                    Err(err) => return Err(err.into()),
                }
            }
        }

        if rx_pace_steps == 0 && !rom_flow_paused {
            let sio_idle = runtime.sio.as_ref().is_none_or(|sio| {
                sio.pending_receive().is_empty() && sio.pending_delayed_receive().is_empty()
            });
            if sio_idle {
                if let Some(byte) = pending_read.pop_front() {
                    runtime.queue_sio_receive_byte(byte);
                    rx_pace_steps = PCLINK_SERIAL_RX_PACE_STEPS;
                }
            }
        }
        runtime.step(1)?;
        rx_pace_steps = rx_pace_steps.saturating_sub(1);
        let mut completed_tx = Vec::new();
        if let Some(sio) = runtime.sio.as_mut() {
            while let Some(byte) = sio.complete_transmit(&mut runtime.memory) {
                completed_tx.push(byte);
            }
        }
        for byte in completed_tx {
            if byte == PCLINK_SERIAL_XOFF {
                rom_flow_paused = true;
            } else if byte == PCLINK_SERIAL_XON {
                rom_flow_paused = false;
            }
            pending_write.push_back(byte);
            rom_tx_bytes = rom_tx_bytes.saturating_add(1);
        }

        if let Some(client) = stream.as_mut() {
            while !pending_write.is_empty() {
                let buf: Vec<u8> = pending_write.iter().take(256).copied().collect();
                match client.write(&buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        for _ in 0..n {
                            pending_write.pop_front();
                        }
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                    Err(err)
                        if matches!(
                            err.kind(),
                            ErrorKind::ConnectionReset | ErrorKind::BrokenPipe
                        ) =>
                    {
                        close_stream = true;
                        break;
                    }
                    Err(err) => return Err(err.into()),
                }
            }
        }

        if close_stream {
            stream = None;
            pending_write.clear();
            served += 1;
            println!(
                "[pclink-serial] client {served}/{client_count} disconnected ({host_rx_bytes} host->rom byte(s), {rom_tx_bytes} rom->host byte(s))"
            );
        }
    }

    println!("[pclink-serial] served {client_count} client(s)");
    runtime.step(PCLINK_SERIAL_POST_CLIENT_SETTLE_STEPS)?;
    Ok(())
}

fn perfetto_part_path(base: &Path, part: u32) -> PathBuf {
    if part == 0 {
        return base.to_path_buf();
    }
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pc-e500");
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let ext = base.extension().and_then(|e| e.to_str()).unwrap_or("");
    let filename = if ext.is_empty() {
        format!("{stem}.part{part:03}")
    } else {
        format!("{stem}.part{part:03}.{ext}")
    };
    parent.join(filename)
}

fn rotate_perfetto_trace(base: &Path, part: u32) {
    let mut guard = PERFETTO_TRACER.enter();
    if let Some(tracer) = guard.take() {
        if let Err(err) = tracer.finish() {
            eprintln!("[perfetto] failed to save trace chunk: {err}");
        }
    }
    let next_path = perfetto_part_path(base, part);
    guard.replace(Some(PerfettoTracer::new(next_path)));
}

async fn sleep_for_cycles(cycles: u64) {
    if cycles == 0 {
        sleep_cycles(1).await;
        return;
    }
    sleep_cycles(cycles).await;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OffWakeGate {
    NotOff,
    WaitingForOnKey,
    WokeOnKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HaltWakeBoundary {
    NotHalted,
    WaitingForInterrupt,
    WokeForInterrupt,
}

/// Apply the standalone loop's provisional OFF wake policy before generic IRQ
/// delivery. Non-ONK status and pending bookkeeping are retained verbatim;
/// only an asserted ONKI bit transitions the CPU back to running.
fn apply_off_wake_gate(state: &mut LlamaState, bus: &mut StandaloneBus) -> OffWakeGate {
    if !state.is_off() {
        return OffWakeGate::NotOff;
    }

    let isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
    if (isr & ISR_ONKI) == 0 {
        return OffWakeGate::WaitingForOnKey;
    }

    state.set_halted(false);
    bus.irq_pending = true;
    bus.timer.irq_pending = true;
    bus.timer.irq_isr = isr;
    bus.last_irq_src = Some("ONK".to_string());
    bus.timer.irq_source = Some("ONK".to_string());
    bus.timer.last_fired = Some("ONK".to_string());
    OffWakeGate::WokeOnKey
}

/// Model HALT wake as an idle scheduling boundary. A status bit may cancel
/// HALT and arm delivery, but the interrupted PC is not delivered or executed
/// until the following scheduling pass.
fn apply_halt_wake_boundary(state: &mut LlamaState, bus: &mut StandaloneBus) -> HaltWakeBoundary {
    if state.is_off() || !state.is_halted() {
        return HaltWakeBoundary::NotHalted;
    }

    // Timers continue while HALTed and may provide the wake source for this
    // idle boundary. Match the Python runtime's tick-before-wake ordering.
    bus.tick_timers_only(bus.cycle_count);
    let isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
    bus.cycle_count = bus.cycle_count.wrapping_add(1);
    if isr == 0 {
        return HaltWakeBoundary::WaitingForInterrupt;
    }

    state.set_halted(false);
    bus.irq_pending = true;
    bus.timer.irq_pending = true;
    bus.timer.irq_isr = isr;
    bus.last_irq_src = None;
    for (mask, src) in [
        (ISR_ONKI, "ONK"),
        (ISR_KEYI, "KEY"),
        (ISR_STI, "STI"),
        (ISR_MTI, "MTI"),
    ] {
        if (isr & mask) != 0 {
            bus.last_irq_src = Some(src.to_string());
            bus.timer.irq_source = Some(src.to_string());
            break;
        }
    }
    HaltWakeBoundary::WokeForInterrupt
}

/// Decide whether the current PC can execute on this scheduling pass using
/// only callback-free state. A low-power wake is an idle boundary, while an
/// already-unmasked interrupt replaces the dormant/current PC on the next
/// running pass.
fn current_instruction_requires_silent_preflight(state: &LlamaState, bus: &StandaloneBus) -> bool {
    if state.is_halted() {
        return false;
    }

    let imr = bus
        .memory
        .read_internal_byte_silent(IMEM_IMR_OFFSET)
        .unwrap_or(0);
    let mut isr = bus
        .memory
        .read_internal_byte_silent(IMEM_ISR_OFFSET)
        .unwrap_or(0);
    let raw_kil = if bus
        .memory
        .read_internal_byte_silent(IMEM_LCC_OFFSET)
        .unwrap_or(0)
        & 0x04
        == 0
    {
        bus.keyboard.compute_physical_kil()
    } else {
        0
    };
    if raw_kil != 0 {
        isr |= ISR_KEYI;
    }
    if bus.pending_onk {
        isr |= ISR_ONKI;
    }
    let irq_replaces_pc = !bus.in_interrupt && (imr & IMR_MASTER) != 0 && (imr & isr) != 0;
    !irq_replaces_pc
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SilentInstructionPreflight {
    source_pc: u32,
    opcode: u8,
    instruction_len: u8,
    provenance: (usize, u64),
}

fn prepare_preflighted_transfer_and_tick(
    opcode: u8,
    state: &LlamaState,
    bus: &mut StandaloneBus,
    preflight: SilentInstructionPreflight,
    run_timer_cycles: bool,
    pre_tick_done: bool,
) -> Result<Option<ValidatedVectorTransfer>, &'static str> {
    if state.pc() & ADDRESS_MASK != preflight.source_pc || opcode != preflight.opcode {
        return Err("silent instruction preflight does not match architectural fetch");
    }
    if bus.vector_transfer_provenance() != preflight.provenance {
        return Err("silent instruction preflight memory mapping changed");
    }
    debug_assert!(preflight.instruction_len > 0);
    let transfer = match opcode {
        0xFE => Some(prepare_validated_vector(INTERRUPT_VECTOR_ADDR, state, bus)?),
        0xFF => Some(fetch_validated_vector(ROM_RESET_VECTOR_ADDR, state, bus)?),
        _ => None,
    };
    if run_timer_cycles && !pre_tick_done {
        bus.tick_timers_only(bus.cycle_count);
    }
    Ok(transfer)
}

fn preflight_and_tick_instruction(
    executor: &LlamaExecutor,
    opcode: u8,
    state: &LlamaState,
    bus: &mut StandaloneBus,
    run_timer_cycles: bool,
    pre_tick_done: bool,
) -> Result<Option<ValidatedVectorTransfer>, &'static str> {
    let instruction_len = executor.validate_before_scheduling_with_length(opcode, state, bus)?;
    let source_pc = state.pc() & ADDRESS_MASK;
    if (0..u32::from(instruction_len)).any(|offset| {
        !bus.instruction_byte_is_stable_for_context(
            source_pc.wrapping_add(offset) & ADDRESS_MASK,
            source_pc,
        )
    }) {
        return Err("callback-backed instruction bytes cannot cross scheduler tick");
    }
    match opcode {
        0xFE => {
            validate_stable_vector_transfer(INTERRUPT_VECTOR_ADDR, state, bus)?;
        }
        0xFF => {
            validate_stable_vector_transfer(ROM_RESET_VECTOR_ADDR, state, bus)?;
        }
        _ => {}
    }
    prepare_preflighted_transfer_and_tick(
        opcode,
        state,
        bus,
        SilentInstructionPreflight {
            source_pc,
            opcode,
            instruction_len,
            provenance: bus.vector_transfer_provenance(),
        },
        run_timer_cycles,
        pre_tick_done,
    )
}

/// Decode and validate the instruction at the state's current PC without
/// consuming an architectural read or advancing trace-replay input.
fn preflight_current_instruction_silently(
    executor: &LlamaExecutor,
    state: &LlamaState,
    bus: &mut StandaloneBus,
) -> Result<SilentInstructionPreflight, &'static str> {
    let source_pc = state.pc() & ADDRESS_MASK;
    let opcode = bus
        .peek_byte_silent_at(source_pc, source_pc)
        .ok_or(sc62015_core::llama::eval::SILENT_PEEK_UNAVAILABLE_ERROR)?;
    let instruction_len = executor.validate_before_scheduling_with_length(opcode, state, bus)?;
    if (0..u32::from(instruction_len)).any(|offset| {
        !bus.instruction_byte_is_stable_for_context(
            source_pc.wrapping_add(offset) & ADDRESS_MASK,
            source_pc,
        )
    }) {
        return Err("callback-backed instruction bytes cannot cross scheduler tick");
    }
    match opcode {
        0xFE => {
            validate_stable_vector_transfer(INTERRUPT_VECTOR_ADDR, state, bus)?;
        }
        0xFF => {
            validate_stable_vector_transfer(ROM_RESET_VECTOR_ADDR, state, bus)?;
        }
        _ => {}
    }
    Ok(SilentInstructionPreflight {
        source_pc,
        opcode,
        instruction_len,
        provenance: bus.vector_transfer_provenance(),
    })
}

/// Validate a vector and every decoded destination byte with the exact
/// PC-sensitive context used by silent decode. No architectural vector read is
/// performed here.
fn validate_stable_vector_transfer(
    vector_addr: u32,
    state: &LlamaState,
    bus: &mut StandaloneBus,
) -> Result<(u32, u8), &'static str> {
    let (target, target_len) = validate_vector_transfer_with_length(vector_addr, state, bus)?;
    let source_pc = state.pc() & ADDRESS_MASK;
    let vector_unstable = (0..3).any(|offset| {
        !bus.instruction_byte_is_stable_for_context(
            vector_addr.wrapping_add(offset) & ADDRESS_MASK,
            source_pc,
        )
    });
    let target_unstable = (0..u32::from(target_len)).any(|offset| {
        !bus.instruction_byte_is_stable_for_context(
            target.wrapping_add(offset) & ADDRESS_MASK,
            target,
        )
    });
    if vector_unstable || target_unstable {
        return Err("callback-backed vector/target cannot cross scheduler tick");
    }
    Ok((target, target_len))
}

fn runtime_set_physical_matrix_event(runtime: &mut CoreRuntime, code: u8, release: bool) {
    runtime.set_physical_matrix_key(code, !release);
}

fn runtime_tap_event(
    runtime: &mut CoreRuntime,
    code: u8,
    hold_instructions: usize,
) -> Result<(), Box<dyn Error>> {
    runtime_set_physical_matrix_event(runtime, code, false);
    runtime.step(hold_instructions)?;
    runtime_set_physical_matrix_event(runtime, code, true);
    Ok(())
}

fn runtime_tap_key(
    runtime: &mut CoreRuntime,
    key: AutoKeyKind,
    hold_instructions: usize,
    after_instructions: usize,
) -> Result<(), Box<dyn Error>> {
    runtime_apply_key_event(runtime, key, false);
    runtime.step(hold_instructions)?;
    runtime_apply_key_event(runtime, key, true);
    runtime.step(after_instructions)?;
    Ok(())
}

fn runtime_apply_key_event(runtime: &mut CoreRuntime, key: AutoKeyKind, release: bool) {
    match key {
        AutoKeyKind::Matrix(code) => {
            runtime.set_physical_matrix_key(code, !release);
        }
        AutoKeyKind::Chord { modifier, code } => {
            if release {
                runtime_apply_key_event(runtime, AutoKeyKind::Matrix(code), true);
                runtime_apply_key_event(runtime, AutoKeyKind::Matrix(modifier), true);
            } else {
                runtime_apply_key_event(runtime, AutoKeyKind::Matrix(modifier), false);
                runtime_apply_key_event(runtime, AutoKeyKind::Matrix(code), false);
            }
        }
        AutoKeyKind::Event(code) => runtime_set_physical_matrix_event(runtime, code, release),
        AutoKeyKind::InputEvent(code) => {
            if release {
                return;
            }
            runtime.queue_translated_key_event(code);
        }
        AutoKeyKind::OnKey => {
            if release {
                runtime.release_on_key();
            } else {
                runtime.press_on_key();
            }
        }
    }
}

fn runtime_lcd_lines(
    runtime: &CoreRuntime,
    text_decoder: Option<&DeviceTextDecoder>,
) -> Vec<String> {
    let Some(decoder) = text_decoder else {
        return Vec::new();
    };
    let Some(lcd) = runtime.lcd.as_deref() else {
        return Vec::new();
    };
    decoder.decode_display_text(lcd)
}

fn write_runtime_lcd_capture(
    model: DeviceModel,
    runtime: &CoreRuntime,
    text_decoder: Option<&DeviceTextDecoder>,
    capture_png: Option<&PathBuf>,
    capture_json: Option<&PathBuf>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let lcd_lines = runtime_lcd_lines(runtime, text_decoder);
    if let Some(path) = capture_png {
        let lcd = runtime.lcd.as_deref().ok_or("missing LCD runtime")?;
        write_lcd_png(
            path,
            &lcd_pixels(lcd),
            LCD_CAPTURE_SCALE,
            lcd_annunciators(model, &runtime.memory).as_ref(),
        )?;
        println!("Wrote LCD PNG capture: {}", path.display());
    }
    if let Some(path) = capture_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let capture = json!({
            "executed": runtime.instruction_count(),
            "pc": format!("0x{:05X}", runtime.state.pc() & ADDRESS_MASK),
            "halted": runtime.state.is_halted(),
            "lcd_lines": lcd_lines,
            "lcd_annunciators": lcd_annunciators(model, &runtime.memory),
        });
        fs::write(path, serde_json::to_string_pretty(&capture)?)?;
        println!("Wrote LCD JSON capture: {}", path.display());
    }
    Ok(lcd_lines)
}

fn run_runtime_key_seq(
    runtime: &mut CoreRuntime,
    raw_key_seq: &str,
    max_instructions: u64,
    model: DeviceModel,
    text_decoder: Option<&DeviceTextDecoder>,
    log_enabled: bool,
    diagnostics: &mut RuntimeDiagnostics,
) -> Result<u64, Box<dyn Error>> {
    let actions = parse_key_seq(raw_key_seq, KEY_SEQ_DEFAULT_HOLD, model)
        .map_err(|err| format!("--key-seq: {err}"))?;
    if actions.is_empty() {
        return Ok(0);
    }
    let mut needs_screen_state = false;
    let mut needs_screen_text = false;
    for action in &actions {
        match action.kind {
            KeySeqKind::WaitScreenChange
            | KeySeqKind::WaitScreenEmpty
            | KeySeqKind::WaitScreenDraw => needs_screen_state = true,
            KeySeqKind::WaitText => needs_screen_text = true,
            _ => {}
        }
    }
    let mut runner = KeySeqRunner::new(actions);
    runner.set_log_enabled(log_enabled);
    let start = runtime.instruction_count();
    let mut elapsed = 0_u64;
    while elapsed < max_instructions {
        let schedule_index = start.saturating_add(elapsed);
        let screen_state = if needs_screen_state || needs_screen_text {
            let lcd = runtime.lcd.as_deref().ok_or("missing LCD runtime")?;
            capture_screen_state(lcd, text_decoder, needs_screen_text)
        } else {
            ScreenState::default()
        };
        let events = runner.step(schedule_index, !runtime.state.is_off(), &screen_state);
        for event in events {
            match event.kind {
                KeySeqEventKind::Press => {
                    if let Some(key) = event.key {
                        runtime_apply_key_event(runtime, key, false);
                    }
                }
                KeySeqEventKind::Release => {
                    if let Some(key) = event.key {
                        runtime_apply_key_event(runtime, key, true);
                    }
                }
                KeySeqEventKind::Log => println!("{}", event.message),
            }
        }
        if runner.is_complete() {
            println!("key-seq: completed at {schedule_index}");
            return Ok(elapsed);
        }
        let now = schedule_index;
        let mut step_count = 1_u64;
        if let Some(release_at) = runner.active_key.map(|_| runner.active_release_at) {
            if release_at > now {
                step_count = release_at - now;
            }
        } else if runner.action_index < runner.actions.len() {
            let action = &runner.actions[runner.action_index];
            match action.kind {
                KeySeqKind::WaitOp if action.op_target_set && action.op_target > now => {
                    step_count = action.op_target - now;
                }
                KeySeqKind::WaitText | KeySeqKind::WaitScreenChange => {
                    step_count = 1_000;
                }
                KeySeqKind::WaitScreenEmpty | KeySeqKind::WaitScreenDraw => {
                    step_count = 1_000;
                }
                _ => {}
            }
        }
        let remaining = max_instructions.saturating_sub(elapsed);
        let bounded = step_count.min(remaining).max(1);
        let consumed = step_runtime_boundaries(runtime, bounded, diagnostics)?;
        elapsed = elapsed.saturating_add(consumed);
        if diagnostics.stopped {
            return Ok(elapsed);
        }
    }
    Err(format!("--key-seq did not complete within {max_instructions} instruction(s)").into())
}

fn iq7000_seed_memo_lines(raw: &str) -> Vec<String> {
    raw.replace("\\n", "\n")
        .split(['\n', '|'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_ascii_uppercase())
        .collect()
}

fn type_iq7000_memo_lines_via_rom_ui(
    runtime: &mut CoreRuntime,
    label: &str,
    lines: &[String],
) -> Result<(), Box<dyn Error>> {
    if lines.is_empty() {
        return Err(format!("{label} is empty").into());
    }
    for (line_idx, line) in lines.iter().enumerate() {
        if line_idx != 0 {
            runtime_tap_key(runtime, AutoKeyKind::Event(0x3D), 35_000, 70_000)?;
        }
        for ch in line.chars() {
            let Some(key) = generated_key_seq_key(DeviceModel::Iq7000, ch) else {
                return Err(
                    format!("{label} contains unmapped IQ-7000 input character {ch:?}").into(),
                );
            };
            runtime_tap_key(runtime, key, 35_000, 55_000)?;
        }
    }
    runtime_tap_key(runtime, AutoKeyKind::Event(0x45), 35_000, 1_000_000)?; // ENTER/store
    Ok(())
}

fn seed_iq7000_memos_via_rom_ui(
    runtime: &mut CoreRuntime,
    memos: &[String],
) -> Result<(), Box<dyn Error>> {
    if memos.is_empty() {
        return Ok(());
    }
    println!(
        "[iq7000-seed] typing {} MEMO entr{} through the IQ-7000 ROM UI",
        memos.len(),
        if memos.len() == 1 { "y" } else { "ies" }
    );
    for (idx, raw) in memos.iter().enumerate() {
        let lines = iq7000_seed_memo_lines(raw);
        type_iq7000_memo_lines_via_rom_ui(
            runtime,
            &format!("--iq7000-seed-memo #{}", idx + 1),
            &lines,
        )?;
        println!("[iq7000-seed] stored MEMO {}", idx + 1);
    }
    Ok(())
}

fn run_iq7000_pclink_ui_path(
    args: &Args,
    rom_bytes: &[u8],
    text_decoder: Option<&DeviceTextDecoder>,
    iq7000_clock_seed: Option<&Iq7000RtcSeed>,
) -> Result<(), Box<dyn Error>> {
    if args.snapshot_in.is_some() {
        return Err("--iq7p-enter-pclink cannot be combined with --snapshot-in".into());
    }

    let mut runtime = CoreRuntime::for_model(args.model, rom_bytes)?;
    args.card.resolve(args.model).apply(&mut runtime.memory)?;
    if let Some(seed) = iq7000_clock_seed {
        runtime.set_iq7000_clock_seed_yyyymmddhhmm(seed.clock.as_ascii())?;
    }
    runtime.power_on_reset()?;
    if let Some(seed) = iq7000_clock_seed {
        runtime.set_iq7000_clock_seed_yyyymmddhhmm(seed.clock.as_ascii())?;
    }

    runtime.step(500_000)?;
    runtime_tap_event(&mut runtime, 0x08, 1_000)?; // MEMO
    runtime.step(160_000)?;
    seed_iq7000_memos_via_rom_ui(&mut runtime, &args.iq7000_seed_memo)?;
    if !args.iq7000_seed_memo.is_empty() {
        runtime_tap_event(&mut runtime, 0x08, 1_000)?; // return to a fresh MEMO prompt
        runtime.step(250_000)?;
    }
    runtime_set_physical_matrix_event(&mut runtime, 0x02, false); // physical SHIFT down
    runtime.step(80_000)?;
    runtime_set_physical_matrix_event(&mut runtime, 0x1D, false); // shifted C / OPTION
    runtime.step(500_000)?;
    runtime_set_physical_matrix_event(&mut runtime, 0x1D, true);
    runtime_set_physical_matrix_event(&mut runtime, 0x02, true); // SHIFT up before selecting 4
    runtime.step(80_000)?;
    runtime_tap_event(&mut runtime, 0x22, 35_000)?; // physical 4 -> PC LINK
    runtime.step(800_000)?;

    let lcd_lines = runtime_lcd_lines(&runtime, text_decoder);
    let lcd_text = lcd_lines.join("\n");
    if !lcd_text.contains("LINK READY") {
        return Err(
            format!("IQ-7000 PC-Link UI did not reach LINK READY; LCD={lcd_text:?}").into(),
        );
    }
    println!("[pclink] IQ-7000 ROM UI reached PC-Link mode");
    println!("LCD (decoded text):");
    for line in &lcd_lines {
        println!("  {line}");
    }

    let has_post_key_seq = args
        .key_seq
        .as_ref()
        .is_some_and(|raw| !raw.trim().is_empty());
    let ready_capture_png = args.iq7p_ready_capture_png.as_ref().or_else(|| {
        (!has_post_key_seq)
            .then_some(args.capture_png.as_ref())
            .flatten()
    });
    let ready_capture_json = args.iq7p_ready_capture_json.as_ref().or_else(|| {
        (!has_post_key_seq)
            .then_some(args.capture_json.as_ref())
            .flatten()
    });
    write_runtime_lcd_capture(
        args.model,
        &runtime,
        text_decoder,
        ready_capture_png,
        ready_capture_json,
    )?;

    if let Some(bind) = args.pclink_serial_listen.as_ref() {
        serve_iq7000_pclink_serial_clients(bind, args.pclink_serial_clients, &mut runtime)?;
    }

    if let Some(raw_key_seq) = args
        .key_seq
        .as_ref()
        .map(|raw| raw.trim())
        .filter(|raw| !raw.is_empty())
    {
        let mut diagnostics = RuntimeDiagnostics::from_args(args)?;
        run_runtime_key_seq(
            &mut runtime,
            raw_key_seq,
            args.steps,
            args.model,
            text_decoder,
            args.key_seq_log,
            &mut diagnostics,
        )?;
        let final_lines = write_runtime_lcd_capture(
            args.model,
            &runtime,
            text_decoder,
            args.capture_png.as_ref(),
            args.capture_json.as_ref(),
        )?;
        println!("LCD after key-seq (decoded text):");
        for line in &final_lines {
            println!("  {line}");
        }
    }
    if let Some(path) = args.snapshot_out.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        runtime.save_snapshot(path)?;
        println!("Saved snapshot to {}", path.display());
    }
    let final_lcd_lines = runtime_lcd_lines(&runtime, text_decoder);
    write_runtime_diagnostic_artifacts(args, &runtime, &final_lcd_lines)?;

    Ok(())
}

fn core_runtime_unsupported_options(args: &Args) -> Vec<&'static str> {
    let mut unsupported = Vec::new();
    if args.lcd_log || args.lcd_log_limit.is_some() {
        unsupported.push("--lcd-log/--lcd-log-limit");
    }
    if args.dump_bus_trace.is_some() {
        unsupported.push("--dump-bus-trace");
    }
    if args.snapshot_in.is_some() || args.snapshot_out.is_some() {
        unsupported.push("--snapshot-in/--snapshot-out");
    }
    if args.turnon2_resume || args.turnon_profile.is_some() {
        unsupported.push("--turnon2-resume/--turnon-profile");
    }
    if args.reset_trace_card {
        unsupported.push("--reset-trace-card");
    }
    if args.reset_trace2_main_display || args.reset_trace2_profile.is_some() {
        unsupported.push("--reset-trace2-main-display/--reset-trace2-profile");
    }
    if !args.iq7000_seed_memo.is_empty() {
        unsupported.push("--iq7000-seed-memo (requires --iq7p-enter-pclink)");
    }
    unsupported
}

struct RuntimeDiagnostics {
    stop_pc: Option<u32>,
    trace_pcs: Vec<u32>,
    trace_pc_window: u64,
    trace_regs: bool,
    trace_pc_counts: HashMap<u32, u64>,
    trace_window_active: u64,
    trace_window_anchor: Option<u32>,
    stopped: bool,
}

impl RuntimeDiagnostics {
    fn from_args(args: &Args) -> Result<Self, Box<dyn Error>> {
        let stop_pc = args.stop_pc.as_deref().map(parse_address).transpose()?;
        let trace_pcs = args
            .trace_pc
            .iter()
            .map(|raw| parse_address(raw))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            stop_pc: stop_pc.map(|pc| pc & ADDRESS_MASK),
            trace_pcs: trace_pcs.into_iter().map(|pc| pc & ADDRESS_MASK).collect(),
            trace_pc_window: args.trace_pc_window.unwrap_or(0),
            trace_regs: args.trace_regs,
            trace_pc_counts: HashMap::new(),
            trace_window_active: 0,
            trace_window_anchor: None,
            stopped: false,
        })
    }

    fn is_active(&self) -> bool {
        self.stop_pc.is_some() || !self.trace_pcs.is_empty()
    }

    fn log_pc(&self, runtime: &CoreRuntime, label: &str, pc: u32, extra: &str) {
        let imr = runtime
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(0);
        let isr = runtime
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        if self.trace_regs {
            let a = runtime.state.get_reg(RegName::A) & 0xFF;
            let f = runtime.state.get_reg(RegName::F) & 0xFF;
            let s = runtime.state.get_reg(RegName::S) & ADDRESS_MASK;
            let y = runtime.state.get_reg(RegName::Y) & ADDRESS_MASK;
            let ssr = runtime
                .memory
                .read_internal_byte(IMEM_SSR_OFFSET)
                .unwrap_or(0);
            println!(
                "[{label}] {extra}pc=0x{pc:05X} imr=0x{imr:02X} isr=0x{isr:02X} \
                 ssr=0x{ssr:02X} a=0x{a:02X} f=0x{f:02X} sp=0x{s:05X} y=0x{y:05X}"
            );
        } else {
            println!("[{label}] {extra}pc=0x{pc:05X} imr=0x{imr:02X} isr=0x{isr:02X}");
        }
    }

    fn before_boundary(&mut self, runtime: &CoreRuntime) {
        if runtime.state.is_halted() || runtime.state.is_off() {
            return;
        }
        let pc = runtime.state.pc() & ADDRESS_MASK;
        if self.trace_pcs.contains(&pc) {
            let count = self
                .trace_pc_counts
                .entry(pc)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            let count = *count;
            if count <= 10 || count.is_multiple_of(1000) {
                self.log_pc(runtime, "pc-trace", pc, &format!("hits={count} "));
            }
            if self.trace_pc_window != 0 {
                self.trace_window_active = self.trace_pc_window;
                self.trace_window_anchor = Some(pc);
            }
        } else if self.trace_window_active != 0 {
            let anchor = self
                .trace_window_anchor
                .map(|anchor| format!("0x{anchor:05X}"))
                .unwrap_or_else(|| "n/a".to_string());
            self.log_pc(
                runtime,
                "pc-trace-window",
                pc,
                &format!("anchor={anchor} remaining={} ", self.trace_window_active),
            );
            self.trace_window_active = self.trace_window_active.saturating_sub(1);
        }
    }

    fn after_boundary(&mut self, runtime: &CoreRuntime, instruction_count_before: u64) {
        if runtime.instruction_count() == instruction_count_before {
            return;
        }
        if self
            .stop_pc
            .is_some_and(|stop| runtime.state.pc() & ADDRESS_MASK == stop)
        {
            self.stopped = true;
        }
    }
}

fn step_runtime_boundaries(
    runtime: &mut CoreRuntime,
    mut boundaries: u64,
    diagnostics: &mut RuntimeDiagnostics,
) -> Result<u64, Box<dyn Error>> {
    let mut consumed = 0_u64;
    while boundaries != 0 {
        if diagnostics.stopped {
            break;
        }
        let chunk = if diagnostics.is_active() {
            1
        } else {
            usize::try_from(boundaries).unwrap_or(usize::MAX)
        };
        diagnostics.before_boundary(runtime);
        let instruction_count_before = runtime.instruction_count();
        runtime.step(chunk)?;
        diagnostics.after_boundary(runtime, instruction_count_before);
        boundaries -= chunk as u64;
        consumed = consumed.saturating_add(chunk as u64);
    }
    Ok(consumed)
}

fn validate_lcd_expectations(args: &Args, lcd_lines: &[String]) -> Result<(), Box<dyn Error>> {
    let mut failures = Vec::new();
    for raw in &args.expect_row {
        match parse_expected_row(raw) {
            Ok((idx, expected)) => {
                let actual = lcd_lines.get(idx).cloned().unwrap_or_default();
                if !actual.contains(&expected) {
                    failures.push(format!(
                        "expect-row failed: row {idx} missing substring '{expected}' (got '{actual}')"
                    ));
                }
            }
            Err(error) => failures.push(error),
        }
    }
    for needle in &args.expect_text {
        if !lcd_lines.iter().any(|line| line.contains(needle)) {
            failures.push(format!(
                "expect-text failed: substring '{needle}' not found in LCD text"
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join(" | ").into())
    }
}

fn run_core_runtime_path(
    args: &Args,
    rom_bytes: &[u8],
    text_decoder: Option<&DeviceTextDecoder>,
    iq7000_clock_seed: Option<&Iq7000RtcSeed>,
) -> Result<(), Box<dyn Error>> {
    let unsupported = core_runtime_unsupported_options(args);
    if !unsupported.is_empty() {
        return Err(format!(
            "the shared core runtime does not yet expose specialized diagnostic option(s) {}; rerun explicitly with --runtime legacy",
            unsupported.join(", ")
        )
        .into());
    }

    let timer_profile = args.model.timer_profile();
    let card_profile = args.card.resolve(args.model);
    eprintln!("[runtime] shared-core");
    eprintln!(
        "[timing] model={} timebase_hz={} unit=sc62015-relative mti_short={} mti_long={} sti_short={} sti_long={} source={}",
        args.model.label(),
        timer_profile.timebase_hz,
        timer_profile.mti_period,
        timer_profile.mti_long_period,
        timer_profile.sti_period,
        timer_profile.sti_long_period,
        timer_profile.provenance_label()
    );
    eprintln!(
        "[card] model={} mode={}",
        args.model.label(),
        if card_profile.is_present() {
            "blank-writable-64k"
        } else {
            "absent"
        }
    );

    let mut runtime = CoreRuntime::for_model(args.model, rom_bytes)?;
    card_profile.apply(&mut runtime.memory)?;
    runtime.timer.enabled = !args.disable_timers;
    if let Some(seed) = iq7000_clock_seed {
        runtime.set_iq7000_clock_seed_yyyymmddhhmm(seed.clock.as_ascii())?;
    }
    runtime.power_on_reset()?;
    if let Some(seed) = iq7000_clock_seed {
        runtime.set_iq7000_clock_seed_yyyymmddhhmm(seed.clock.as_ascii())?;
    }
    let mut diagnostics = RuntimeDiagnostics::from_args(args)?;

    if args.perfetto {
        if let Ok(symbols) = load_bnida_names(args.model, args.bnida.clone()) {
            if !symbols.is_empty() {
                set_call_ui_function_names(symbols);
            }
        }
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(existing) = guard.take() {
            guard.replace(Some(existing));
            return Err("a Perfetto trace is already active".into());
        }
        guard.replace(Some(PerfettoTracer::new(args.perfetto_path.clone())));
    }

    let started = Instant::now();
    let execution_result = (|| -> Result<(), Box<dyn Error>> {
        let consumed = if let Some(raw_key_seq) = args
            .key_seq
            .as_ref()
            .map(|raw| raw.trim())
            .filter(|raw| !raw.is_empty())
        {
            run_runtime_key_seq(
                &mut runtime,
                raw_key_seq,
                args.steps,
                args.model,
                text_decoder,
                args.key_seq_log,
                &mut diagnostics,
            )?
        } else {
            0
        };
        step_runtime_boundaries(
            &mut runtime,
            args.steps.saturating_sub(consumed),
            &mut diagnostics,
        )?;
        Ok(())
    })();

    let trace_result = if args.perfetto {
        let tracer = PERFETTO_TRACER.enter().take();
        tracer
            .map(|tracer| tracer.finish())
            .transpose()
            .map_err(|error| -> Box<dyn Error> { Box::new(error) })
    } else {
        Ok(None)
    };
    execution_result?;
    trace_result?;

    let lcd_lines = write_runtime_lcd_capture(
        args.model,
        &runtime,
        text_decoder,
        args.capture_png.as_ref(),
        args.capture_json.as_ref(),
    )?;
    write_runtime_diagnostic_artifacts(args, &runtime, &lcd_lines)?;
    println!("LCD (decoded text):");
    for line in &lcd_lines {
        println!("  {line}");
    }
    println!(
        "Executed {} instruction(s) across at most {} scheduler boundaries; PC=0x{:05X}",
        runtime.instruction_count(),
        args.steps,
        runtime.state.pc() & ADDRESS_MASK
    );
    if args.perf {
        let rate = runtime.instruction_count() as f64 / started.elapsed().as_secs_f64();
        println!(
            "Perf: {:.2} MIPS ({} instr / {:.3?})",
            rate / 1_000_000.0,
            runtime.instruction_count(),
            started.elapsed()
        );
    }
    validate_lcd_expectations(args, &lcd_lines)
}

fn run(mut args: Args) -> Result<(), Box<dyn Error>> {
    apply_scenario(&mut args)?;

    if args.iq7p_enter_pclink && args.model != DeviceModel::Iq7000 {
        return Err("--iq7p-enter-pclink is only supported for --model iq-7000".into());
    }
    if args.pclink_serial_listen.is_some() && !args.iq7p_enter_pclink {
        return Err("--pclink-serial-listen requires --iq7p-enter-pclink".into());
    }

    if args.turnon2_resume && args.model != DeviceModel::PcE500Jp {
        return Err("--turnon2-resume is only supported for --model pc-e500-jp".into());
    }
    if args.reset_trace2_main_display && args.model != DeviceModel::PcE500Jp {
        return Err("--reset-trace2-main-display is only supported for --model pc-e500-jp".into());
    }
    if args.snapshot_in.is_some() && (args.turnon2_resume || args.turnon_profile.is_some()) {
        return Err(
            "--snapshot-in cannot be combined with --turnon2-resume/--turnon-profile".into(),
        );
    }
    if args.snapshot_in.is_some()
        && (args.reset_trace2_main_display || args.reset_trace2_profile.is_some())
    {
        return Err(
            "--snapshot-in cannot be combined with --reset-trace2-main-display/--reset-trace2-profile"
                .into(),
        );
    }
    if args.reset_trace_card
        && (args.reset_trace2_main_display || args.reset_trace2_profile.is_some())
    {
        return Err(
            "--reset-trace-card cannot be combined with --reset-trace2-main-display/--reset-trace2-profile"
                .into(),
        );
    }
    let iq7000_clock_seed = if args.model == DeviceModel::Iq7000 {
        parse_iq7000_rtc_arg(&args.iq7000_rtc).map_err(|err| format!("--iq7000-rtc: {err}"))?
    } else {
        None
    };

    let rom_path = args
        .rom
        .clone()
        .unwrap_or_else(|| default_rom_path(args.model));
    let rom_bytes = load_rom(&rom_path)?;
    let text_decoder = args.model.text_decoder(&rom_bytes);
    if args.iq7p_enter_pclink {
        if args.runtime == RuntimeEngine::Legacy {
            return Err("--iq7p-enter-pclink requires --runtime core".into());
        }
        return run_iq7000_pclink_ui_path(
            &args,
            &rom_bytes,
            text_decoder.as_ref(),
            iq7000_clock_seed.as_ref(),
        );
    }
    if args.runtime == RuntimeEngine::Core {
        eprintln!(
            "[rom] model={} path={}",
            args.model.label(),
            rom_path.display()
        );
        return run_core_runtime_path(
            &args,
            &rom_bytes,
            text_decoder.as_ref(),
            iq7000_clock_seed.as_ref(),
        );
    }
    eprintln!("[runtime] legacy-diagnostic");
    let turnon_profile = if args.turnon2_resume || args.turnon_profile.is_some() {
        load_turnon_resume_profile(args.model, args.turnon_profile.clone())?
    } else {
        None
    };
    let reset_trace2_profile =
        if args.reset_trace2_main_display || args.reset_trace2_profile.is_some() {
            load_reset_trace2_main_display_profile(args.model, args.reset_trace2_profile.clone())?
        } else {
            None
        };
    let trace_resume_mode = turnon_profile.is_some() || reset_trace2_profile.is_some();

    let log_lcd = args.lcd_log;
    let log_lcd_limit = args.lcd_log_limit.unwrap_or(50);
    let log_dbg = |_msg: &str| {};
    let debug_probe_ranges = parse_debug_probe_ranges(&args.debug_probe_range)
        .map_err(|err| format!("--debug-probe-range: {err}"))?;
    let wants_debug_probe = args.debug_probe_json.is_some()
        || args.capture_json.is_some()
        || !debug_probe_ranges.is_empty();

    let perfetto_base_path = args.perfetto_path.clone();

    if args.perfetto {
        // Install BNIDA-derived function names (if available) so the "Functions" track labels
        // resolve to stable names instead of sub_XXXXXX fallbacks.
        if let Ok(symbols) = load_bnida_names(args.model, args.bnida.clone()) {
            if !symbols.is_empty() {
                set_call_ui_function_names(symbols);
            }
        }
        // Chunk long traces to avoid OOM (retrobus-perfetto buffers in memory).
        // The output will be `${perfetto_path}.partNNN.perfetto-trace` for each chunk.
        rotate_perfetto_trace(&perfetto_base_path, 0);
    }

    eprintln!(
        "[rom] model={} path={}",
        args.model.label(),
        rom_path.display()
    );

    let mut key_seq_runner = KeySeqRunner::new(Vec::new());
    let mut use_key_seq = false;
    let mut needs_screen_state = false;
    let mut needs_screen_text = false;
    if let Some(raw) = args
        .key_seq
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let actions = parse_key_seq(raw, KEY_SEQ_DEFAULT_HOLD, args.model)
            .map_err(|err| format!("--key-seq: {err}"))?;
        if !actions.is_empty() {
            for action in &actions {
                match action.kind {
                    KeySeqKind::WaitScreenChange
                    | KeySeqKind::WaitScreenEmpty
                    | KeySeqKind::WaitScreenDraw => needs_screen_state = true,
                    KeySeqKind::WaitText => needs_screen_text = true,
                    _ => {}
                }
            }
            key_seq_runner.reset(actions);
            key_seq_runner.set_log_enabled(args.key_seq_log);
            use_key_seq = true;
        }
    }
    let perfetto_chunk_size: u64 = 0;

    // Parity: do not auto-strobe; rely on ROM strobes.
    let trace_kbd = false;

    let stop_pc = if let Some(pc_str) = args.stop_pc.as_ref() {
        Some(parse_address(pc_str)?)
    } else {
        None
    };
    let trace_pcs: Vec<u32> = args
        .trace_pc
        .iter()
        .map(|raw| parse_address(raw))
        .collect::<Result<_, _>>()?;
    let trace_pc_window = args.trace_pc_window.unwrap_or(0);
    let use_full_system_image = matches!(args.model, DeviceModel::PcE500Jp);

    let mut memory = MemoryImage::new();
    match args.model {
        DeviceModel::Iq7000 => iq7000::load_iq7000_rom_image_into_memory(&mut memory, &rom_bytes),
        DeviceModel::PcE500Jp if use_full_system_image => {
            load_pce500_system_image_into_memory(&mut memory, &rom_bytes);
        }
        _ => {
            // Default standalone path mirrors the legacy Python top-window load.
            load_pce500_rom_window_into_memory(&mut memory, &rom_bytes);
        }
    }
    let readonly_ranges = if matches!(args.model, DeviceModel::Iq7000) {
        vec![(iq7000::ROM_READONLY_START, iq7000::ROM_READONLY_END)]
    } else {
        vec![
            (NO_RAM_WINDOW_START as u32, NO_RAM_WINDOW_END as u32),
            (
                ROM_WINDOW_START as u32,
                (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32,
            ),
        ]
    };
    memory.set_readonly_ranges(readonly_ranges);
    memory.set_keyboard_bridge(false);

    let card_profile = if trace_resume_mode {
        DeviceMemoryCardProfile::Absent
    } else {
        args.card.resolve(args.model)
    };
    card_profile.apply(&mut memory)?;
    eprintln!(
        "[card] model={} mode={}",
        args.model.label(),
        if card_profile.is_present() {
            "blank-writable-64k"
        } else {
            "absent"
        }
    );

    let timer_profile = args.model.timer_profile();
    eprintln!(
        "[timing] model={} timebase_hz={} unit=sc62015-relative mti_short={} mti_long={} sti_short={} sti_long={} source={}",
        args.model.label(),
        timer_profile.timebase_hz,
        timer_profile.mti_period,
        timer_profile.mti_long_period,
        timer_profile.sti_period,
        timer_profile.sti_long_period,
        timer_profile.provenance_label()
    );
    let perfetto = args.perfetto.then(|| {
        let mut irq_path = args.perfetto_path.clone();
        let stem = irq_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pc-e500");
        irq_path.set_file_name(format!("{stem}.irq.perfetto-trace"));
        log_dbg(&format!("irq perfetto path: {}", irq_path.display()));
        IrqPerfetto::new(irq_path)
    });
    let lcd_kind = args.model.lcd_kind();
    let mut lcd = create_lcd(lcd_kind);
    sc62015_core::device::configure_lcd_char_tracing(lcd.as_mut(), args.model, &rom_bytes);
    let mut bus = StandaloneBus::new(
        memory,
        lcd,
        timer_profile.new_context(!args.disable_timers),
        log_lcd,
        log_lcd_limit,
        trace_kbd,
        perfetto,
        None,
        None,
    );
    if let Some(seed) = iq7000_clock_seed.as_ref() {
        eprintln!(
            "[rtc] IQ-7000 clock workspace {} ({})",
            seed.clock.as_ascii(),
            seed.source
        );
        bus.install_iq7000_clock_seed(seed.clone());
    }
    if let Some(path) = args.dump_bus_trace.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = fs::File::create(path)?;
        bus.set_bus_trace(Some(BufWriter::new(file)));
    }
    configure_bus_for_model(&mut bus, args.model);
    if args.reset_trace_card {
        bus.enable_reset_trace_card();
    }
    // Keep default timer-driven scans unless tests override the flag.
    bus.timer.set_preserve_phase(false);
    let mut state = LlamaState::new();
    let executor = LlamaExecutor::new();
    let mut base_instruction_count: u64 = 0;
    if let Some(snapshot_path) = args.snapshot_in.as_ref() {
        let metadata =
            load_snapshot_state(snapshot_path, &mut bus, &mut state, args.model, &rom_bytes)?;
        base_instruction_count = metadata.instruction_count;
        if args.perfetto {
            set_perf_instr_counter(base_instruction_count);
        }
    } else if let Some(profile) = turnon_profile.as_ref() {
        apply_turnon_resume_profile(&mut bus, &mut state, profile);
    } else if let Some(profile) = reset_trace2_profile.as_ref() {
        apply_reset_trace2_main_display_profile(&mut bus, &mut state, profile)
            .map_err(|err| format!("reset-trace2 profile: {err}"))?;
        base_instruction_count = profile.seed_instruction_count;
        if args.perfetto {
            set_perf_instr_counter(base_instruction_count);
        }
    } else {
        if args.model.is_pce500_family() && !use_full_system_image {
            bus.strobe_all_columns();
        }
        power_on_reset(&mut bus, &mut state)
            .map_err(|error| std::io::Error::new(ErrorKind::InvalidData, error))?;
        // power_on_reset seeds PC from the ROM reset vector at 0xFFFFD.
        if use_full_system_image {
            seed_pce500_bootstrap_imem(&mut bus.memory);
        }
    }
    bus.reapply_iq7000_clock_seed();
    if use_key_seq {
        bus.keyboard.set_repeat_enabled(false);
    }

    let start = Instant::now();
    let max_steps = if args.turnon2_resume && args.steps == DEFAULT_RUN_STEPS {
        turnon_profile
            .as_ref()
            .and_then(|profile| profile.target_instruction_count)
            .unwrap_or(args.steps)
    } else {
        args.steps
    };
    let perfetto_enabled = args.perfetto;
    let trace_regs = args.trace_regs;
    let wants_lcd_trace = args.dump_lcd_trace.is_some();
    let model = args.model;
    let perfetto_base_path_run = perfetto_base_path.clone();
    let turnon_profile_run = turnon_profile.clone();
    let reset_trace2_profile_run = reset_trace2_profile.clone();

    let summary_slot: Rc<RefCell<Option<RunSummary>>> = Rc::new(RefCell::new(None));
    let summary_slot_run = summary_slot.clone();
    let run_error_slot: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let run_error_slot_run = run_error_slot.clone();
    let mut driver = AsyncDriver::new();
    let snapshot_out = args.snapshot_out.clone();
    let base_instruction_count = base_instruction_count;

    driver.spawn(async move {
        let mut bus = bus;
        let mut state = state;
        let mut executor = executor;
        let text_decoder = text_decoder;
        let turnon_profile = turnon_profile_run;
        let reset_trace2_profile = reset_trace2_profile_run;
        let mut key_seq_runner = key_seq_runner;
        let use_key_seq = use_key_seq;
        let needs_screen_state = needs_screen_state;
        let needs_screen_text = needs_screen_text;

        let mut executed: u64 = base_instruction_count;
        let mut perfetto_part: u32 = 1;
        let mut trace_resume_target_applied = false;
        let mut reset_trace2_target_applied = false;

        let mut trace_pc_counts: HashMap<u32, u64> = HashMap::new();
        let mut trace_window_active: u64 = 0;
        let mut trace_window_anchor: Option<u32> = None;
        let mut run_fault: Option<String> = None;
        let perfetto_dbg = false;
        let log_dbg = |_msg: &str| {};

        log_dbg(&format!("entering execute loop for {max_steps} steps"));
        while executed < max_steps {
            let pre_tick_done = false;
            // Silent vector-target validation and stability classification use
            // the current instruction context. Establish it before IRQ
            // preflight, then refresh it below if delivery changes PC.
            bus.set_pc(state.pc());
            bus.set_instr_index(executed);
            // Ensure vector table is patched once before executing instructions.
            if !bus.vec_patched {
                bus.maybe_patch_vectors();
            }

            // Validate the current instruction, including operand- and
            // stack-dependent fail-closed checks, entirely through the silent
            // peek path before validating any asynchronous transfer. This
            // makes a malformed current instruction authoritative and leaves
            // trace-replay input untouched on rejection.
            let current_pc = state.pc() & ADDRESS_MASK;
            let should_preflight_current =
                current_instruction_requires_silent_preflight(&state, &bus);
            let silent_current = if should_preflight_current {
                match preflight_current_instruction_silently(&executor, &state, &mut bus) {
                    Ok(preflight) => Some(preflight),
                    Err(error) => {
                        let message = format!(
                            "error preflighting current instruction at PC=0x{current_pc:05X}: {error}"
                        );
                        eprintln!("{message}");
                        run_fault = Some(message);
                        break;
                    }
                }
            } else {
                None
            };
            if use_key_seq {
                let screen_state = if needs_screen_state || needs_screen_text {
                    capture_screen_state(bus.lcd(), text_decoder.as_ref(), needs_screen_text)
                } else {
                    ScreenState::default()
                };
                let events = key_seq_runner.step(executed, !state.is_off(), &screen_state);
                for event in events {
                    match event.kind {
                        KeySeqEventKind::Press => {
                            if let Some(key) = event.key {
                                match key {
                                    AutoKeyKind::Matrix(code) => bus.press_key(code),
                                    AutoKeyKind::Chord { modifier, code } => {
                                        bus.press_key(modifier);
                                        bus.press_key(code);
                                    }
                                    AutoKeyKind::Event(code) => bus.inject_matrix_event(code, false),
                                    AutoKeyKind::InputEvent(code) => bus.inject_input_event(code),
                                    AutoKeyKind::OnKey => bus.press_on_key(),
                                }
                            }
                        }
                        KeySeqEventKind::Release => {
                            if let Some(key) = event.key {
                                match key {
                                    AutoKeyKind::Matrix(code) => bus.release_key(code),
                                    AutoKeyKind::Chord { modifier, code } => {
                                        bus.release_key(code);
                                        bus.release_key(modifier);
                                    }
                                    AutoKeyKind::Event(code) => bus.inject_matrix_event(code, true),
                                    AutoKeyKind::InputEvent(_) => {}
                                    AutoKeyKind::OnKey => bus.clear_on_key(),
                                }
                            }
                        }
                        KeySeqEventKind::Log => {
                            println!("{}", event.message);
                        }
                    }
                }
            }

            // OFF wake filtering must run before generic IRQ delivery.  A
            // pending KEYI/MTI is retained but cannot bypass the ONKI-only
            // wake gate.
            match apply_off_wake_gate(&mut state, &mut bus) {
                OffWakeGate::NotOff => {}
                OffWakeGate::WaitingForOnKey | OffWakeGate::WokeOnKey => {
                    // OFF consumes an idle boundary whether it remains stopped
                    // or wakes. Delivery/execution starts on the next pass.
                    bus.cycle_count = bus.cycle_count.wrapping_add(1);
                    sleep_for_cycles(1).await;
                    continue;
                }
            }

            match apply_halt_wake_boundary(&mut state, &mut bus) {
                HaltWakeBoundary::NotHalted => {}
                HaltWakeBoundary::WaitingForInterrupt
                | HaltWakeBoundary::WokeForInterrupt => {
                    sleep_for_cycles(1).await;
                    continue;
                }
            }

            if bus.irq_pending() {
                // A deliverable asynchronous IRQ replaces, but does not erase,
                // the fall-through opcode fetch. Hardware performs exactly one
                // byte read here and does not decode operands.
                let irq_pc = state.pc() & ADDRESS_MASK;
                let _discarded_opcode = bus.load(irq_pc, 8);
                // S is a 20-bit external pointer. Interrupt-frame bytes wrap
                // independently at FFFFF -> 00000 even when S starts below 5.
                if let Err(error) = bus.deliver_irq(&mut state) {
                    let message =
                        format!("error delivering IRQ at PC=0x{:05X}: {error}", state.pc());
                    eprintln!("{message}");
                    run_fault = Some(message);
                    break;
                }
            }
            let pc = state.pc();
            bus.set_pc(pc);
            bus.set_instr_index(executed);
            bus.update_trace_resume_state();
            if !trace_pcs.is_empty() && trace_pcs.contains(&pc) {
                let count = trace_pc_counts
                    .entry(pc)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if *count <= 10 || count.is_multiple_of(1000) {
                    let imr = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
                    let isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                    if trace_regs {
                        let a = state.get_reg(RegName::A) & 0xFF;
                        let f = state.get_reg(RegName::F) & 0xFF;
                        let s = state.get_reg(RegName::S) & 0x0F_FFFF;
                        let y = state.get_reg(RegName::Y) & 0x0F_FFFF;
                        let ssr = bus.memory.read_internal_byte(IMEM_SSR_OFFSET).unwrap_or(0);
                        println!(
                            "[pc-trace] pc=0x{pc:05X} hits={hits} imr=0x{imr:02X} isr=0x{isr:02X} ssr=0x{ssr:02X} onk={onk} a=0x{a:02X} f=0x{f:02X} sp=0x{s:06X} y=0x{y:06X}",
                            pc = pc,
                            hits = count,
                            imr = imr,
                            isr = isr,
                            ssr = ssr,
                            onk = bus.pending_onk,
                            a = a,
                            f = f,
                            s = s,
                            y = y
                        );
                    } else {
                        println!(
                            "[pc-trace] pc=0x{pc:05X} hits={hits} imr=0x{imr:02X} isr=0x{isr:02X}",
                            pc = pc,
                            hits = count,
                            imr = imr,
                            isr = isr
                        );
                    }
                }
                if trace_pc_window > 0 {
                    trace_window_active = trace_pc_window;
                    trace_window_anchor = Some(pc);
                }
            } else if trace_window_active > 0 {
                let imr = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
                let isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                let anchor = trace_window_anchor
                    .map(|p| format!("0x{p:05X}"))
                    .unwrap_or_else(|| "n/a".to_string());
                if trace_regs {
                    let a = state.get_reg(RegName::A) & 0xFF;
                    let f = state.get_reg(RegName::F) & 0xFF;
                    let s = state.get_reg(RegName::S) & 0x0F_FFFF;
                    let y = state.get_reg(RegName::Y) & 0x0F_FFFF;
                    let ssr = bus.memory.read_internal_byte(IMEM_SSR_OFFSET).unwrap_or(0);
                    println!(
                        "[pc-trace-window] anchor={anchor} pc=0x{pc:05X} remaining={} imr=0x{imr:02X} isr=0x{isr:02X} ssr=0x{ssr:02X} onk={onk} a=0x{a:02X} f=0x{f:02X} sp=0x{s:06X} y=0x{y:06X}",
                        trace_window_active,
                        pc = pc,
                        imr = imr,
                        isr = isr,
                        ssr = ssr,
                        onk = bus.pending_onk,
                        a = a,
                        f = f,
                        s = s,
                        y = y
                    );
                } else {
                    println!(
                        "[pc-trace-window] anchor={anchor} pc=0x{pc:05X} remaining={} imr=0x{imr:02X} isr=0x{isr:02X}",
                        trace_window_active,
                        pc = pc,
                        imr = imr,
                        isr = isr
                    );
                }
                trace_window_active = trace_window_active.saturating_sub(1);
            }
            if bus.log_lcd && bus.log_lcd_count < 50 && executed.is_multiple_of(1000) {
                let imr = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
                let isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
                println!(
                    "[pc] pc=0x{pc:05X} imr=0x{imr:02X} isr=0x{isr:02X}",
                    pc = pc,
                    imr = imr,
                    isr = isr
                );
            }
            let opcode = bus.load(pc, 8) as u8;
            if pc == current_pc {
                if let Some(preflight) = silent_current {
                    if opcode != preflight.opcode {
                        let message = format!(
                            "error fetching opcode at PC=0x{pc:05X}: architectural fetch \
                             0x{opcode:02X} disagrees with silent preflight 0x{:02X}",
                            preflight.opcode
                        );
                        eprintln!("{message}");
                        run_fault = Some(message);
                        break;
                    }
                }
            }
            let run_timer_cycles = !state.is_off();
            let prepared_result = match silent_current.filter(|proof| proof.source_pc == pc) {
                Some(preflight) => prepare_preflighted_transfer_and_tick(
                    opcode,
                    &state,
                    &mut bus,
                    preflight,
                    run_timer_cycles,
                    pre_tick_done,
                ),
                None => preflight_and_tick_instruction(
                    &executor,
                    opcode,
                    &state,
                    &mut bus,
                    run_timer_cycles,
                    pre_tick_done,
                ),
            };
            let prepared_transfer = match prepared_result {
                Ok(transfer) => transfer,
                Err(err) => {
                    let message =
                        format!("error executing opcode 0x{opcode:02X} at PC=0x{pc:05X}: {err}");
                    eprintln!("{message}");
                    run_fault = Some(message);
                    break;
                }
            };
            if perfetto_dbg {
                eprintln!("[perfetto-debug] executing opcode=0x{opcode:02X}");
            }
            match executor
                .execute_with_vector_transfer(
                    opcode,
                    &mut state,
                    &mut bus,
                    prepared_transfer,
                )
            {
                Ok(_instr_len) => {
                    bus.handle_irq_return(opcode, &state);
                    bus.finalize_instruction();
                    bus.apply_deferred_key_irq();
                    if run_timer_cycles && opcode != 0xEF {
                        bus.cycle_count = bus.cycle_count.wrapping_add(1);
                    }
                    executed += 1;
                    if perfetto_chunk_size > 0
                        && perfetto_enabled
                        && executed.is_multiple_of(perfetto_chunk_size)
                        && executed > 0
                    {
                        rotate_perfetto_trace(&perfetto_base_path_run, perfetto_part);
                        perfetto_part = perfetto_part.saturating_add(1);
                    }
                    if let Some(stop) = stop_pc {
                        if state.pc() == stop {
                            break;
                        }
                    }
                    if !trace_resume_target_applied {
                        if let Some(profile) = turnon_profile.as_ref() {
                            if let Some(target_instruction_count) = profile.target_instruction_count
                            {
                                if executed >= target_instruction_count {
                                    match apply_turnon_resume_target_lcd(&mut bus, profile) {
                                        Ok(applied) => {
                                            trace_resume_target_applied = applied;
                                        }
                                        Err(err) => {
                                            eprintln!(
                                                "warning: failed to apply turnon2 target LCD snapshot: {err}"
                                            );
                                        }
                                    }
                                    if trace_resume_target_applied {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if !reset_trace2_target_applied {
                        if let Some(profile) = reset_trace2_profile.as_ref() {
                            if let Some(target_instruction_count) = profile.target_instruction_count
                            {
                                if executed >= target_instruction_count {
                                    match apply_lcd_snapshot(
                                        &mut bus,
                                        profile.target_lcd_meta.as_ref(),
                                        &profile.target_lcd_payload,
                                    ) {
                                        Ok(applied) => {
                                            reset_trace2_target_applied = applied;
                                        }
                                        Err(err) => {
                                            eprintln!(
                                                "warning: failed to apply reset-trace2 target LCD snapshot: {err}"
                                            );
                                        }
                                    }
                                    if reset_trace2_target_applied {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if state.is_halted() {
                        continue;
                    }
                }
                Err(err) => {
                    let message =
                        format!("error executing opcode 0x{opcode:02X} at PC=0x{pc:05X}: {err}");
                    eprintln!("{message}");
                    bus.poisoned.get_or_insert_with(|| err.to_string());
                    run_fault = Some(message);
                    if perfetto_dbg {
                        eprintln!(
                            "[perfetto-debug] execute error at step {} opcode=0x{opcode:02X}: {err}",
                            executed + 1
                        );
                    }
                    break;
                }
            }
            if perfetto_dbg {
                eprintln!(
                    "[perfetto-debug] step {} complete: pc=0x{:05X} cycles={}",
                    executed,
                    state.pc() & ADDRESS_MASK,
                    bus.cycle_count
                );
            }
        }

        if let Some(error) = run_fault {
            bus.finish_perfetto();
            bus.finish_bus_trace();
            *run_error_slot_run.borrow_mut() = Some(error);
            emit_event(DriverEvent::User(CPU_DONE_EVENT));
            return;
        }

        if let Some(snapshot_path) = snapshot_out.as_ref() {
            bus.reapply_iq7000_clock_seed();
            if let Err(err) =
                save_snapshot_state(snapshot_path, &bus, &state, executed, args.model)
            {
                *run_error_slot_run.borrow_mut() = Some(format!(
                    "failed to save snapshot to {}: {err}",
                    snapshot_path.display()
                ));
            } else {
                println!("Saved snapshot to {}", snapshot_path.display());
            }
        }

        bus.finish_perfetto();
        bus.finish_bus_trace();

        let imr_mem = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        let isr_mem = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let imr_reg = (state.get_reg(RegName::IMR) & 0xFF) as u8;
        let lcd_stats = bus.lcd().stats();

        let lcd_lines = text_decoder
            .as_ref()
            .map(|decoder| decoder.decode_display_text(bus.lcd()))
            .unwrap_or_default();
        let lcd_trace = if wants_lcd_trace {
            let trace = bus.lcd().display_trace_buffer();
            let trace = trace.map(|row| row.to_vec()).to_vec();
            let vram = bus.lcd().display_vram_bytes();
            let vram = vram.map(|row| row.to_vec()).to_vec();
            Some(LcdTraceDump {
                executed,
                pc: state.pc(),
                halted: state.is_halted(),
                lcd_lines: lcd_lines.clone(),
                vram,
                trace,
            })
        } else {
            None
        };

        let debug_probe = wants_debug_probe.then(|| {
            build_debug_probe(model, &bus, &state, executed, &debug_probe_ranges)
        });

        let summary = RunSummary {
            executed,
            pc: state.pc(),
            halted: state.is_halted(),
            lcd_writes: bus.lcd_writes,
            imr_mem,
            isr_mem,
            imr_reg,
            lcd_stats,
            lcd_lines,
            lcd_pixels: lcd_pixels(bus.lcd()),
            lcd_annunciators: lcd_annunciators(model, &bus.memory),
            lcd_trace,
            debug_probe,
        };
        *summary_slot_run.borrow_mut() = Some(summary);
        emit_event(DriverEvent::User(CPU_DONE_EVENT));
    });

    loop {
        let result = driver.run_for(u64::MAX);
        if matches!(result.event, DriverEvent::User(CPU_DONE_EVENT)) {
            break;
        }
    }

    if let Some(error) = run_error_slot.borrow_mut().take() {
        return Err(error.into());
    }

    let elapsed = start.elapsed();
    let summary = summary_slot
        .borrow_mut()
        .take()
        .ok_or("missing run summary")?;

    println!(
        "Executed {} instruction(s) in {:.3?} (PC=0x{:05X}, halted={}, lcd_writes={})",
        summary.executed, elapsed, summary.pc, summary.halted, summary.lcd_writes
    );
    println!(
        "Final IMR=0x{:02X} (mem=0x{:02X}) ISR=0x{:02X}",
        summary.imr_reg, summary.imr_mem, summary.isr_mem
    );

    println!(
        "LCD stats: on={:?} instr={:?} data={:?} cs(both/left/right)=({}/{}/{})",
        summary.lcd_stats.chip_on,
        summary.lcd_stats.instruction_counts,
        summary.lcd_stats.data_write_counts,
        summary.lcd_stats.cs_both_count,
        summary.lcd_stats.cs_left_count,
        summary.lcd_stats.cs_right_count
    );

    println!("LCD (decoded text):");
    for line in &summary.lcd_lines {
        println!("  {}", line);
    }

    if let Some(path) = &args.dump_lcd_trace {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let dump = summary.lcd_trace.as_ref().ok_or("missing LCD trace data")?;
        fs::write(path, serde_json::to_string_pretty(dump)?)?;
        println!("Wrote LCD trace dump: {}", path.display());
    }
    if let Some(path) = &args.dump_bus_trace {
        println!("Wrote bus trace dump: {}", path.display());
    }

    if let Some(path) = &args.debug_probe_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let probe = summary
            .debug_probe
            .as_ref()
            .ok_or("missing debug probe data")?;
        fs::write(path, serde_json::to_string_pretty(probe)?)?;
        println!("Wrote debug probe JSON: {}", path.display());
    }

    if let Some(path) = &args.capture_png {
        write_lcd_png(
            path,
            &summary.lcd_pixels,
            LCD_CAPTURE_SCALE,
            summary.lcd_annunciators.as_ref(),
        )?;
        println!("Wrote LCD PNG capture: {}", path.display());
    }

    if let Some(path) = &args.capture_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let capture = json!({
            "executed": summary.executed,
            "pc": format!("0x{:05X}", summary.pc),
            "halted": summary.halted,
            "lcd_writes": summary.lcd_writes,
            "iq7000_rtc": iq7000_clock_seed.as_ref().map(|seed| json!({
                "source": seed.source.as_str(),
                "yyyymmddhhmm": seed.clock.as_ascii(),
            })),
            "imr_mem": format!("0x{:02X}", summary.imr_mem),
            "isr_mem": format!("0x{:02X}", summary.isr_mem),
            "imr_reg": format!("0x{:02X}", summary.imr_reg),
            "lcd_lines": &summary.lcd_lines,
            "lcd_pixels": &summary.lcd_pixels,
            "lcd_annunciators": &summary.lcd_annunciators,
            "debug_probe": &summary.debug_probe,
        });
        fs::write(path, serde_json::to_string_pretty(&capture)?)?;
        println!("Wrote LCD JSON capture: {}", path.display());
    }

    if args.perf {
        let instrs_per_sec = if elapsed.as_secs_f64() > 0.0 {
            (summary.executed as f64) / elapsed.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "Perf: {:.2} MIPS ({} instr / {:.3?})",
            instrs_per_sec / 1_000_000.0,
            summary.executed,
            elapsed
        );
    }

    let mut failures = Vec::new();
    for raw in &args.expect_row {
        match parse_expected_row(raw) {
            Ok((idx, expected)) => {
                let actual = summary.lcd_lines.get(idx).cloned().unwrap_or_default();
                if !actual.contains(&expected) {
                    failures.push(format!(
                        "expect-row failed: row {idx} missing substring '{expected}' (got '{actual}')"
                    ));
                }
            }
            Err(err) => failures.push(err),
        }
    }
    for needle in &args.expect_text {
        if !summary.lcd_lines.iter().any(|line| line.contains(needle)) {
            failures.push(format!(
                "expect-text failed: substring '{needle}' not found in LCD text"
            ));
        }
    }
    if !failures.is_empty() {
        eprintln!("FAIL: {}", failures.join(" | "));
        std::process::exit(1);
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("fatal: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sc62015_core::llama::state::PowerState;
    use sc62015_core::pce500::ROM_RESET_VECTOR_ADDR;

    fn test_standalone_bus() -> StandaloneBus {
        StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        )
    }

    #[test]
    fn hw002_zero_i_wait_passes_side_effect_free_preflight() {
        let mut bus = test_standalone_bus();
        bus.cycle_count = 7;
        bus.timer.enabled = true;
        bus.timer.next_mti = 0;
        bus.timer.next_sti = 0;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 0);
        let executor = LlamaExecutor::new();
        let isr_before = bus.memory.read_internal_byte(IMEM_ISR_OFFSET);

        let transfer =
            preflight_and_tick_instruction(&executor, 0xEF, &state, &mut bus, false, false)
                .expect("HW-002 WAIT I=0 must pass scheduler preflight");

        assert!(transfer.is_none());
        assert_eq!(bus.cycle_count, 7);
        assert_eq!(bus.timer.next_mti, 0);
        assert_eq!(bus.timer.next_sti, 0);
        assert_eq!(bus.memory.read_internal_byte(IMEM_ISR_OFFSET), isr_before);
        assert!(!bus.timer.irq_pending);
    }

    #[test]
    fn tcl_preflight_accepts_without_timer_mutation() {
        let mut bus = test_standalone_bus();
        bus.cycle_count = 9;
        bus.timer.enabled = true;
        bus.timer.next_mti = 0;
        bus.timer.next_sti = 0;
        let state = LlamaState::new();
        let executor = LlamaExecutor::new();
        let isr_before = bus.memory.read_internal_byte(IMEM_ISR_OFFSET);

        let transfer =
            preflight_and_tick_instruction(&executor, 0xCE, &state, &mut bus, false, false)
                .expect("TCL is supported by the standalone timer bus");

        assert!(transfer.is_none());
        assert_eq!(bus.cycle_count, 9);
        assert_eq!(bus.timer.next_mti, 0);
        assert_eq!(bus.timer.next_sti, 0);
        assert_eq!(bus.memory.read_internal_byte(IMEM_ISR_OFFSET), isr_before);
        assert!(!bus.timer.irq_pending);
    }

    #[test]
    fn reserved_preflight_rejects_before_timer_tick() {
        let mut bus = test_standalone_bus();
        bus.cycle_count = 11;
        bus.timer.enabled = true;
        bus.timer.next_mti = 0;
        bus.timer.next_sti = 0;
        let state = LlamaState::new();
        let executor = LlamaExecutor::new();
        let isr_before = bus.memory.read_internal_byte(IMEM_ISR_OFFSET);

        let error = preflight_and_tick_instruction(&executor, 0x20, &state, &mut bus, true, false)
            .expect_err("reserved opcode must fail before scheduler mutation");

        assert_eq!(error, "invalid or reserved opcode");
        assert_eq!(bus.cycle_count, 11);
        assert_eq!(bus.timer.next_mti, 0);
        assert_eq!(bus.timer.next_sti, 0);
        assert_eq!(bus.memory.read_internal_byte(IMEM_ISR_OFFSET), isr_before);
        assert!(!bus.timer.irq_pending);
    }

    #[test]
    fn trace_resume_irq_target_stability_uses_the_target_context() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        let source_pc = 0x012345;
        let target_pc = 0x023456;
        state.set_pc(source_pc);
        bus.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR, (target_pc & 0xFF) as u8);
        bus.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 1, ((target_pc >> 8) & 0xFF) as u8);
        bus.memory
            .write_external_byte(INTERRUPT_VECTOR_ADDR + 2, ((target_pc >> 16) & 0x0F) as u8);
        bus.install_trace_resume_reads(
            target_pc,
            vec![TurnonResumeByte {
                addr: target_pc,
                value: 0x00,
            }],
        );

        // This is the context setup performed at the top of the standalone
        // loop, before its early asynchronous-IRQ preflight.
        bus.set_pc(state.pc());
        bus.set_instr_index(17);
        let (target, target_len) =
            validate_vector_transfer_with_length(INTERRUPT_VECTOR_ADDR, &state, &mut bus)
                .expect("silent IRQ target validation");

        assert_eq!(target, target_pc);
        assert_eq!(target_len, 1);
        assert!(
            bus.instruction_byte_is_stable(target),
            "the source context alone does not activate target-anchored replay"
        );
        assert!(
            !bus.instruction_byte_is_stable_for_context(target, target),
            "the exact target context must quarantine the replay-provided byte"
        );
        assert_eq!(
            validate_stable_vector_transfer(INTERRUPT_VECTOR_ADDR, &state, &mut bus),
            Err("callback-backed vector/target cannot cross scheduler tick")
        );
    }

    #[test]
    fn trace_resume_malformed_current_preflight_consumes_no_replay_or_bus_read() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        state.set_pc(0x012345);
        bus.install_trace_resume_reads(
            state.pc(),
            vec![TurnonResumeByte {
                addr: state.pc(),
                value: 0x20,
            }],
        );
        bus.set_pc(state.pc());
        bus.set_instr_index(23);
        let executor = LlamaExecutor::new();
        let reads_before = bus.memory.memory_read_count();

        let error = preflight_current_instruction_silently(&executor, &state, &mut bus)
            .expect_err("reserved replay opcode must fail silently");

        assert_eq!(error, "invalid or reserved opcode");
        assert_eq!(bus.trace_resume_read_index, 0);
        assert!(!bus.trace_resume_read_enabled);
        assert_eq!(bus.memory.memory_read_count(), reads_before);
    }

    #[test]
    fn silent_instruction_preflight_is_bound_to_the_memory_mapping() {
        let mut bus = test_standalone_bus();
        let state = LlamaState::new();
        bus.memory.write_external_byte(0, 0x00);
        let executor = LlamaExecutor::new();
        let proof = preflight_current_instruction_silently(&executor, &state, &mut bus)
            .expect("preflight NOP");

        bus.memory.add_ram_overlay(0x200, 1, "mapping-change");
        let error =
            prepare_preflighted_transfer_and_tick(0x00, &state, &mut bus, proof, false, false)
                .expect_err("a mapping change must invalidate the earlier proof");

        assert_eq!(error, "silent instruction preflight memory mapping changed");
        assert_eq!(bus.cycle_count, 0);
    }

    #[test]
    fn malformed_operand_preflight_rejects_before_timer_tick() {
        for opcode in [0x11, 0xE3, 0xEB] {
            let mut bus = test_standalone_bus();
            bus.cycle_count = 13;
            bus.timer.enabled = true;
            bus.timer.next_mti = 0;
            bus.timer.next_sti = 0;
            let mut state = LlamaState::new();
            state.set_reg(RegName::I, 1);
            state.set_reg(RegName::U, 0x180);
            state.set_reg(RegName::S, 0x190);
            let bytes: &[u8] = match opcode {
                0x11 => &[0x11, 0x00],
                0xE3 => &[0xE3, 0x10, 0x00],
                0xEB => &[0xEB, 0x10, 0x00],
                _ => unreachable!(),
            };
            bus.memory.write_external_slice(0, bytes);
            let executor = LlamaExecutor::new();
            let reads_before = bus.memory.memory_read_count();
            let writes_before = bus.memory.memory_write_count();
            let isr_before = bus
                .memory
                .read_internal_byte_silent(IMEM_ISR_OFFSET)
                .unwrap_or(0);
            let u_before = state.get_reg(RegName::U);
            let s_before = state.get_reg(RegName::S);

            preflight_and_tick_instruction(&executor, opcode, &state, &mut bus, true, false)
                .expect_err("malformed/data-dependent instruction must fail preflight");

            assert_eq!(bus.cycle_count, 13, "opcode {opcode:02X}");
            assert_eq!(bus.timer.next_mti, 0, "opcode {opcode:02X}");
            assert_eq!(bus.timer.next_sti, 0, "opcode {opcode:02X}");
            assert_eq!(
                bus.memory.read_internal_byte_silent(IMEM_ISR_OFFSET),
                Some(isr_before),
                "opcode {opcode:02X}"
            );
            assert_eq!(state.get_reg(RegName::U), u_before, "opcode {opcode:02X}");
            assert_eq!(state.get_reg(RegName::S), s_before, "opcode {opcode:02X}");
            assert!(!bus.timer.irq_pending, "opcode {opcode:02X}");
            assert_eq!(bus.memory.memory_read_count(), reads_before);
            assert_eq!(bus.memory.memory_write_count(), writes_before);
        }
    }

    #[test]
    fn hw011_pop_f_images_pass_side_effect_free_preflight() {
        for (opcode, stack_reg, stack_addr) in
            [(0x3E, RegName::U, 0x180), (0x5F, RegName::S, 0x190)]
        {
            let mut bus = test_standalone_bus();
            bus.memory.write_external_byte(0, opcode);
            bus.memory.write_external_byte(stack_addr, 0xFC);
            let mut state = LlamaState::new();
            state.set_reg(stack_reg, stack_addr);
            state.set_reg(RegName::F, 0x03);
            let executor = LlamaExecutor::new();
            let reads_before = bus.memory.memory_read_count();
            let writes_before = bus.memory.memory_write_count();

            let transfer =
                preflight_and_tick_instruction(&executor, opcode, &state, &mut bus, false, false)
                    .expect("HW-011 POPU/POPS F upper bits normalize during execution");

            assert!(transfer.is_none(), "opcode {opcode:02X}");
            assert_eq!(state.get_reg(stack_reg), stack_addr, "opcode {opcode:02X}");
            assert_eq!(state.get_reg(RegName::F), 0x03, "opcode {opcode:02X}");
            assert_eq!(bus.memory.memory_read_count(), reads_before);
            assert_eq!(bus.memory.memory_write_count(), writes_before);
        }
    }

    #[test]
    fn hw011_reti_f_image_passes_side_effect_free_preflight() {
        let mut bus = test_standalone_bus();
        bus.memory.write_external_byte(0, 0x01);
        bus.memory.write_external_byte(0x190, 0xA5);
        bus.memory.write_external_byte(0x191, 0xFC);
        bus.memory.write_external_byte(0x192, 0x00);
        bus.memory.write_external_byte(0x193, 0x02);
        bus.memory.write_external_byte(0x194, 0x00);
        bus.memory.write_external_byte(0x200, 0x00);
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x190);
        state.set_reg(RegName::F, 0x03);
        let executor = LlamaExecutor::new();
        let reads_before = bus.memory.memory_read_count();
        let writes_before = bus.memory.memory_write_count();

        let transfer =
            preflight_and_tick_instruction(&executor, 0x01, &state, &mut bus, false, false)
                .expect("HW-011 RETI upper F bits normalize during execution");

        assert!(transfer.is_none());
        assert_eq!(state.get_reg(RegName::S), 0x190);
        assert_eq!(state.get_reg(RegName::F), 0x03);
        assert_eq!(bus.memory.memory_read_count(), reads_before);
        assert_eq!(bus.memory.memory_write_count(), writes_before);
    }

    #[test]
    fn software_reset_rejects_dynamic_vector_or_target_before_timer_tick() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        for dynamic_vector in [true, false] {
            let mut bus = test_standalone_bus();
            bus.cycle_count = 19;
            bus.timer.enabled = true;
            bus.timer.next_mti = 0;
            bus.timer.next_sti = 0;
            let mut state = LlamaState::new();
            state.set_pc(0);
            let target = 0x00200;
            bus.memory.write_external_byte(0, 0xFF);
            bus.memory
                .write_external_byte(ROM_RESET_VECTOR_ADDR, (target & 0xFF) as u8);
            bus.memory
                .write_external_byte(ROM_RESET_VECTOR_ADDR + 1, ((target >> 8) & 0xFF) as u8);
            bus.memory
                .write_external_byte(ROM_RESET_VECTOR_ADDR + 2, ((target >> 16) & 0x0F) as u8);
            bus.memory.write_external_byte(target, 0x00);
            if dynamic_vector {
                bus.memory
                    .set_python_ranges(vec![(ROM_RESET_VECTOR_ADDR, ROM_RESET_VECTOR_ADDR + 2)]);
            } else {
                bus.memory.set_python_ranges(vec![(target, target)]);
            }
            bus.host_peek = Some(Box::new(move |addr| {
                Some(match addr & ADDRESS_MASK {
                    ROM_RESET_VECTOR_ADDR => (target & 0xFF) as u8,
                    addr if addr == ROM_RESET_VECTOR_ADDR + 1 => ((target >> 8) & 0xFF) as u8,
                    addr if addr == ROM_RESET_VECTOR_ADDR + 2 => ((target >> 16) & 0x0F) as u8,
                    addr if addr == target => 0x00,
                    _ => 0x00,
                })
            }));
            let architectural_reads = Arc::new(AtomicUsize::new(0));
            let read_count = Arc::clone(&architectural_reads);
            bus.host_read = Some(Box::new(move |_addr| {
                read_count.fetch_add(1, Ordering::Relaxed);
                Some(0)
            }));
            let executor = LlamaExecutor::new();
            let memory_reads_before = bus.memory.memory_read_count();
            let memory_writes_before = bus.memory.memory_write_count();

            let error =
                preflight_and_tick_instruction(&executor, 0xFF, &state, &mut bus, true, false)
                    .expect_err("dynamic RESET vector state must fail before scheduler work");

            assert_eq!(
                error, "callback-backed vector/target cannot cross scheduler tick",
                "dynamic_vector={dynamic_vector}"
            );
            assert_eq!(bus.cycle_count, 19);
            assert_eq!(bus.timer.next_mti, 0);
            assert_eq!(bus.timer.next_sti, 0);
            assert_eq!(state.pc(), 0);
            assert_eq!(architectural_reads.load(Ordering::Relaxed), 0);
            assert_eq!(bus.memory.memory_read_count(), memory_reads_before);
            assert_eq!(bus.memory.memory_write_count(), memory_writes_before);
        }
    }

    #[test]
    fn off_gate_blocks_deliverable_non_onk_without_destroying_state() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        state.set_power_state(PowerState::Off);
        bus.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY | IMR_MTI);
        let status = ISR_KEYI | ISR_MTI;
        bus.memory.write_internal_byte(IMEM_ISR_OFFSET, status);
        bus.irq_pending = true;
        bus.timer.irq_pending = true;
        bus.last_irq_src = Some("KEY".to_string());
        bus.timer.irq_source = Some("KEY".to_string());
        bus.timer.last_fired = Some("MTI".to_string());
        bus.timer.irq_isr = status;
        bus.timer.key_irq_latched = true;

        // Without the OFF gate this pending image is deliverable by the
        // generic IRQ path. The loop must stop here instead.
        assert!(bus.irq_pending());
        let result = apply_off_wake_gate(&mut state, &mut bus);

        assert_eq!(result, OffWakeGate::WaitingForOnKey);
        assert!(state.is_off());
        assert_eq!(bus.memory.read_internal_byte(IMEM_ISR_OFFSET), Some(status));
        assert!(bus.irq_pending);
        assert!(bus.timer.irq_pending);
        assert_eq!(bus.last_irq_src.as_deref(), Some("KEY"));
        assert_eq!(bus.timer.irq_source.as_deref(), Some("KEY"));
        assert_eq!(bus.timer.last_fired.as_deref(), Some("MTI"));
        assert_eq!(bus.timer.irq_isr, status);
        assert!(bus.timer.key_irq_latched);
    }

    #[test]
    fn off_gate_wakes_only_on_onki_and_preserves_full_status_image() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        state.set_power_state(PowerState::Off);
        let status = ISR_KEYI | ISR_MTI | ISR_ONKI;
        bus.memory.write_internal_byte(IMEM_ISR_OFFSET, status);
        bus.timer.key_irq_latched = true;

        let result = apply_off_wake_gate(&mut state, &mut bus);

        assert_eq!(result, OffWakeGate::WokeOnKey);
        assert_eq!(state.power_state(), PowerState::Running);
        assert_eq!(bus.memory.read_internal_byte(IMEM_ISR_OFFSET), Some(status));
        assert!(bus.irq_pending);
        assert!(bus.timer.irq_pending);
        assert_eq!(bus.timer.irq_isr, status);
        assert_eq!(bus.last_irq_src.as_deref(), Some("ONK"));
        assert_eq!(bus.timer.irq_source.as_deref(), Some("ONK"));
        assert_eq!(bus.timer.last_fired.as_deref(), Some("ONK"));
        assert!(bus.timer.key_irq_latched);
    }

    #[test]
    fn halted_pending_irq_wakes_on_an_idle_boundary_before_delivery() {
        let mut bus = test_standalone_bus();
        bus.timer.enabled = false;
        bus.cycle_count = 41;
        bus.memory
            .write_internal_byte(IMEM_IMR_OFFSET, IMR_MASTER | IMR_KEY);
        bus.memory.write_internal_byte(IMEM_ISR_OFFSET, ISR_KEYI);
        bus.irq_pending = true;
        bus.timer.irq_pending = true;

        let mut state = LlamaState::new();
        state.set_pc(0x12345);
        state.set_reg(RegName::S, 0x23456);
        state.set_halted(true);
        let external_before = bus.memory.external_slice().to_vec();
        assert!(
            !current_instruction_requires_silent_preflight(&state, &bus),
            "a dormant PC cannot execute on the HALT wake boundary"
        );

        let result = apply_halt_wake_boundary(&mut state, &mut bus);

        assert_eq!(result, HaltWakeBoundary::WokeForInterrupt);
        assert_eq!(state.power_state(), PowerState::Running);
        assert_eq!(state.pc(), 0x12345, "wake must not deliver the vector");
        assert_eq!(state.get_reg(RegName::S), 0x23456);
        assert_eq!(bus.cycle_count, 42, "wake consumes one idle cycle");
        assert_eq!(bus.delivered_irq_count, 0);
        assert_eq!(bus.memory.external_slice(), external_before);
        assert!(bus.irq_pending, "delivery remains armed for the next pass");
        assert_eq!(bus.last_irq_src.as_deref(), Some("KEY"));
        assert!(
            !current_instruction_requires_silent_preflight(&state, &bus),
            "the armed unmasked IRQ replaces the dormant PC on the next pass"
        );

        bus.memory.write_internal_byte(IMEM_IMR_OFFSET, 0);
        assert!(
            current_instruction_requires_silent_preflight(&state, &bus),
            "a masked wake source leaves the resumed PC executable"
        );
    }

    #[test]
    fn on_key_sets_isr_and_triggers_pending_irq() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        // Enable ONK in IMR and set master bit.
        bus.memory
            .write_internal_byte(super::IMEM_IMR_OFFSET, super::IMR_MASTER | super::IMR_ONK);
        // CI is an independent external input and must survive ON-key changes.
        bus.memory
            .write_internal_byte(super::IMEM_SSR_OFFSET, super::SSR_CI);
        // Assert ONK input and ISR bit.
        bus.press_on_key();
        let ssr = bus
            .memory
            .read_internal_byte(super::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(
            ssr & super::SSR_CI,
            super::SSR_CI,
            "SSR.CI should retain its independently supplied level"
        );
        assert_eq!(
            ssr & super::SSR_ONK,
            super::SSR_ONK,
            "SSR.ONK should latch after ON key"
        );
        let isr = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_ne!(isr & super::ISR_ONKI, 0, "ONKI should latch after ON key");
        // irq_pending should fire with ONK masked in.
        assert!(
            bus.irq_pending(),
            "ONK should make IRQ pending when unmasked"
        );
        // Simulate firmware clearing ISR while ON key remains latched: irq_pending should
        // reassert ONKI (level-triggered) to avoid losing the event.
        bus.in_interrupt = true;
        bus.active_irq_mask = super::ISR_ONKI;
        if let Some(cur_isr) = bus.memory.read_internal_byte(super::IMEM_ISR_OFFSET) {
            bus.memory
                .write_internal_byte(super::IMEM_ISR_OFFSET, cur_isr & !super::ISR_ONKI);
        }
        let pending = bus.irq_pending();
        let isr_after = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_ne!(
            isr_after & super::ISR_ONKI,
            0,
            "pending_onk should reassert ONKI after clear"
        );
        assert!(
            !pending,
            "nested IRQ delivery should be suppressed while in_interrupt"
        );

        bus.clear_on_key();
        let released_ssr = bus
            .memory
            .read_internal_byte(super::IMEM_SSR_OFFSET)
            .unwrap_or(0);
        assert_eq!(
            released_ssr & (super::SSR_CI | super::SSR_ONK),
            super::SSR_CI,
            "releasing ON must not clear the independent SSR.CI input"
        );
    }

    #[test]
    fn selected_auto_key_asserts_raw_keyi_at_scheduling_boundary() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 1, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.timer.set_keyboard_irq_enabled(true);
        bus.strobe_all_columns();

        let isr_before = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(isr_before & super::ISR_KEYI, 0);

        bus.press_key(super::PF1_CODE);

        let isr_after = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(
            isr_after & super::ISR_KEYI,
            0,
            "auto key press should not assert KEYI before scan"
        );
        assert_eq!(
            bus.keyboard.fifo_len(),
            0,
            "auto key press should not enqueue FIFO immediately"
        );
        assert!(
            !bus.timer.key_irq_latched,
            "auto key press should not latch KEYI before scan"
        );
        assert!(
            !bus.pending_kil,
            "auto key press should not mark KIL pending before scan"
        );

        let _delivery_selected = bus.irq_pending();

        let isr_scan = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_ne!(
            isr_scan & super::ISR_KEYI,
            0,
            "raw selected KIL should assert KEYI before debounce"
        );
    }

    #[test]
    fn ksd_masks_kil_without_consuming_pending_key() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 1, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.timer.set_keyboard_irq_enabled(true);
        bus.strobe_all_columns();
        bus.press_key(super::PF1_CODE);
        bus.advance_cycles(6);

        let fifo_before = bus.keyboard.fifo_snapshot();
        assert!(!fifo_before.is_empty(), "scan should queue the pressed key");
        assert!(bus.timer.key_irq_latched);
        assert!(bus.pending_kil);

        bus.memory.write_internal_byte(IMEM_LCC_OFFSET, 0x04);
        let kil_addr = INTERNAL_MEMORY_START + IMEM_KIL_OFFSET;
        assert_eq!(bus.load(kil_addr, 8), 0, "LCC.KSD should mask KIL");
        assert_eq!(
            bus.keyboard.fifo_snapshot(),
            fifo_before,
            "a KSD-masked read must preserve the queued key"
        );
        assert!(bus.timer.key_irq_latched);
        assert!(bus.pending_kil);

        bus.memory.write_internal_byte(IMEM_LCC_OFFSET, 0);
        assert_ne!(
            bus.load(kil_addr, 8),
            0,
            "the selected pressed key should be visible after KSD clears"
        );
    }

    #[test]
    fn exact_input_event_does_not_manufacture_raw_matrix_keyi() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Iq7000Vram),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.timer.set_keyboard_irq_enabled(true);

        bus.inject_input_event(0xA3);

        assert_eq!(bus.keyboard.fifo_snapshot(), vec![0xA3]);
        let isr = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(isr & super::ISR_KEYI, 0);
        assert!(bus.pending_kil);
        assert!(!bus.irq_pending);
    }

    #[test]
    fn irq_pending_reasserts_when_isr_bits_set() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.memory
            .write_internal_byte(super::IMEM_IMR_OFFSET, super::IMR_MASTER | super::IMR_MTI);
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_MTI);
        bus.irq_pending = false;
        bus.in_interrupt = false;

        assert!(
            bus.irq_pending(),
            "ISR bits should reassert irq_pending when unmasked"
        );
        assert!(bus.irq_pending, "irq_pending latch should be set");
    }

    #[test]
    fn raw_keyi_reasserts_in_handler_until_physical_release() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.strobe_all_columns();
        bus.press_key(super::PF1_CODE);
        bus.in_interrupt = true;
        bus.memory.write_internal_byte(super::IMEM_ISR_OFFSET, 0);

        assert!(!bus.irq_pending(), "nested delivery remains deferred");
        assert_eq!(
            bus.memory
                .read_internal_byte(super::IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & super::ISR_KEYI,
            super::ISR_KEYI
        );

        bus.release_key(super::PF1_CODE);
        bus.memory.write_internal_byte(super::IMEM_ISR_OFFSET, 0);
        bus.irq_pending = false;
        assert!(!bus.irq_pending());
        assert_eq!(
            bus.memory
                .read_internal_byte(super::IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & super::ISR_KEYI,
            0
        );
    }

    #[test]
    fn per_instruction_scan_keeps_fifo_separate_from_raw_keyi() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.scan_on_timer = false;
        bus.keyboard.set_press_threshold(1);
        bus.strobe_all_columns();

        bus.press_key(super::PF1_CODE);
        let isr_before = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(isr_before & super::ISR_KEYI, 0);

        bus.finalize_instruction();
        bus.apply_deferred_key_irq();

        let isr_after = bus
            .memory
            .read_internal_byte(super::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(isr_after & super::ISR_KEYI, 0);
        assert!(bus.timer.key_irq_latched);
        assert!(bus.pending_kil);
        let _delivery_selected = bus.irq_pending();
        assert_ne!(
            bus.memory
                .read_internal_byte(super::IMEM_ISR_OFFSET)
                .unwrap_or(0)
                & super::ISR_KEYI,
            0,
            "raw held matrix level should assert KEYI"
        );
    }

    #[test]
    fn deliver_irq_prefers_onk_when_masked_in() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x0200);
        // Enable master + ONK mask.
        bus.memory
            .write_internal_byte(super::IMEM_IMR_OFFSET, super::IMR_MASTER | super::IMR_ONK);
        // Assert ONK pending.
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_ONKI);
        bus.pending_onk = true;
        bus.irq_pending = true;
        assert!(bus.irq_pending(), "ONK pending should signal irq_pending");
        bus.deliver_irq(&mut state).expect("deliver ONK IRQ");
        assert_eq!(
            bus.active_irq_mask,
            super::ISR_ONKI,
            "ONK should be the active IRQ mask"
        );
        assert_eq!(bus.last_irq_src.as_deref(), Some("ONK"));
        assert!(bus.in_interrupt);
    }

    #[test]
    fn deliver_irq_uses_complete_iq7000_rom_priority() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x0200);
        bus.memory.write_internal_byte(
            super::IMEM_IMR_OFFSET,
            super::IMR_MASTER
                | super::IMR_RX
                | super::IMR_EX
                | super::IMR_TX
                | super::IMR_ONK
                | super::IMR_KEY
                | super::IMR_STI
                | super::IMR_MTI,
        );
        bus.memory.write_internal_byte(
            super::IMEM_ISR_OFFSET,
            super::ISR_RXI
                | super::ISR_EXI
                | super::ISR_TXI
                | super::ISR_ONKI
                | super::ISR_KEYI
                | super::ISR_STI
                | super::ISR_MTI,
        );
        bus.pending_onk = true;
        bus.irq_pending = true;

        assert!(bus.irq_pending());
        assert_eq!(bus.last_irq_src.as_deref(), Some("RX"));
        bus.deliver_irq(&mut state)
            .expect("deliver highest-priority IRQ");

        assert_eq!(bus.active_irq_mask, super::ISR_RXI);
        assert_eq!(bus.last_irq_src.as_deref(), Some("RX"));
        assert!(bus.in_interrupt);
    }

    #[test]
    fn deliver_irq_poisoned_after_stack_overwrites_validated_vector() {
        let mut bus = test_standalone_bus();
        let mut state = LlamaState::new();
        state.set_pc(0x034567);
        state.set_reg(RegName::S, 0x0FFFFF);
        state.set_reg(RegName::F, 0x03);
        let imr = super::IMR_MASTER | super::IMR_ONK;
        state.set_reg(RegName::IMR, u32::from(imr));
        bus.memory.write_external_byte(0x0FFFFA, 0x00);
        bus.memory.write_external_byte(0x0FFFFB, 0x01);
        bus.memory.write_external_byte(0x0FFFFC, 0x00);
        bus.memory.write_external_byte(0x000100, 0x00);
        bus.memory.write_internal_byte(super::IMEM_IMR_OFFSET, imr);
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_ONKI);
        bus.pending_onk = true;
        bus.irq_pending = true;

        let error = bus
            .deliver_irq(&mut state)
            .expect_err("post-frame vector change must fail");

        assert_eq!(error, sc62015_core::llama::eval::VECTOR_UPPER_NIBBLE_ERROR);
        assert!(bus.poisoned.is_some());
        assert_eq!(state.get_reg(RegName::S), 0x0FFFFA);
        assert_eq!(
            state.get_reg(RegName::IMR),
            u32::from(imr & !super::IMR_MASTER)
        );
        assert!(reject_unrepresented_snapshot_runtime(&bus)
            .expect_err("poisoned runtime must not be snapshotted")
            .to_string()
            .contains("poisoned fail-stop runtime state"));
        assert_eq!(
            bus.deliver_irq(&mut state)
                .expect_err("poisoned runtime must reject continued delivery"),
            "standalone runtime is poisoned; power-on reset required"
        );
    }

    #[test]
    fn deliver_irq_rejects_noncanonical_vector_before_frame_or_metadata_mutation() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_pc(0x012345);
        state.set_reg(RegName::S, 0x000240);
        state.set_reg(RegName::F, 0x03);
        state.set_reg(RegName::IMR, u32::from(super::IMR_MASTER | super::IMR_ONK));
        bus.memory.write_external_byte(0x0FFFFA, 0x78);
        bus.memory.write_external_byte(0x0FFFFB, 0x56);
        bus.memory.write_external_byte(0x0FFFFC, 0xF4);
        bus.memory
            .write_internal_byte(super::IMEM_IMR_OFFSET, super::IMR_MASTER | super::IMR_ONK);
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_ONKI);
        bus.pending_onk = true;
        bus.irq_pending = true;
        bus.last_irq_src = Some("ONK".to_string());

        let external_before = bus.memory.external_slice().to_vec();
        let internal_before = bus.memory.internal_slice().to_vec();
        let writes_before = bus.memory.memory_write_count();
        let err = bus
            .deliver_irq(&mut state)
            .expect_err("noncanonical IRQ vector must be quarantined");

        assert_eq!(err, sc62015_core::llama::eval::VECTOR_UPPER_NIBBLE_ERROR);
        assert_eq!(state.pc(), 0x012345);
        assert_eq!(state.get_reg(RegName::S), 0x000240);
        assert_eq!(state.get_reg(RegName::F), 0x03);
        assert_eq!(
            state.get_reg(RegName::IMR),
            u32::from(super::IMR_MASTER | super::IMR_ONK)
        );
        assert_eq!(bus.memory.external_slice(), external_before);
        assert_eq!(bus.memory.internal_slice(), internal_before);
        assert_eq!(bus.memory.memory_write_count(), writes_before);
        assert!(bus.pending_onk);
        assert!(bus.irq_pending);
        assert!(!bus.in_interrupt);
        assert_eq!(bus.last_irq_src.as_deref(), Some("ONK"));
        assert_eq!(bus.active_irq_mask, 0);
        assert_eq!(bus.delivered_irq_count, 0);
        assert!(!bus.vec_patched);
    }

    #[test]
    fn deliver_irq_wraps_each_frame_byte_at_the_20_bit_boundary() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_pc(0x034567);
        state.set_reg(RegName::S, 0x000002);
        state.set_reg(RegName::F, 0x03);
        bus.memory.write_external_byte(0x0FFFFA, 0x78);
        bus.memory.write_external_byte(0x0FFFFB, 0x56);
        bus.memory.write_external_byte(0x0FFFFC, 0x02);
        let imr = super::IMR_MASTER | super::IMR_ONK;
        bus.memory.write_internal_byte(super::IMEM_IMR_OFFSET, imr);
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_ONKI);
        bus.pending_onk = true;
        bus.irq_pending = true;

        bus.deliver_irq(&mut state).expect("deliver wrapped IRQ");

        assert_eq!(state.get_reg(RegName::S), 0x0FFFFD);
        assert_eq!(state.get_reg(RegName::PC), 0x025678);
        for (address, expected) in [
            (0x0FFFFF, 0x67),
            (0x000000, 0x45),
            (0x000001, 0x03),
            (0x0FFFFE, 0x03),
            (0x0FFFFD, imr),
        ] {
            assert_eq!(
                bus.memory.load(address, 8),
                Some(u32::from(expected)),
                "wrapped frame byte at 0x{address:05X}"
            );
        }
    }

    #[test]
    fn deliver_irq_does_not_force_masked_onk_over_unmasked_timer() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x0200);
        bus.memory
            .write_internal_byte(super::IMEM_IMR_OFFSET, super::IMR_MASTER | super::IMR_MTI);
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_ONKI | super::ISR_MTI);
        bus.pending_onk = true;
        bus.irq_pending = true;

        bus.deliver_irq(&mut state).expect("deliver timer IRQ");

        assert_eq!(
            bus.active_irq_mask,
            super::ISR_MTI,
            "masked ONK must not override the ROM-visible unmasked timer source"
        );
        assert_eq!(bus.last_irq_src.as_deref(), Some("MTI"));
    }

    #[test]
    fn deliver_irq_prefers_sub_timer_over_main_timer() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x0200);
        bus.memory.write_internal_byte(
            super::IMEM_IMR_OFFSET,
            super::IMR_MASTER | super::IMR_STI | super::IMR_MTI,
        );
        bus.memory
            .write_internal_byte(super::IMEM_ISR_OFFSET, super::ISR_STI | super::ISR_MTI);
        bus.irq_pending = true;

        bus.deliver_irq(&mut state).expect("deliver sub-timer IRQ");

        assert_eq!(bus.active_irq_mask, super::ISR_STI);
        assert_eq!(bus.last_irq_src.as_deref(), Some("STI"));
    }

    #[test]
    fn retf_does_not_clear_in_interrupt() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        state.set_pc(0x12345);
        bus.in_interrupt = true;
        bus.active_irq_mask = super::ISR_MTI;
        bus.last_irq_src = Some("MTI".to_string());

        bus.handle_irq_return(0x07, &state);

        assert!(bus.in_interrupt);
        assert_eq!(bus.active_irq_mask, super::ISR_MTI);
        assert!(bus.last_irq_src.is_some());
    }

    #[test]
    fn reset_vector_matches_python_address() {
        let reset_addr = ROM_RESET_VECTOR_ADDR as usize;
        let vector_bytes = [0x45u8, 0x23, 0x01]; // little-endian 0x012345
        let max_addr = reset_addr;
        let mut rom = vec![0u8; max_addr + 3];
        for (i, byte) in vector_bytes.iter().enumerate() {
            rom[reset_addr + i] = *byte;
        }
        let expected_pc = (vector_bytes[0] as u32)
            | ((vector_bytes[1] as u32) << 8)
            | ((vector_bytes[2] as u32) << 16);

        let mut memory = MemoryImage::new();
        memory.load_external(&rom);
        let mut bus = StandaloneBus::new(
            memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(false, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state = LlamaState::new();
        power_on_reset(&mut bus, &mut state).expect("valid ROM reset vector");
        let pc_mask = 0x0F_FFFFu32;
        assert_eq!(
            state.pc(),
            expected_pc & pc_mask,
            "power_on_reset should use the reset vector at 0xFFFFD"
        );

        let runner_vec = (rom[ROM_RESET_VECTOR_ADDR as usize] as u32)
            | ((rom[ROM_RESET_VECTOR_ADDR as usize + 1] as u32) << 8)
            | ((rom[ROM_RESET_VECTOR_ADDR as usize + 2] as u32) << 16);
        state.set_pc(runner_vec & pc_mask);
        assert_eq!(
            state.pc(),
            expected_pc & pc_mask,
            "standalone runner PC seed must honour the PC-E500 reset vector"
        );
    }

    #[test]
    fn reset_trace_card_uses_blank_writable_ce1_and_zero_filled_ce6() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(false, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.enable_reset_trace_card();

        assert_eq!(bus.load(0x010012, 8), 0x00);
        assert_eq!(bus.load(0x040000, 8), 0x00);
        assert_eq!(bus.load(0x040001, 8), 0x00);

        bus.store(0x010012, 8, 0x99);
        bus.store(0x040000, 8, 0x4D);
        bus.store(0x040001, 8, 0xCA);

        assert_eq!(bus.load(0x010012, 8), 0x00);
        assert_eq!(bus.load(0x040000, 8), 0x4D);
        assert_eq!(bus.load(0x040001, 8), 0xCA);
    }

    #[test]
    fn advance_cycles_ticks_mti() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 1, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.timer.kb_irq_enabled = false;
        assert_eq!(bus.cycle_count, 0);
        assert!(!bus.timer.irq_pending);

        bus.advance_cycles(1);

        assert_eq!(bus.cycle_count, 1);
        assert!(bus.timer.irq_pending);
        assert_eq!(bus.timer.irq_source.as_deref(), Some("MTI"));
        assert_eq!(bus.last_irq_src.as_deref(), Some("MTI"));
    }

    #[test]
    fn key_seq_parses_waiters_and_hold() {
        let actions = parse_key_seq(
            "pf1:20,wait-op:5,wait-text:MAIN MENU,wait-power:off,wait-screen-change,wait-screen-empty,wait-screen-draw",
            100,
            DeviceModel::PcE500,
        )
        .expect("parse key seq");
        assert_eq!(actions.len(), 7);
        assert_eq!(actions[0].kind, KeySeqKind::Press);
        assert_eq!(actions[0].hold, 20);
        assert_eq!(actions[1].kind, KeySeqKind::WaitOp);
        assert_eq!(actions[2].kind, KeySeqKind::WaitText);
        assert_eq!(actions[3].kind, KeySeqKind::WaitPower);
        assert_eq!(actions[4].kind, KeySeqKind::WaitScreenChange);
        assert_eq!(actions[5].kind, KeySeqKind::WaitScreenEmpty);
        assert_eq!(actions[6].kind, KeySeqKind::WaitScreenDraw);
    }

    #[test]
    fn key_seq_accepts_space_alias() {
        let actions = parse_key_seq("space", 10, DeviceModel::PcE500).expect("parse key seq");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, KeySeqKind::Press);
        let code =
            KeyboardMatrix::matrix_code_for_key_name("KEY_SPACE").expect("expected KEY_SPACE");
        assert_eq!(actions[0].key, Some(AutoKeyKind::Matrix(code)));
    }

    #[test]
    fn key_seq_uses_generated_shifted_pc_input_map() {
        let actions = parse_key_seq("!", 10, DeviceModel::PcE500).expect("parse key seq");
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0].key,
            Some(AutoKeyKind::Chord {
                modifier: 0x06,
                code: 0x01,
            })
        );
    }

    #[test]
    fn key_seq_uses_generated_iq_input_map() {
        let actions = parse_key_seq("0", 10, DeviceModel::Iq7000).expect("parse key seq");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].key, Some(AutoKeyKind::Event(0x20)));
    }

    #[test]
    fn key_seq_accepts_named_iq_controls() {
        let actions = parse_key_seq(
            "memo,text:XMAS\\nPRESENTS,memo-enter,search-down",
            10,
            DeviceModel::Iq7000,
        )
        .expect("parse key seq");
        assert_eq!(actions[0].key, Some(AutoKeyKind::Event(0x08)));
        assert_eq!(actions[5].key, Some(AutoKeyKind::Event(0x3D)));
        assert_eq!(
            actions[actions.len() - 2].key,
            Some(AutoKeyKind::Event(0x45))
        );
        assert_eq!(
            actions[actions.len() - 1].key,
            Some(AutoKeyKind::Event(0x0B))
        );
    }

    #[test]
    fn key_seq_accepts_iq_shift_and_caps_controls() {
        let actions = parse_key_seq("shift,function,caps,caps-off", 10, DeviceModel::Iq7000)
            .expect("parse key seq");
        assert_eq!(actions.len(), 4);
        assert_eq!(actions[0].key, Some(AutoKeyKind::Event(0x01)));
        assert_eq!(actions[1].key, Some(AutoKeyKind::Event(0x04)));
        assert_eq!(actions[2].key, Some(AutoKeyKind::Event(0x09)));
        assert_eq!(actions[3].key, Some(AutoKeyKind::Event(0x09)));
    }

    #[test]
    fn key_seq_accepts_event_down_up_for_shifted_iq_menus() {
        let actions = parse_key_seq(
            "down:event:0x02,option,up:event:0x02",
            10,
            DeviceModel::Iq7000,
        )
        .expect("parse key seq");
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].kind, KeySeqKind::KeyDown);
        assert_eq!(actions[0].key, Some(AutoKeyKind::Event(0x02)));
        assert_eq!(actions[1].kind, KeySeqKind::Press);
        assert_eq!(actions[1].key, Some(AutoKeyKind::Event(0x1D)));
        assert_eq!(actions[2].kind, KeySeqKind::KeyUp);
        assert_eq!(actions[2].key, Some(AutoKeyKind::Event(0x02)));
    }

    #[test]
    fn pclink_serial_peer_drives_host_byte_msb_first() {
        let mut memory = MemoryImage::new();
        let mut peer = Iq7000PclinkSerialPeer::default();
        peer.attach(&mut memory);
        peer.queue_host_bytes(&[0xA5]);

        peer.before_instruction(&mut memory);
        assert_eq!(
            memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
            0
        );

        let bits = [true, false, true, false, false, true, false, true];
        for (idx, bit) in bits.iter().copied().enumerate() {
            peer.observe_eoh_write(false, &mut memory);
            assert_eq!(
                memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA
                    != 0,
                bit
            );
            peer.observe_eoh_write(true, &mut memory);
            assert_ne!(
                memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
                0
            );
            for _ in 0..IQ7000_PACOM_RELEASE_STEPS {
                peer.before_instruction(&mut memory);
            }
            if idx < bits.len() - 1 {
                assert_eq!(
                    memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
                    0
                );
            }
        }

        assert!(peer.host_tx.is_none());
        assert_ne!(
            memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
            0
        );
    }

    #[test]
    fn pclink_serial_peer_collects_rom_byte_msb_first() {
        let mut memory = MemoryImage::new();
        let mut peer = Iq7000PclinkSerialPeer::default();
        peer.attach(&mut memory);

        let bits = [true, false, false, false, false, false, false, false];
        peer.observe_eoh_write(false, &mut memory);
        for (idx, bit) in bits.iter().copied().enumerate() {
            peer.observe_eoh_write(bit, &mut memory);
            assert_ne!(
                memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
                0
            );
            peer.observe_eoh_write(true, &mut memory);
            if idx < bits.len() - 1 {
                assert_eq!(
                    memory.read_internal_byte(IMEM_EIH_OFFSET).unwrap_or(0) & IQ7000_PACOM_EIH_DATA,
                    0
                );
                peer.observe_eoh_write(false, &mut memory);
            }
        }

        assert_eq!(peer.pop_rom_byte(), Some(0x80));
        assert!(peer.rom_rx.is_none());
    }

    #[test]
    fn iq7000_lcd_annunciators_reads_shadow_and_workspace_state() {
        let mut memory = MemoryImage::new();
        memory
            .store(IQ7000_KEY_STATE_ADDR, 8, IQ7000_CAPS_ANNUNCIATOR as u32)
            .expect("store key state");
        let ann = iq7000_lcd_annunciators(&memory);
        assert_eq!(ann.state_raw, IQ7000_CAPS_ANNUNCIATOR);
        assert_eq!(ann.shadow_raw, 0);
        assert_eq!(ann.raw_union, IQ7000_CAPS_ANNUNCIATOR);
        assert_eq!(ann.unmapped_state, 0);
        assert_eq!(ann.unmapped_shadow, 0);
        assert_eq!(ann.unmapped_union, 0);
        assert!(ann.caps);
        assert!(!ann.shift);

        memory
            .store(
                IQ7000_ANNUNCIATOR_SHADOW_ADDR,
                8,
                IQ7000_SHIFT_ANNUNCIATOR as u32,
            )
            .expect("store annunciator shadow");
        let ann = iq7000_lcd_annunciators(&memory);
        assert_eq!(ann.state_raw, IQ7000_CAPS_ANNUNCIATOR);
        assert_eq!(ann.shadow_raw, IQ7000_SHIFT_ANNUNCIATOR);
        assert_eq!(
            ann.raw_union,
            IQ7000_SHIFT_ANNUNCIATOR | IQ7000_CAPS_ANNUNCIATOR
        );
        assert_eq!(ann.unmapped_state, 0);
        assert_eq!(ann.unmapped_shadow, 0);
        assert_eq!(ann.unmapped_union, 0);
        assert!(ann.shift);
        assert!(ann.caps);
    }

    #[test]
    fn iq7000_png_capture_adds_annunciator_strip() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("iq7000_lcd_annunciators_{stamp}.png"));
        let pixels = vec![vec![0u8, 1u8], vec![1u8, 0u8]];
        let ann = LcdAnnunciators {
            state_raw: 0,
            shadow_raw: IQ7000_CAPS_ANNUNCIATOR,
            raw_union: IQ7000_CAPS_ANNUNCIATOR,
            unmapped_state: 0,
            unmapped_shadow: 0,
            unmapped_union: 0,
            shift: false,
            caps: true,
        };
        write_lcd_png(&path, &pixels, 1, Some(&ann)).expect("write png");
        let png = std::fs::read(&path).expect("read png");
        let width = u32::from_be_bytes(png[16..20].try_into().expect("width"));
        let height = u32::from_be_bytes(png[20..24].try_into().expect("height"));
        assert_eq!(width, 2);
        assert_eq!(height, 9);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn iq7000_unmapped_annunciator_bit_is_reported_but_not_drawn() {
        let mut memory = MemoryImage::new();
        memory
            .store(IQ7000_ANNUNCIATOR_SHADOW_ADDR, 8, 0x80)
            .expect("store raw annunciator shadow");
        let ann = iq7000_lcd_annunciators(&memory);
        assert_eq!(ann.state_raw, 0);
        assert_eq!(ann.shadow_raw, 0x80);
        assert_eq!(ann.raw_union, 0x80);
        assert_eq!(ann.unmapped_state, 0);
        assert_eq!(ann.unmapped_shadow, 0x80);
        assert_eq!(ann.unmapped_union, 0x80);
        assert!(!ann.shift);
        assert!(!ann.caps);

        let pixels = vec![vec![0u8; 96]; 64];
        let baseline = LcdAnnunciators {
            state_raw: 0,
            shadow_raw: 0,
            raw_union: 0,
            unmapped_state: 0,
            unmapped_shadow: 0,
            unmapped_union: 0,
            shift: false,
            caps: false,
        };
        assert_eq!(
            pixels_with_iq7000_annunciators(&pixels, Some(&ann)),
            pixels_with_iq7000_annunciators(&pixels, Some(&baseline))
        );
    }

    #[test]
    fn iq7000_lcd_pixels_render_vram_as_96_by_64() {
        let mut lcd = create_lcd(sc62015_core::LcdKind::Iq7000Vram);
        lcd.write(0x405A, 0x80);
        lcd.write(0x605A, 0x40);

        let pixels = lcd_pixels(lcd.as_ref());

        assert_eq!(pixels.len(), 64);
        assert!(pixels.iter().all(|row| row.len() == 96));
        assert_eq!(pixels[0][5], 1);
        assert_eq!(pixels[33][5], 1);
    }

    #[test]
    fn debug_probe_range_parser_accepts_named_hex_range() {
        assert_eq!(
            parse_debug_probe_range("storage@0x1fd00:0x40").expect("range"),
            DebugProbeRange {
                name: "storage".to_string(),
                addr: 0x1FD00,
                len: 0x40,
            }
        );
    }

    #[test]
    fn scenario_key_seq_many_joins_with_semicolon() {
        let seq = ScenarioKeySeq::Many(vec![
            "memo".to_string(),
            "text:HELLO".to_string(),
            "memo-enter".to_string(),
        ]);
        assert_eq!(seq.join(), "memo;text:HELLO;memo-enter");
    }

    #[test]
    fn key_seq_accepts_exact_digitizer_events() {
        let actions =
            parse_key_seq("digitizer:0xA3", 10, DeviceModel::PcE500).expect("parse key seq");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, KeySeqKind::Press);
        assert_eq!(actions[0].key, Some(AutoKeyKind::InputEvent(0xA3)));
    }

    #[test]
    fn key_seq_wait_op_is_relative() {
        let actions =
            parse_key_seq("wait-op:5,pf1", 10, DeviceModel::PcE500).expect("parse key seq");
        let mut runner = KeySeqRunner::new(actions);
        let screen = ScreenState::default();
        let events = runner.step(10, true, &screen);
        assert!(events.is_empty());
        let events = runner.step(15, true, &screen);
        assert!(events.is_empty(), "wait-op completes but does not press");
        let events = runner.step(16, true, &screen);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, KeySeqEventKind::Press);
    }

    #[test]
    fn key_seq_wait_screen_change_tracks_baseline() {
        let actions = parse_key_seq("wait-screen-change,pf1", 10, DeviceModel::PcE500)
            .expect("parse key seq");
        let mut runner = KeySeqRunner::new(actions);
        let screen = ScreenState {
            valid: true,
            is_blank: true,
            signature: 1,
            text_valid: false,
            text: String::new(),
        };
        let events = runner.step(0, true, &screen);
        assert!(events.is_empty());
        let events = runner.step(1, true, &screen);
        assert!(events.is_empty());
        let screen_changed = ScreenState {
            signature: 2,
            ..screen
        };
        let events = runner.step(2, true, &screen_changed);
        assert!(events.is_empty());
        let events = runner.step(3, true, &screen_changed);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, KeySeqEventKind::Press);
    }

    #[test]
    fn iq7000_rtc_fixed_seed_populates_clock_workspace() {
        use sc62015_core::iq7000::{
            CLOCK_INITIALIZED_FLAG as IQ7000_CLOCK_INITIALIZED_FLAG,
            CLOCK_WORKSPACE_START as IQ7000_CLOCK_WORKSPACE_START,
        };

        let seed = parse_iq7000_rtc_arg("202604252052")
            .expect("parse fixed RTC")
            .expect("seed enabled");
        assert_eq!(seed.clock.as_ascii(), "202604252052");
        assert_eq!(
            seed.clock.read(IQ7000_CLOCK_WORKSPACE_START + 4, 16),
            Some(u16::from_le_bytes(*b"04") as u32)
        );

        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Iq7000Vram),
            TimerContext::new(false, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.install_iq7000_clock_seed(seed);

        assert_eq!(bus.load(IQ7000_CLOCK_WORKSPACE_START, 8), b'2' as u32);
        assert_eq!(
            bus.load(IQ7000_CLOCK_WORKSPACE_START + 10, 16),
            u16::from_le_bytes(*b"52") as u32
        );
        assert_eq!(
            bus.memory
                .load(IQ7000_CLOCK_INITIALIZED_FLAG, 8)
                .unwrap_or(0),
            1
        );
    }

    #[test]
    fn iq7000_rtc_seed_validates_range_and_off_mode() {
        assert!(parse_iq7000_rtc_arg("off").expect("off parses").is_none());
        assert!(parse_iq7000_rtc_arg("202613010000").is_err());
        assert!(parse_iq7000_rtc_arg("202601012460").is_err());
        assert!(parse_iq7000_rtc_arg("202601010059").is_ok());
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    #[test]
    fn snapshot_roundtrip_restores_state() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut memory = MemoryImage::new();
        let _ = memory.store(0x2000, 8, 0x12);
        memory.write_internal_byte(0x10, 0x34);
        memory
            .load_memory_card_with_writable(&vec![0xA5; 8192], false)
            .unwrap();

        let mut bus = StandaloneBus::new(
            memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.cycle_count = 1234;
        bus.timer.next_mti = 111;
        bus.timer.next_sti = 222;
        bus.pending_onk = true;
        assert_eq!(
            bus.memory
                .read_internal_byte(super::IMEM_SSR_OFFSET)
                .unwrap_or(0)
                & super::SSR_ONK,
            0,
            "standalone physical ON-key level must remain outside raw SSR storage"
        );

        let mut state = LlamaState::new();
        state.set_reg(RegName::PC, 0x12345);
        state.set_reg(RegName::A, 0x56);
        state.call_depth_inc();
        state.set_call_sub_level(2);
        state.set_power_state(PowerState::Halted);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let snapshot_path = std::env::temp_dir().join(format!("pce500_snapshot_{stamp}.pcsnap"));
        save_snapshot_state(&snapshot_path, &bus, &state, 42, DeviceModel::PcE500)
            .expect("save snapshot");

        let mut target_memory = MemoryImage::new();
        target_memory.set_memory_card_slot_present(false);
        let mut bus2 = StandaloneBus::new(
            target_memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state2 = LlamaState::new();
        let rom_bytes = vec![0u8; 0x100000];
        let meta = load_snapshot_state(
            &snapshot_path,
            &mut bus2,
            &mut state2,
            DeviceModel::PcE500,
            &rom_bytes,
        )
        .expect("load snapshot");

        assert_eq!(meta.instruction_count, 42);
        assert_eq!(state2.get_reg(RegName::PC) & ADDRESS_MASK, 0x12345);
        assert_eq!(state2.get_reg(RegName::A) & 0xFF, 0x56);
        assert_eq!(state2.power_state(), PowerState::Halted);
        assert_eq!(state2.call_sub_level(), 2);
        assert_eq!(bus2.cycle_count, 1234);
        assert!(bus2.pending_onk, "held ON-key level must round-trip");
        assert_eq!(
            bus2.memory
                .read_internal_byte(super::IMEM_SSR_OFFSET)
                .unwrap_or(0)
                & super::SSR_ONK,
            0,
            "restoring the physical level must leave raw SSR storage unchanged"
        );
        assert_eq!(bus2.memory.load(0x2000, 8).unwrap_or(0), 0x12);
        assert_eq!(bus2.memory.read_internal_byte(0x10).unwrap_or(0), 0x34);
        let card = bus2
            .memory
            .memory_card_snapshot()
            .unwrap()
            .expect("snapshot present card overrides active absent slot");
        assert_eq!(card.mode, sc62015_core::memory::MemoryCardMode::Present);
        assert_eq!(card.capacity, 8192);
        assert!(!card.writable);
        assert_eq!(card.payload, vec![0xA5; 8192]);

        let _ = std::fs::remove_file(snapshot_path);
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    #[test]
    fn snapshot_timer_next_is_exact() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.cycle_count = 1234;
        bus.timer.next_mti = 9999;
        bus.timer.next_sti = 8888;

        let state = LlamaState::new();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let snapshot_path =
            std::env::temp_dir().join(format!("pce500_snapshot_timer_{stamp}.pcsnap"));
        save_snapshot_state(&snapshot_path, &bus, &state, 0, DeviceModel::PcE500)
            .expect("save snapshot");

        let loaded = snapshot::load_snapshot(&snapshot_path).expect("load snapshot");
        let meta = loaded.metadata;
        assert_eq!(meta.device_model, Some(DeviceModel::PcE500));
        assert_eq!(meta.timer.next_mti, 9999);
        assert_eq!(meta.timer.next_sti, 8888);

        let _ = std::fs::remove_file(snapshot_path);
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    #[test]
    fn snapshot_v4_serializes_cards_and_rejects_unrepresented_device_state() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let snapshot_path =
            std::env::temp_dir().join(format!("pce500_snapshot_reject_device_{stamp}.pcsnap"));
        let _ = std::fs::remove_file(&snapshot_path);

        let mut memory = MemoryImage::new();
        memory
            .load_memory_card(&vec![0xA5; 0x1_0000])
            .expect("install memory card");
        let bus = StandaloneBus::new(
            memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        save_snapshot_state(
            &snapshot_path,
            &bus,
            &LlamaState::new(),
            0,
            DeviceModel::PcE500,
        )
        .expect("present card is represented by snapshot v4");
        let loaded = snapshot::load_snapshot(&snapshot_path).unwrap();
        let card = loaded.memory_card.expect("typed present card");
        assert_eq!(card.mode, sc62015_core::memory::MemoryCardMode::Present);
        assert_eq!(card.capacity, 0x1_0000);
        assert!(card.writable);
        assert_eq!(card.payload, vec![0xA5; 0x1_0000]);

        let mut absent_card_memory = MemoryImage::new();
        absent_card_memory
            .load_memory_card_with_writable(&vec![0x5A; 0x4000], false)
            .unwrap();
        absent_card_memory.set_memory_card_slot_present(false);
        let absent_card_bus = StandaloneBus::new(
            absent_card_memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        save_snapshot_state(
            &snapshot_path,
            &absent_card_bus,
            &LlamaState::new(),
            0,
            DeviceModel::PcE500,
        )
        .expect("absent card and retained medium are represented by snapshot v4");
        let loaded = snapshot::load_snapshot(&snapshot_path).unwrap();
        let card = loaded.memory_card.expect("typed absent card");
        assert_eq!(card.mode, sc62015_core::memory::MemoryCardMode::Absent);
        assert_eq!(card.capacity, 0x4000);
        assert!(!card.writable);
        assert_eq!(card.payload, vec![0x5A; 0x4000]);

        let mut active_present_memory = MemoryImage::new();
        active_present_memory
            .load_memory_card_with_writable(&vec![0xC3; 0x8000], true)
            .unwrap();
        let mut active_present_bus = StandaloneBus::new(
            active_present_memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        load_snapshot_state(
            &snapshot_path,
            &mut active_present_bus,
            &mut LlamaState::new(),
            DeviceModel::PcE500,
            &vec![0; sc62015_core::EXTERNAL_SPACE],
        )
        .expect("snapshot absent card overrides active present card");
        let restored = active_present_bus
            .memory
            .memory_card_snapshot()
            .unwrap()
            .unwrap();
        assert_eq!(restored.mode, sc62015_core::memory::MemoryCardMode::Absent);
        assert_eq!(restored.capacity, 0x4000);
        assert!(!restored.writable);
        assert_eq!(restored.payload, vec![0x5A; 0x4000]);
        let valid_archive = std::fs::read(&snapshot_path).expect("read valid destination archive");

        let mut generic_overlay_memory = MemoryImage::new();
        generic_overlay_memory.add_ram_overlay(0x8000, 16, "snapshot-test");
        let generic_overlay_bus = StandaloneBus::new(
            generic_overlay_memory,
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let error = save_snapshot_state(
            &snapshot_path,
            &generic_overlay_bus,
            &LlamaState::new(),
            0,
            DeviceModel::PcE500,
        )
        .expect_err("generic overlays remain outside snapshot v4");
        assert!(
            error.to_string().contains("generic memory-overlay")
                || error.to_string().contains("attestation")
        );
        assert_eq!(
            std::fs::read(&snapshot_path).unwrap(),
            valid_archive,
            "failed save must preserve the previous destination"
        );

        let mut rtc_bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Iq7000Vram),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        rtc_bus.install_iq7000_clock_seed(
            parse_iq7000_rtc_arg("202604252052")
                .expect("RTC seed parses")
                .expect("RTC is enabled"),
        );
        let error = save_snapshot_state(
            &snapshot_path,
            &rtc_bus,
            &LlamaState::new(),
            0,
            DeviceModel::Iq7000,
        )
        .expect_err("unserialized RTC protocol state must fail closed");
        assert!(error.to_string().contains("RTC protocol"));
        assert_eq!(
            std::fs::read(&snapshot_path).unwrap(),
            valid_archive,
            "a second failed save must still preserve the previous destination"
        );
        let _ = std::fs::remove_file(snapshot_path);
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    #[test]
    fn iq7000_snapshot_load_preserves_exact_model_ranges() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let ranges = vec![(0x90000, 0x9FFFF), (0x80000, 0x8FFFF)];
        let canonical_ranges =
            snapshot::canonical_snapshot_ranges("test readonly", &ranges).unwrap();
        let mut source_memory = MemoryImage::new();
        source_memory.set_readonly_ranges(ranges.clone());
        let source_bus = StandaloneBus::new(
            source_memory,
            create_lcd(sc62015_core::LcdKind::Iq7000Vram),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let snapshot_path =
            std::env::temp_dir().join(format!("iq7000_snapshot_ranges_{stamp}.pcsnap"));
        save_snapshot_state(
            &snapshot_path,
            &source_bus,
            &LlamaState::new(),
            0,
            DeviceModel::Iq7000,
        )
        .expect("save IQ-7000 snapshot");

        let mut target_memory = MemoryImage::new();
        target_memory.set_readonly_ranges(ranges.clone());
        let mut target_bus = StandaloneBus::new(
            target_memory,
            create_lcd(sc62015_core::LcdKind::Iq7000Vram),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut target_state = LlamaState::new();
        load_snapshot_state(
            &snapshot_path,
            &mut target_bus,
            &mut target_state,
            DeviceModel::Iq7000,
            &[],
        )
        .expect("load IQ-7000 snapshot");

        assert_eq!(
            target_bus.memory.readonly_ranges(),
            canonical_ranges.as_slice()
        );
        target_bus
            .memory
            .store(0x00100, 8, 0xA5)
            .expect("IQ-7000 low RAM remains writable");
        assert_eq!(target_bus.memory.load(0x00100, 8), Some(0xA5));
        let _ = std::fs::remove_file(snapshot_path);
    }

    #[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
    #[test]
    fn snapshot_roundtrip_preserves_off_state() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 10, 20),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        bus.cycle_count = 55;

        let mut state = LlamaState::new();
        state.set_pc(0x22222);
        state.set_power_state(PowerState::Off);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let snapshot_path =
            std::env::temp_dir().join(format!("pce500_snapshot_off_{stamp}.pcsnap"));
        save_snapshot_state(&snapshot_path, &bus, &state, 7, DeviceModel::PcE500)
            .expect("save snapshot");

        let mut bus2 = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 0, 0),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        let mut state2 = LlamaState::new();
        let rom_bytes = vec![0u8; 0x100000];
        let meta = load_snapshot_state(
            &snapshot_path,
            &mut bus2,
            &mut state2,
            DeviceModel::PcE500,
            &rom_bytes,
        )
        .expect("load snapshot");

        assert_eq!(meta.instruction_count, 7);
        assert_eq!(state2.power_state(), PowerState::Off);
        assert_eq!(state2.get_reg(RegName::PC) & ADDRESS_MASK, 0x22222);

        let _ = std::fs::remove_file(snapshot_path);
    }

    #[test]
    fn pc_e500_configures_keyboard_scan_mode() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 1, 1),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        assert!(bus.scan_on_timer, "scan_on_timer defaults to true");
        assert!(!bus.timer_finalize_clamp, "timer clamp defaults to false");
        configure_bus_for_model(&mut bus, DeviceModel::PcE500);
        assert!(!bus.scan_on_timer, "PC-E500 should scan per instruction");
        assert!(
            !bus.timer_finalize_clamp,
            "timer clamp should remain disabled by default"
        );
        let snap = bus.keyboard.snapshot_state();
        assert_eq!(snap.press_threshold, 1);
    }

    #[test]
    fn keyboard_trace_resolves_relocatable_iocs_fifo() {
        let mut bus = StandaloneBus::new(
            MemoryImage::new(),
            create_lcd(sc62015_core::LcdKind::Hd61202),
            TimerContext::new(true, 1, 1),
            false,
            0,
            false,
            None,
            None,
            None,
        );
        assert_eq!(bus.keyboard_fifo_addresses(), None);
        bus.memory
            .store(PCE500_IOCS_WS_PTR_ADDR, 24, 0x00BF9B4)
            .expect("store IOCS workspace pointer");
        bus.memory
            .store(0x00BF9B6, 16, 0x0050)
            .expect("store FIFO offset");

        assert_eq!(
            bus.keyboard_fifo_addresses(),
            Some((0x00BFA04, 0x00BF9B8, 0x00BF9B9))
        );
        assert!(bus.is_keyboard_fifo_address(0x00BF9B8));
        assert!(bus.is_keyboard_fifo_address(0x00BF9B9));
        assert!(bus.is_keyboard_fifo_address(0x00BFA04));
        assert!(bus.is_keyboard_fifo_address(0x00BFA13));
        assert!(!bus.is_keyboard_fifo_address(0x00BFC96));

        bus.memory
            .store(PCE500_IOCS_WS_PTR_ADDR, 24, 0x00BFA00)
            .expect("relocate IOCS workspace pointer");
        bus.memory
            .store(0x00BFA02, 16, 0x0170)
            .expect("store relocated FIFO offset");
        assert_eq!(
            bus.keyboard_fifo_addresses(),
            Some((0x00BFB70, 0x00BFA04, 0x00BFA05))
        );
    }

    #[test]
    fn cli_defaults_to_the_shared_core_runtime_and_model_card_policy() {
        let args = Args::try_parse_from(["pce500-llama"]).expect("parse defaults");
        assert_eq!(args.runtime, RuntimeEngine::Core);
        assert_eq!(args.card, CardMode::Auto);
        assert!(core_runtime_unsupported_options(&args).is_empty());
    }

    #[test]
    fn specialized_diagnostics_require_an_explicit_legacy_runtime() {
        let args = Args::try_parse_from(["pce500-llama", "--snapshot-out", "state.pcsnap"])
            .expect("parse snapshot option");
        assert_eq!(
            core_runtime_unsupported_options(&args),
            vec!["--snapshot-in/--snapshot-out"]
        );
    }

    #[test]
    fn model_independent_diagnostics_use_the_shared_runtime() {
        let args = Args::try_parse_from([
            "pce500-llama",
            "--stop-pc",
            "0x12345",
            "--trace-pc",
            "0x10000",
            "--trace-regs",
            "--dump-lcd-trace",
            "lcd.json",
            "--debug-probe-json",
            "probe.json",
            "--debug-probe-range",
            "work@0x100:0x10",
        ])
        .expect("parse shared diagnostics");
        assert!(core_runtime_unsupported_options(&args).is_empty());
    }

    #[test]
    fn shared_runtime_stop_pc_halts_the_cli_boundary_loop() {
        let mut runtime = CoreRuntime::new();
        runtime.memory.write_external_slice(0, &[0x00, 0x00]);
        runtime.state.set_pc(0);
        let args = Args::try_parse_from(["pce500-llama", "--stop-pc", "1"]).expect("parse stop PC");
        let mut diagnostics = RuntimeDiagnostics::from_args(&args).expect("diagnostics");

        let consumed = step_runtime_boundaries(&mut runtime, 100, &mut diagnostics)
            .expect("execute until stop PC");

        assert_eq!(consumed, 1);
        assert_eq!(runtime.instruction_count(), 1);
        assert_eq!(runtime.state.pc(), 1);
        assert!(diagnostics.stopped);
    }

    #[test]
    fn runtime_key_sequence_budget_advances_while_cpu_is_off() {
        let mut runtime = CoreRuntime::new();
        runtime.state.set_power_state(PowerState::Off);
        let args = Args::try_parse_from(["pce500-llama"]).expect("parse defaults");
        let mut diagnostics = RuntimeDiagnostics::from_args(&args).expect("diagnostics");

        let consumed = run_runtime_key_seq(
            &mut runtime,
            "wait-op:2,on:1",
            10,
            DeviceModel::PcE500,
            None,
            false,
            &mut diagnostics,
        )
        .expect("OFF-state key sequence must remain bounded");

        assert!(consumed <= 10);
        assert!(!runtime.state.is_off());
    }
}
