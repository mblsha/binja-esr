// PY_SOURCE: sc62015/pysc62015/emulator.py

use clap::Parser;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::llama::state::mask_for;
use sc62015_core::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_RXD_OFFSET};
use sc62015_core::{
    pce500::DEFAULT_MTI_PERIOD, pce500::DEFAULT_STI_PERIOD, pce500::ROM_WINDOW_START,
    timer::TimerContext, CoreRuntime, DeviceModel, LoopDetectorConfig,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fs;
use std::io::{stdout, IsTerminal, Write};
use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const IQ7000_TEXT_ROWS: usize = 8;
const IQ7000_TEXT_COLS: usize = 16;
const STATUS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const PF_KEY_HOLD_STEPS: u64 = 2_000;
const PF_AUTO_HOLD_STEPS: u64 = 40_000;
const ON_AUTO_HOLD_CYCLES: u64 = 20_000;
const BASIC_REPL_HUB_PC: u32 = 0x00FFE09;
const BASIC_WARM_START_PC: u32 = 0x00F9C94;
const AUTO_TYPE_START_DELAY_STEPS: u64 = 20_000;
const AUTO_TYPE_GAP_STEPS: u64 = 5_000;
const BASIC_KEY_CODE: u8 = 0x04;
const IMR_MASTER: u8 = 0x80;
const IMR_KEY: u8 = 0x04;
const ISR_KEYI: u8 = 0x04;
const PF1_CODE: u8 = 0x56;
const OFF_WAIT_PC: u32 = 0x00F1036;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum CardMode {
    Present,
    Absent,
}

#[derive(Parser, Debug)]
#[command(
    name = "sc62015-lcd",
    about = "Render decoded LCD text in a terminal window."
)]
struct Args {
    /// ROM model/profile (affects default ROM + LCD decoder).
    #[arg(long, value_enum, default_value_t = DeviceModel::DEFAULT)]
    model: DeviceModel,

    /// ROM image to load (defaults to repo-symlinked ROM for --model).
    #[arg(long, value_name = "PATH")]
    rom: Option<PathBuf>,

    /// Memory card slot state (PC-E500).
    #[arg(long, value_enum, default_value_t = CardMode::Present)]
    card: CardMode,

    /// Number of instructions to execute before exiting (0 = run until Ctrl+C).
    #[arg(long, default_value_t = 0)]
    steps: u64,

    /// Number of instructions between LCD refresh checks.
    #[arg(long, default_value_t = 20_000)]
    refresh_steps: u64,

    /// Number of instructions between input polls (0 = only poll each refresh).
    #[arg(long, default_value_t = 1_000)]
    input_steps: u64,

    /// Sleep this many milliseconds after each refresh check (0 = no sleep).
    #[arg(long, default_value_t = 0)]
    sleep_ms: u64,

    /// Disable timers (MTI/STI) while running.
    #[arg(long, default_value_t = false)]
    disable_timers: bool,

    /// Do not use the alternate screen buffer (useful in tmux capture panes).
    #[arg(long, default_value_t = false)]
    no_alt_screen: bool,

    /// Force raw-mode + key polling even if stdin/stdout are not TTYs.
    #[arg(long, default_value_t = false)]
    force_tty: bool,

    /// Map digits 1-5 to PF1-PF5 (disables typing those digits as characters).
    #[arg(long, default_value_t = false)]
    pf_numbers: bool,

    /// BNIDA JSON file with function names to show in the status line.
    #[arg(long, value_name = "PATH")]
    bnida: Option<PathBuf>,

    /// Force-enable KEY IRQ delivery when injecting keys (debug helper).
    #[arg(long, default_value_t = false)]
    force_key_irq: bool,

    /// Auto-run the PF1 twice boot flow (wait for S2(CARD), PF1, wait for S1(MAIN), PF1 again).
    #[arg(long, default_value_t = false)]
    pf1_twice: bool,

    /// Stub-return immediately from IOCS dispatch (debug helper).
    #[arg(long, default_value_t = false)]
    stub_iocs: bool,

    /// Skip delay_* routines by forcing a fast return (debug helper).
    #[arg(long, default_value_t = false)]
    fast_delay: bool,

    /// Stub-return from SIO routines (debug helper).
    #[arg(long, default_value_t = false)]
    stub_sio: bool,

    /// Fast-clear external RAM ranges when the ROM calls clear_external_ram_* helpers.
    #[arg(long, default_value_t = false)]
    fast_init: bool,

    /// Jump to the BASIC entry point after the PF1 flow (debug helper).
    #[arg(long, default_value_t = false)]
    jump_basic: bool,

    /// Auto-type a string once BASIC is reached (debug helper).
    #[arg(long)]
    auto_type: Option<String>,

    /// Auto-press the BASIC key after PF1 completes (debug helper).
    #[arg(long, default_value_t = false)]
    auto_basic: bool,

    /// Delay before auto-pressing BASIC (in steps).
    #[arg(long, default_value_t = AUTO_TYPE_START_DELAY_STEPS)]
    auto_basic_delay: u64,

    /// Delay before auto-typing (in steps).
    #[arg(long, default_value_t = AUTO_TYPE_START_DELAY_STEPS)]
    auto_type_delay: u64,

    /// Write loop report JSON on exit (defaults to loop_report_<epoch>.json).
    #[arg(long, value_name = "PATH")]
    loop_report: Option<PathBuf>,
}

struct TerminalGuard {
    use_tty: bool,
    use_alt: bool,
}

impl TerminalGuard {
    fn enter(use_tty: bool, use_alt: bool) -> Result<Self, Box<dyn Error>> {
        let mut out = stdout();
        if use_tty {
            crossterm::terminal::enable_raw_mode()?;
            if use_alt {
                crossterm::execute!(out, EnterAlternateScreen)?;
            }
            crossterm::execute!(out, Hide, Clear(ClearType::All))?;
            out.flush()?;
        }
        Ok(Self { use_tty, use_alt })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.use_tty {
            let mut out = stdout();
            let _ = crossterm::execute!(out, Show);
            if self.use_alt {
                let _ = crossterm::execute!(out, LeaveAlternateScreen);
            }
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
}

fn default_rom_path(model: DeviceModel) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../data/{}", model.rom_basename()))
}

fn lcd_geometry(model: DeviceModel) -> (usize, usize) {
    match model {
        DeviceModel::PcE500 => {
            let rows = sc62015_core::lcd::LCD_DISPLAY_ROWS / 8;
            let cols = sc62015_core::lcd::LCD_DISPLAY_COLS / 6;
            (rows, cols)
        }
        DeviceModel::Iq7000 => (IQ7000_TEXT_ROWS, IQ7000_TEXT_COLS),
    }
}

fn normalize_lines(mut lines: Vec<String>, line_count: usize, width: usize) -> Vec<String> {
    if lines.len() > line_count {
        lines.truncate(line_count);
    }
    while lines.len() < line_count {
        lines.push(String::new());
    }
    for line in &mut lines {
        if line.len() > width {
            line.truncate(width);
        }
        if line.len() < width {
            let pad = width.saturating_sub(line.len());
            line.push_str(&" ".repeat(pad));
        }
    }
    lines
}

fn render_frame(
    lines: &[String],
    status: &str,
    extra_lines: &[String],
    use_tty: bool,
) -> Result<(), Box<dyn Error>> {
    let mut out = stdout();
    if use_tty {
        let (cols, _) = crossterm::terminal::size().unwrap_or((0, 0));
        let max_cols = cols.saturating_sub(1) as usize;
        crossterm::queue!(out, MoveTo(0, 0), Clear(ClearType::All))?;
        for (row, line) in lines.iter().enumerate() {
            let view = if max_cols > 0 {
                line.chars().take(max_cols).collect::<String>()
            } else {
                line.clone()
            };
            crossterm::queue!(out, MoveTo(0, row as u16), Clear(ClearType::CurrentLine))?;
            write!(out, "{view}")?;
        }
        let status_view = if max_cols > 0 {
            status.chars().take(max_cols).collect::<String>()
        } else {
            status.to_string()
        };
        let status_row = lines.len() as u16 + 1;
        crossterm::queue!(out, MoveTo(0, status_row), Clear(ClearType::CurrentLine))?;
        write!(out, "{status_view}")?;
        out.flush()?;
        return Ok(());
    }
    for line in lines {
        writeln!(out, "{line}")?;
    }
    writeln!(out)?;
    writeln!(out, "{status}")?;
    if !extra_lines.is_empty() {
        for line in extra_lines {
            writeln!(out, "{line}")?;
        }
    }
    out.flush()?;
    Ok(())
}

fn render_status_line(status: &str, row: u16, use_tty: bool) -> Result<(), Box<dyn Error>> {
    if !use_tty {
        return Ok(());
    }
    let mut out = stdout();
    let (cols, _) = crossterm::terminal::size().unwrap_or((0, 0));
    let max_cols = cols.saturating_sub(1) as usize;
    let status_view = if max_cols > 0 {
        status.chars().take(max_cols).collect::<String>()
    } else {
        status.to_string()
    };
    crossterm::queue!(out, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
    write!(out, "{status_view}")?;
    out.flush()?;
    Ok(())
}

fn format_status(
    runtime: &CoreRuntime,
    executed: u64,
    last_key: &Option<String>,
    last_key_step: u64,
    symbols: Option<&SymbolMap>,
) -> String {
    let pc = runtime.state.pc() & 0x000f_ffff;
    let power_state = if runtime.state.is_off() {
        "OFF"
    } else if runtime.state.is_halted() {
        "HALT"
    } else {
        "RUN"
    };
    let label = format_symbol(pc, symbols);
    let pc_display = format!("pc=0x{pc:05X} {label}");
    let key_status = last_key
        .as_ref()
        .map(|label| format!(" last_key={label}@{last_key_step}"))
        .unwrap_or_default();
    format!("{pc_display} steps={executed} state={power_state}{key_status} (Ctrl+C to exit)")
}

fn resolve_symbol(addr: u32, symbols: Option<&SymbolMap>) -> Option<(u32, String, u32)> {
    let map = symbols?;
    let (base, name) = map.range(..=addr).next_back()?;
    let offset = addr.saturating_sub(*base);
    Some((*base, name.clone(), offset))
}

fn resolve_function(
    addr: u32,
    symbols: Option<&SymbolMap>,
    functions: Option<&FunctionSet>,
) -> Option<(u32, String, u32)> {
    let funcs = functions?;
    let base = *funcs.range(..=addr).next_back()?;
    let name = symbols
        .and_then(|map| map.get(&base).cloned())
        .unwrap_or_else(|| format!("sub_{base:05X}"));
    let offset = addr.saturating_sub(base);
    Some((base, name, offset))
}

fn format_symbol(addr: u32, symbols: Option<&SymbolMap>) -> String {
    if let Some((_base, name, offset)) = resolve_symbol(addr, symbols) {
        if offset == 0 {
            return name;
        }
        return format!("{name}+0x{offset:X}");
    }
    format!("sub_{addr:05X}")
}

fn format_call_stack_lines(frames: &[u32], symbols: Option<&SymbolMap>) -> Vec<String> {
    if frames.is_empty() {
        return vec!["Call stack: (empty)".to_string()];
    }
    let mut out = Vec::with_capacity(frames.len() + 1);
    out.push("Call stack:".to_string());
    for (idx, frame) in frames.iter().enumerate() {
        let addr = frame & 0x000f_ffff;
        let label = format_symbol(addr, symbols);
        out.push(format!("{idx:02}: {label} (0x{addr:05X})"));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn format_debug_lines(
    runtime: &CoreRuntime,
    symbols: Option<&SymbolMap>,
    functions: Option<&FunctionSet>,
    last_key: &Option<String>,
    last_key_step: u64,
    pending_releases: &[PendingRelease],
    halted_steps: u64,
    pf1_stage: Option<Pf1TwiceStage>,
) -> Vec<String> {
    let kb_irq = if runtime.timer.kb_irq_enabled {
        "on"
    } else {
        "off"
    };
    let key_latch = if runtime.timer.key_irq_latched {
        "on"
    } else {
        "off"
    };
    let irq_pending = if runtime.timer.irq_pending {
        "on"
    } else {
        "off"
    };
    let in_irq = if runtime.timer.in_interrupt {
        "on"
    } else {
        "off"
    };
    let imr = runtime.memory.read_internal_byte(0xFB).unwrap_or(0);
    let isr = runtime.memory.read_internal_byte(0xFC).unwrap_or(0);
    let kil = runtime.memory.read_internal_byte(0xF2).unwrap_or(0);
    let imr_reg = runtime.state.get_reg(RegName::IMR) & 0xFF;
    let instr = runtime.instruction_count();
    let cycles = runtime.cycle_count();
    let power_state = if runtime.state.is_off() {
        "OFF"
    } else if runtime.state.is_halted() {
        "HALT"
    } else {
        "RUN"
    };
    let fifo_len = runtime
        .keyboard
        .as_ref()
        .map(|kb| kb.fifo_len())
        .unwrap_or(0);
    let last = last_key.as_deref().unwrap_or("—");
    let last_step = if last_key.is_some() {
        format!("{last_key_step}")
    } else {
        "—".to_string()
    };
    let pending = if pending_releases.is_empty() {
        "—".to_string()
    } else {
        pending_releases
            .iter()
            .map(|entry| format!("{:02X}", entry.code))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut lines = Vec::new();
    if let Some(line) = format_iocs_debug_line(runtime, symbols) {
        lines.push(line);
    }
    lines.extend([
        format!(
            "KB: irq={kb_irq} latch={key_latch} pending={irq_pending} in_irq={in_irq} imr=0x{imr:02X} imr_reg=0x{imr_reg:02X} isr=0x{isr:02X} kil=0x{kil:02X} fifo={fifo_len}"
        ),
        format!(
            "CPU: instr={instr} cycles={cycles} state={power_state} halted_steps={halted_steps}"
        ),
        format!("Key: last={last}@{last_step} pending=[{pending}]"),
        format!(
            "Auto: pf1_twice={}",
            pf1_stage
                .map(|stage| format!("{stage:?}"))
                .unwrap_or_else(|| "off".to_string())
        ),
    ]);
    let loop_line = match runtime
        .loop_detector()
        .and_then(|det| det.current_summary())
    {
        Some(summary) => {
            let mut line = format!(
                "Loop: start=0x{start:05X} len={len} reps={reps}",
                start = summary.start_pc,
                len = summary.len,
                reps = summary.repeats
            );
            let alt_count = summary.candidate_lengths.len().saturating_sub(1);
            if alt_count > 0 {
                line.push_str(&format!(" alts={alt_count}"));
            }
            line
        }
        None => "Loop: (none)".to_string(),
    };
    lines.push(loop_line);
    if let Some(detector) = runtime.loop_detector() {
        if detector.current_summary().is_some() {
            if let Some(report) = detector.last_report() {
                let mut functions_seen = BTreeMap::<u32, String>::new();
                for entry in &report.trace {
                    if entry.mainline_index.is_none() {
                        continue;
                    }
                    let pc = entry.pc_before & 0x000f_ffff;
                    if let Some((base, name, _)) = resolve_function(pc, symbols, functions) {
                        functions_seen.entry(base).or_insert(name);
                    } else if let Some((base, name, _)) = resolve_symbol(pc, symbols) {
                        functions_seen.entry(base).or_insert(name);
                    }
                }
                if !functions_seen.is_empty() {
                    let names_list = functions_seen.into_values().collect::<Vec<_>>();
                    let count = names_list.len();
                    let names = names_list.join(", ");
                    lines.push(format!("Loop fns({count}): {names}"));
                }
            }
        }
    }
    lines
}

#[allow(clippy::too_many_arguments)]
fn format_extra_lines(
    runtime: &CoreRuntime,
    symbols: Option<&SymbolMap>,
    functions: Option<&FunctionSet>,
    last_key: &Option<String>,
    last_key_step: u64,
    pending_releases: &[PendingRelease],
    halted_steps: u64,
    pf1_stage: Option<Pf1TwiceStage>,
) -> Vec<String> {
    let mut lines = format_call_stack_lines(runtime.state.call_stack(), symbols);
    lines.extend(format_debug_lines(
        runtime,
        symbols,
        functions,
        last_key,
        last_key_step,
        pending_releases,
        halted_steps,
        pf1_stage,
    ));
    lines
}

fn render_extra_lines(
    lines: &[String],
    start_row: u16,
    use_tty: bool,
    prev_lines: &mut usize,
) -> Result<(), Box<dyn Error>> {
    if !use_tty {
        return Ok(());
    }
    let mut out = stdout();
    let (cols, _) = crossterm::terminal::size().unwrap_or((0, 0));
    let max_cols = cols.saturating_sub(1) as usize;
    for (idx, line) in lines.iter().enumerate() {
        let view = if max_cols > 0 {
            line.chars().take(max_cols).collect::<String>()
        } else {
            line.clone()
        };
        crossterm::queue!(
            out,
            MoveTo(0, start_row.saturating_add(idx as u16)),
            Clear(ClearType::CurrentLine)
        )?;
        write!(out, "{view}")?;
    }
    if *prev_lines > lines.len() {
        for extra in lines.len()..*prev_lines {
            crossterm::queue!(
                out,
                MoveTo(0, start_row.saturating_add(extra as u16)),
                Clear(ClearType::CurrentLine)
            )?;
        }
    }
    out.flush()?;
    *prev_lines = lines.len();
    Ok(())
}

fn decode_row0(
    text_decoder: &Option<sc62015_core::device::DeviceTextDecoder>,
    runtime: &CoreRuntime,
) -> String {
    match (text_decoder, runtime.lcd.as_deref()) {
        (Some(decoder), Some(lcd)) => decoder
            .decode_display_text(lcd)
            .first()
            .cloned()
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn strip_leading_line_comments(raw: &str) -> String {
    let lines = raw.split('\n');
    let mut output = Vec::new();
    let mut skip = true;
    for line in lines {
        if skip && line.trim_start().starts_with("//") {
            continue;
        }
        skip = false;
        output.push(line);
    }
    output.join("\n")
}

fn default_bnida_path(model: DeviceModel) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    match model {
        DeviceModel::PcE500 => root.join("rom-analysis/pc-e500/s3-en/bnida.json"),
        DeviceModel::Iq7000 => root.join("rom-analysis/iq-7000/bnida.json"),
    }
}

fn default_loop_report_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    PathBuf::from(format!(
        "loop_report_{}_{}.json",
        stamp.as_secs(),
        stamp.subsec_nanos()
    ))
}

fn ensure_term() {
    let term = std::env::var("TERM").unwrap_or_default();
    if term.trim().is_empty() {
        std::env::set_var("TERM", "xterm-256color");
    }
}

fn load_bnida_symbols(path: &PathBuf) -> Result<SymbolMap, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let cleaned = strip_leading_line_comments(&raw);
    let parsed: BnidaJson = serde_json::from_str(&cleaned)?;
    let mut out = BTreeMap::new();
    for (key, value) in parsed.names.unwrap_or_default() {
        let addr = key.trim().parse::<u32>()? & 0x000f_ffff;
        let name = value.trim().to_string();
        if !name.is_empty() {
            out.insert(addr, name);
        }
    }
    Ok(out)
}

fn load_bnida_functions(path: &PathBuf) -> Result<FunctionSet, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let cleaned = strip_leading_line_comments(&raw);
    let parsed: BnidaJson = serde_json::from_str(&cleaned)?;
    let mut out = BTreeSet::new();
    for addr in parsed.functions.unwrap_or_default() {
        out.insert(addr & 0x000f_ffff);
    }
    Ok(out)
}

fn in_iocs_dispatch(runtime: &CoreRuntime, symbols: Option<&SymbolMap>) -> bool {
    let pc = runtime.state.pc() & 0x000f_ffff;
    if let Some((_base, name, _)) = resolve_symbol(pc, symbols) {
        if name.contains("iocs_dispatch") {
            return true;
        }
    }
    if let Some(frame) = runtime.state.call_stack().last().copied() {
        if let Some((_base, name, _)) = resolve_symbol(frame, symbols) {
            return name.contains("iocs_dispatch");
        }
    }
    false
}

fn in_delay_routine(runtime: &CoreRuntime, symbols: Option<&SymbolMap>) -> bool {
    let pc = runtime.state.pc() & 0x000f_ffff;
    if let Some((_base, name, _)) = resolve_symbol(pc, symbols) {
        if name.starts_with("delay_") {
            return true;
        }
    }
    if let Some(frame) = runtime.state.call_stack().last().copied() {
        if let Some((_base, name, _)) = resolve_symbol(frame, symbols) {
            return name.starts_with("delay_");
        }
    }
    false
}

fn in_sio_routine(runtime: &CoreRuntime, symbols: Option<&SymbolMap>) -> bool {
    let pc = runtime.state.pc() & 0x000f_ffff;
    if let Some((_base, name, _)) = resolve_symbol(pc, symbols) {
        if name.starts_with("sio_") || name == "halt_with_keyboard_matrix_active_f1742" {
            return true;
        }
    }
    if let Some(frame) = runtime.state.call_stack().last().copied() {
        if let Some((_base, name, _)) = resolve_symbol(frame, symbols) {
            if name.starts_with("sio_") || name == "halt_with_keyboard_matrix_active_f1742" {
                return true;
            }
        }
    }
    false
}

fn in_clear_external_ram(runtime: &CoreRuntime, symbols: Option<&SymbolMap>) -> bool {
    let pc = runtime.state.pc() & 0x000f_ffff;
    if let Some((_base, name, _)) = resolve_symbol(pc, symbols) {
        if name.starts_with("clear_external_ram_") {
            return true;
        }
    }
    if let Some(frame) = runtime.state.call_stack().last().copied() {
        if let Some((_base, name, _)) = resolve_symbol(frame, symbols) {
            if name.starts_with("clear_external_ram_") {
                return true;
            }
        }
    }
    false
}

fn fast_clear_pce500_ram(runtime: &mut CoreRuntime, cleared: &mut bool, zero_buf: &[u8]) {
    if *cleared {
        return;
    }
    runtime.memory.write_external_slice(0, zero_buf);
    *cleared = true;
}

fn format_iocs_debug_line(runtime: &CoreRuntime, symbols: Option<&SymbolMap>) -> Option<String> {
    if !in_iocs_dispatch(runtime, symbols) {
        return None;
    }
    let a = runtime.state.get_reg(RegName::A);
    let b = runtime.state.get_reg(RegName::B);
    let ba = runtime.state.get_reg(RegName::BA);
    let i = runtime.state.get_reg(RegName::I);
    let u = runtime.state.get_reg(RegName::U);
    let s = runtime.state.get_reg(RegName::S);
    let f = runtime.state.get_reg(RegName::F);
    let fc = runtime.state.get_reg(RegName::FC);
    let fz = runtime.state.get_reg(RegName::FZ);
    Some(format!(
        "IOCS: A=0x{a:02X} B=0x{b:02X} BA=0x{ba:04X} I=0x{i:04X} U=0x{u:06X} S=0x{s:06X} F=0x{f:02X} FC={fc} FZ={fz}"
    ))
}

fn pop_stack_value(runtime: &mut CoreRuntime, bits: u8) -> u32 {
    let bytes = bits.div_ceil(8);
    let mask = mask_for(RegName::S);
    let mut value = 0u32;
    let mut sp = runtime.state.get_reg(RegName::S);
    for i in 0..bytes {
        let byte = runtime
            .memory
            .load_with_pc(sp, 8, Some(runtime.state.pc()))
            .unwrap_or(0)
            & 0xFF;
        value |= byte << (8 * i);
        sp = sp.wrapping_add(1) & mask;
    }
    runtime.state.set_reg(RegName::S, sp);
    value
}

fn force_ret(runtime: &mut CoreRuntime) {
    let pc_before = runtime.state.pc();
    let ret = pop_stack_value(runtime, 16);
    let current_page = pc_before & 0xFF0000;
    let _ = runtime.state.pop_call_page();
    let dest = (current_page | (ret & 0xFFFF)) & 0xFFFFF;
    runtime.state.set_pc(dest);
    runtime.state.call_depth_dec();
    let _ = runtime.state.pop_call_stack();
}

fn force_retf(runtime: &mut CoreRuntime) {
    let ret = pop_stack_value(runtime, 24);
    let dest = ret & 0xFFFFF;
    runtime.state.set_pc(dest);
    runtime.state.call_depth_dec();
    let _ = runtime.state.pop_call_stack();
}

fn force_return_auto(runtime: &mut CoreRuntime) {
    let call_depth = runtime.state.call_stack().len();
    let page_depth = runtime.state.call_page_depth();
    if page_depth < call_depth {
        force_retf(runtime);
    } else {
        force_ret(runtime);
    }
}

fn push_return_16(runtime: &mut CoreRuntime, addr: u32) {
    let mask = mask_for(RegName::S);
    let sp = runtime.state.get_reg(RegName::S);
    let new_sp = sp.wrapping_sub(2) & mask;
    let addr16 = addr & 0xFFFF;
    let _ = runtime
        .memory
        .store_with_pc(new_sp, 8, addr16 & 0xFF, Some(runtime.state.pc()));
    let _ = runtime.memory.store_with_pc(
        new_sp.wrapping_add(1),
        8,
        (addr16 >> 8) & 0xFF,
        Some(runtime.state.pc()),
    );
    runtime.state.set_reg(RegName::S, new_sp);
}

fn jump_to_basic_loop(runtime: &mut CoreRuntime) {
    push_return_16(runtime, BASIC_REPL_HUB_PC);
    runtime.state.set_pc(BASIC_WARM_START_PC);
    runtime.state.set_halted(false);
    runtime.state.reset_call_metrics();
}

struct StubReturnConfig<'a> {
    symbols: Option<&'a SymbolMap>,
    stub_iocs: bool,
    fast_delay: bool,
    stub_sio: bool,
    fast_init: bool,
    ram_zero_buf: Option<&'a [u8]>,
}

fn apply_stub_returns(
    runtime: &mut CoreRuntime,
    ram_cleared: &mut bool,
    config: StubReturnConfig<'_>,
) -> bool {
    if config.stub_iocs && in_iocs_dispatch(runtime, config.symbols) {
        force_return_auto(runtime);
        return true;
    }
    if config.fast_delay && in_delay_routine(runtime, config.symbols) {
        force_return_auto(runtime);
        return true;
    }
    if config.stub_sio && in_sio_routine(runtime, config.symbols) {
        runtime.state.set_reg(RegName::FC, 0);
        runtime.state.set_reg(RegName::FZ, 0);
        runtime.memory.write_internal_byte(IMEM_RXD_OFFSET, 0x41);
        runtime.memory.write_internal_byte(0xD5, 0x41);
        force_return_auto(runtime);
        return true;
    }
    if config.fast_init {
        if let Some(buf) = config.ram_zero_buf {
            if in_clear_external_ram(runtime, config.symbols) {
                fast_clear_pce500_ram(runtime, ram_cleared, buf);
                force_return_auto(runtime);
                return true;
            }
        }
    }
    false
}

struct KeyFeedback {
    label: Option<String>,
    quit: bool,
}

struct PendingRelease {
    code: u8,
    due_step: u64,
}

type SymbolMap = BTreeMap<u32, String>;
type FunctionSet = BTreeSet<u32>;

#[derive(serde::Deserialize)]
struct BnidaJson {
    names: Option<HashMap<String, String>>,
    #[serde(default)]
    functions: Option<Vec<u32>>,
}

#[derive(Clone, Copy, Debug)]
enum Pf1TwiceStage {
    WaitBootText,
    Press1,
    WaitMainText,
    Press2,
    WaitNextText,
    Done,
}

fn matrix_code_for_char(ch: char) -> Option<u8> {
    let upper = ch.to_ascii_uppercase();
    match upper {
        'A' => Some(0x03),
        'B' => Some(0x15),
        'C' => Some(0x0D),
        'D' => Some(0x0B),
        'E' => Some(0x09),
        'F' => Some(0x12),
        'G' => Some(0x13),
        'H' => Some(0x1A),
        'I' => Some(0x20),
        'J' => Some(0x1B),
        'K' => Some(0x22),
        'L' => Some(0x23),
        'M' => Some(0x1D),
        'N' => Some(0x1C),
        'O' => Some(0x21),
        'P' => Some(0x50),
        'Q' => Some(0x01),
        'R' => Some(0x10),
        'S' => Some(0x0A),
        'T' => Some(0x11),
        'U' => Some(0x19),
        'V' => Some(0x14),
        'W' => Some(0x08),
        'X' => Some(0x0C),
        'Y' => Some(0x18),
        'Z' => Some(0x05),
        '0' => Some(0x2F),
        '1' => Some(0x2E),
        '2' => Some(0x36),
        '3' => Some(0x3E),
        '4' => Some(0x2D),
        '5' => Some(0x35),
        '6' => Some(0x3D),
        '7' => Some(0x2C),
        '8' => Some(0x34),
        '9' => Some(0x3C),
        ' ' => Some(0x16),
        '.' => Some(0x3F),
        ',' => Some(0x24),
        ';' => Some(0x25),
        '+' => Some(0x47),
        '-' => Some(0x46),
        '*' => Some(0x45),
        '/' => Some(0x44),
        '=' => Some(0x4F),
        '(' => Some(0x4B),
        ')' => Some(0x48),
        _ => None,
    }
}

fn matrix_code_for_ctrl_digit(digit: char) -> Option<u8> {
    match digit {
        '1' => Some(0x56),
        '2' => Some(0x55),
        '3' => Some(0x54),
        '4' => Some(0x53),
        '5' => Some(0x52),
        _ => None,
    }
}

fn inject_key(
    runtime: &mut CoreRuntime,
    code: u8,
    executed: u64,
    pending_releases: &mut Vec<PendingRelease>,
    hold_steps: u64,
    force_key_irq: bool,
) {
    if force_key_irq && !runtime.timer.kb_irq_enabled {
        runtime.timer.kb_irq_enabled = true;
    }
    if let Some(kb) = runtime.keyboard.as_mut() {
        let kb_irq_enabled = runtime.timer.kb_irq_enabled || force_key_irq;
        let _ = kb.inject_matrix_event(code, false, &mut runtime.memory, kb_irq_enabled);
        if kb_irq_enabled {
            runtime.timer.key_irq_latched = true;
            if let Some(cur) = runtime.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                runtime
                    .memory
                    .write_internal_byte(IMEM_ISR_OFFSET, cur | ISR_KEYI);
            }
            runtime.timer.irq_pending = true;
            if runtime.timer.irq_source.is_none() && !runtime.timer.in_interrupt {
                runtime.timer.irq_source = Some("KEY".to_string());
            }
        }
        if hold_steps == 0 {
            let _ = kb.inject_matrix_event(code, true, &mut runtime.memory, kb_irq_enabled);
        } else {
            pending_releases.retain(|pending| pending.code != code);
            pending_releases.push(PendingRelease {
                code,
                due_step: executed.saturating_add(hold_steps),
            });
        }
    }
    if force_key_irq {
        let current = runtime
            .memory
            .read_internal_byte(IMEM_IMR_OFFSET)
            .unwrap_or(0);
        let next = current | IMR_MASTER | IMR_KEY;
        if next != current {
            runtime.memory.write_internal_byte(IMEM_IMR_OFFSET, next);
            runtime.state.set_reg(RegName::IMR, next as u32);
        }
    }
}

fn apply_pending_releases(
    runtime: &mut CoreRuntime,
    pending_releases: &mut Vec<PendingRelease>,
    executed: u64,
) {
    if pending_releases.is_empty() {
        return;
    }
    let Some(kb) = runtime.keyboard.as_mut() else {
        pending_releases.clear();
        return;
    };
    let kb_irq_enabled = runtime.timer.kb_irq_enabled;
    let mut idx = 0;
    while idx < pending_releases.len() {
        if pending_releases[idx].due_step <= executed {
            let code = pending_releases[idx].code;
            let _ = kb.inject_matrix_event(code, true, &mut runtime.memory, kb_irq_enabled);
            pending_releases.swap_remove(idx);
        } else {
            idx += 1;
        }
    }
}

fn handle_key_event(
    runtime: &mut CoreRuntime,
    key: KeyEvent,
    pf_numbers: bool,
    executed: u64,
    pending_releases: &mut Vec<PendingRelease>,
    pending_on_release: &mut Option<u64>,
    force_key_irq: bool,
) -> KeyFeedback {
    if key.kind != KeyEventKind::Press {
        return KeyFeedback {
            label: None,
            quit: false,
        };
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        if let KeyCode::Char(ch) = key.code {
            if ch == 'c' || ch == 'C' {
                return KeyFeedback {
                    label: None,
                    quit: true,
                };
            }
            if ch == 'o' || ch == 'O' {
                runtime.press_on_key();
                *pending_on_release =
                    Some(runtime.cycle_count().saturating_add(ON_AUTO_HOLD_CYCLES));
                return KeyFeedback {
                    label: Some("ON".to_string()),
                    quit: false,
                };
            }
            if let Some(code) = matrix_code_for_ctrl_digit(ch) {
                inject_key(
                    runtime,
                    code,
                    executed,
                    pending_releases,
                    PF_KEY_HOLD_STEPS,
                    force_key_irq,
                );
                return KeyFeedback {
                    label: Some(format!("PF{}", ch)),
                    quit: false,
                };
            }
        }
    }
    match key.code {
        KeyCode::Enter => {
            inject_key(runtime, 0x4F, executed, pending_releases, 0, force_key_irq);
            return KeyFeedback {
                label: Some("=".to_string()),
                quit: false,
            };
        }
        KeyCode::Backspace => {
            inject_key(runtime, 0x4D, executed, pending_releases, 0, force_key_irq);
            return KeyFeedback {
                label: Some("BS".to_string()),
                quit: false,
            };
        }
        KeyCode::Delete => {
            inject_key(runtime, 0x4C, executed, pending_releases, 0, force_key_irq);
            return KeyFeedback {
                label: Some("DEL".to_string()),
                quit: false,
            };
        }
        KeyCode::F(1) => {
            inject_key(
                runtime,
                0x56,
                executed,
                pending_releases,
                PF_KEY_HOLD_STEPS,
                force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF1".to_string()),
                quit: false,
            };
        }
        KeyCode::F(2) => {
            inject_key(
                runtime,
                0x55,
                executed,
                pending_releases,
                PF_KEY_HOLD_STEPS,
                force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF2".to_string()),
                quit: false,
            };
        }
        KeyCode::F(3) => {
            inject_key(
                runtime,
                0x54,
                executed,
                pending_releases,
                PF_KEY_HOLD_STEPS,
                force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF3".to_string()),
                quit: false,
            };
        }
        KeyCode::F(4) => {
            inject_key(
                runtime,
                0x53,
                executed,
                pending_releases,
                PF_KEY_HOLD_STEPS,
                force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF4".to_string()),
                quit: false,
            };
        }
        KeyCode::F(5) => {
            inject_key(
                runtime,
                0x52,
                executed,
                pending_releases,
                PF_KEY_HOLD_STEPS,
                force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF5".to_string()),
                quit: false,
            };
        }
        KeyCode::Char(ch) => {
            if pf_numbers {
                if let Some(code) = matrix_code_for_ctrl_digit(ch) {
                    inject_key(
                        runtime,
                        code,
                        executed,
                        pending_releases,
                        PF_KEY_HOLD_STEPS,
                        force_key_irq,
                    );
                    return KeyFeedback {
                        label: Some(format!("PF{}", ch)),
                        quit: false,
                    };
                }
            }
            if let Some(code) = matrix_code_for_char(ch) {
                inject_key(runtime, code, executed, pending_releases, 0, force_key_irq);
                return KeyFeedback {
                    label: Some(ch.to_string()),
                    quit: false,
                };
            }
        }
        _ => {}
    }
    KeyFeedback {
        label: None,
        quit: false,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.refresh_steps == 0 {
        return Err("refresh_steps must be > 0".into());
    }
    let rom_path = args.rom.unwrap_or_else(|| default_rom_path(args.model));
    let rom_bytes = fs::read(&rom_path)?;

    let mut runtime = CoreRuntime::new();
    runtime.set_device_model(args.model)?;
    args.model.configure_runtime(&mut runtime, &rom_bytes)?;
    runtime
        .memory
        .set_memory_card_slot_present(matches!(args.card, CardMode::Present));
    if args.disable_timers {
        *runtime.timer = TimerContext::new(false, 0, 0);
    } else {
        *runtime.timer =
            TimerContext::new(true, DEFAULT_MTI_PERIOD as i32, DEFAULT_STI_PERIOD as i32);
    }
    let loop_config = LoopDetectorConfig {
        detect_stride: args.refresh_steps,
        ..Default::default()
    };
    runtime.enable_loop_detector(loop_config);
    runtime.power_on_reset();

    let text_decoder = args.model.text_decoder(&rom_bytes);
    let (line_count, width) = lcd_geometry(args.model);
    let bnida_path = args.bnida.or_else(|| {
        let candidate = default_bnida_path(args.model);
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    });
    let symbols = bnida_path
        .as_ref()
        .and_then(|path| load_bnida_symbols(path).ok());
    let function_addrs = bnida_path
        .as_ref()
        .and_then(|path| load_bnida_functions(path).ok());
    let mut last_lines: Vec<String> = Vec::new();
    let mut first_draw = true;
    let mut last_key: Option<String> = None;
    let mut last_key_step: u64 = 0;
    let mut pending_releases: Vec<PendingRelease> = Vec::new();
    let mut pending_on_release: Option<u64> = None;
    let mut auto_type_queue: Vec<char> =
        args.auto_type.clone().unwrap_or_default().chars().collect();
    let mut auto_type_next_step: Option<u64> = None;
    let mut auto_basic_step: Option<u64> = None;
    let mut auto_basic_pending = args.auto_basic;
    let mut jump_basic_pending = args.jump_basic;
    let mut halted_steps: u64 = 0;
    let mut last_lcd_check: u64 = 0;
    let mut pf1_stage = if args.pf1_twice {
        Pf1TwiceStage::WaitBootText
    } else {
        Pf1TwiceStage::Done
    };
    let mut main_row0_seen: Option<String> = None;
    let mut fast_init_cleared = false;
    let fast_init_buf = if args.fast_init && matches!(args.model, DeviceModel::PcE500) {
        Some(vec![0u8; ROM_WINDOW_START])
    } else {
        None
    };
    let lcd_check_interval: u64 = 5_000;
    ensure_term();
    let use_tty = args.force_tty || (stdout().is_terminal() && std::io::stdin().is_terminal());
    let use_alt = use_tty && !args.no_alt_screen;

    let _guard = TerminalGuard::enter(use_tty, use_alt)?;
    let status_row = line_count as u16 + 1;
    let mut last_status_draw = Instant::now();
    let mut last_stack_lines = 0usize;
    let stack_row = status_row.saturating_add(1);
    if use_tty {
        let blank_lines = normalize_lines(Vec::new(), line_count, width);
        let status = format_status(&runtime, 0, &last_key, last_key_step, symbols.as_ref());
        let extra_lines = format_extra_lines(
            &runtime,
            symbols.as_ref(),
            function_addrs.as_ref(),
            &last_key,
            last_key_step,
            &pending_releases,
            halted_steps,
            Some(pf1_stage).filter(|stage| !matches!(stage, Pf1TwiceStage::Done)),
        );
        render_frame(&blank_lines, &status, &extra_lines, use_tty)?;
        render_extra_lines(&extra_lines, stack_row, use_tty, &mut last_stack_lines)?;
        last_lines = blank_lines;
        first_draw = false;
        last_status_draw = Instant::now();
    }

    let mut executed: u64 = 0;
    let mut running = true;
    while running {
        if args.steps > 0 && executed >= args.steps {
            break;
        }
        let mut dirty = false;
        let mut remaining = if args.steps == 0 {
            args.refresh_steps
        } else {
            (args.steps - executed).min(args.refresh_steps)
        };
        if remaining == 0 {
            break;
        }
        while remaining > 0 {
            let mut did_stub = false;
            if apply_stub_returns(
                &mut runtime,
                &mut fast_init_cleared,
                StubReturnConfig {
                    symbols: symbols.as_ref(),
                    stub_iocs: args.stub_iocs,
                    fast_delay: args.fast_delay,
                    stub_sio: args.stub_sio,
                    fast_init: args.fast_init,
                    ram_zero_buf: fast_init_buf.as_deref(),
                },
            ) {
                executed = executed.saturating_add(1);
                remaining = remaining.saturating_sub(1);
                dirty = true;
                did_stub = true;
            }
            let chunk = if args.input_steps == 0 {
                remaining
            } else {
                remaining.min(args.input_steps)
            };
            if !did_stub {
                runtime.step(chunk as usize)?;
                executed = executed.saturating_add(chunk);
                remaining = remaining.saturating_sub(chunk);
            }
            apply_pending_releases(&mut runtime, &mut pending_releases, executed);
            if let Some(release_cycle) = pending_on_release {
                if runtime.cycle_count() >= release_cycle {
                    runtime.release_on_key();
                    pending_on_release = None;
                }
            }
            if let Some(step) = auto_basic_step {
                if executed >= step {
                    inject_key(
                        &mut runtime,
                        BASIC_KEY_CODE,
                        executed,
                        &mut pending_releases,
                        PF_KEY_HOLD_STEPS,
                        args.force_key_irq,
                    );
                    last_key = Some("BASIC".to_string());
                    last_key_step = executed;
                    auto_basic_step = None;
                    if auto_type_next_step.is_none() && !auto_type_queue.is_empty() {
                        auto_type_next_step = Some(executed.saturating_add(args.auto_type_delay));
                    }
                    dirty = true;
                }
            }
            if let Some(next_step) = auto_type_next_step {
                if executed >= next_step {
                    let mut sent = false;
                    while let Some(ch) = auto_type_queue.first().copied() {
                        auto_type_queue.remove(0);
                        let code = match ch {
                            '\n' | '\r' => Some(0x4F),
                            _ => matrix_code_for_char(ch),
                        };
                        if let Some(code) = code {
                            inject_key(
                                &mut runtime,
                                code,
                                executed,
                                &mut pending_releases,
                                0,
                                args.force_key_irq,
                            );
                            last_key = Some(ch.to_string());
                            last_key_step = executed;
                            auto_type_next_step =
                                Some(executed.saturating_add(AUTO_TYPE_GAP_STEPS));
                            sent = true;
                            dirty = true;
                            break;
                        }
                    }
                    if !sent {
                        auto_type_next_step = None;
                    }
                }
            }
            if runtime.state.is_halted() {
                halted_steps = halted_steps.saturating_add(chunk);
            } else {
                halted_steps = 0;
            }
            if args.pf1_twice
                && matches!(pf1_stage, Pf1TwiceStage::Done)
                && pending_on_release.is_none()
                && runtime.state.is_halted()
                && (runtime.state.pc() & 0x000f_ffff) == OFF_WAIT_PC
            {
                runtime.press_on_key();
                pending_on_release =
                    Some(runtime.cycle_count().saturating_add(ON_AUTO_HOLD_CYCLES));
                last_key = Some("ON(auto)".to_string());
                last_key_step = executed;
                dirty = true;
            }
            let should_check_lcd = !matches!(pf1_stage, Pf1TwiceStage::Done)
                || ((auto_basic_pending || jump_basic_pending) && auto_basic_step.is_none());
            if should_check_lcd && executed.saturating_sub(last_lcd_check) >= lcd_check_interval {
                last_lcd_check = executed;
                let row0 = decode_row0(&text_decoder, &runtime);
                match pf1_stage {
                    Pf1TwiceStage::WaitBootText => {
                        if row0.contains("S2(CARD):") {
                            pf1_stage = Pf1TwiceStage::Press1;
                        }
                    }
                    Pf1TwiceStage::Press1 => {
                        inject_key(
                            &mut runtime,
                            PF1_CODE,
                            executed,
                            &mut pending_releases,
                            PF_AUTO_HOLD_STEPS,
                            args.force_key_irq,
                        );
                        last_key = Some("PF1(auto)".to_string());
                        last_key_step = executed;
                        pf1_stage = Pf1TwiceStage::WaitMainText;
                    }
                    Pf1TwiceStage::WaitMainText => {
                        if row0.contains("S1(MAIN):") {
                            main_row0_seen = Some(row0.clone());
                            pf1_stage = Pf1TwiceStage::Press2;
                        }
                    }
                    Pf1TwiceStage::Press2 => {
                        inject_key(
                            &mut runtime,
                            PF1_CODE,
                            executed,
                            &mut pending_releases,
                            PF_AUTO_HOLD_STEPS,
                            args.force_key_irq,
                        );
                        last_key = Some("PF1(auto2)".to_string());
                        last_key_step = executed;
                        if args.jump_basic {
                            jump_to_basic_loop(&mut runtime);
                            pf1_stage = Pf1TwiceStage::Done;
                            jump_basic_pending = false;
                            if auto_type_next_step.is_none() && !auto_type_queue.is_empty() {
                                auto_type_next_step =
                                    Some(executed.saturating_add(args.auto_type_delay));
                            }
                            dirty = true;
                        } else {
                            if args.auto_basic {
                                auto_basic_step =
                                    Some(executed.saturating_add(args.auto_basic_delay));
                            }
                            pf1_stage = Pf1TwiceStage::WaitNextText;
                        }
                    }
                    Pf1TwiceStage::WaitNextText => {
                        if let Some(main) = main_row0_seen.as_ref() {
                            if !row0.trim().is_empty() && row0 != *main {
                                pf1_stage = Pf1TwiceStage::Done;
                                if args.auto_basic {
                                    auto_basic_step =
                                        Some(executed.saturating_add(args.auto_basic_delay));
                                    auto_basic_pending = false;
                                }
                            }
                        }
                    }
                    Pf1TwiceStage::Done => {}
                }
                if (auto_basic_pending || jump_basic_pending)
                    && auto_basic_step.is_none()
                    && (!args.pf1_twice || matches!(pf1_stage, Pf1TwiceStage::Done))
                    && (row0.contains("S2(CARD):") || row0.contains("S1(MAIN):"))
                {
                    if jump_basic_pending {
                        jump_to_basic_loop(&mut runtime);
                        jump_basic_pending = false;
                        if auto_type_next_step.is_none() && !auto_type_queue.is_empty() {
                            auto_type_next_step =
                                Some(executed.saturating_add(args.auto_basic_delay));
                        }
                        dirty = true;
                    } else {
                        auto_basic_step = Some(executed.saturating_add(args.auto_basic_delay));
                        auto_basic_pending = false;
                    }
                }
            }

            if use_tty {
                while event::poll(Duration::from_millis(0))? {
                    if let Event::Key(key) = event::read()? {
                        let feedback = handle_key_event(
                            &mut runtime,
                            key,
                            args.pf_numbers,
                            executed,
                            &mut pending_releases,
                            &mut pending_on_release,
                            args.force_key_irq,
                        );
                        if feedback.quit {
                            running = false;
                            break;
                        }
                        if let Some(label) = feedback.label {
                            last_key = Some(label);
                            last_key_step = executed;
                            dirty = true;
                        }
                    }
                }
            }
            if use_tty && !first_draw && last_status_draw.elapsed() >= STATUS_UPDATE_INTERVAL {
                let status = format_status(
                    &runtime,
                    executed,
                    &last_key,
                    last_key_step,
                    symbols.as_ref(),
                );
                let extra_lines = format_extra_lines(
                    &runtime,
                    symbols.as_ref(),
                    function_addrs.as_ref(),
                    &last_key,
                    last_key_step,
                    &pending_releases,
                    halted_steps,
                    Some(pf1_stage).filter(|stage| !matches!(stage, Pf1TwiceStage::Done)),
                );
                render_status_line(&status, status_row, use_tty)?;
                render_extra_lines(&extra_lines, stack_row, use_tty, &mut last_stack_lines)?;
                last_status_draw = Instant::now();
            }
            if !running {
                break;
            }
            if dirty {
                break;
            }
            if args.steps > 0 && executed >= args.steps {
                break;
            }
        }

        let lines = match (&text_decoder, runtime.lcd.as_deref()) {
            (Some(decoder), Some(lcd)) => decoder.decode_display_text(lcd),
            _ => Vec::new(),
        };
        let lines = normalize_lines(lines, line_count, width);
        if first_draw || dirty || lines != last_lines {
            let status = format_status(
                &runtime,
                executed,
                &last_key,
                last_key_step,
                symbols.as_ref(),
            );
            let extra_lines = format_extra_lines(
                &runtime,
                symbols.as_ref(),
                function_addrs.as_ref(),
                &last_key,
                last_key_step,
                &pending_releases,
                halted_steps,
                Some(pf1_stage).filter(|stage| !matches!(stage, Pf1TwiceStage::Done)),
            );
            render_frame(&lines, &status, &extra_lines, use_tty)?;
            render_extra_lines(&extra_lines, stack_row, use_tty, &mut last_stack_lines)?;
            last_lines = lines;
            first_draw = false;
            last_status_draw = Instant::now();
        }

        if args.sleep_ms > 0 {
            sleep(Duration::from_millis(args.sleep_ms));
        }
    }

    if let Some(detector) = runtime.loop_detector() {
        if let Some(report) = detector.last_report() {
            let path = args
                .loop_report
                .clone()
                .unwrap_or_else(default_loop_report_path);
            let json = serde_json::to_string_pretty(report)?;
            fs::write(&path, json)?;
            eprintln!("[loop] report saved to {}", path.display());
        }
    }

    Ok(())
}
