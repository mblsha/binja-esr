// PY_SOURCE: sc62015/pysc62015/emulator.py

#[path = "sc62015_lcd/host.rs"]
mod host;

use chrono::{Datelike, Timelike, Utc};
use clap::Parser;
use crossterm::{
    cursor::MoveTo,
    event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    terminal::{Clear, ClearType},
};
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::llama::state::mask_for;
use sc62015_core::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET, IMEM_RXD_OFFSET};
use sc62015_core::native_ui::ControlState;
use sc62015_core::pacing::{ExecutionMode, Pacer, HOST_SLICE_TARGET_US};
use sc62015_core::{
    iq7000_annunciators::Iq7000Annunciators, pce500::ROM_WINDOW_START, CoreRuntime,
    DeviceMemoryCardProfile, DeviceModel, LoopDetectorConfig,
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
const PF_KEY_HOLD_STEPS: u64 = 40_000;
const CHAR_KEY_HOLD_STEPS: u64 = 10_000;
const SHIFTED_CHORD_LEAD_STEPS: u64 = 8_000;
const ON_AUTO_HOLD_BOUNDARIES: u64 = 20_000;
const BASIC_REPL_HUB_PC: u32 = 0x00FFE09;
const BASIC_WARM_START_PC: u32 = 0x00F9C94;
const AUTO_TYPE_START_DELAY_STEPS: u64 = 20_000;
const AUTO_TYPE_GAP_STEPS: u64 = 20_000;
const BASIC_KEY_CODE: u8 = 0x04;
const ENTER_KEY_CODE: u8 = 0x27;
const DELETE_KEY_CODE: u8 = 0x4C;
const BACKSPACE_KEY_CODE: u8 = 0x4D;
const IQ7000_SHIFT_EVENT_CODE: u8 = 0x01;
const IQ7000_FUNCTION_EVENT_CODE: u8 = 0x04;
const IQ7000_CAPS_EVENT_CODE: u8 = 0x09;
const IMR_MASTER: u8 = 0x80;
const IMR_KEY: u8 = 0x04;
const ISR_KEYI: u8 = 0x04;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum CardMode {
    Auto,
    Present,
    Absent,
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
    #[arg(long, value_enum, default_value_t = CardMode::Auto)]
    card: CardMode,

    /// Scheduler-boundary budget before exiting (0 = run until Ctrl+C).
    #[arg(long, default_value_t = 0)]
    steps: u64,

    /// Host execution mode. Interactive uses an uncalibrated nominal timebase.
    /// Deterministic requires finite --steps and a fixed IQ RTC seed; live guest keys are disabled.
    #[arg(long, value_enum, default_value_t = ExecutionMode::Interactive)]
    mode: ExecutionMode,

    /// Maximum normal LCD refresh cadence, independent of CPU speed (1..=60).
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..=60))]
    target_fps: u64,

    /// Scheduler-boundary budget between LCD refresh checks.
    #[arg(long, default_value_t = 20_000)]
    refresh_steps: u64,

    /// Maximum boundary budget between input polls (0 = use refresh budget).
    /// Host execution also yields between small batches after about 4 ms.
    #[arg(long, default_value_t = 1_000)]
    input_steps: u64,

    /// Enable instruction-history/loop diagnostics (off for normal interaction).
    #[arg(long, default_value_t = false)]
    loop_diagnostics: bool,

    /// Show expensive call-stack/keyboard/loop details (off during normal interaction).
    #[arg(long, default_value_t = false)]
    debug_state: bool,

    /// Legacy extra delay after LCD checks. Sleep is split into <=4 ms waits to service controls.
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

    /// IQ-7000 clock seed: host UTC, off, or UTC YYYYMMDDHHMM.
    #[arg(long, value_name = "host|off|YYYYMMDDHHMM", default_value = "host")]
    iq7000_rtc: String,

    /// Synthetically assert KEYI when injecting keys (debug-only override).
    #[arg(long, default_value_t = false)]
    force_key_irq: bool,

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

    /// Enable loop diagnostics and write the report JSON here on exit.
    #[arg(long, value_name = "PATH")]
    loop_report: Option<PathBuf>,
}

fn default_rom_path(model: DeviceModel) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../data/{}", model.rom_basename()))
}

fn lcd_geometry(model: DeviceModel) -> (usize, usize) {
    match model {
        DeviceModel::PcE500 | DeviceModel::PcE500Jp => {
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
        *line = line.chars().take(width).collect();
        if line.chars().count() < width {
            let pad = width.saturating_sub(line.chars().count());
            line.push_str(&" ".repeat(pad));
        }
    }
    lines
}

fn render_frame(
    out: &mut impl Write,
    lines: &[String],
    status: &str,
    extra_lines: &[String],
    use_tty: bool,
) -> Result<(), Box<dyn Error>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_modes_require_explicit_reproducibility_inputs() {
        for model in ["pc-e500", "iq-7000"] {
            let defaults = Args::try_parse_from(["sc62015-lcd", "--model", model]).unwrap();
            assert_eq!(defaults.mode, ExecutionMode::Interactive);
            validate_execution_args(&defaults).unwrap();
            let mut deterministic =
                Args::try_parse_from(["sc62015-lcd", "--model", model, "--mode", "deterministic"])
                    .unwrap();
            assert!(validate_execution_args(&deterministic).is_err());
            deterministic.steps = 1000;
            if model == "iq-7000" {
                assert!(validate_execution_args(&deterministic).is_err());
            }
            deterministic.iq7000_rtc = "202609060000".into();
            validate_execution_args(&deterministic).unwrap();
        }
        assert!(Args::try_parse_from(["sc62015-lcd", "--target-fps", "0"]).is_err());
        assert!(Args::try_parse_from(["sc62015-lcd", "--target-fps", "61"]).is_err());
    }

    #[test]
    fn native_deadlines_are_exact_across_large_small_and_overdue_budgets() {
        let presses = [PendingPress {
            code: 3,
            due_step: 120,
            hold_steps: 10,
            force_key_irq: false,
        }];
        let releases = [PendingRelease {
            code: 4,
            due_step: 130,
        }];
        assert_eq!(
            limit_input_deadline(1000, 100, &presses, &releases, [Some(115)]),
            15
        );
        assert_eq!(
            limit_input_deadline(2, 100, &presses, &releases, [Some(115)]),
            2
        );
        assert_eq!(
            limit_input_deadline(1000, 121, &presses, &releases, [None]),
            0
        );
        assert_eq!(limit_input_deadline(1000, 121, &[], &releases, [None]), 9);
        assert_eq!(limit_input_deadline(1000, 121, &[], &[], [None]), 1000);
    }

    #[test]
    fn native_on_tap_deadline_uses_executed_boundaries_not_frozen_cpu_clock() {
        let mut runtime = CoreRuntime::new();
        runtime.state.power_off();
        let mut deadline = None;
        let feedback = handle_key_event(
            &mut runtime,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            false,
            1000,
            &mut Vec::new(),
            &mut Vec::new(),
            KeyEventOptions {
                pending_on_release: &mut deadline,
                force_key_irq: false,
                model: DeviceModel::PcE500,
            },
        );
        assert_eq!(feedback.label.as_deref(), Some("ON"));
        assert!(runtime.physical_on_key_pressed());
        assert_eq!(runtime.cycle_count(), 0);
        assert_eq!(deadline, Some(1000 + ON_AUTO_HOLD_BOUNDARIES));
    }

    #[test]
    fn loop_diagnostics_are_opt_in_for_both_models() {
        for model in ["pc-e500", "iq-7000"] {
            let normal = Args::try_parse_from(["sc62015-lcd", "--model", model]).unwrap();
            assert!(!normal.loop_diagnostics);
            assert!(!normal.debug_state);
            assert!(normal.loop_report.is_none());
            let diagnostics =
                Args::try_parse_from(["sc62015-lcd", "--model", model, "--loop-diagnostics"])
                    .unwrap();
            assert!(diagnostics.loop_diagnostics);
        }
    }

    #[test]
    fn focus_and_exit_cleanup_release_contacts_and_cancel_future_taps() {
        for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
            let mut runtime = CoreRuntime::for_model(model, &[]).unwrap();
            let mut releases = Vec::new();
            let mut presses = vec![PendingPress {
                code: 2,
                due_step: 120,
                hold_steps: 10,
                force_key_irq: false,
            }];
            let mut on_release = Some(140);
            inject_key(&mut runtime, 3, 100, &mut releases, 30, false);
            runtime.press_on_key();
            assert!(!runtime
                .keyboard
                .as_ref()
                .unwrap()
                .pressed_matrix_codes()
                .is_empty());
            release_host_inputs(&mut runtime, &mut releases, &mut presses, &mut on_release);
            assert!(runtime
                .keyboard
                .as_ref()
                .unwrap()
                .pressed_matrix_codes()
                .is_empty());
            assert!(!runtime.physical_on_key_pressed());
            assert!(releases.is_empty());
            assert!(presses.is_empty());
            assert_eq!(on_release, None);
            assert!(!apply_pending_presses(
                &mut runtime,
                &mut presses,
                &mut releases,
                1000
            ));
        }
    }

    fn iq7000_runtime() -> CoreRuntime {
        let mut runtime = CoreRuntime::new();
        runtime
            .set_device_model(DeviceModel::Iq7000)
            .expect("set IQ-7000 model");
        runtime
    }

    #[test]
    fn iq7000_status_line_reports_shift_caps_annunciators() {
        let mut runtime = iq7000_runtime();
        assert_eq!(format_iq7000_annunciator_status(&runtime), " lcd=none");
        runtime
            .memory
            .store(
                sc62015_core::iq7000_annunciators::IQ7000_ANNUNCIATOR_SHADOW_ADDRS[0],
                8,
                (sc62015_core::iq7000_annunciators::IQ7000_SHIFT
                    | sc62015_core::iq7000_annunciators::IQ7000_CAPS) as u32,
            )
            .expect("store annunciator shadow");
        assert_eq!(
            format_iq7000_annunciator_status(&runtime),
            " lcd=SHIFT,CAPS,DESYNC"
        );

        runtime
            .memory
            .store(
                sc62015_core::iq7000_annunciators::IQ7000_ANNUNCIATOR_STATE_ADDRS[0],
                8,
                0x98,
            )
            .expect("store matching annunciator state");
        runtime
            .memory
            .store(
                sc62015_core::iq7000_annunciators::IQ7000_ANNUNCIATOR_SHADOW_ADDRS[0],
                8,
                0x98,
            )
            .expect("store battery annunciator shadow");
        assert_eq!(
            format_iq7000_annunciator_status(&runtime),
            " lcd=BATT,SHIFT,CAPS"
        );

        runtime
            .memory
            .store(
                sc62015_core::iq7000_annunciators::IQ7000_ANNUNCIATOR_SHADOW_ADDRS[1],
                8,
                0x80,
            )
            .expect("store unknown annunciator shadow");
        assert!(
            format_iq7000_annunciator_status(&runtime).contains("UNMAPPED_SHADOW:[00, 80, 00, 00]")
        );
    }

    #[test]
    fn iq7000_tui_function_keys_inject_shift_caps_events() {
        let mut runtime = iq7000_runtime();
        let mut pending_on_release = None;
        let mut pending_releases = Vec::new();
        let mut pending_presses = Vec::new();

        let feedback = handle_key_event(
            &mut runtime,
            KeyEvent::new(KeyCode::F(6), KeyModifiers::NONE),
            false,
            0,
            &mut pending_releases,
            &mut pending_presses,
            KeyEventOptions {
                pending_on_release: &mut pending_on_release,
                force_key_irq: false,
                model: DeviceModel::Iq7000,
            },
        );
        assert_eq!(feedback.label.as_deref(), Some("SHIFT"));

        let feedback = handle_key_event(
            &mut runtime,
            KeyEvent::new(KeyCode::F(7), KeyModifiers::NONE),
            false,
            0,
            &mut pending_releases,
            &mut pending_presses,
            KeyEventOptions {
                pending_on_release: &mut pending_on_release,
                force_key_irq: false,
                model: DeviceModel::Iq7000,
            },
        );
        assert_eq!(feedback.label.as_deref(), Some("CAPS"));

        let feedback = handle_key_event(
            &mut runtime,
            KeyEvent::new(KeyCode::F(8), KeyModifiers::NONE),
            false,
            0,
            &mut pending_releases,
            &mut pending_presses,
            KeyEventOptions {
                pending_on_release: &mut pending_on_release,
                force_key_irq: false,
                model: DeviceModel::Iq7000,
            },
        );
        assert_eq!(feedback.label.as_deref(), Some("FUNCTION"));

        let fifo = runtime.keyboard.as_ref().expect("keyboard").fifo_snapshot();
        assert_eq!(
            fifo,
            vec![
                IQ7000_SHIFT_EVENT_CODE,
                IQ7000_CAPS_EVENT_CODE,
                IQ7000_FUNCTION_EVENT_CODE,
            ]
        );
    }
}

fn render_status_line(
    out: &mut impl Write,
    status: &str,
    row: u16,
    use_tty: bool,
) -> Result<(), Box<dyn Error>> {
    if !use_tty {
        return Ok(());
    }
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
    pacer: &Pacer,
    control: ControlState,
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
    let iq_status = format_iq7000_annunciator_status(runtime);
    let ui_state = if control.quit {
        "STOPPED"
    } else if control.paused {
        "PAUSED"
    } else {
        "RUNNING"
    };
    format!("{ui_state} ack={} boundaries={executed} [{}, uncalibrated] {pc_display} cpu={power_state}{key_status}{iq_status} dropped_host_ms={} (Ctrl+P pause/resume; Ctrl+C quit)", control.revision, pacer.mode().label(), pacer.dropped_host_ns() / 1_000_000)
}

fn format_iq7000_annunciator_status(runtime: &CoreRuntime) -> String {
    if runtime.device_model() != DeviceModel::Iq7000 {
        return String::new();
    }
    let annunciators = Iq7000Annunciators::read(&runtime.memory);
    let names = [
        ("BATT", annunciators.batt),
        ("CARD", annunciators.card),
        ("EDIT", annunciators.edit),
        ("SHIFT", annunciators.shift),
        ("CAPS", annunciators.caps),
        ("*", annunciators.secret_data),
        ("S", annunciators.secret_mode),
        ("BEEP", annunciators.key_beep),
        ("ALARM", annunciators.alarm),
        ("UP", annunciators.more_up),
        ("DOWN", annunciators.more_down),
        ("LEFT", annunciators.more_left),
        ("RIGHT", annunciators.more_right),
    ];
    let active = names
        .into_iter()
        .filter_map(|(name, enabled)| enabled.then_some(name))
        .collect::<Vec<_>>();
    let mut status = format!(
        " lcd={}",
        if active.is_empty() {
            "none".to_string()
        } else {
            active.join(",")
        }
    );
    if annunciators.unmapped_shadow_bytes != [0; 4] {
        status.push_str(&format!(
            ",UNMAPPED_SHADOW:{:02X?}",
            annunciators.unmapped_shadow_bytes
        ));
    }
    if annunciators.desynchronized {
        status.push_str(",DESYNC");
    }
    status
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
    ));
    lines
}

fn render_extra_lines(
    out: &mut impl Write,
    lines: &[String],
    start_row: u16,
    use_tty: bool,
    prev_lines: &mut usize,
) -> Result<(), Box<dyn Error>> {
    if !use_tty {
        return Ok(());
    }
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

fn decode_display_text(
    text_decoder: &Option<sc62015_core::device::DeviceTextDecoder>,
    runtime: &CoreRuntime,
) -> String {
    match (text_decoder, runtime.lcd.as_deref()) {
        (Some(decoder), Some(lcd)) => decoder.decode_display_text(lcd).join("\n"),
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
        DeviceModel::PcE500 => root.join("rom-analysis/pc-e500/en/bnida.json"),
        DeviceModel::PcE500Jp => root.join("rom-analysis/pc-e500/jp/bnida.json"),
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
        if name.starts_with("delay_") || name == "boot_all_cleared_delay_loop_f7fac" {
            return true;
        }
    }
    if let Some(frame) = runtime.state.call_stack().last().copied() {
        if let Some((_base, name, _)) = resolve_symbol(frame, symbols) {
            return name.starts_with("delay_") || name == "boot_all_cleared_delay_loop_f7fac";
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

struct PendingPress {
    code: u8,
    due_step: u64,
    hold_steps: u64,
    force_key_irq: bool,
}

fn limit_input_deadline(
    requested: u64,
    executed: u64,
    presses: &[PendingPress],
    releases: &[PendingRelease],
    other: impl IntoIterator<Item = Option<u64>>,
) -> u64 {
    presses
        .iter()
        .map(|entry| entry.due_step)
        .chain(releases.iter().map(|entry| entry.due_step))
        .chain(other.into_iter().flatten())
        .fold(requested, |budget, deadline| {
            budget.min(deadline.saturating_sub(executed))
        })
}

struct KeyEventOptions<'a> {
    pending_on_release: &'a mut Option<u64>,
    force_key_irq: bool,
    model: DeviceModel,
}

enum CharKey {
    Single(u8),
    Shifted { modifier: u8, code: u8 },
}

type SymbolMap = BTreeMap<u32, String>;
type FunctionSet = BTreeSet<u32>;

#[derive(serde::Deserialize)]
struct BnidaJson {
    names: Option<HashMap<String, String>>,
    #[serde(default)]
    functions: Option<Vec<u32>>,
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

fn char_key_for_tui(ch: char) -> Option<CharKey> {
    let shifted = |code| CharKey::Shifted {
        modifier: 0x06, // SHIFT
        code,
    };
    match ch {
        '!' => Some(shifted(0x01)),               // Q
        '"' => Some(shifted(0x08)),               // W
        '#' => Some(shifted(0x09)),               // E
        '$' => Some(shifted(0x10)),               // R
        '%' => Some(shifted(0x11)),               // T
        '&' => Some(shifted(0x18)),               // Y
        '\'' => Some(shifted(0x19)),              // U
        '<' => Some(shifted(0x20)),               // I
        '>' => Some(shifted(0x21)),               // O
        '@' => Some(shifted(0x50)),               // P
        '[' => Some(shifted(0x03)),               // A
        ']' => Some(shifted(0x0A)),               // S
        '{' => Some(shifted(0x0B)),               // D
        '}' => Some(shifted(0x12)),               // F
        '\\' | '\u{00A5}' => Some(shifted(0x13)), // G: backslash on EN, yen on JP
        '|' => Some(shifted(0x1A)),               // H
        '~' => Some(shifted(0x1B)),               // J
        '_' => Some(shifted(0x22)),               // K
        '^' => Some(shifted(0x23)),               // L
        '?' => Some(shifted(0x24)),               // ,
        ':' => Some(shifted(0x25)),               // ;
        _ => matrix_code_for_char(ch).map(CharKey::Single),
    }
}

fn auto_type_gap_for_char(ch: char) -> u64 {
    match char_key_for_tui(ch) {
        Some(CharKey::Shifted { .. }) => SHIFTED_CHORD_LEAD_STEPS
            .saturating_add(CHAR_KEY_HOLD_STEPS)
            .saturating_add(AUTO_TYPE_GAP_STEPS),
        _ => AUTO_TYPE_GAP_STEPS,
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
        kb.press_matrix_code(code, &mut runtime.memory);
        if force_key_irq {
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
            kb.release_matrix_code(code, &mut runtime.memory);
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

fn inject_input_event(runtime: &mut CoreRuntime, code: u8, force_key_irq: bool) -> bool {
    if force_key_irq && !runtime.timer.kb_irq_enabled {
        runtime.timer.kb_irq_enabled = true;
    }
    let Some(kb) = runtime.keyboard.as_mut() else {
        return false;
    };
    let kb_irq_enabled = runtime.timer.kb_irq_enabled || force_key_irq;
    let events = kb.inject_input_event(code, &mut runtime.memory, kb_irq_enabled);
    if events == 0 {
        return false;
    }
    if force_key_irq {
        runtime.timer.key_irq_latched = true;
        runtime.timer.irq_pending = true;
        if runtime.timer.irq_source.is_none() && !runtime.timer.in_interrupt {
            runtime.timer.irq_source = Some("KEY".to_string());
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
    true
}

fn inject_char_key(
    runtime: &mut CoreRuntime,
    ch: char,
    executed: u64,
    pending_releases: &mut Vec<PendingRelease>,
    pending_presses: &mut Vec<PendingPress>,
    hold_steps: u64,
    force_key_irq: bool,
) -> bool {
    match char_key_for_tui(ch) {
        Some(CharKey::Single(code)) => {
            inject_key(
                runtime,
                code,
                executed,
                pending_releases,
                hold_steps,
                force_key_irq,
            );
            true
        }
        Some(CharKey::Shifted { modifier, code }) => {
            let modifier_hold = hold_steps
                .saturating_add(SHIFTED_CHORD_LEAD_STEPS)
                .saturating_add(1_000);
            inject_key(
                runtime,
                modifier,
                executed,
                pending_releases,
                modifier_hold,
                force_key_irq,
            );
            pending_presses.push(PendingPress {
                code,
                due_step: executed.saturating_add(SHIFTED_CHORD_LEAD_STEPS),
                hold_steps,
                force_key_irq,
            });
            true
        }
        None => false,
    }
}

fn apply_pending_presses(
    runtime: &mut CoreRuntime,
    pending_presses: &mut Vec<PendingPress>,
    pending_releases: &mut Vec<PendingRelease>,
    executed: u64,
) -> bool {
    let mut pressed = false;
    let mut idx = 0;
    while idx < pending_presses.len() {
        if pending_presses[idx].due_step <= executed {
            let pending = pending_presses.swap_remove(idx);
            inject_key(
                runtime,
                pending.code,
                executed,
                pending_releases,
                pending.hold_steps,
                pending.force_key_irq,
            );
            pressed = true;
        } else {
            idx += 1;
        }
    }
    pressed
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
    let mut idx = 0;
    while idx < pending_releases.len() {
        if pending_releases[idx].due_step <= executed {
            let code = pending_releases[idx].code;
            kb.release_matrix_code(code, &mut runtime.memory);
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
    pending_presses: &mut Vec<PendingPress>,
    options: KeyEventOptions<'_>,
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
                *options.pending_on_release =
                    Some(executed.saturating_add(ON_AUTO_HOLD_BOUNDARIES));
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
                    options.force_key_irq,
                );
                return KeyFeedback {
                    label: Some(format!("PF{}", ch)),
                    quit: false,
                };
            }
        }
    }
    match key.code {
        KeyCode::CapsLock if options.model == DeviceModel::Iq7000 => {
            if inject_input_event(runtime, IQ7000_CAPS_EVENT_CODE, options.force_key_irq) {
                return KeyFeedback {
                    label: Some("CAPS".to_string()),
                    quit: false,
                };
            }
        }
        KeyCode::Enter => {
            inject_key(
                runtime,
                ENTER_KEY_CODE,
                executed,
                pending_releases,
                CHAR_KEY_HOLD_STEPS,
                options.force_key_irq,
            );
            return KeyFeedback {
                label: Some("ENTER".to_string()),
                quit: false,
            };
        }
        KeyCode::Backspace => {
            inject_key(
                runtime,
                BACKSPACE_KEY_CODE,
                executed,
                pending_releases,
                CHAR_KEY_HOLD_STEPS,
                options.force_key_irq,
            );
            return KeyFeedback {
                label: Some("BS".to_string()),
                quit: false,
            };
        }
        KeyCode::Delete => {
            inject_key(
                runtime,
                DELETE_KEY_CODE,
                executed,
                pending_releases,
                CHAR_KEY_HOLD_STEPS,
                options.force_key_irq,
            );
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
                options.force_key_irq,
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
                options.force_key_irq,
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
                options.force_key_irq,
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
                options.force_key_irq,
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
                options.force_key_irq,
            );
            return KeyFeedback {
                label: Some("PF5".to_string()),
                quit: false,
            };
        }
        KeyCode::F(6) if options.model == DeviceModel::Iq7000 => {
            if inject_input_event(runtime, IQ7000_SHIFT_EVENT_CODE, options.force_key_irq) {
                return KeyFeedback {
                    label: Some("SHIFT".to_string()),
                    quit: false,
                };
            }
        }
        KeyCode::F(7) if options.model == DeviceModel::Iq7000 => {
            if inject_input_event(runtime, IQ7000_CAPS_EVENT_CODE, options.force_key_irq) {
                return KeyFeedback {
                    label: Some("CAPS".to_string()),
                    quit: false,
                };
            }
        }
        KeyCode::F(8) if options.model == DeviceModel::Iq7000 => {
            if inject_input_event(runtime, IQ7000_FUNCTION_EVENT_CODE, options.force_key_irq) {
                return KeyFeedback {
                    label: Some("FUNCTION".to_string()),
                    quit: false,
                };
            }
        }
        KeyCode::Char(ch) => {
            if ch == '\u{8}' || ch == '\u{7f}' {
                inject_key(
                    runtime,
                    BACKSPACE_KEY_CODE,
                    executed,
                    pending_releases,
                    CHAR_KEY_HOLD_STEPS,
                    options.force_key_irq,
                );
                return KeyFeedback {
                    label: Some("BS".to_string()),
                    quit: false,
                };
            }
            if pf_numbers {
                if let Some(code) = matrix_code_for_ctrl_digit(ch) {
                    inject_key(
                        runtime,
                        code,
                        executed,
                        pending_releases,
                        PF_KEY_HOLD_STEPS,
                        options.force_key_irq,
                    );
                    return KeyFeedback {
                        label: Some(format!("PF{}", ch)),
                        quit: false,
                    };
                }
            }
            if inject_char_key(
                runtime,
                ch,
                executed,
                pending_releases,
                pending_presses,
                CHAR_KEY_HOLD_STEPS,
                options.force_key_irq,
            ) {
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

fn iq7000_host_rtc_seed() -> String {
    let now = Utc::now();
    format!(
        "{:04}{:02}{:02}{:02}{:02}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute()
    )
}

fn apply_iq7000_rtc_arg(
    runtime: &mut CoreRuntime,
    model: DeviceModel,
    raw: &str,
) -> Result<(), Box<dyn Error>> {
    if model != DeviceModel::Iq7000 {
        return Ok(());
    }
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("off") || trimmed.eq_ignore_ascii_case("none") {
        runtime.clear_iq7000_clock_seed();
        return Ok(());
    }
    let seed = if trimmed.eq_ignore_ascii_case("host") || trimmed.is_empty() {
        iq7000_host_rtc_seed()
    } else {
        trimmed.to_string()
    };
    runtime
        .set_iq7000_clock_seed_yyyymmddhhmm(&seed)
        .map_err(|err| format!("--iq7000-rtc: {err}"))?;
    Ok(())
}

fn validate_execution_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.refresh_steps == 0 {
        return Err("refresh_steps must be > 0".into());
    }
    if Instant::now()
        .checked_add(Duration::from_millis(args.sleep_ms))
        .is_none()
    {
        return Err("sleep_ms exceeds the host clock range".into());
    }
    if args.mode == ExecutionMode::Deterministic {
        if args.steps == 0 {
            return Err("deterministic mode requires a finite --steps budget".into());
        }
        if args.model == DeviceModel::Iq7000
            && (args.iq7000_rtc.trim().is_empty()
                || args.iq7000_rtc.trim().eq_ignore_ascii_case("host"))
        {
            return Err("deterministic IQ-7000 execution requires --iq7000-rtc YYYYMMDDHHMM (or off), not a host-time seed".into());
        }
    }
    Ok(())
}

struct UiView<'a> {
    decoder: &'a Option<sc62015_core::DeviceTextDecoder>,
    symbols: Option<&'a SymbolMap>,
    functions: Option<&'a FunctionSet>,
    debug: bool,
}

struct UiProgress<'a> {
    executed: u64,
    last_key: &'a Option<String>,
    last_key_step: u64,
    releases: &'a [PendingRelease],
    halted_steps: u64,
    control: ControlState,
}

impl UiView<'_> {
    fn frame(&self, runtime: &CoreRuntime, pacer: &Pacer, progress: UiProgress<'_>) -> host::Frame {
        host::Frame {
            lcd: self
                .decoder
                .as_ref()
                .zip(runtime.lcd.as_deref())
                .map(|(decoder, lcd)| decoder.capture_text_frame(lcd)),
            status: format_status(
                runtime,
                progress.executed,
                progress.last_key,
                progress.last_key_step,
                self.symbols,
                pacer,
                progress.control,
            ),
            extra: if self.debug {
                format_extra_lines(
                    runtime,
                    self.symbols,
                    self.functions,
                    progress.last_key,
                    progress.last_key_step,
                    progress.releases,
                    progress.halted_steps,
                )
            } else {
                Vec::new()
            },
            final_log: None,
        }
    }
}

fn release_host_inputs(
    runtime: &mut CoreRuntime,
    releases: &mut Vec<PendingRelease>,
    presses: &mut Vec<PendingPress>,
    on_release: &mut Option<u64>,
) {
    releases.clear();
    presses.clear();
    *on_release = None;
    if let Some(keyboard) = &mut runtime.keyboard {
        for code in keyboard.pressed_matrix_codes() {
            keyboard.release_matrix_code(code, &mut runtime.memory);
        }
    }
    runtime.release_on_key(); // Physical contact only; RTC wake remains independent.
}

fn main() -> std::process::ExitCode {
    match run_native() {
        Ok(code) => code,
        Err(error) => {
            host::report_error_bounded(format!("[execution] {error}"));
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_native() -> Result<std::process::ExitCode, Box<dyn Error>> {
    let args = Args::parse();
    validate_execution_args(&args)?;
    let mut pacer = Pacer::for_model(args.model, args.mode);
    let mut warnings = vec![format!("[execution] {}: {} compatibility timing units/s, not hardware calibrated; IQ uses the PC fallback. RTC follows guest elapsed time, not paused host time.", args.mode.label(), pacer.timebase_hz())];
    if args.mode == ExecutionMode::Deterministic {
        warnings.push("[execution] explicit --steps budget; live guest keys disabled (Ctrl+P pause/resume, Ctrl+C quit)".into());
    }
    let rom_path = args.rom.unwrap_or_else(|| default_rom_path(args.model));
    let rom_bytes = fs::read(&rom_path)?;

    let mut runtime = CoreRuntime::for_model(args.model, &rom_bytes)?;
    apply_iq7000_rtc_arg(&mut runtime, args.model, &args.iq7000_rtc)?;
    if args.disable_timers {
        runtime.timer.enabled = false;
    }
    args.card.resolve(args.model).apply(&mut runtime.memory)?;
    if args.loop_diagnostics || args.loop_report.is_some() {
        let loop_config = LoopDetectorConfig {
            detect_stride: args.refresh_steps,
            ..Default::default()
        };
        runtime.enable_loop_detector(loop_config);
    }
    runtime.power_on_reset()?;
    apply_iq7000_rtc_arg(&mut runtime, args.model, &args.iq7000_rtc)?;

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
    let mut first_draw = true;
    let mut last_key: Option<String> = None;
    let mut last_key_step: u64 = 0;
    let mut pending_releases: Vec<PendingRelease> = Vec::new();
    let mut pending_presses: Vec<PendingPress> = Vec::new();
    let mut pending_on_release: Option<u64> = None;
    let mut auto_type_queue: Vec<char> =
        args.auto_type.clone().unwrap_or_default().chars().collect();
    let mut auto_type_next_step: Option<u64> = None;
    let mut auto_basic_step: Option<u64> = None;
    let mut auto_basic_pending = args.auto_basic;
    let mut jump_basic_pending = args.jump_basic;
    let mut halted_steps: u64 = 0;
    let mut last_lcd_check: u64 = 0;
    let mut fast_init_cleared = false;
    let fast_init_buf = if args.fast_init && args.model.is_pce500_family() {
        Some(vec![0u8; ROM_WINDOW_START])
    } else {
        None
    };
    let lcd_check_interval: u64 = 5_000;
    ensure_term();
    let use_tty = args.force_tty || (stdout().is_terminal() && std::io::stdin().is_terminal());
    let use_alt = use_tty && !args.no_alt_screen;

    let mut host = host::NativeHost::new(
        text_decoder.clone(),
        (line_count, width),
        use_tty,
        use_alt,
        warnings,
    )?;
    let view = UiView {
        decoder: &text_decoder,
        symbols: symbols.as_ref(),
        functions: function_addrs.as_ref(),
        debug: args.debug_state,
    };
    let mut observed = host.controls.state();
    let mut last_status_draw = Instant::now();
    host.publish(view.frame(
        &runtime,
        &pacer,
        UiProgress {
            executed: 0,
            last_key: &last_key,
            last_key_step,
            releases: &pending_releases,
            halted_steps,
            control: observed,
        },
    ));

    let mut executed: u64 = 0;
    let mut running = true;
    let host_epoch = Instant::now();
    let host_target = Duration::from_micros(HOST_SLICE_TARGET_US);
    let display_interval = Duration::from_secs_f64(1.0 / args.target_fps as f64);
    let mut last_display_check = Instant::now();
    let mut legacy_delay_until: Option<Instant> = None;
    let execution: Result<(), Box<dyn Error>> = (|| {
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
                let requested = host.controls.state();
                if requested != observed {
                    if requested.release_epoch != observed.release_epoch {
                        release_host_inputs(
                            &mut runtime,
                            &mut pending_releases,
                            &mut pending_presses,
                            &mut pending_on_release,
                        );
                    }
                    if requested.paused != observed.paused {
                        pacer.rebase();
                    }
                    observed = requested;
                    dirty = true;
                }
                if observed.quit {
                    running = false;
                    break;
                }
                if let Some(error) = host.stats().error {
                    return Err(format!("terminal output failed: {error}").into());
                }
                let previously_executed = executed;
                let now = Instant::now();
                let delayed = legacy_delay_until.is_some_and(|deadline| now < deadline);
                if !delayed && legacy_delay_until.take().is_some() {
                    pacer.rebase();
                }
                let mut host_wait = if delayed || observed.paused {
                    host_target
                } else {
                    Duration::ZERO
                };
                let mut did_stub = false;
                if !delayed
                    && !observed.paused
                    && apply_stub_returns(
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
                    )
                {
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
                let chunk = limit_input_deadline(
                    chunk,
                    executed,
                    &pending_presses,
                    &pending_releases,
                    [
                        pending_on_release,
                        auto_basic_step,
                        auto_type_next_step,
                        ((auto_basic_pending
                            || jump_basic_pending
                            || (auto_type_next_step.is_none() && !auto_type_queue.is_empty()))
                            && auto_basic_step.is_none())
                        .then_some(last_lcd_check.saturating_add(lcd_check_interval)),
                    ],
                );
                if !did_stub && !delayed && !observed.paused {
                    let slice_start = Instant::now();
                    let limit = usize::try_from(chunk).unwrap_or(usize::MAX);
                    let used = if args.mode == ExecutionMode::Deterministic {
                        runtime
                            .run_slice(limit, |_| {
                                slice_start.elapsed() >= host_target
                                    || host.controls.changed(observed)
                            })?
                            .progress
                            .boundary_budget_used
                    } else {
                        let result = runtime.run_automatic_slice(
                            &mut pacer,
                            u64::try_from(host_epoch.elapsed().as_nanos()).unwrap_or(u64::MAX),
                            limit,
                            |_| {
                                slice_start.elapsed() >= host_target
                                    || host.controls.changed(observed)
                            },
                        )?;
                        host_wait = Duration::from_nanos(result.plan.wait_ns);
                        result
                            .slice
                            .map_or(0, |slice| slice.progress.boundary_budget_used)
                    } as u64;
                    executed = executed.saturating_add(used);
                    remaining = remaining.saturating_sub(used);
                }
                if !observed.paused {
                    if apply_pending_presses(
                        &mut runtime,
                        &mut pending_presses,
                        &mut pending_releases,
                        executed,
                    ) {
                        dirty = true;
                    }
                    apply_pending_releases(&mut runtime, &mut pending_releases, executed);
                    if let Some(release_boundary) = pending_on_release {
                        if executed >= release_boundary {
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
                                auto_type_next_step =
                                    Some(executed.saturating_add(args.auto_type_delay));
                            }
                            dirty = true;
                        }
                    }
                    if let Some(next_step) = auto_type_next_step {
                        if executed >= next_step {
                            let mut sent = false;
                            while let Some(ch) = auto_type_queue.first().copied() {
                                auto_type_queue.remove(0);
                                let did_inject = match ch {
                                    '\n' | '\r' => {
                                        inject_key(
                                            &mut runtime,
                                            ENTER_KEY_CODE,
                                            executed,
                                            &mut pending_releases,
                                            CHAR_KEY_HOLD_STEPS,
                                            args.force_key_irq,
                                        );
                                        true
                                    }
                                    _ => inject_char_key(
                                        &mut runtime,
                                        ch,
                                        executed,
                                        &mut pending_releases,
                                        &mut pending_presses,
                                        CHAR_KEY_HOLD_STEPS,
                                        args.force_key_irq,
                                    ),
                                };
                                if did_inject {
                                    last_key = Some(ch.to_string());
                                    last_key_step = executed;
                                    auto_type_next_step =
                                        Some(executed.saturating_add(auto_type_gap_for_char(ch)));
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
                        halted_steps = halted_steps
                            .saturating_add(executed.saturating_sub(previously_executed));
                    } else {
                        halted_steps = 0;
                    }
                    let should_check_lcd = (auto_basic_pending
                        || jump_basic_pending
                        || (auto_type_next_step.is_none() && !auto_type_queue.is_empty()))
                        && auto_basic_step.is_none();
                    if should_check_lcd
                        && executed.saturating_sub(last_lcd_check) >= lcd_check_interval
                    {
                        last_lcd_check = executed;
                        let row0 = decode_row0(&text_decoder, &runtime);
                        let display_text = decode_display_text(&text_decoder, &runtime);
                        if auto_type_next_step.is_none()
                            && !auto_type_queue.is_empty()
                            && display_text.contains('>')
                        {
                            auto_type_next_step =
                                Some(executed.saturating_add(args.auto_type_delay));
                            dirty = true;
                            continue;
                        }
                        let row_has_menu = row0.contains("S2(CARD):") || row0.contains("S1(MAIN):");
                        if jump_basic_pending && auto_basic_step.is_none() && row_has_menu {
                            jump_to_basic_loop(&mut runtime);
                            jump_basic_pending = false;
                            if auto_type_next_step.is_none() && !auto_type_queue.is_empty() {
                                auto_type_next_step =
                                    Some(executed.saturating_add(args.auto_basic_delay));
                            }
                            dirty = true;
                        } else if auto_basic_pending && auto_basic_step.is_none() && row_has_menu {
                            auto_basic_step = Some(executed.saturating_add(args.auto_basic_delay));
                            auto_basic_pending = false;
                        }
                    }
                } // Freeze budget-scheduled taps and automation while paused.

                if use_tty {
                    let (epoch, keys) = host.controls.take_batch();
                    if epoch != observed.release_epoch {
                        release_host_inputs(
                            &mut runtime,
                            &mut pending_releases,
                            &mut pending_presses,
                            &mut pending_on_release,
                        );
                        observed.release_epoch = epoch;
                        dirty = true;
                    }
                    for key in keys {
                        let current = host.controls.state();
                        if current.release_epoch != epoch || current.quit {
                            break;
                        }
                        if args.mode == ExecutionMode::Deterministic {
                            continue;
                        }
                        let feedback = handle_key_event(
                            &mut runtime,
                            key,
                            args.pf_numbers,
                            executed,
                            &mut pending_releases,
                            &mut pending_presses,
                            KeyEventOptions {
                                pending_on_release: &mut pending_on_release,
                                force_key_irq: args.force_key_irq,
                                model: args.model,
                            },
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
                if use_tty && !first_draw && last_status_draw.elapsed() >= STATUS_UPDATE_INTERVAL {
                    dirty = true;
                }
                if !running {
                    break;
                }
                if dirty {
                    break;
                }
                if last_display_check.elapsed() >= display_interval {
                    break;
                }
                if args.steps > 0 && executed >= args.steps {
                    break;
                }
                if !host_wait.is_zero() {
                    // Rounding tiny waits up avoids a sub-millisecond busy wake
                    // loop. Any resulting host credit is accounted by Rust next time.
                    sleep(host_wait.max(Duration::from_millis(1)).min(host_target));
                }
            }

            if !first_draw
                && !dirty
                && running
                && (args.steps == 0 || executed < args.steps)
                && last_display_check.elapsed() < display_interval
            {
                continue;
            }
            last_display_check = Instant::now();

            let mut frame = view.frame(
                &runtime,
                &pacer,
                UiProgress {
                    executed,
                    last_key: &last_key,
                    last_key_step,
                    releases: &pending_releases,
                    halted_steps,
                    control: observed,
                },
            );
            if host.controls.dropped_inputs() > 0 {
                frame.status.push_str(&format!(
                    " INPUT OVERFLOW TOTAL: {} discarded; old contacts cleared",
                    host.controls.dropped_inputs()
                ));
            }
            if !host.publish(frame) {
                return Err("terminal output worker stopped".into());
            }
            first_draw = false;
            last_status_draw = Instant::now();

            if args.sleep_ms > 0 && legacy_delay_until.is_none() {
                legacy_delay_until =
                    Instant::now().checked_add(Duration::from_millis(args.sleep_ms));
            }
        }
        Ok(())
    })();

    let fault = execution
        .err()
        .map(|error| error.to_string())
        .or_else(|| host.controls.error());
    release_host_inputs(
        &mut runtime,
        &mut pending_releases,
        &mut pending_presses,
        &mut pending_on_release,
    );
    observed.quit = true;
    let mut final_frame = view.frame(
        &runtime,
        &pacer,
        UiProgress {
            executed,
            last_key: &last_key,
            last_key_step,
            releases: &pending_releases,
            halted_steps,
            control: observed,
        },
    );
    let mut log = format!("[execution] finished boundaries={executed} retired={} cpu_timing={} elapsed_timing={} dropped_host_ns={}", runtime.instruction_count(), runtime.cycle_count(), runtime.elapsed_timing_units(), pacer.dropped_host_ns());
    if let Some(error) = &fault {
        log.push_str(&format!("\n[execution] FAULT: {error}"));
        final_frame.status = format!("FAULT: {error}; {}", final_frame.status);
    }
    final_frame.final_log = Some(log);
    host.publish(final_frame);
    if !host.finish() {
        host::report_error_bounded("[execution] terminal output could not finish within the shutdown deadline, or terminal restoration failed; final display may be incomplete".into());
        return Ok(std::process::ExitCode::from(2));
    }
    if let Some(error) = fault {
        return Err(error.into());
    }
    if let Some(detector) = runtime.loop_detector() {
        if let Some(report) = detector.last_report() {
            let path = args
                .loop_report
                .clone()
                .unwrap_or_else(default_loop_report_path);
            let json = serde_json::to_string_pretty(report)?;
            fs::write(&path, json)?;
            host::report_error_bounded(format!("[loop] report saved to {}", path.display()));
        }
    }

    Ok(std::process::ExitCode::SUCCESS)
}
