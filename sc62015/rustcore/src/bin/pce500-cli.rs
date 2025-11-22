use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser};
use once_cell::sync::Lazy;
use sc62015_core::{
    apply_registers, collect_registers, execute_step, load_snapshot as core_load_snapshot,
    now_timestamp, register_width, save_snapshot as core_save_snapshot, BoundInstrBuilder,
    BoundInstrView, BusProfiler, ExecManifestEntry, HostMemory, LayoutEntryView, ManifestEntryView,
    MemoryImage, SnapshotLoad, SnapshotMetadata, TimerContext, ADDRESS_MASK, LCD_DISPLAY_COLS,
    LCD_DISPLAY_ROWS, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START,
    INTERNAL_SPACE, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use std::time::{Duration, Instant};

mod generated {
    pub mod types {
        include!("../../generated/types.rs");
    }
    pub mod payload {
        include!("../../generated/handlers.rs");
    }
    pub mod opcode_index {
        include!("../../generated/opcode_index.rs");
    }
}

use generated::opcode_index::OPCODE_INDEX;
use generated::types::{BoundInstrRepr, ManifestEntry, PreInfo};

static MANIFEST: Lazy<Vec<ManifestEntry>> =
    Lazy::new(|| serde_json::from_str(generated::payload::PAYLOAD).expect("manifest json"));
static OPCODE_LOOKUP: Lazy<sc62015_core::OpcodeLookup<'static, ManifestEntry>> = Lazy::new(|| {
    sc62015_core::OpcodeLookup::new(&*MANIFEST, OPCODE_INDEX)
});

#[derive(Parser, Debug)]
#[command(name = "pce500-cli")]
#[command(about = "Experimental pure-Rust CLI for SC62015/PC-E500", long_about = None)]
struct Args {
    /// Path to PC-E500 ROM image
    #[arg(long)]
    rom: Option<PathBuf>,

    /// Load snapshot before execution
    #[arg(long)]
    load_snapshot: Option<PathBuf>,

    /// Save snapshot after execution
    #[arg(long)]
    save_snapshot: Option<PathBuf>,

    /// Number of instructions to execute
    #[arg(long, default_value_t = 15_000)]
    steps: u64,

    /// Enable fast mode (keeps timers enabled; currently informational)
    #[arg(long, action = ArgAction::SetTrue)]
    fast: bool,

    /// Execution timeout in seconds (not yet enforced)
    #[arg(long, default_value_t = 0)]
    timeout_secs: u64,

    /// Emit LCD text (not yet implemented; enables LCD device)
    #[arg(long, action = ArgAction::SetTrue)]
    lcd_text: bool,

    /// Emit Perfetto trace (placeholder)
    #[arg(long, action = ArgAction::SetTrue)]
    perfetto: bool,

    /// Path to write Perfetto traceEvents JSON (convert via scripts/convert_trace_json_to_perfetto.py)
    #[arg(long, default_value = "pc-e500-trace.json")]
    trace_file: PathBuf,

    /// Enable auto-key (placeholder)
    #[arg(long, action = ArgAction::SetTrue)]
    auto_key: bool,

    /// Preset: rust-fast sets fast mode and zero timeout
    #[arg(long, default_value = "")]
    preset: String,

    /// Emit a per-opcode timing table after the run
    #[arg(long, action = ArgAction::SetTrue)]
    profile_opcodes: bool,
}

struct LocalHost;

impl HostMemory for LocalHost {
    fn load(&mut self, _space: rust_scil::bus::Space, _addr: u32, _bits: u8) -> u32 {
        0
    }

    fn store(
        &mut self,
        _space: rust_scil::bus::Space,
        _addr: u32,
        _bits: u8,
        _value: u32,
    ) {
    }

    fn read_byte(&mut self, _address: u32) -> u8 {
        0
    }

    fn notify_lcd_write(&mut self, _address: u32, _value: u32) {}
}

#[derive(Default)]
struct StatsProfiler {
    bus_load: u64,
    bus_store: u64,
    python_load: u64,
    python_store: u64,
}

impl BusProfiler for StatsProfiler {
    fn record_bus_load(&mut self) {
        self.bus_load += 1;
    }
    fn record_bus_store(&mut self) {
        self.bus_store += 1;
    }
    fn record_python_load(&mut self, _address: u32) {
        self.python_load += 1;
    }
    fn record_python_store(&mut self, _address: u32) {
        self.python_store += 1;
    }
}

// LCD text decoding constants derived from the Python display helpers.
const FONT_BASE: u32 = 0x00F2215;
const GLYPH_WIDTH: usize = 5;
const GLYPH_HEIGHT: usize = 7;
const GLYPH_STRIDE: usize = 6; // five glyph bytes + spacer
const GLYPH_COUNT: usize = 96; // ASCII 0x20-0x7F
const CHAR_COLUMNS: usize = 40;
const CHAR_ROWS: usize = 4;
const PIXELS_PER_CHAR_COL: usize = 6;

impl ManifestEntryView for ManifestEntry {
    type Layout = generated::types::LayoutEntry;

    fn opcode(&self) -> u8 {
        self.opcode as u8
    }

    fn pre(&self) -> Option<(String, String)> {
        self.pre
            .as_ref()
            .map(|info| (info.first.clone(), info.second.clone()))
    }

    fn binder(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.binder
    }

    fn instr(&self) -> &serde_json::Value {
        &self.instr
    }

    fn layout(&self) -> &[Self::Layout] {
        &self.layout
    }
}

impl ExecManifestEntry for ManifestEntry {
    fn mnemonic(&self) -> &str {
        &self.mnemonic
    }

    fn family(&self) -> Option<&str> {
        self.family.as_deref()
    }
}

impl LayoutEntryView for generated::types::LayoutEntry {
    fn key(&self) -> &str {
        &self.key
    }

    fn kind(&self) -> &str {
        &self.kind
    }

    fn meta(&self) -> &std::collections::HashMap<String, serde_json::Value> {
        &self.meta
    }
}

impl BoundInstrView for BoundInstrRepr {
    fn opcode(&self) -> u32 {
        self.opcode
    }

    fn operands(&self) -> &std::collections::HashMap<String, serde_json::Value> {
        &self.operands
    }

    fn pre(&self) -> Option<(String, String)> {
        self.pre
            .as_ref()
            .map(|pre| (pre.first.clone(), pre.second.clone()))
    }
}

impl BoundInstrBuilder for BoundInstrRepr {
    fn from_parts(
        opcode: u32,
        mnemonic: &str,
        family: Option<&str>,
        length: u8,
        pre: Option<(String, String)>,
        operands: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            opcode,
            mnemonic: mnemonic.to_string(),
            family: family.map(|f| f.to_string()),
            length,
            pre: pre.map(|(first, second)| PreInfo { first, second }),
            operands,
        }
    }
}

impl sc62015_core::OpcodeIndexView for generated::opcode_index::OpcodeIndexEntry {
    fn opcode(&self) -> u8 {
        self.opcode
    }

    fn pre(&self) -> Option<(String, String)> {
        self.pre
            .map(|pre| (pre.first.to_string(), pre.second.to_string()))
    }

    fn manifest_index(&self) -> usize {
        self.manifest_index
    }
}

fn font_glyph_columns(memory: &MemoryImage, code: u8) -> Option<[u8; GLYPH_WIDTH]> {
    if code < 0x20 || code >= 0x20 + GLYPH_COUNT as u8 {
        return None;
    }
    let idx = (code - 0x20) as u32;
    let base = FONT_BASE + idx * GLYPH_STRIDE as u32;
    let mut cols = [0u8; GLYPH_WIDTH];
    for offset in 0..GLYPH_WIDTH {
        let byte = memory.read_byte(base + offset as u32)?;
        cols[offset] = byte & 0x7F;
    }
    Some(cols)
}

fn build_glyph_reverse_lookup(
    memory: &MemoryImage,
) -> Result<HashMap<[u8; GLYPH_WIDTH], char>> {
    let mut lookup = HashMap::new();
    for code in 0x20u8..(0x20 + GLYPH_COUNT as u8) {
        let glyph = font_glyph_columns(memory, code).ok_or_else(|| {
            anyhow::anyhow!(format!(
                "missing glyph data for code 0x{code:02X} at base 0x{base:06X}",
                base = FONT_BASE
            ))
        })?;
        lookup.insert(glyph, code as char);
        let mut inverted = glyph;
        for byte in inverted.iter_mut() {
            *byte = !*byte & 0x7F;
        }
        lookup.entry(inverted).or_insert(code as char);
    }
    Ok(lookup)
}

fn decode_display_text(
    lcd: &sc62015_core::LcdController,
    memory: &MemoryImage,
) -> Result<Vec<String>> {
    let reverse_lookup = build_glyph_reverse_lookup(memory)?;
    let buffer = lcd.display_buffer();
    let height = LCD_DISPLAY_ROWS.min(buffer.len());
    let width = buffer.first().map(|row| row.len()).unwrap_or(0);
    let width = LCD_DISPLAY_COLS.min(width);
    let mut lines = Vec::new();
    // Match Python layout: stitch L+R chips into 240 columns, then decode 4 rows.
    for page in 0..CHAR_ROWS {
        let row_base = page * GLYPH_HEIGHT;
        if row_base + GLYPH_HEIGHT > height {
            break;
        }
        let mut row_chars = String::new();
        for char_idx in 0..CHAR_COLUMNS {
            let col_base = char_idx * PIXELS_PER_CHAR_COL;
            if col_base + GLYPH_WIDTH > width {
                row_chars.push(' ');
                continue;
            }
            let mut columns = [0u8; GLYPH_WIDTH];
            for glyph_col in 0..GLYPH_WIDTH {
                let col = col_base + glyph_col;
                let mut bits = 0u8;
                for row in 0..GLYPH_HEIGHT {
                    // Mirror Python decoder: buffer holds 1 for lit pixels, so set bit on zero.
                    if buffer[row_base + row][col] == 0 {
                        bits |= 1 << row;
                    }
                }
                columns[glyph_col] = bits & 0x7F;
            }
            let ch = reverse_lookup.get(&columns).copied().unwrap_or('?');
            row_chars.push(ch);
        }
        lines.push(row_chars.trim_end().to_string());
    }
    Ok(lines)
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if args.preset.eq_ignore_ascii_case("rust-fast") {
        args.fast = true;
        args.timeout_secs = 0;
    }
    if args.auto_key {
        eprintln!("note: --auto-key is not yet implemented in the Rust CLI");
    }
    let mut memory = MemoryImage::new();
    memory.set_python_ranges(Vec::new());
    memory.set_keyboard_bridge(true);

    let mut rom_loaded = false;
    let rom_path = args.rom.clone().or_else(|| {
        let default = PathBuf::from("data/pc-e500.bin");
        if default.exists() {
            Some(default)
        } else {
            None
        }
    });
    if let Some(rom_path) = rom_path.as_ref() {
        let bytes = fs::read(rom_path)
            .with_context(|| format!("failed to read ROM {}", rom_path.display()))?;
        memory.load_external(&bytes);
        memory.set_readonly_ranges(vec![(0, bytes.len().saturating_sub(1) as u32)]);
        rom_loaded = true;
        println!("Loaded ROM from {}", rom_path.display());
    } else {
        println!("No ROM provided; running without ROM overlay");
    }

    let mut state = rust_scil::state::State::default();
    let mut timer = TimerContext::new(true, 500, 5000);
    let mut lcd = args.lcd_text.then(sc62015_core::LcdController::new);
    let mut keyboard = Some(sc62015_core::KeyboardMatrix::new());
    let mut instruction_count: u64 = 0;
    let mut cycle_count: u64 = 0;
    let mut tracer = if args.perfetto {
        Some(sc62015_core::PerfettoTracer::new(args.trace_file.clone()))
    } else {
        None
    };
    let timeout = if args.timeout_secs > 0 {
        Some(Duration::from_secs(args.timeout_secs))
    } else {
        None
    };
    let start_time = Instant::now();

    if let Some(path) = args.load_snapshot.as_ref() {
        let loaded = core_load_snapshot(path, &mut memory)
            .with_context(|| format!("failed to load snapshot {}", path.display()))?;
        apply_snapshot(&mut state, &mut timer, &loaded)?;
        instruction_count = loaded.metadata.instruction_count;
        cycle_count = loaded.metadata.cycle_count;
        // Carry over fast_mode flag if present so timer/IRQ cadence matches.
        if loaded.metadata.fast_mode {
            println!("Loaded snapshot with fast_mode=true");
        }
        if loaded.metadata.lcd.is_some() && lcd.is_none() {
            lcd = Some(sc62015_core::LcdController::new());
        }
        if let (Some(meta), Some(payload)) =
            (loaded.metadata.lcd.as_ref(), loaded.lcd_payload.as_ref())
        {
            if let Some(lcd_dev) = lcd.as_mut() {
                lcd_dev
                    .load_snapshot(meta, payload.as_slice())
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }
    }
    // Initialise PC from reset vector when starting from ROM and no snapshot set it.
    if rom_loaded {
        let current_pc = state.get_reg("PC", register_width("PC")) & ADDRESS_MASK;
        if current_pc == 0 {
            if let Some(pc) = read_reset_vector(&memory) {
                state.set_reg("PC", pc, register_width("PC"));
            }
        }
    }

    let mut host = LocalHost;
    let mut profiler = StatsProfiler::default();
    let mut dur_exec_ns: u128 = 0;
    let mut dur_kbd_ns: u128 = 0;
    let mut dur_timer_ns: u128 = 0;
    let mut opcode_stats: HashMap<u8, (u64, u128)> = HashMap::new();

    for _ in 0..args.steps {
        if state.halted {
            break;
        }
        if let Some(limit) = timeout {
            if start_time.elapsed() >= limit {
                println!("Timeout reached after {:.2}s", start_time.elapsed().as_secs_f64());
                break;
            }
        }
        let t_exec = Instant::now();
        let (opcode, _len) = execute_step::<BoundInstrRepr, ManifestEntry, LocalHost, StatsProfiler>(
            &mut state,
            &mut memory,
            &mut host,
            &*MANIFEST,
            &*OPCODE_LOOKUP,
            keyboard.as_mut().map(|k| k as &mut sc62015_core::KeyboardMatrix),
            lcd.as_mut().map(|l| l as &mut sc62015_core::LcdController),
            tracer.as_mut().map(|t| t as &mut sc62015_core::PerfettoTracer),
            Some(&mut profiler),
            instruction_count,
        )
        .map_err(|e| anyhow::anyhow!(format!("execute failed: {e}")))?;
        dur_exec_ns += t_exec.elapsed().as_nanos();
        opcode_stats
            .entry(opcode)
            .and_modify(|(c, t)| {
                *c += 1;
                *t += t_exec.elapsed().as_nanos();
            })
            .or_insert((1, t_exec.elapsed().as_nanos()));

        if let Some(kbd) = keyboard.as_mut() {
            let t_kbd = Instant::now();
            if kbd.scan_tick() > 0 {
                kbd.write_fifo_to_memory(&mut memory);
            }
            dur_kbd_ns += t_kbd.elapsed().as_nanos();
        }
        let t_timer = Instant::now();
        timer.tick_timers(&mut memory, &mut cycle_count);
        dur_timer_ns += t_timer.elapsed().as_nanos();
        instruction_count = instruction_count.wrapping_add(1);
    }

    if let Some(tr) = tracer {
        tr.finish()?;
    }

    println!(
        "Executed {} instructions (cycles {})",
        instruction_count, cycle_count
    );
    if instruction_count > 0 {
        let exec_ms = dur_exec_ns as f64 / 1_000_000.0;
        let kbd_ms = dur_kbd_ns as f64 / 1_000_000.0;
        let timer_ms = dur_timer_ns as f64 / 1_000_000.0;
        println!(
            "Timing breakdown (ms): exec {:.3}, keyboard {:.3}, timer {:.3} (per-instr exec {:.3} µs)",
            exec_ms,
            kbd_ms,
            timer_ms,
            (dur_exec_ns as f64 / instruction_count as f64) / 1000.0
        );
        println!(
            "Bus stats: loads {} (python {}), stores {} (python {})",
            profiler.bus_load, profiler.python_load, profiler.bus_store, profiler.python_store
        );
        if args.profile_opcodes {
            let mut rows: Vec<(u8, u64, u128)> = opcode_stats
                .into_iter()
                .map(|(opc, (count, nanos))| (opc, count, nanos))
                .collect();
            rows.sort_by_key(|&(_, _, nanos)| std::cmp::Reverse(nanos));
            println!("Top opcodes by total wall time:");
            for (idx, (opc, count, nanos)) in rows.iter().take(10).enumerate() {
                let avg_us = (*nanos as f64 / *count as f64) / 1000.0;
                let total_ms = *nanos as f64 / 1_000_000.0;
                println!(
                    "{:>2}: opcode 0x{:02X} count {:>5} total {:>8.3} ms avg {:>6.3} µs",
                    idx + 1,
                    opc,
                    count,
                    total_ms,
                    avg_us
                );
            }
        }
    }

    if args.lcd_text {
        println!("\nLCD TEXT:");
        if let Some(lcd_dev) = lcd.as_ref() {
            match decode_display_text(lcd_dev, &memory) {
                Ok(lines) if !lines.is_empty() => {
                    for (idx, line) in lines.iter().enumerate() {
                        println!("ROW{idx}: {line}");
                    }
                }
                Ok(_) => println!("<no text decoded>"),
                Err(err) => println!("<decode failed: {err}>"),
            }
        } else {
            println!("<LCD device disabled>");
        }
    }

    if let Some(path) = args.save_snapshot.as_ref() {
        let (timer_info, interrupt_info) = timer.snapshot_info();
        let (lcd_meta, lcd_payload) = if let Some(lcd_dev) = lcd.as_ref() {
            let (meta, payload) = lcd_dev.export_snapshot();
            (Some(meta), payload)
        } else {
            (None, Vec::new())
        };
        let metadata = SnapshotMetadata {
            magic: SNAPSHOT_MAGIC.to_string(),
            version: SNAPSHOT_VERSION,
            backend: "rust-cli".to_string(),
            created: now_timestamp(),
            instruction_count,
            cycle_count,
            pc: state.get_reg("PC", register_width("PC")) & ADDRESS_MASK,
            timer: timer_info,
            interrupts: interrupt_info,
            fallback_ranges: memory.python_ranges().to_vec(),
            readonly_ranges: memory.readonly_ranges().to_vec(),
            internal_ram: (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32),
            imem: (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32),
            memory_dump_pc: 0,
            fast_mode: args.fast,
            memory_image_size: memory.external_len(),
            lcd_payload_size: lcd_payload.len(),
            lcd: lcd_meta,
        };
        let registers = collect_registers(&state);
        let lcd_payload_ref = if lcd_payload.is_empty() {
            None
        } else {
            Some(lcd_payload.as_slice())
        };
        core_save_snapshot(path, &metadata, &registers, &memory, lcd_payload_ref)
            .with_context(|| format!("failed to save snapshot {}", path.display()))?;
    }

    Ok(())
}

fn apply_snapshot(
    state: &mut rust_scil::state::State,
    timer: &mut TimerContext,
    loaded: &SnapshotLoad,
) -> Result<()> {
    apply_registers(state, &loaded.registers);
    timer.apply_snapshot_info(&loaded.metadata.timer, &loaded.metadata.interrupts);
    if let Some(lcd_props) = loaded.metadata.lcd.as_ref() {
        if lcd_props.get("width").is_none() {
            bail!("snapshot lcd metadata missing width");
        }
    }
    Ok(())
}

fn read_reset_vector(memory: &MemoryImage) -> Option<u32> {
    // PC-E500 reset vector stored at 0xFFFFD..0xFFFFF (little-endian 24-bit)
    let base = 0x0F_FFFD;
    let b0 = memory.read_byte(base)? as u32;
    let b1 = memory.read_byte(base + 1)? as u32;
    let b2 = memory.read_byte(base + 2)? as u32;
    Some(((b2 << 16) | (b1 << 8) | b0) & ADDRESS_MASK)
}
