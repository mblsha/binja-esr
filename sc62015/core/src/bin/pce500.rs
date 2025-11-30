// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/cli.py

use clap::Parser;
use retrobus_perfetto::{AnnotationValue, PerfettoTraceBuilder, TrackId};
#[cfg(feature = "llama-tests")]
use sc62015_core::PERFETTO_TRACER;
use sc62015_core::{
    keyboard::KeyboardMatrix,
    lcd::{LcdController, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS},
    llama::{
        eval::{power_on_reset, LlamaBus, LlamaExecutor},
        opcodes::RegName,
        state::LlamaState,
    },
    memory::MemoryImage,
    timer::TimerContext,
    ADDRESS_MASK, INTERNAL_MEMORY_START,
};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const FONT_BASE: usize = 0x00F2215;
const GLYPH_WIDTH: usize = 5;
const GLYPH_STRIDE: usize = GLYPH_WIDTH + 1;
const GLYPH_COUNT: usize = 96;
const ROWS_PER_CELL: usize = 8;
const COLS_PER_CELL: usize = 6;
const IMEM_IMR_OFFSET: u32 = 0xFB;
const IMEM_ISR_OFFSET: u32 = 0xFC;
const IMEM_KOL_OFFSET: u32 = 0xF0;
const IMEM_KOH_OFFSET: u32 = 0xF1;
const IMEM_KIL_OFFSET: u32 = 0xF2;
const IMEM_LCC_OFFSET: u32 = 0xFE;
const FIFO_BASE_ADDR: u32 = 0x00BFC96;
const FIFO_TAIL_ADDR: u32 = 0x00BFC9E;
const VEC_RANGE_START: u32 = 0x00BFCC6;
const VEC_RANGE_END: u32 = 0x00BFCCC;
const HANDLER_JP_Y_PC: u32 = 0x0F2053;
const ISR_KEYI: u8 = 0x04;
const ISR_MTI: u8 = 0x01;
const ISR_STI: u8 = 0x02;
const IMR_MASTER: u8 = 0x80;
const IMR_KEY: u8 = 0x04;
const IMR_MTI: u8 = 0x01;
const IMR_STI: u8 = 0x02;
const PF1_CODE: u8 = 0x56; // col=10, row=6
const PF2_CODE: u8 = 0x55; // col=10, row=5
const PF2_MENU_PC: u32 = 0x0F1FBF; // observed Python PC after PF2 menu renders
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;
const ROM_RESET_VECTOR_ADDR: u32 = 0xFFFFD;

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

    fn finish(self) -> Result<(), String> {
        self.builder
            .save(&self.path)
            .map_err(|e| format!("perfetto save: {e}"))
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "pce500-llama",
    about = "Standalone Rust LLAMA runner for the PC-E500 ROM (no Python harness)."
)]
struct Args {
    /// Number of instructions to execute before exiting.
    #[arg(long, default_value_t = 20_000)]
    steps: u64,

    /// ROM image to load (defaults to the repo-symlinked PC-E500 ROM).
    #[arg(long, value_name = "PATH", default_value_os_t = default_rom_path())]
    rom: PathBuf,

    /// Run until PF1 menu (ignores steps limit).
    #[arg(long, default_value_t = false)]
    pf1: bool,

    /// Run PF2 (does not clear RAM) instead of PF1; ignored if --pf1 is set.
    #[arg(long, default_value_t = false, conflicts_with = "pf1")]
    pf2: bool,

    /// Optional explicit matrix code to auto-press (overrides --pf1/--pf2). Accepts
    /// 'pf1', 'pf2', decimal, or hex (e.g., 0x56).
    #[arg(long, value_name = "CODE")]
    auto_key: Option<String>,

    /// Auto-press after this many executed instructions.
    #[arg(long, default_value_t = 15_000)]
    auto_after: u64,

    /// Auto-release after this many additional instructions (if pressed).
    #[arg(long, default_value_t = 1_000)]
    auto_release_after: u64,

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

    /// Dump keyboard stats (strobe count, IRQ count, FIFO) on exit.
    #[arg(long, default_value_t = false)]
    debug_keyboard: bool,

    /// Force KOL/KOH strobes each tick (helps auto-key flows when ROM isn't strobing).
    #[arg(long, default_value_t = false)]
    force_strobe: bool,

    /// Inject a synthetic KIL read/strobe on auto-key press to simulate ROM polling.
    #[arg(long, default_value_t = false)]
    bridge_kil: bool,

    /// Only scan keyboard on timer IRQs (skip per-instruction scans).
    #[arg(long, default_value_t = false)]
    scan_on_timer: bool,

    /// If the ROM never reads KIL, perform one synthetic KIL read when KEYI is pending.
    #[arg(long, default_value_t = false)]
    force_kil_consume: bool,

    /// Trace IMEM keyboard/IRQ reads/writes (KOL/KOH/KIL/IMR/ISR) with PC context.
    #[arg(long, default_value_t = false)]
    trace_kbd: bool,

    /// Bridge PF2 by jumping to the PF2 menu PC after pressing PF2 (diagnostic only).
    #[arg(long, default_value_t = false)]
    force_pf2_jump: bool,

    /// Emit a Perfetto trace with IRQ/IMR/ISR events.
    #[arg(long, default_value_t = false)]
    perfetto: bool,

    /// Path to write the Perfetto trace.
    #[arg(long, value_name = "PATH", default_value = "pc-e500.perfetto-trace")]
    perfetto_path: PathBuf,
}

fn default_rom_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pc-e500.bin")
}

struct StandaloneBus {
    memory: MemoryImage,
    lcd: LcdController,
    timer: TimerContext,
    cycle_count: u64,
    keyboard: KeyboardMatrix,
    lcd_writes: u64,
    log_lcd: bool,
    log_lcd_count: u32,
    log_lcd_limit: u32,
    in_interrupt: bool,
    delivered_irq_count: u32,
    pending_kil: bool,
    last_kbd_access: Option<String>,
    kil_reads: u32,
    rom_kil_reads: u32,
    rom_koh_reads: u32,
    rom_kol_reads: u32,
    trace_kbd: bool,
    last_pc: u32,
    vec_patched: bool,
    perfetto: Option<IrqPerfetto>,
    last_irq_src: Option<String>,
    active_irq_mask: u8,
}

impl StandaloneBus {
    fn new(
        memory: MemoryImage,
        lcd: LcdController,
        timer: TimerContext,
        log_lcd: bool,
        log_lcd_limit: u32,
        trace_kbd: bool,
        perfetto: Option<IrqPerfetto>,
    ) -> Self {
        Self {
            memory,
            lcd,
            timer,
            cycle_count: 0,
            keyboard: KeyboardMatrix::new(),
            lcd_writes: 0,
            log_lcd,
            log_lcd_count: 0,
            log_lcd_limit,
            in_interrupt: false,
            delivered_irq_count: 0,
            pending_kil: false,
            last_kbd_access: None,
            kil_reads: 0,
            rom_kil_reads: 0,
            rom_koh_reads: 0,
            rom_kol_reads: 0,
            trace_kbd,
            last_pc: 0,
            vec_patched: false,
            perfetto,
            last_irq_src: None,
            active_irq_mask: 0,
        }
    }

    fn lcd(&self) -> &LcdController {
        &self.lcd
    }

    fn keyboard_stats(&self) -> (u32, u32, Vec<u8>) {
        (
            self.keyboard.strobe_count(),
            self.keyboard.irq_count(),
            self.keyboard.fifo_snapshot(),
        )
    }

    fn delivered_irq_count(&self) -> u32 {
        self.delivered_irq_count
    }

    fn set_pc(&mut self, pc: u32) {
        self.last_pc = pc & ADDRESS_MASK;
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

    fn trace_fifo_access(&self, kind: &str, addr: u32, bits: u8, value: u32) {
        if !self.trace_kbd {
            return;
        }
        if !(FIFO_BASE_ADDR..=FIFO_TAIL_ADDR).contains(&addr) {
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

    /// Parity: leave vectors to the ROM; no patching.
    fn maybe_patch_vectors(&mut self) {
        self.vec_patched = true;
    }

    fn tick_keyboard(&mut self) {
        // Parity: scan only when called by timer cadence; assert KEYI when events are queued.
        let events = self.keyboard.scan_tick();
        if events > 0 {
            self.keyboard.write_fifo_to_memory(&mut self.memory);
            self.raise_key_irq();
            self.pending_kil = true;
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

    fn raise_key_irq(&mut self) {
        if let Some(cur) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
            let new = cur | ISR_KEYI;
            self.memory.write_internal_byte(IMEM_ISR_OFFSET, new);
        }
    }

    fn press_key(&mut self, code: u8) {
        self.keyboard.press_matrix_code(code, &mut self.memory);
        self.raise_key_irq();
        self.pending_kil = true;
    }

    fn release_key(&mut self, code: u8) {
        self.keyboard.release_matrix_code(code, &mut self.memory);
        self.pending_kil = false;
        self.raise_key_irq();
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
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        if self.in_interrupt {
            let active = self.active_irq_mask;
            let still_active = active != 0 && (isr & active) != 0;
            // If we appear stuck in an interrupt but the active bit cleared, drop the in-progress
            // marker so pending IRQs can be delivered, matching the Python emulator’s recovery.
            if (imr & IMR_MASTER) != 0 && !still_active {
                self.in_interrupt = false;
                self.active_irq_mask = 0;
            } else {
                #[cfg(feature = "llama-tests")]
                {
                    if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                        if let Some(tracer) = guard.as_mut() {
                            tracer.record_kio_read(Some(self.last_pc), IMEM_IMR_OFFSET as u8, imr);
                            tracer.record_kio_read(Some(self.last_pc), IMEM_ISR_OFFSET as u8, isr);
                            tracer.record_kio_read(Some(self.last_pc), 0xEF, 0);
                        }
                    }
                }
                return false;
            }
        }
        // Master bit set means interrupts enabled. Keyboard IRQs are level-triggered, so allow
        // them to pend even if IRM is still masked to mirror the Python emulator.
        let irm_enabled = (imr & IMR_MASTER) != 0 || (isr & ISR_KEYI != 0);
        if !irm_enabled {
            #[cfg(feature = "llama-tests")]
            {
                if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                    if let Some(tracer) = guard.as_mut() {
                        tracer.record_kio_read(Some(self.last_pc), IMEM_IMR_OFFSET as u8, imr);
                        tracer.record_kio_read(Some(self.last_pc), IMEM_ISR_OFFSET as u8, isr);
                        tracer.record_kio_read(Some(self.last_pc), 0xEF, 0);
                    }
                }
            }
            return false;
        }
        // Per-bit masks are "enabled when set".
        let pending = ((isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0))
            || ((isr & ISR_MTI != 0) && (imr & IMR_MTI != 0))
            || ((isr & ISR_STI != 0) && (imr & IMR_STI != 0));
        // If IMR masks everything, remember pending ISRs so we can restore on RETI.
        if !pending && isr != 0 && self.active_irq_mask == 0 {
            self.last_irq_src = Some("masked".to_string());
        }
        // Trace pending decision for visibility (Perfetto + console).
        #[cfg(feature = "llama-tests")]
        {
            if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_kio_read(Some(self.last_pc), IMEM_IMR_OFFSET as u8, imr);
                    tracer.record_kio_read(Some(self.last_pc), IMEM_ISR_OFFSET as u8, isr);
                    tracer.record_kio_read(Some(self.last_pc), 0xEE, pending as u8);
                    // Encode src mask: bit0=KEY, bit1=MTI, bit2=STI
                    let mask = (((isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0)) as u8)
                        | ((((isr & ISR_MTI != 0) && (imr & IMR_MTI != 0)) as u8) << 1)
                        | ((((isr & ISR_STI != 0) && (imr & IMR_STI != 0)) as u8) << 2);
                    tracer.record_kio_read(Some(self.last_pc), 0xED, mask);
                }
            }
            if std::env::var("IRQ_TRACE").as_deref() == Ok("1") {
                println!(
                    "[irq-pending] pc=0x{:05X} imr=0x{:02X} isr=0x{:02X} pending={} in_irq={}",
                    self.last_pc, imr, isr, pending, self.in_interrupt
                );
            }
        }
        pending
    }

    fn idle_tick(&mut self) {
        // Advance timers once; scan keyboard when MTI fires to mirror Python’s timer hook.
        let (mti, sti) = self.timer.tick_timers(&mut self.memory, &mut self.cycle_count);
        if mti {
            self.tick_keyboard();
        }
        if sti && self.keyboard.fifo_len() > 0 {
            // Parity: surface timer-driven keyboard IRQs when events are pending.
            if let Some(cur) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                if (cur & ISR_KEYI) == 0 {
                    self.memory
                        .write_internal_byte(IMEM_ISR_OFFSET, cur | ISR_KEYI);
                }
            }
        }
    }

    #[cfg(feature = "llama-tests")]
    #[allow(dead_code)]
    fn trace_kio(&self, pc: u32, offset: u8, value: u8) {
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                tracer.record_kio_read(Some(pc), offset, value);
            }
        }
    }

    fn log_irq_delivery(&mut self, src: Option<&str>, vec: u32, imr: u8, isr: u8, pc: u32) {
        #[cfg(feature = "llama-tests")]
        {
            if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_kio_read(Some(pc), IMEM_IMR_OFFSET as u8, imr);
                    tracer.record_kio_read(Some(pc), IMEM_ISR_OFFSET as u8, isr);
                    tracer.record_kio_read(Some(pc), 0xFF, (vec & 0xFF) as u8);
                }
            }
        }
        if std::env::var("IRQ_TRACE").as_deref() == Ok("1") {
            println!(
                "[irq-deliver] pc=0x{pc:05X} src={src:?} vec=0x{vec:05X} imr=0x{imr:02X} isr=0x{isr:02X}",
                pc = pc,
                src = src,
                vec = vec & ADDRESS_MASK,
                imr = imr,
                isr = isr
            );
        }
    }

    fn deliver_irq(&mut self, state: &mut LlamaState) {
        // Mirror the IR intrinsic: push PC, F, IMR, clear IRM, jump to vector.
        fn push_stack(
            memory: &mut MemoryImage,
            state: &mut LlamaState,
            reg: RegName,
            value: u32,
            bits: u8,
        ) {
            let bytes = bits.div_ceil(8);
            let sp = state.get_reg(reg);
            let new_sp = sp.wrapping_sub(bytes as u32) & 0x00FF_FFFF;
            for i in 0..bytes {
                let byte = ((value >> (8 * i)) & 0xFF) as u8;
                let _ = memory.store(new_sp + i as u32, 8, byte as u32);
            }
            state.set_reg(reg, new_sp);
        }

        let pc = state.pc() & ADDRESS_MASK;
        push_stack(&mut self.memory, state, RegName::S, pc, 24);
        let f = state.get_reg(RegName::F) & 0xFF;
        push_stack(&mut self.memory, state, RegName::S, f, 8);
        let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
        let imr = (self.memory.load(imr_addr, 8).unwrap_or(0) & 0xFF) as u8;
        push_stack(&mut self.memory, state, RegName::S, imr as u32, 8);
        // Clear IRM (bit7) on entry to match Python/IR semantics.
        let cleared_imr = imr & 0x7F;
        let _ = self.memory.store(imr_addr, 8, cleared_imr as u32);
        state.set_reg(RegName::IMR, u32::from(cleared_imr));

        // Ensure the vector table is patched before reading the vector.
        self.maybe_patch_vectors();
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let vec = (self.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap_or(0))
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 1, 8).unwrap_or(0) << 8)
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 2, 8).unwrap_or(0) << 16);
        state.set_pc(vec & ADDRESS_MASK);
        state.set_halted(false);
        self.in_interrupt = true;
        let (src, mask) = if (isr & ISR_KEYI != 0) && (imr & IMR_KEY != 0) {
            (Some("KEY"), ISR_KEYI)
        } else if (isr & ISR_MTI != 0) && (imr & IMR_MTI != 0) {
            (Some("MTI"), ISR_MTI)
        } else if (isr & ISR_STI != 0) && (imr & IMR_STI != 0) {
            (Some("STI"), ISR_STI)
        } else {
            (None, 0)
        };
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
        if let Some(tracer) = self.perfetto.take() {
            if let Err(err) = tracer.finish() {
                eprintln!("[perfetto] failed to save trace: {err}");
            }
        }
    }
}

fn mask_bits(bits: u8) -> u32 {
    if bits == 0 || bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

impl LlamaBus for StandaloneBus {
    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        let addr = addr & ADDRESS_MASK;
        if bits == 0 {
            return 0;
        }
        let kbd_offset = MemoryImage::internal_offset(addr);
        if let Some(offset) = kbd_offset {
            if offset == IMEM_KIL_OFFSET {
                // Honor KSD (keyboard strobe disable) bit in LCC (bit 2).
                let lcc = self.memory.read_internal_byte(IMEM_LCC_OFFSET).unwrap_or(0);
                if (lcc & 0x04) != 0 {
                    self.trace_kbd_access("read-ksd-masked", addr, offset, bits, 0);
                    return 0;
                }
            }
            // E-port inputs (PEI1/PEI2) are host-driven in Python; return 0 when not bridged.
            if matches!(offset, 0xF5 | 0xF6) {
                self.trace_kbd_access("read-eport", addr, offset, bits, 0);
                return 0;
            }
            if let Some(byte) = self.keyboard.handle_read(offset, &mut self.memory) {
                match offset {
                    IMEM_KIL_OFFSET => self.kil_reads = self.kil_reads.saturating_add(1),
                    IMEM_KOH_OFFSET => self.rom_koh_reads = self.rom_koh_reads.saturating_add(1),
                    IMEM_KOL_OFFSET => self.rom_kol_reads = self.rom_kol_reads.saturating_add(1),
                    _ => {}
                }
                if offset == IMEM_KIL_OFFSET {
                    self.pending_kil = false;
                    // Parity: assert KEYI when FIFO has pending events on read.
                    if self.keyboard.fifo_len() > 0 {
                        if let Some(cur) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                            if (cur & ISR_KEYI) == 0 {
                                self.memory
                                    .write_internal_byte(IMEM_ISR_OFFSET, cur | ISR_KEYI);
                            }
                        }
                    }
                    // Emit perfetto event for KIL read with PC/value.
                    #[cfg(feature = "llama-tests")]
                    {
                        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                            if let Some(tracer) = guard.as_mut() {
                                tracer.record_kio_read(Some(self.last_pc), offset as u8, byte);
                            }
                        }
                    }
                    println!(
                        "[kil-read-llama] pc=0x{:06X} val=0x{:02X}",
                        self.last_pc, byte
                    );
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
            ) && self.trace_kbd
            {
                // Trace fallthrough reads (handled by memory, not keyboard).
                if let Some(val) = self.memory.read_internal_byte(offset) {
                    self.trace_kbd_access("read-fallthrough", addr, offset, bits, val as u32);
                }
            }
        }
        if self.trace_kbd && (VEC_RANGE_START..=VEC_RANGE_END).contains(&addr) {
            println!(
                "[vec-trace-read] pc=0x{pc:05X} addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                pc = self.last_pc,
                addr = addr,
                bits = bits,
                val = self.memory.load(addr, bits).unwrap_or(0) & mask_bits(bits)
            );
        }
        self.memory
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
                if (FIFO_BASE_ADDR..=FIFO_TAIL_ADDR).contains(&addr) {
                    self.trace_fifo_access("read", addr, bits, val);
                }
                val & mask_bits(bits)
            })
            .unwrap_or(0)
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        let addr = addr & ADDRESS_MASK;
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
            if matches!(offset, 0xF5 | 0xF6) {
                // Ignore writes to host-driven E-port inputs for parity.
                self.trace_kbd_access("write-eport", addr, offset, bits, value);
                return;
            }
            if self
                .keyboard
                .handle_write(offset, value as u8, &mut self.memory)
            {
                // Parity: KIO writes can enqueue events; assert KEYI if FIFO has entries.
                if (offset == IMEM_KOL_OFFSET || offset == IMEM_KOH_OFFSET)
                    && self.keyboard.fifo_len() > 0
                {
                    if let Some(cur) = self.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                        if (cur & ISR_KEYI) == 0 {
                            self.memory
                                .write_internal_byte(IMEM_ISR_OFFSET, cur | ISR_KEYI);
                        }
                    }
                    self.pending_kil = true;
                }
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
                    }
                }
                return;
            }
        }
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
            }
            return;
        }
        if MemoryImage::is_internal(addr) {
            self.trace_imem_access("write", addr, bits, value);
        }
        if (FIFO_BASE_ADDR..=FIFO_TAIL_ADDR).contains(&addr) {
            self.trace_fifo_access("write", addr, bits, value);
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
        if let Some((offset, prev)) = imr_isr_prev {
            if let Some(cur) = self.memory.read_internal_byte(offset) {
                if cur != prev {
                    self.log_imem_write(offset, prev, cur);
                }
            }
        }
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        base & ADDRESS_MASK
    }
}

struct FontMap {
    glyphs: HashMap<[u8; GLYPH_WIDTH], char>,
}

impl FontMap {
    fn from_rom(rom: &[u8]) -> Self {
        let mut glyphs = HashMap::new();
        for index in 0..GLYPH_COUNT {
            let start = FONT_BASE + index * GLYPH_STRIDE;
            if start + GLYPH_WIDTH > rom.len() {
                break;
            }
            let mut pattern = [0u8; GLYPH_WIDTH];
            pattern.copy_from_slice(&rom[start..start + GLYPH_WIDTH]);
            let codepoint = 0x20 + index as u32;
            if let Some(ch) = char::from_u32(codepoint) {
                glyphs.insert(pattern, ch);
                // Accept inverted glyphs to mirror the Python text decoder's tolerance for
                // polarity differences in the LCD buffer.
                let mut inverted = [0u8; GLYPH_WIDTH];
                for (dest, src) in inverted.iter_mut().zip(pattern) {
                    *dest = (!src) & 0x7F;
                }
                glyphs.entry(inverted).or_insert(ch);
            }
        }
        Self { glyphs }
    }

    fn resolve(&self, pattern: &[u8; GLYPH_WIDTH]) -> char {
        *self.glyphs.get(pattern).unwrap_or(&'?')
    }
}

fn cell_patterns(
    buffer: &[[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS],
) -> Vec<Vec<[u8; GLYPH_WIDTH]>> {
    let rows = LCD_DISPLAY_ROWS / ROWS_PER_CELL;
    let cols = LCD_DISPLAY_COLS / COLS_PER_CELL;
    let mut patterns = Vec::with_capacity(rows);

    for row in 0..rows {
        let mut row_patterns = Vec::with_capacity(cols);
        let y0 = row * ROWS_PER_CELL;
        for col in 0..cols {
            let x0 = col * COLS_PER_CELL;
            let mut pattern = [0u8; GLYPH_WIDTH];
            for dx in 0..GLYPH_WIDTH {
                let mut column = 0u8;
                for dy in 0..ROWS_PER_CELL {
                    // Match Python text decoder: treat 0 as lit pixel when building columns.
                    if buffer[y0 + dy][x0 + dx] == 0 {
                        column |= 1 << dy;
                    }
                }
                pattern[dx] = column & 0x7F;
            }
            row_patterns.push(pattern);
        }
        patterns.push(row_patterns);
    }
    patterns
}

fn decode_lcd_text(lcd: &LcdController, font: &FontMap) -> Vec<String> {
    let buffer = lcd.display_buffer();
    let patterns = cell_patterns(&buffer);
    let mut lines = Vec::with_capacity(patterns.len());
    for row in patterns {
        let text: String = row.iter().map(|p| font.resolve(p)).collect();
        lines.push(text.trim_end().to_string());
    }
    lines
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

fn parse_matrix_code(raw: &str) -> Result<Option<u8>, Box<dyn Error>> {
    let lowered = raw.trim().to_lowercase();
    if lowered == "pf1" {
        return Ok(Some(PF1_CODE));
    }
    if lowered == "pf2" {
        return Ok(Some(PF2_CODE));
    }
    if let Some(hex) = lowered.strip_prefix("0x") {
        let value = u8::from_str_radix(hex, 16)?;
        return Ok(Some(value));
    }
    if let Ok(value) = lowered.parse::<u8>() {
        return Ok(Some(value));
    }
    Err(format!("could not parse matrix code '{raw}'").into())
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

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let rom_bytes = load_rom(&args.rom)?;
    let font = FontMap::from_rom(&rom_bytes);

    let log_lcd_env = env::var("RUST_LCD_TRACE").is_ok();
    let log_lcd = args.lcd_log || log_lcd_env;
    let log_lcd_limit = args
        .lcd_log_limit
        .or_else(|| {
            env::var("RUST_LCD_TRACE_MAX")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
        })
        .unwrap_or(50);

    let auto_key = if args.pf1 {
        Some(PF1_CODE)
    } else if args.pf2 {
        Some(PF2_CODE)
    } else if let Some(raw) = args.auto_key.as_ref() {
        parse_matrix_code(raw)?
    } else {
        None
    };
    // Parity: do not auto-strobe; rely on ROM strobes.
    let trace_kbd = args.trace_kbd;

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
    let mut trace_pc_counts: HashMap<u32, u64> = HashMap::new();
    let trace_pc_window = args.trace_pc_window.unwrap_or(0);
    let mut trace_window_active: u64 = 0;
    let mut trace_window_anchor: Option<u32> = None;
    let trace_regs = args.trace_regs;
    let mut last_y_trace: Option<u32> = None;

    let mut memory = MemoryImage::new();
    // Load only ROM region (top 256KB) to mirror Python; leave RAM zeroed.
    let rom_start: usize = 0xC0000;
    let rom_len: usize = 0x40000;
    let src_start = rom_bytes.len().saturating_sub(rom_len);
    let slice = &rom_bytes[src_start..];
    let copy_len = slice.len().min(rom_len);
    memory.write_external_slice(rom_start, &slice[slice.len() - copy_len..]);
    memory.set_readonly_ranges(vec![(rom_start as u32, (rom_start + rom_len - 1) as u32)]);
    memory.set_keyboard_bridge(false);

    // Timer periods align with Python harness defaults (fast ≈500 cycles, slow ≈5000)
    // unless disabled for debugging.
    let perfetto = args
        .perfetto
        .then(|| IrqPerfetto::new(args.perfetto_path.clone()));
    let mut bus = StandaloneBus::new(
        memory,
        LcdController::new(),
        if args.disable_timers {
            TimerContext::new(false, 0, 0)
        } else {
            // Match Python cadence: MTI≈500 cycles, STI≈5000 cycles.
            TimerContext::new(true, 500, 5_000)
        },
        log_lcd,
        log_lcd_limit,
        trace_kbd,
        perfetto,
    );
    bus.strobe_all_columns();
    if auto_key.is_some() {
        bus.keyboard.set_repeat_enabled(false);
    }
    let mut state = LlamaState::new();
    let mut executor = LlamaExecutor::new();
    power_on_reset(&mut bus, &mut state);
    // Align PC with ROM reset vector.
    let reset_vec = rom_bytes
        .get(ROM_RESET_VECTOR_ADDR as usize)
        .copied()
        .unwrap_or(0) as u32
        | ((rom_bytes
            .get(ROM_RESET_VECTOR_ADDR as usize + 1)
            .copied()
            .unwrap_or(0) as u32)
            << 8)
        | ((rom_bytes
            .get(ROM_RESET_VECTOR_ADDR as usize + 2)
            .copied()
            .unwrap_or(0) as u32)
            << 16);
    state.set_pc(reset_vec & ADDRESS_MASK);
    // Leave IMR at reset defaults; the ROM will initialize vectors/masks.

    let start = Instant::now();
    let mut executed: u64 = 0;
    let auto_press_step: u64 = args.auto_after;
    let mut auto_release_after: u64 = args.auto_release_after;
    if args.pf2 && args.auto_release_after == 1_000 {
        auto_release_after = args.steps;
    }
    let mut pressed_key = false;
    let mut release_at: u64 = 0;
    let max_steps = args.steps;
    let mut _halted_after_pf1 = false;
    for _ in 0..max_steps {
        // Ensure vector table is patched once before executing instructions.
        if !bus.vec_patched {
            bus.maybe_patch_vectors();
        }
        // Drive timers each iteration; keyboard scan occurs on timer cadence inside idle_tick.
        bus.idle_tick();
        if bus.irq_pending() {
            bus.deliver_irq(&mut state);
        } else if state.is_halted() {
            continue;
        }
        let pc = state.pc();
        bus.set_pc(pc);
        if pc == HANDLER_JP_Y_PC {
            let y = state.get_reg(RegName::Y) & 0xFFFFFF;
            if last_y_trace != Some(y) {
                println!(
                    "[jp-y-trace] pc=0x{pc:05X} y=0x{y:06X} imr=0x{imr:02X} isr=0x{isr:02X}",
                    pc = pc,
                    y = y,
                    imr = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0),
                    isr = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0),
                );
                last_y_trace = Some(y);
            }
        }
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
                    let s = state.get_reg(RegName::S) & 0xFFFFFF;
                    let y = state.get_reg(RegName::Y) & 0xFFFFFF;
                    println!(
                        "[pc-trace] pc=0x{pc:05X} hits={hits} imr=0x{imr:02X} isr=0x{isr:02X} a=0x{a:02X} f=0x{f:02X} sp=0x{s:06X} y=0x{y:06X}",
                        pc = pc,
                        hits = count,
                        imr = imr,
                        isr = isr,
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
                let s = state.get_reg(RegName::S) & 0xFFFFFF;
                let y = state.get_reg(RegName::Y) & 0xFFFFFF;
                println!(
                    "[pc-trace-window] anchor={anchor} pc=0x{pc:05X} remaining={} imr=0x{imr:02X} isr=0x{isr:02X} a=0x{a:02X} f=0x{f:02X} sp=0x{s:06X} y=0x{y:06X}",
                    trace_window_active,
                    pc = pc,
                    imr = imr,
                    isr = isr,
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
        // Simulate WAIT timing like the Python emulator: if the opcode is WAIT (0xEF),
        // advance timers/keyboard scans for I cycles so timer-driven scans are not starved.
        let wait_cycles = if opcode == 0xEF {
            (state.get_reg(RegName::I) & 0xFFFF) as usize
        } else {
            0
        };
        if let Some(code) = auto_key {
            if !pressed_key && executed >= auto_press_step {
                bus.press_key(code);
                if args.force_pf2_jump && code == PF2_CODE {
                    state.set_pc(PF2_MENU_PC);
                }
                pressed_key = true;
                release_at = executed + auto_release_after;
            } else if pressed_key && executed >= release_at {
                bus.release_key(code);
                pressed_key = false;
            }
        }
        match executor.execute(opcode, &mut state, &mut bus) {
            Ok(_) => {
                if opcode == 0x01 {
                    // RETI completes interrupt service.
                    let last_src = bus.last_irq_src.clone();
                    bus.log_irq_event(
                        "IRQ_Return",
                        last_src.as_deref(),
                        [
                            ("pc", AnnotationValue::Pointer(state.pc() as u64)),
                            (
                                "imr",
                                AnnotationValue::UInt(
                                    bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0)
                                        as u64,
                                ),
                            ),
                            (
                                "isr",
                                AnnotationValue::UInt(
                                    bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0)
                                        as u64,
                                ),
                            ),
                        ],
                    );
                    // Clear the ISR bit that triggered this interrupt (parity with Python emulator).
                    if bus.active_irq_mask != 0 {
                        if let Some(cur_isr) = bus.memory.read_internal_byte(IMEM_ISR_OFFSET) {
                            let new_isr = cur_isr & !bus.active_irq_mask;
                            bus.memory.write_internal_byte(IMEM_ISR_OFFSET, new_isr);
                        }
                        if bus.active_irq_mask == ISR_KEYI {
                            bus.pending_kil = false;
                        }
                    }
                    bus.in_interrupt = false;
                    bus.last_irq_src = None;
                    bus.active_irq_mask = 0;
                }
                if wait_cycles > 0 {
                    for _ in 0..wait_cycles {
                        bus.idle_tick();
                    }
                }
                executed += 1;
                if let Some(stop) = stop_pc {
                    if state.pc() == stop {
                        break;
                    }
                }
                if state.is_halted() {
                    continue;
                }
            }
            Err(err) => {
                eprintln!("error executing opcode 0x{opcode:02X} at PC=0x{pc:05X}: {err}");
                break;
            }
        }
    }
    let elapsed = start.elapsed();

    println!(
        "Executed {executed} instruction(s) in {:.3?} (PC=0x{:05X}, halted={}, lcd_writes={})",
        elapsed,
        state.pc(),
        state.is_halted(),
        bus.lcd_writes
    );
    let imr_mem = bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
    let isr_mem = bus.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
    let imr_reg = state.get_reg(RegName::IMR) & 0xFF;
    println!("Final IMR=0x{imr_reg:02X} (mem=0x{imr_mem:02X}) ISR=0x{isr_mem:02X}");

    let lcd_stats = bus.lcd().stats();
    println!(
        "LCD stats: on={:?} instr={:?} data={:?} cs(both/left/right)=({}/{}/{})",
        lcd_stats.chip_on,
        lcd_stats.instruction_counts,
        lcd_stats.data_write_counts,
        lcd_stats.cs_both_count,
        lcd_stats.cs_left_count,
        lcd_stats.cs_right_count
    );

    if args.debug_keyboard {
        let (strobe_count, irq_count, fifo) = bus.keyboard_stats();
        let kol = bus
            .memory
            .read_internal_byte(IMEM_KOL_OFFSET)
            .unwrap_or_default();
        let koh = bus
            .memory
            .read_internal_byte(IMEM_KOH_OFFSET)
            .unwrap_or_default();
        let kil = bus
            .memory
            .read_internal_byte(IMEM_KIL_OFFSET)
            .unwrap_or_default();
        println!(
            "Keyboard: strobe_count={} irq_count={} delivered_irq_count={} kil_reads={} rom_reads(kol/koh/kil)=({}/{}/{}) fifo={:?} KOL=0x{kol:02X} KOH=0x{koh:02X} KIL=0x{kil:02X}",
            strobe_count,
            irq_count,
            bus.delivered_irq_count(),
            bus.kil_reads,
            bus.rom_kol_reads,
            bus.rom_koh_reads,
            bus.rom_kil_reads,
            fifo
        );
    }

    let lcd_lines = decode_lcd_text(bus.lcd(), &font);
    println!("LCD (decoded text):");
    for line in &lcd_lines {
        println!("  {}", line);
    }

    bus.finish_perfetto();

    if args.perf {
        let instrs_per_sec = if elapsed.as_secs_f64() > 0.0 {
            (executed as f64) / elapsed.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "Perf: {:.2} MIPS ({} instr / {:.3?})",
            instrs_per_sec / 1_000_000.0,
            executed,
            elapsed
        );
    }

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
            Err(err) => failures.push(err),
        }
    }
    for needle in &args.expect_text {
        if !lcd_lines.iter().any(|line| line.contains(needle)) {
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
