use clap::Parser;
use sc62015_core::{
    keyboard::KeyboardMatrix,
    lcd::{LcdController, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS},
    llama::{
        eval::{LlamaBus, LlamaExecutor},
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
const IMEM_USR_OFFSET: u32 = 0xF8;
const IMEM_SSR_OFFSET: u32 = 0xFD;
const IMEM_LCC_OFFSET: u32 = 0xFE;
const IMEM_SCR_OFFSET: u32 = 0xFD;
const IMEM_KOL_OFFSET: u32 = 0xF0;
const IMEM_KOH_OFFSET: u32 = 0xF1;
const ISR_KEYI: u8 = 0x04;
const ISR_MTI: u8 = 0x01;
const ISR_STI: u8 = 0x02;
const IMR_MASTER: u8 = 0x80;
const IMR_KEY: u8 = 0x04;
const IMR_MTI: u8 = 0x01;
const IMR_STI: u8 = 0x02;
const PF1_CODE: u8 = 0x56; // col=10, row=6
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;
const ROM_RESET_VECTOR_ADDR: u32 = 0xFFFFD;

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

    /// Run until PF1 menu (ignores step limit).
    #[arg(long, default_value_t = false)]
    pf1: bool,
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
}

impl StandaloneBus {
    fn new(memory: MemoryImage, lcd: LcdController, timer: TimerContext) -> Self {
        let log_lcd_limit = env::var("RUST_LCD_TRACE_MAX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(50);
        Self {
            memory,
            lcd,
            timer,
            cycle_count: 0,
            keyboard: KeyboardMatrix::new(),
            lcd_writes: 0,
            log_lcd: env::var("RUST_LCD_TRACE").is_ok(),
            log_lcd_count: 0,
            log_lcd_limit,
            in_interrupt: false,
        }
    }

    fn lcd(&self) -> &LcdController {
        &self.lcd
    }

    fn tick_keyboard(&mut self) {
        if self.keyboard.scan_tick() > 0 {
            self.keyboard.write_fifo_to_memory(&mut self.memory);
            self.raise_key_irq();
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
    }

    fn release_key(&mut self, code: u8) {
        self.keyboard.release_matrix_code(code, &mut self.memory);
        self.raise_key_irq();
    }

    fn irq_pending(&mut self) -> bool {
        let isr = self.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0);
        let imr = self.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0);
        // Master bit cleared means interrupts enabled.
        if self.in_interrupt || (imr & IMR_MASTER) != 0 {
            return false;
        }
        // Per-bit masks are "mask when set"; deliver only when mask bit is 0.
        let mut any = false;
        if (isr & ISR_KEYI != 0) && (imr & IMR_KEY == 0) {
            any = true;
        }
        if (isr & ISR_MTI != 0) && (imr & IMR_MTI == 0) {
            any = true;
        }
        if (isr & ISR_STI != 0) && (imr & IMR_STI == 0) {
            any = true;
        }
        any
    }

    fn idle_tick(&mut self) {
        self.timer
            .tick_timers(&mut self.memory, &mut self.cycle_count);
        self.tick_keyboard();
    }

    fn deliver_irq(&mut self, state: &mut LlamaState) {
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
        let cleared_imr = imr & 0x7F;
        let _ = self.memory.store(imr_addr, 8, u32::from(cleared_imr));
        state.set_reg(RegName::IMR, cleared_imr as u32);

        let vec = (self.memory.load(INTERRUPT_VECTOR_ADDR, 8).unwrap_or(0))
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 1, 8).unwrap_or(0) << 8)
            | (self.memory.load(INTERRUPT_VECTOR_ADDR + 2, 8).unwrap_or(0) << 16);
        state.set_pc(vec & ADDRESS_MASK);
        state.set_halted(false);
        self.in_interrupt = true;
        self.memory.write_internal_byte(IMEM_ISR_OFFSET, 0);
        if self.log_lcd && self.log_lcd_count < 50 {
            println!(
                "[irq] delivered: vec=0x{vec:05X} imr=0x{imr:02X} pc_prev=0x{pc:05X}",
                vec = vec & ADDRESS_MASK,
                imr = cleared_imr,
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
        self.timer
            .tick_timers(&mut self.memory, &mut self.cycle_count);
        self.tick_keyboard();
        if let Some(byte) = self
            .keyboard
            .handle_read(addr.saturating_sub(INTERNAL_MEMORY_START), &mut self.memory)
        {
            return byte as u32;
        }
        if self.lcd.handles(addr) {
            return self.lcd.read(addr) & mask_bits(bits);
        }
        self.memory.load(addr, bits).unwrap_or(0) & mask_bits(bits)
    }

    fn store(&mut self, addr: u32, bits: u8, value: u32) {
        let addr = addr & ADDRESS_MASK;
        self.timer
            .tick_timers(&mut self.memory, &mut self.cycle_count);
        self.tick_keyboard();
        if self.keyboard.handle_write(
            addr.saturating_sub(INTERNAL_MEMORY_START),
            value as u8,
            &mut self.memory,
        ) {
            return;
        }
        if bits == 8 && self.lcd.handles(addr) {
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
            return;
        }
        let _ = self.memory.store(addr, bits, value);
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
                    if buffer[y0 + dy][x0 + dx] == 0 {
                        column |= 1 << dy;
                    }
                }
                pattern[dx] = column;
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

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let rom_bytes = load_rom(&args.rom)?;
    let font = FontMap::from_rom(&rom_bytes);

    let mut memory = MemoryImage::new();
    memory.load_external(&rom_bytes);
    memory.set_keyboard_bridge(true);
    // Seed IMR/ISR like the Python harness bootstrap so timer/key IRQs can deliver.
    memory.write_internal_byte(IMEM_IMR_OFFSET, 0x00);
    memory.write_internal_byte(IMEM_ISR_OFFSET, 0x00);
    // Align USR/SSR defaults with Python intrinsic reset.
    memory.write_internal_byte(IMEM_USR_OFFSET, 0x18);
    memory.write_internal_byte(IMEM_SSR_OFFSET, 0x00);
    memory.write_internal_byte(IMEM_LCC_OFFSET, 0x00);
    memory.write_internal_byte(IMEM_SCR_OFFSET, 0x00);

    let mut bus = StandaloneBus::new(
        memory,
        LcdController::new(),
        TimerContext::new(true, 8192, 8192),
    );
    bus.strobe_all_columns();
    let mut state = LlamaState::new();
    let mut executor = LlamaExecutor::new();
    // Align PC with ROM reset vector.
    let reset_vec = (bus.memory.load(ROM_RESET_VECTOR_ADDR, 8).unwrap_or(0))
        | (bus.memory.load(ROM_RESET_VECTOR_ADDR + 1, 8).unwrap_or(0) << 8)
        | (bus.memory.load(ROM_RESET_VECTOR_ADDR + 2, 8).unwrap_or(0) << 16);
    state.set_pc(reset_vec & ADDRESS_MASK);

    let start = Instant::now();
    let mut executed: u64 = 0;
    let auto_press_step: u64 = 15_000;
    let auto_release_after: u64 = 40_000;
    let mut pressed_pf1 = false;
    let mut release_at: u64 = 0;
    let max_steps = if args.pf1 { u64::MAX } else { args.steps };
    let mut _halted_after_pf1 = false;
    #[allow(clippy::manual_is_multiple_of)]
    for _ in 0..max_steps {
        let imr_reg = state.get_reg(RegName::IMR) & 0xFF;
        if bus.memory.read_internal_byte(IMEM_IMR_OFFSET).unwrap_or(0) != imr_reg as u8 {
            bus.memory
                .write_internal_byte(IMEM_IMR_OFFSET, imr_reg as u8);
        }
        bus.idle_tick();
        if bus.irq_pending() {
            bus.deliver_irq(&mut state);
            bus.in_interrupt = false;
        } else if state.is_halted() {
            continue;
        }
        let pc = state.pc();
        if bus.log_lcd && bus.log_lcd_count < 50 && executed % 1000 == 0 {
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
        if !pressed_pf1 && executed >= auto_press_step {
            bus.press_key(PF1_CODE);
            pressed_pf1 = true;
            release_at = executed + auto_release_after;
        } else if pressed_pf1 && executed >= release_at {
            bus.release_key(PF1_CODE);
            pressed_pf1 = false;
        }
        match executor.execute(opcode, &mut state, &mut bus) {
            Ok(_) => {
                executed += 1;
                if args.pf1 && state.pc() == 0x0F172F {
                    _halted_after_pf1 = true;
                    break;
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

    let lcd_lines = decode_lcd_text(bus.lcd(), &font);
    println!("LCD (decoded text):");
    for line in lcd_lines {
        println!("  {}", line);
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
