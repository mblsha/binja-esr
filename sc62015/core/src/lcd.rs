// PY_SOURCE: pce500/display/hd61202.py:HD61202
// PY_SOURCE: pce500/display/controller_wrapper.py:HD61202Controller

use crate::{
    llama::eval::{
        perfetto_instr_context, perfetto_last_call_stack, perfetto_last_pc, PerfettoCallStack,
    },
    llama::{opcodes::RegName, state::mask_for},
    PERFETTO_TRACER,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

const LCD_WIDTH: usize = 64;
const LCD_PAGES: usize = 8;
pub const LCD_CHIP_ROWS: usize = LCD_PAGES * 8;
pub const LCD_CHIP_COLS: usize = LCD_WIDTH;
#[allow(dead_code)]
const LCD_RANGE_LOW: u32 = 0x2000;
#[allow(dead_code)]
const LCD_RANGE_HIGH: u32 = 0xA000;
pub const LCD_DISPLAY_ROWS: usize = 32;
pub const LCD_DISPLAY_COLS: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LcdKind {
    #[serde(rename = "hd61202")]
    Hd61202,
    #[serde(rename = "unknown")]
    Unknown,
}

impl LcdKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hd61202 => "hd61202",
            Self::Unknown => "unknown",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim() {
            "hd61202" => Self::Hd61202,
            "unknown" => Self::Unknown,
            _ => Self::Unknown,
        }
    }
}

pub fn lcd_kind_from_snapshot_meta(metadata: &Value, default: LcdKind) -> LcdKind {
    metadata
        .get("kind")
        .and_then(|v| v.as_str())
        .map(LcdKind::parse)
        .unwrap_or(default)
}

pub trait LcdHal: Send {
    fn kind(&self) -> LcdKind;
    fn reset(&mut self);
    fn handles(&self, address: u32) -> bool;
    fn read(&mut self, address: u32) -> Option<u8>;
    fn write(&mut self, address: u32, value: u8);
    fn read_placeholder(&self, address: u32) -> u32;

    fn begin_display_write_capture(&mut self);
    fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite>;

    fn display_buffer(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS];
    fn chip_display_buffer(&self, chip_index: usize) -> [[u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS];
    fn display_vram_bytes(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_PAGES];
    fn display_trace_buffer(&self) -> [[LcdWriteTrace; LCD_DISPLAY_COLS]; LCD_PAGES];

    fn stats(&self) -> LcdStats;

    fn export_snapshot(&self) -> (Value, Vec<u8>);
    fn load_snapshot(&mut self, metadata: &Value, payload: &[u8]) -> Result<(), String>;
}

pub fn create_lcd(kind: LcdKind) -> Box<dyn LcdHal> {
    match kind {
        LcdKind::Hd61202 => Box::new(LcdController::new()),
        LcdKind::Unknown => Box::new(UnknownLcdController::new()),
    }
}

// The LCD controller is mirrored at 0x2000 and 0xA000. Bits 0-3 encode
// R/W, DI, and CS the same way the Python hd61202 decoder does:
//  bit0: 0=write, 1=read
//  bit1: 0=instruction, 1=data
//  bit2-3: chip select (00=both, 01=right/CS2, 10=left/CS1, 11=none)
const LCD_ADDR_HI_LEFT: u32 = 0x0A000;
const LCD_ADDR_HI_RIGHT: u32 = 0x02000;

/// Map an internal LCD overlay offset (0x00-0x0F) to the standard controller address space.
/// Parity: Python exposes the controller via an IMEM remap; treat it like the 0x2000 mirror.
pub fn overlay_addr(offset: u32) -> u32 {
    LCD_ADDR_HI_RIGHT + (offset & 0x0FFF)
}

#[derive(Clone, Copy, Default)]
struct Hd61202State {
    on: bool,
    start_line: u8,
    page: u8,
    y_address: u8,
    busy: bool,
}

struct Hd61202Chip {
    state: Hd61202State,
    vram: [[u8; LCD_WIDTH]; LCD_PAGES],
    vram_trace: [[LcdWriteTrace; LCD_WIDTH]; LCD_PAGES],
    instruction_count: u32,
    data_write_count: u32,
    data_read_count: u32,
}

impl Default for Hd61202Chip {
    fn default() -> Self {
        Self {
            state: Hd61202State::default(),
            vram: [[0; LCD_WIDTH]; LCD_PAGES],
            vram_trace: [[LcdWriteTrace::default(); LCD_WIDTH]; LCD_PAGES],
            instruction_count: 0,
            data_write_count: 0,
            data_read_count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LcdWriteTrace {
    pub pc: u32,
    pub call_stack: PerfettoCallStack,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LcdDisplayWrite {
    pub page: u8,
    pub col: u16,
    pub value: u8,
    pub trace: LcdWriteTrace,
}

pub struct UnknownLcdController {
    write_count: u64,
}

impl UnknownLcdController {
    pub fn new() -> Self {
        Self { write_count: 0 }
    }
}

impl Hd61202Chip {
    fn write_instruction(&mut self, instr: LcdInstruction, data: u8) {
        self.instruction_count = self.instruction_count.wrapping_add(1);
        self.state.busy = true;
        match instr {
            LcdInstruction::OnOff => {
                self.state.on = (data & 1) != 0;
            }
            LcdInstruction::StartLine => {
                self.state.start_line = data & 0b0011_1111;
            }
            LcdInstruction::SetPage => {
                self.state.page = data & 0b0000_0111;
            }
            LcdInstruction::SetYAddress => {
                self.state.y_address = data & 0b0011_1111;
            }
        }
    }

    fn write_data(&mut self, data: u8, trace: LcdWriteTrace) {
        self.data_write_count = self.data_write_count.wrapping_add(1);
        let page = (self.state.page as usize) % LCD_PAGES;
        let y = (self.state.y_address as usize) % LCD_WIDTH;
        self.vram[page][y] = data;
        self.vram_trace[page][y] = trace;
        self.state.y_address = ((self.state.y_address as usize + 1) % LCD_WIDTH) as u8;
        self.state.busy = true;
    }

    #[allow(dead_code)]
    fn read_status(&mut self) -> u8 {
        // Python HD61202 always reports ready; only ON flag is surfaced.
        let on = if self.state.on { 0x40 } else { 0x00 };
        self.state.busy = false;
        on
    }

    #[allow(dead_code)]
    fn read_data(&mut self) -> u8 {
        self.data_read_count = self.data_read_count.wrapping_add(1);
        let page = (self.state.page as usize) % LCD_PAGES;
        let y = (self.state.y_address as usize) % LCD_WIDTH;
        let value = self.vram[page][y];
        self.state.y_address = ((self.state.y_address as usize + 1) % LCD_WIDTH) as u8;
        value
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChipSelect {
    Left,  // CS1
    Right, // CS2
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadWrite {
    Write = 0,
    Read = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DataInstruction {
    Instruction,
    Data,
}

#[derive(Clone, Copy)]
enum LcdInstruction {
    OnOff,
    StartLine,
    SetPage,
    SetYAddress,
}

enum CommandKind {
    Instruction(LcdInstruction, u8),
    Data(u8),
}

struct LcdCommand {
    cs: ChipSelect,
    kind: CommandKind,
}

fn decode_access(address: u32) -> Option<(ChipSelect, DataInstruction, ReadWrite)> {
    let addr_hi = address & 0x0F000;
    // Parity: tolerate accesses anywhere in the mirrored window by folding to the low nibble.
    let addr_lo = address & 0x000F;
    if addr_hi != LCD_ADDR_HI_LEFT && addr_hi != LCD_ADDR_HI_RIGHT {
        return None;
    }
    let rw = if (addr_lo & 1) == 0 {
        ReadWrite::Write
    } else {
        ReadWrite::Read
    };
    let di = if (addr_lo >> 1) & 1 == 0 {
        DataInstruction::Instruction
    } else {
        DataInstruction::Data
    };
    let cs = match (addr_lo >> 2) & 0b11 {
        0b00 => ChipSelect::Both,
        0b01 => ChipSelect::Right,
        0b10 => ChipSelect::Left,
        0b11 => return None,
        _ => return None,
    };
    Some((cs, di, rw))
}

fn parse_command(address: u32, value: u8) -> Option<LcdCommand> {
    let (cs, di, _rw) = decode_access(address)?;
    let kind = if di == DataInstruction::Instruction {
        let instr = match value >> 6 {
            0b00 => LcdInstruction::OnOff,
            0b01 => LcdInstruction::SetYAddress,
            0b10 => LcdInstruction::SetPage,
            0b11 => LcdInstruction::StartLine,
            _ => return None,
        };
        let mut data = value & 0b0011_1111;
        data = match instr {
            LcdInstruction::OnOff => data & 1,
            LcdInstruction::SetPage => data & 0b0000_0111,
            _ => data,
        };
        CommandKind::Instruction(instr, data)
    } else {
        CommandKind::Data(value)
    };

    Some(LcdCommand { cs, kind })
}

pub struct LcdController {
    chips: [Hd61202Chip; 2],
    cs_both_count: u32,
    cs_left_count: u32,
    cs_right_count: u32,
    last_status: Option<u8>,
    display_write_capture: Option<HashMap<u16, LcdDisplayWrite>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LcdStats {
    pub chip_on: [bool; 2],
    pub instruction_counts: [u32; 2],
    pub data_write_counts: [u32; 2],
    pub cs_both_count: u32,
    pub cs_left_count: u32,
    pub cs_right_count: u32,
}

impl LcdController {
    pub fn kind(&self) -> LcdKind {
        LcdKind::Hd61202
    }

    pub fn new() -> Self {
        Self {
            chips: [Hd61202Chip::default(), Hd61202Chip::default()],
            cs_both_count: 0,
            cs_left_count: 0,
            cs_right_count: 0,
            last_status: None,
            display_write_capture: None,
        }
    }

    pub fn reset(&mut self) {
        self.chips = [Hd61202Chip::default(), Hd61202Chip::default()];
        self.cs_both_count = 0;
        self.cs_left_count = 0;
        self.cs_right_count = 0;
        self.last_status = None;
        self.display_write_capture = None;
    }

    pub fn begin_display_write_capture(&mut self) {
        self.display_write_capture = Some(HashMap::new());
    }

    pub fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite> {
        let Some(map) = self.display_write_capture.take() else {
            return Vec::new();
        };
        let mut out: Vec<LcdDisplayWrite> = map.into_values().collect();
        out.sort_by_key(|ev| ((ev.page as u32) << 16) | (ev.col as u32));
        out
    }

    fn record_display_write_capture(
        &mut self,
        chip_index: usize,
        page: u8,
        chip_col: u8,
        value: u8,
        trace: LcdWriteTrace,
    ) {
        let Some(map) = self.display_write_capture.as_mut() else {
            return;
        };
        let Some(display_col) = map_chip_col_to_display_col(chip_index, page, chip_col) else {
            return;
        };
        let key = ((page as u16) << 8) | (display_col & 0xFF);
        map.insert(
            key,
            LcdDisplayWrite {
                page,
                col: display_col,
                value,
                trace,
            },
        );
    }

    fn chip_indices(cs: ChipSelect) -> &'static [usize] {
        match cs {
            ChipSelect::Both => &[0, 1],
            ChipSelect::Right => &[1],
            ChipSelect::Left => &[0],
        }
    }

    pub fn handles(&self, address: u32) -> bool {
        let addr = address & 0x00FF_FFFF;
        (0x0000_2000..=0x0000_2FFF).contains(&addr) || (0x0000_A000..=0x0000_AFFF).contains(&addr)
    }

    pub fn write(&mut self, address: u32, value: u8) {
        if let Some(command) = parse_command(address, value) {
            let ctx = perfetto_instr_context();
            let op_index = ctx.map(|(idx, _)| idx);
            let pc = ctx.map(|(_, pc)| pc).or(Some(perfetto_last_pc()));
            let call_stack = perfetto_last_call_stack();
            let trace = LcdWriteTrace {
                pc: pc.unwrap_or(0) & mask_for(RegName::PC),
                call_stack,
            };
            match command.cs {
                ChipSelect::Both => self.cs_both_count = self.cs_both_count.wrapping_add(1),
                ChipSelect::Left => self.cs_left_count = self.cs_left_count.wrapping_add(1),
                ChipSelect::Right => self.cs_right_count = self.cs_right_count.wrapping_add(1),
            }
            for idx in Self::chip_indices(command.cs) {
                let chip = &mut self.chips[*idx];
                match command.kind {
                    CommandKind::Instruction(instr, data) => {
                        let column_snapshot = chip.state.y_address;
                        chip.write_instruction(instr, data);
                        let name = match instr {
                            LcdInstruction::OnOff => "LCD_ON_OFF",
                            LcdInstruction::StartLine => "LCD_START_LINE",
                            LcdInstruction::SetPage => "LCD_SET_PAGE",
                            LcdInstruction::SetYAddress => "LCD_SET_Y_ADDRESS",
                        };
                        let mut guard = PERFETTO_TRACER.enter();
                        guard.with_some(|tracer| {
                            tracer.record_lcd_event(
                                name,
                                address & 0x00FF_FFFF,
                                data,
                                *idx,
                                chip.state.page,
                                column_snapshot,
                                pc,
                                op_index,
                            );
                        });
                    }
                    CommandKind::Data(data) => {
                        let page_before = chip.state.page;
                        let column_before = chip.state.y_address;
                        chip.write_data(data, trace);
                        let column = (column_before as usize % LCD_WIDTH) as u8;
                        self.record_display_write_capture(*idx, page_before, column, data, trace);
                        let mut guard = PERFETTO_TRACER.enter();
                        guard.with_some(|tracer| {
                            tracer.record_lcd_event(
                                "VRAM_Write",
                                address & 0x00FF_FFFF,
                                data,
                                *idx,
                                page_before,
                                column,
                                pc,
                                op_index,
                            );
                        });
                    }
                }
            }
        }
    }

    /// Return the last recorded write trace for a chip's VRAM addressing unit (page, column).
    pub fn vram_write_trace(
        &self,
        chip_index: usize,
        page: u8,
        column: u8,
    ) -> Option<LcdWriteTrace> {
        let chip = self.chips.get(chip_index)?;
        let page = (page as usize) % LCD_PAGES;
        let column = (column as usize) % LCD_WIDTH;
        Some(chip.vram_trace[page][column])
    }

    /// Return a display-mapped [page][col] trace buffer (8 pages x 240 columns) that matches
    /// the pixel buffer mapping used by `display_buffer()`, including `START_LINE` scrolling.
    ///
    /// Note: when `START_LINE` is not page-aligned (`start_line % 8 != 0`), a displayed byte is
    /// composed from two underlying VRAM pages. In that case the trace is taken from the source
    /// page that contributes the majority of bits to the displayed byte.
    pub fn display_trace_buffer(&self) -> [[LcdWriteTrace; LCD_DISPLAY_COLS]; LCD_PAGES] {
        let mut out = [[LcdWriteTrace::default(); LCD_DISPLAY_COLS]; LCD_PAGES];
        let left = &self.chips[0];
        let right = &self.chips[1];

        copy_trace_region(&mut out, right, 0, 0..64, 0, false);
        copy_trace_region(&mut out, left, 0, 0..56, 64, false);
        copy_trace_region(&mut out, left, 4, 0..56, 120, true);
        copy_trace_region(&mut out, right, 4, 0..64, 176, true);
        out
    }

    /// Return a display-mapped [page][col] VRAM byte buffer (8 pages x 240 columns) that matches
    /// the chip-mirror layout used by `display_buffer()` and `display_trace_buffer()`, including
    /// `START_LINE` scrolling (bytes are derived when `start_line % 8 != 0`).
    pub fn display_vram_bytes(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_PAGES] {
        let mut out = [[0u8; LCD_DISPLAY_COLS]; LCD_PAGES];
        let left = &self.chips[0];
        let right = &self.chips[1];

        copy_vram_region(&mut out, right, 0, 0..64, 0, false);
        copy_vram_region(&mut out, left, 0, 0..56, 64, false);
        copy_vram_region(&mut out, left, 4, 0..56, 120, true);
        copy_vram_region(&mut out, right, 4, 0..64, 176, true);
        out
    }

    pub fn read(&mut self, address: u32) -> Option<u8> {
        let (_cs, _di, rw) = decode_access(address)?;
        if rw != ReadWrite::Read {
            return None;
        }
        // Parity: Python controller wrapper always returns 0xFF and does not update counters/state.
        Some(0xFF)
    }

    pub fn export_snapshot(&self) -> (Value, Vec<u8>) {
        let mut meta = json!({
            "kind": self.kind(),
            "chip_count": self.chips.len(),
            "pages": LCD_PAGES,
            "width": LCD_WIDTH,
            "chips": [],
            "cs_both_count": self.cs_both_count,
            "cs_left_count": self.cs_left_count,
            "cs_right_count": self.cs_right_count,
        });
        let mut chips_meta = Vec::with_capacity(self.chips.len());
        let mut payload = Vec::with_capacity(self.chips.len() * LCD_PAGES * LCD_WIDTH);
        for chip in &self.chips {
            chips_meta.push(json!({
                "on": chip.state.on,
                "start_line": chip.state.start_line,
                "page": chip.state.page,
                "y_address": chip.state.y_address,
                "instruction_count": chip.instruction_count,
                "data_write_count": chip.data_write_count,
                "data_read_count": chip.data_read_count,
            }));
            for page in &chip.vram {
                for byte in page {
                    payload.push(*byte);
                }
            }
        }
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("chips".to_string(), Value::Array(chips_meta));
        }
        (meta, payload)
    }

    pub fn load_snapshot(&mut self, metadata: &Value, vram: &[u8]) -> Result<(), String> {
        let chip_count = metadata
            .get("chip_count")
            .and_then(|v| v.as_u64())
            .ok_or("lcd meta missing chip_count")?;
        if chip_count != self.chips.len() as u64 {
            return Err("lcd chip_count mismatch".to_string());
        }
        let pages = metadata
            .get("pages")
            .and_then(|v| v.as_u64())
            .ok_or("lcd meta missing pages")?;
        let width = metadata
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or("lcd meta missing width")?;
        if pages as usize != LCD_PAGES || width as usize != LCD_WIDTH {
            return Err("lcd geometry mismatch".to_string());
        }
        if vram.len() != self.chips.len() * LCD_PAGES * LCD_WIDTH {
            return Err("lcd vram size mismatch".to_string());
        }
        for (idx, chip) in self.chips.iter_mut().enumerate() {
            if let Some(chips) = metadata.get("chips").and_then(|v| v.as_array()) {
                if let Some(meta) = chips.get(idx) {
                    chip.state.on = meta.get("on").and_then(|v| v.as_bool()).unwrap_or(false);
                    chip.state.start_line =
                        meta.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
                    chip.state.page = meta.get("page").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
                    chip.state.y_address =
                        meta.get("y_address").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
                    chip.instruction_count = meta
                        .get("instruction_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    chip.data_write_count = meta
                        .get("data_write_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    chip.data_read_count = meta
                        .get("data_read_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                }
            }
            let start = idx * LCD_PAGES * LCD_WIDTH;
            let end = start + LCD_PAGES * LCD_WIDTH;
            let slice = &vram[start..end];
            for page in 0..LCD_PAGES {
                let base = page * LCD_WIDTH;
                chip.vram[page].copy_from_slice(&slice[base..base + LCD_WIDTH]);
            }
        }
        self.cs_both_count = metadata
            .get("cs_both_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        self.cs_left_count = metadata
            .get("cs_left_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        self.cs_right_count = metadata
            .get("cs_right_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        Ok(())
    }

    pub fn read_placeholder(&self, address: u32) -> u32 {
        if let Some((_cs, _di, rw)) = decode_access(address) {
            if rw == ReadWrite::Read {
                // Mirror Python wrapper: reads are not emulated; always return 0xFF.
                return 0xFF;
            }
        }
        0
    }

    /// Return a 32x240 buffer of pixel-on (1) / pixel-off (0) values.
    /// Mirrors the layout from the Python `get_display_buffer` helper so the
    /// text decoder code can share the same assumptions.
    pub fn display_buffer(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS] {
        let mut buffer = [[0u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS];
        let left = &self.chips[0];
        let right = &self.chips[1];
        copy_region(&mut buffer, right, 0, 0..64, 0, false);
        copy_region(&mut buffer, left, 0, 0..56, 64, false);
        copy_region(&mut buffer, left, 4, 0..56, 120, true);
        copy_region(&mut buffer, right, 4, 0..64, 176, true);
        buffer
    }

    pub fn chip_display_buffer(&self, chip_index: usize) -> [[u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS] {
        let mut out = [[0u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS];
        let Some(chip) = self.chips.get(chip_index) else {
            return out;
        };
        let start_line = (chip.state.start_line as usize) % LCD_CHIP_ROWS;
        for (y_display, row_out) in out.iter_mut().enumerate().take(LCD_CHIP_ROWS) {
            let y_vram = (y_display + start_line) % LCD_CHIP_ROWS;
            let page = y_vram / 8;
            let bit = y_vram % 8;
            for (x, pixel) in row_out.iter_mut().enumerate().take(LCD_CHIP_COLS) {
                *pixel = pixel_on(chip.vram[page][x], bit);
            }
        }
        out
    }

    pub fn stats(&self) -> LcdStats {
        let mut stats = LcdStats::default();
        for (idx, chip) in self.chips.iter().enumerate() {
            stats.chip_on[idx] = chip.state.on;
            stats.instruction_counts[idx] = chip.instruction_count;
            stats.data_write_counts[idx] = chip.data_write_count;
        }
        stats.cs_both_count = self.cs_both_count;
        stats.cs_left_count = self.cs_left_count;
        stats.cs_right_count = self.cs_right_count;
        stats
    }
}

impl LcdHal for LcdController {
    fn kind(&self) -> LcdKind {
        self.kind()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn handles(&self, address: u32) -> bool {
        self.handles(address)
    }

    fn read(&mut self, address: u32) -> Option<u8> {
        self.read(address)
    }

    fn write(&mut self, address: u32, value: u8) {
        self.write(address, value)
    }

    fn read_placeholder(&self, address: u32) -> u32 {
        self.read_placeholder(address)
    }

    fn begin_display_write_capture(&mut self) {
        self.begin_display_write_capture();
    }

    fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite> {
        self.take_display_write_capture()
    }

    fn display_buffer(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS] {
        self.display_buffer()
    }

    fn chip_display_buffer(&self, chip_index: usize) -> [[u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS] {
        self.chip_display_buffer(chip_index)
    }

    fn display_vram_bytes(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_PAGES] {
        self.display_vram_bytes()
    }

    fn display_trace_buffer(&self) -> [[LcdWriteTrace; LCD_DISPLAY_COLS]; LCD_PAGES] {
        self.display_trace_buffer()
    }

    fn stats(&self) -> LcdStats {
        self.stats()
    }

    fn export_snapshot(&self) -> (Value, Vec<u8>) {
        self.export_snapshot()
    }

    fn load_snapshot(&mut self, metadata: &Value, payload: &[u8]) -> Result<(), String> {
        self.load_snapshot(metadata, payload)
    }
}

impl Default for UnknownLcdController {
    fn default() -> Self {
        Self::new()
    }
}

impl LcdHal for UnknownLcdController {
    fn kind(&self) -> LcdKind {
        LcdKind::Unknown
    }

    fn reset(&mut self) {
        self.write_count = 0;
    }

    fn handles(&self, address: u32) -> bool {
        let addr = address & 0x00FF_FFFF;
        (0x0000_2000..=0x0000_2FFF).contains(&addr) || (0x0000_A000..=0x0000_AFFF).contains(&addr)
    }

    fn read(&mut self, address: u32) -> Option<u8> {
        if !self.handles(address) {
            return None;
        }
        if (address & 1) == 1 {
            // Mirror the existing PC-E500 LCD wrapper behaviour: reads return 0xFF.
            return Some(0xFF);
        }
        None
    }

    fn write(&mut self, address: u32, _value: u8) {
        if !self.handles(address) {
            return;
        }
        if (address & 1) == 0 {
            self.write_count = self.write_count.wrapping_add(1);
        }
    }

    fn read_placeholder(&self, address: u32) -> u32 {
        if !self.handles(address) {
            return 0;
        }
        if (address & 1) == 1 {
            return 0xFF;
        }
        0
    }

    fn begin_display_write_capture(&mut self) {}

    fn take_display_write_capture(&mut self) -> Vec<LcdDisplayWrite> {
        Vec::new()
    }

    fn display_buffer(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS] {
        [[0u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS]
    }

    fn chip_display_buffer(&self, _chip_index: usize) -> [[u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS] {
        [[0u8; LCD_CHIP_COLS]; LCD_CHIP_ROWS]
    }

    fn display_vram_bytes(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_PAGES] {
        [[0u8; LCD_DISPLAY_COLS]; LCD_PAGES]
    }

    fn display_trace_buffer(&self) -> [[LcdWriteTrace; LCD_DISPLAY_COLS]; LCD_PAGES] {
        [[LcdWriteTrace::default(); LCD_DISPLAY_COLS]; LCD_PAGES]
    }

    fn stats(&self) -> LcdStats {
        LcdStats::default()
    }

    fn export_snapshot(&self) -> (Value, Vec<u8>) {
        (json!({"kind": self.kind()}), Vec::new())
    }

    fn load_snapshot(&mut self, metadata: &Value, payload: &[u8]) -> Result<(), String> {
        let kind = lcd_kind_from_snapshot_meta(metadata, LcdKind::Unknown);
        if kind != self.kind() {
            return Err("lcd kind mismatch".to_string());
        }
        if !payload.is_empty() {
            return Err("unknown lcd does not accept payload".to_string());
        }
        Ok(())
    }
}

fn copy_region(
    buffer: &mut [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS],
    chip: &Hd61202Chip,
    start_page: usize,
    column_range: std::ops::Range<usize>,
    dest_start_col: usize,
    mirror: bool,
) {
    let start_line = (chip.state.start_line as usize) % LCD_CHIP_ROWS;
    for (row, row_buf) in buffer.iter_mut().enumerate().take(LCD_DISPLAY_ROWS) {
        let y_display = start_page * 8 + row;
        let y_vram = (y_display + start_line) % LCD_CHIP_ROWS;
        let page = y_vram / 8;
        let bit = y_vram % 8;
        if mirror {
            for (dest_offset, src_col) in column_range.clone().rev().enumerate() {
                if let Some(byte) = chip.vram.get(page).and_then(|page| page.get(src_col)) {
                    row_buf[dest_start_col + dest_offset] = pixel_on(*byte, bit);
                }
            }
        } else {
            for (dest_offset, src_col) in column_range.clone().enumerate() {
                if let Some(byte) = chip.vram.get(page).and_then(|page| page.get(src_col)) {
                    row_buf[dest_start_col + dest_offset] = pixel_on(*byte, bit);
                }
            }
        }
    }
}

fn copy_trace_region(
    buffer: &mut [[LcdWriteTrace; LCD_DISPLAY_COLS]; LCD_PAGES],
    chip: &Hd61202Chip,
    start_page: usize,
    column_range: std::ops::Range<usize>,
    dest_start_col: usize,
    mirror: bool,
) {
    let start_line = (chip.state.start_line as usize) % LCD_CHIP_ROWS;
    let page_shift = start_line / 8;
    let bit_shift = start_line % 8;
    for page_offset in 0..(LCD_DISPLAY_ROWS / 8) {
        let out_page = start_page + page_offset;
        if out_page >= LCD_PAGES {
            continue;
        }
        let src_page0 = (out_page + page_shift) % LCD_PAGES;
        let src_page1 = (src_page0 + 1) % LCD_PAGES;
        let src_page = if bit_shift == 0 || bit_shift <= 4 {
            src_page0
        } else {
            src_page1
        };
        if mirror {
            for (dest_offset, src_col) in column_range.clone().rev().enumerate() {
                buffer[out_page][dest_start_col + dest_offset] = chip.vram_trace[src_page][src_col];
            }
        } else {
            for (dest_offset, src_col) in column_range.clone().enumerate() {
                buffer[out_page][dest_start_col + dest_offset] = chip.vram_trace[src_page][src_col];
            }
        }
    }
}

fn copy_vram_region(
    buffer: &mut [[u8; LCD_DISPLAY_COLS]; LCD_PAGES],
    chip: &Hd61202Chip,
    start_page: usize,
    column_range: std::ops::Range<usize>,
    dest_start_col: usize,
    mirror: bool,
) {
    let start_line = (chip.state.start_line as usize) % LCD_CHIP_ROWS;
    let page_shift = start_line / 8;
    let bit_shift = start_line % 8;
    let shift = bit_shift as u32;
    for page_offset in 0..(LCD_DISPLAY_ROWS / 8) {
        let out_page = start_page + page_offset;
        if out_page >= LCD_PAGES {
            continue;
        }
        let src_page0 = (out_page + page_shift) % LCD_PAGES;
        let src_page1 = (src_page0 + 1) % LCD_PAGES;
        if mirror {
            for (dest_offset, src_col) in column_range.clone().rev().enumerate() {
                let byte0 = chip.vram[src_page0][src_col];
                buffer[out_page][dest_start_col + dest_offset] = if bit_shift == 0 {
                    byte0
                } else {
                    let byte1 = chip.vram[src_page1][src_col];
                    (byte0 >> shift) | (byte1 << (8 - shift))
                };
            }
        } else {
            for (dest_offset, src_col) in column_range.clone().enumerate() {
                let byte0 = chip.vram[src_page0][src_col];
                buffer[out_page][dest_start_col + dest_offset] = if bit_shift == 0 {
                    byte0
                } else {
                    let byte1 = chip.vram[src_page1][src_col];
                    (byte0 >> shift) | (byte1 << (8 - shift))
                };
            }
        }
    }
}

fn map_chip_col_to_display_col(chip_index: usize, page: u8, chip_col: u8) -> Option<u16> {
    let page = (page as usize) % LCD_PAGES;
    let chip_col = (chip_col as usize) % LCD_WIDTH;
    if page < 4 {
        match chip_index {
            1 => Some(chip_col as u16),
            0 if chip_col < 56 => Some((64 + chip_col) as u16),
            _ => None,
        }
    } else {
        match chip_index {
            0 if chip_col < 56 => Some((120 + (55 - chip_col)) as u16),
            1 => Some((176 + (63 - chip_col)) as u16),
            _ => None,
        }
    }
}

impl Default for LcdController {
    fn default() -> Self {
        Self::new()
    }
}

fn pixel_on(byte: u8, bit: usize) -> u8 {
    // Match Python helper: pixels are lit when the stored bit is 0.
    if ((byte >> bit) & 1) == 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_read_returns_on_flag_and_clears_busy() {
        let mut chip = Hd61202Chip::default();
        chip.state.on = true;
        chip.state.busy = true;
        let status = chip.read_status();
        assert_eq!(status & 0x40, 0x40);
        assert!(!chip.state.busy);
    }

    #[test]
    fn status_read_when_off_reports_ready_zero() {
        let mut chip = Hd61202Chip::default();
        chip.state.on = false;
        chip.state.busy = true;
        let status = chip.read_status();
        assert_eq!(status, 0x00);
        assert!(!chip.state.busy);
    }

    #[test]
    fn data_read_advances_y_address() {
        let mut chip = Hd61202Chip::default();
        chip.state.page = 0;
        chip.state.y_address = 0;
        chip.vram[0][0] = 0xAA;
        chip.vram[0][1] = 0xBB;
        let first = chip.read_data();
        let second = chip.read_data();
        assert_eq!(first, 0xAA);
        assert_eq!(second, 0xBB);
        assert_eq!(chip.state.y_address, 2);
    }

    #[test]
    fn pixel_on_matches_python_polarity() {
        // Parity with Python: cleared bits are lit, set bits are off.
        assert_eq!(super::pixel_on(0b0000_0001, 0), 0);
        assert_eq!(super::pixel_on(0b0000_0000, 0), 1);
        assert_eq!(super::pixel_on(0b1000_0000, 7), 0);
        assert_eq!(super::pixel_on(0b0111_1111, 7), 1);
    }

    #[test]
    fn handles_mirrors_match_python() {
        let mut lcd = LcdController::new();
        assert!(lcd.handles(0x2000));
        assert!(lcd.handles(0x200F));
        assert!(
            lcd.handles(0x2010),
            "low mirror spans 0x2000-0x2FFF like Python"
        );
        assert!(lcd.handles(0xA000));
        assert!(lcd.handles(0xAFFF));

        // Write ON instruction to right chip via the high mirror (CS=Right).
        lcd.write(0xA004, 0x3F);
        assert!(lcd.chips[1].state.on);
    }

    #[test]
    fn high_offset_addresses_map_to_low_nibble() {
        // Addresses like 0x2100 should decode the same as 0x2000 mirror.
        let mut lcd = LcdController::new();
        // Set Y address to 0 via high offset instruction address (bit1=0 => instruction).
        lcd.write(0x2100, 0x40); // SetYAddress=0
                                 // Write data using high offset data address (bit1=1).
        lcd.write(0x2102, 0xAA);
        // Verify write landed in VRAM for both chips at y=0 page 0.
        assert_eq!(lcd.chips[0].vram[0][0], 0xAA);
        assert_eq!(lcd.chips[1].vram[0][0], 0xAA);
    }

    #[test]
    fn display_write_capture_records_display_mapped_coordinates() {
        let mut lcd = LcdController::new();
        lcd.begin_display_write_capture();

        // Left chip instruction port: 0x2000 + cs(Left=0b10)<<2 + di(Instruction=0)<<1 + rw(Write=0)
        let left_instr = 0x2008;
        // Left chip data port: same but di(Data=1)
        let left_data = 0x200A;

        // Set page=1, y=0, then write one byte.
        lcd.write(left_instr, 0x80 | 0x01);
        lcd.write(left_instr, 0x40);
        lcd.write(left_data, 0xAA);

        let events = lcd.take_display_write_capture();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].page, 1);
        // For page<4: left chip columns 0..55 map to display columns 64..119.
        assert_eq!(events[0].col, 64);
        assert_eq!(events[0].value, 0xAA);
    }

    #[test]
    fn display_write_capture_keeps_only_last_value_per_addressing_point() {
        let mut lcd = LcdController::new();
        lcd.begin_display_write_capture();

        let left_instr = 0x2008;
        let left_data = 0x200A;

        lcd.write(left_instr, 0x80 | 0x01);
        lcd.write(left_instr, 0x40);
        lcd.write(left_data, 0x11);
        // Rewind y back to 0 and overwrite.
        lcd.write(left_instr, 0x40);
        lcd.write(left_data, 0x22);

        let events = lcd.take_display_write_capture();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].value, 0x22);
    }

    #[test]
    fn chip_display_buffer_scrolls_by_start_line() {
        let mut lcd = LcdController::new();
        for chip in lcd.chips.iter_mut() {
            for page in chip.vram.iter_mut() {
                for byte in page.iter_mut() {
                    *byte = 0xFF;
                }
            }
        }
        // Light a single pixel at VRAM row 0, col 0 on the right chip (bit 0 cleared => on).
        lcd.chips[1].vram[0][0] = 0xFE;

        lcd.chips[1].state.start_line = 0;
        let buf0 = lcd.chip_display_buffer(1);
        assert_eq!(buf0[0][0], 1);
        assert_eq!(buf0[63][0], 0);

        lcd.chips[1].state.start_line = 1;
        let buf1 = lcd.chip_display_buffer(1);
        assert_eq!(buf1[0][0], 0);
        assert_eq!(buf1[63][0], 1);
    }

    #[test]
    fn display_buffer_applies_start_line_rotation() {
        let mut lcd = LcdController::new();
        for chip in lcd.chips.iter_mut() {
            for page in chip.vram.iter_mut() {
                for byte in page.iter_mut() {
                    *byte = 0xFF;
                }
            }
        }
        lcd.chips[1].vram[0][0] = 0xFE;

        lcd.chips[1].state.start_line = 0;
        let buf0 = lcd.display_buffer();
        assert_eq!(buf0[0][0], 1);

        lcd.chips[1].state.start_line = 1;
        let buf1 = lcd.display_buffer();
        assert_eq!(buf1[0][0], 0);
        // Chip display row 63 maps to the mirrored right-half segment at row 31, col 239.
        assert_eq!(buf1[31][239], 1);
    }

    #[test]
    fn display_vram_bytes_applies_start_line_page_shift() {
        let mut lcd = LcdController::new();
        lcd.chips[1].vram[0][0] = 0x11;
        lcd.chips[1].vram[1][0] = 0x22;
        lcd.chips[1].vram[4][0] = 0x33;

        lcd.chips[1].state.start_line = 8;
        let vram = lcd.display_vram_bytes();
        assert_eq!(vram[0][0], 0x22);
        assert_eq!(vram[3][0], 0x33);
    }

    #[test]
    fn display_vram_bytes_applies_start_line_bit_shift() {
        let mut lcd = LcdController::new();
        lcd.chips[1].vram[0][0] = 0xAA;
        lcd.chips[1].vram[1][0] = 0x55;

        lcd.chips[1].state.start_line = 1;
        let vram = lcd.display_vram_bytes();
        let expected = (0xAAu8 >> 1) | (0x55u8 << 7);
        assert_eq!(vram[0][0], expected);
    }

    #[test]
    fn display_vram_bytes_mirrored_region_includes_start_line() {
        let mut lcd = LcdController::new();
        lcd.chips[1].vram[4][0] = 0xF0;
        lcd.chips[1].vram[5][0] = 0x0F;

        lcd.chips[1].state.start_line = 1;
        let vram = lcd.display_vram_bytes();
        let expected = (0xF0u8 >> 1) | (0x0Fu8 << 7);
        assert_eq!(vram[4][239], expected);
    }

    #[test]
    fn display_vram_bytes_stay_in_sync_with_display_buffer() {
        let mut lcd = LcdController::new();
        for (chip_idx, chip) in lcd.chips.iter_mut().enumerate() {
            for (page_idx, page) in chip.vram.iter_mut().enumerate() {
                for (col_idx, byte) in page.iter_mut().enumerate() {
                    *byte = (chip_idx as u8)
                        .wrapping_mul(0x40)
                        .wrapping_add((page_idx as u8).wrapping_mul(0x11))
                        .wrapping_add(col_idx as u8);
                }
            }
        }
        lcd.chips[0].state.start_line = 13;
        lcd.chips[1].state.start_line = 37;

        let vram = lcd.display_vram_bytes();
        let pixels = lcd.display_buffer();
        for (row, row_pixels) in pixels.iter().enumerate().take(LCD_DISPLAY_ROWS) {
            let page_left = row / 8;
            let page_right = 4 + row / 8;
            let bit = row % 8;
            for (col, &pixel) in row_pixels.iter().enumerate().take(LCD_DISPLAY_COLS) {
                let page = if col < 120 { page_left } else { page_right };
                assert_eq!(
                    pixel,
                    super::pixel_on(vram[page][col], bit),
                    "row {row} col {col}"
                );
            }
        }
    }

    #[test]
    fn display_trace_buffer_applies_start_line_page_shift() {
        let mut lcd = LcdController::new();
        lcd.chips[1].vram_trace[1][0] = LcdWriteTrace {
            pc: 0x1234,
            ..Default::default()
        };

        lcd.chips[1].state.start_line = 8;
        let trace = lcd.display_trace_buffer();
        assert_eq!(trace[0][0].pc, 0x1234);
    }

    #[test]
    fn display_trace_buffer_selects_majority_source_page_when_bit_shifted() {
        let mut lcd = LcdController::new();
        lcd.chips[1].vram_trace[0][0] = LcdWriteTrace {
            pc: 0x1111,
            ..Default::default()
        };
        lcd.chips[1].vram_trace[1][0] = LcdWriteTrace {
            pc: 0x2222,
            ..Default::default()
        };

        lcd.chips[1].state.start_line = 1;
        let trace0 = lcd.display_trace_buffer();
        assert_eq!(trace0[0][0].pc, 0x1111);

        lcd.chips[1].state.start_line = 7;
        let trace1 = lcd.display_trace_buffer();
        assert_eq!(trace1[0][0].pc, 0x2222);
    }
}
