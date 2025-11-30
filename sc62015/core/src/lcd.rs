use serde_json::{json, Value};
use std::env;

const LCD_WIDTH: usize = 64;
const LCD_PAGES: usize = 8;
#[allow(dead_code)]
const LCD_RANGE_LOW: u32 = 0x2000;
#[allow(dead_code)]
const LCD_RANGE_HIGH: u32 = 0xA000;
pub const LCD_DISPLAY_ROWS: usize = 32;
pub const LCD_DISPLAY_COLS: usize = 240;

// The LCD controller is mirrored at 0x2000 and 0xA000. Bits 0-3 encode
// R/W, DI, and CS the same way the Python hd61202 decoder does:
//  bit0: 0=write, 1=read
//  bit1: 0=instruction, 1=data
//  bit2-3: chip select (00=both, 01=right/CS2, 10=left/CS1, 11=none)
const LCD_ADDR_HI_LEFT: u32 = 0x0A000;
const LCD_ADDR_HI_RIGHT: u32 = 0x02000;

#[derive(Clone, Copy, Default)]
struct Hd61202State {
    on: bool,
    start_line: u8,
    page: u8,
    y_address: u8,
}

struct Hd61202Chip {
    state: Hd61202State,
    vram: [[u8; LCD_WIDTH]; LCD_PAGES],
    instruction_count: u32,
    data_write_count: u32,
    data_read_count: u32,
}

impl Default for Hd61202Chip {
    fn default() -> Self {
        Self {
            state: Hd61202State::default(),
            vram: [[0; LCD_WIDTH]; LCD_PAGES],
            instruction_count: 0,
            data_write_count: 0,
            data_read_count: 0,
        }
    }
}

impl Hd61202Chip {
    fn write_instruction(&mut self, instr: LcdInstruction, data: u8) {
        self.instruction_count = self.instruction_count.wrapping_add(1);
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

    fn write_data(&mut self, data: u8) {
        self.data_write_count = self.data_write_count.wrapping_add(1);
        let page = (self.state.page as usize) % LCD_PAGES;
        let y = (self.state.y_address as usize) % LCD_WIDTH;
        self.vram[page][y] = data;
        self.state.y_address = ((self.state.y_address as usize + 1) % LCD_WIDTH) as u8;
    }

    #[allow(dead_code)]
    fn read_status(&mut self) -> u8 {
        0xFF
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
    if addr_hi != LCD_ADDR_HI_LEFT && addr_hi != LCD_ADDR_HI_RIGHT {
        return None;
    }
    let addr_lo = address & 0x0FFF;
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
    let (cs, di, rw) = decode_access(address)?;
    if rw != ReadWrite::Write {
        return None;
    }

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
    pub fn new() -> Self {
        Self {
            chips: [Hd61202Chip::default(), Hd61202Chip::default()],
            cs_both_count: 0,
            cs_left_count: 0,
            cs_right_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.chips = [Hd61202Chip::default(), Hd61202Chip::default()];
        self.cs_both_count = 0;
        self.cs_left_count = 0;
        self.cs_right_count = 0;
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
        (0x0000_2000..=0x0000_200F).contains(&addr) || (0x0000_A000..=0x0000_AFFF).contains(&addr)
    }

    pub fn write(&mut self, address: u32, value: u8) {
        if let Some(command) = parse_command(address, value) {
            if env::var("RUST_LCD_DEBUG").is_ok() {
                println!("[rust-lcd-device] addr=0x{address:05X} value=0x{value:02X}");
            }
            match command.cs {
                ChipSelect::Both => self.cs_both_count = self.cs_both_count.wrapping_add(1),
                ChipSelect::Left => self.cs_left_count = self.cs_left_count.wrapping_add(1),
                ChipSelect::Right => self.cs_right_count = self.cs_right_count.wrapping_add(1),
            }
            for idx in Self::chip_indices(command.cs) {
                let chip = &mut self.chips[*idx];
                match command.kind {
                    CommandKind::Instruction(instr, data) => chip.write_instruction(instr, data),
                    CommandKind::Data(data) => chip.write_data(data),
                }
            }
        }
    }

    pub fn export_snapshot(&self) -> (Value, Vec<u8>) {
        let mut meta = json!({
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

    pub fn read(&self, address: u32) -> u32 {
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
        if right.state.on {
            copy_region(&mut buffer, right, 0, 0..64, 0, false);
        }
        if left.state.on {
            copy_region(&mut buffer, left, 0, 0..56, 64, false);
            copy_region(&mut buffer, left, 4, 0..56, 120, true);
        }
        if right.state.on {
            copy_region(&mut buffer, right, 4, 0..64, 176, true);
        }
        buffer
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

fn copy_region(
    buffer: &mut [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS],
    chip: &Hd61202Chip,
    start_page: usize,
    column_range: std::ops::Range<usize>,
    dest_start_col: usize,
    mirror: bool,
) {
    for (row, row_buf) in buffer.iter_mut().enumerate().take(LCD_DISPLAY_ROWS) {
        let page = start_page + row / 8;
        let bit = row % 8;
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

impl Default for LcdController {
    fn default() -> Self {
        Self::new()
    }
}

fn pixel_on(byte: u8, bit: usize) -> u8 {
    // Match Python helper: return 1 for lit pixels, 0 for off.
    if ((byte >> bit) & 1) == 0 {
        1
    } else {
        0
    }
}
