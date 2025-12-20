// PY_SOURCE: pce500/display/text_decoder.py:decode_display_text
// PY_SOURCE: pce500/display/font.py

use crate::lcd::{LcdHal, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
use std::collections::HashMap;

const GLYPH_WIDTH: usize = 5;
const GLYPH_STRIDE: usize = 6; // five data columns + spacer
const GLYPH_COUNT: usize = 96; // ASCII 0x20-0x7F
const ROWS_PER_CELL: usize = 8;
const COLS_PER_CELL: usize = 6;

#[derive(Debug, Default, Clone)]
pub struct Pce500FontMap {
    glyphs: HashMap<[u8; GLYPH_WIDTH], char>,
}

impl Pce500FontMap {
    pub fn from_rom(rom: &[u8], font_base_addr: u32, rom_window_start: u32) -> Self {
        let Some(base) = font_base_offset(rom, font_base_addr, rom_window_start) else {
            return Self::default();
        };

        let mut glyphs = HashMap::new();
        for index in 0..GLYPH_COUNT {
            let start = base + index * GLYPH_STRIDE;
            if start + GLYPH_WIDTH > rom.len() {
                break;
            }
            let mut pattern = [0u8; GLYPH_WIDTH];
            pattern.copy_from_slice(&rom[start..start + GLYPH_WIDTH]);
            for byte in &mut pattern {
                *byte &= 0x7F;
            }

            let codepoint = 0x20 + index as u32;
            if let Some(ch) = char::from_u32(codepoint) {
                glyphs.insert(pattern, ch);
                // Match Python decoder: accept inverted glyphs to tolerate polarity differences in
                // the LCD buffer (0/1 for pixel on).
                let mut inverted = [0u8; GLYPH_WIDTH];
                for (dest, src) in inverted.iter_mut().zip(pattern) {
                    *dest = (!src) & 0x7F;
                }
                glyphs.entry(inverted).or_insert(ch);
            }
        }
        Self { glyphs }
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    fn resolve(&self, pattern: &[u8; GLYPH_WIDTH]) -> char {
        *self.glyphs.get(pattern).unwrap_or(&'?')
    }
}

fn font_base_offset(rom: &[u8], font_base_addr: u32, rom_window_start: u32) -> Option<usize> {
    let base = usize::try_from(font_base_addr).ok()?;
    if base < rom.len() {
        return Some(base);
    }
    let window_base = font_base_addr.checked_sub(rom_window_start)?;
    let window_base = usize::try_from(window_base).ok()?;
    if window_base < rom.len() {
        return Some(window_base);
    }
    None
}

pub fn decode_display_text(lcd: &dyn LcdHal, font: &Pce500FontMap) -> Vec<String> {
    let buffer = lcd.display_buffer();
    let char_rows = LCD_DISPLAY_ROWS / ROWS_PER_CELL;
    let char_cols = LCD_DISPLAY_COLS / COLS_PER_CELL;
    let mut lines = Vec::with_capacity(char_rows);

    for page in 0..char_rows {
        let row_base = page * ROWS_PER_CELL;
        let mut row_chars = Vec::with_capacity(char_cols);
        for char_index in 0..char_cols {
            let col_base = char_index * COLS_PER_CELL;
            let mut pattern = [0u8; GLYPH_WIDTH];
            for (glyph_col, pattern_col) in pattern.iter_mut().enumerate() {
                let mut bits = 0u8;
                let col = col_base + glyph_col;
                for dy in 0..ROWS_PER_CELL {
                    // Match Python text decoder: treat 0 as lit pixel when building columns.
                    if buffer[row_base + dy][col] == 0 {
                        bits |= 1 << dy;
                    }
                }
                *pattern_col = bits & 0x7F;
            }
            row_chars.push(font.resolve(&pattern));
        }
        lines.push(row_chars.iter().collect::<String>().trim_end().to_string());
    }
    lines
}
