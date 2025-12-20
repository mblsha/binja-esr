// PY_SOURCE: pce500/display/text_decoder.py:decode_display_text
// PY_SOURCE: pce500/display/font.py

use crate::lcd::{LcdHal, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
use std::collections::HashMap;

const GLYPH_WIDTH: usize = 5;
const GLYPH_STRIDE: usize = 6; // five data columns + spacer
const GLYPH_COUNT: usize = 96; // ASCII 0x20-0x7F
const ROWS_PER_CELL: usize = 8;
const COLS_PER_CELL: usize = 6;

const IQ7000_CELL_BYTES: usize = 6;
const IQ7000_TEXT_COLS: usize = 16;
const IQ7000_TEXT_ROWS: usize = 8;

const IQ7000_LARGE_CELL_HALF_BYTES: usize = 8;
const IQ7000_LARGE_CELL_BYTES: usize = IQ7000_LARGE_CELL_HALF_BYTES * 2;
const IQ7000_LARGE_TEXT_COLS: usize = 12;
const IQ7000_LARGE_TEXT_ROWS: usize = 4;

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

#[derive(Debug, Default, Clone)]
pub struct Iq7000FontMap {
    glyphs: HashMap<[u8; IQ7000_CELL_BYTES], char>,
}

impl Iq7000FontMap {
    pub fn from_rom(rom: &[u8], font_base_addr: u32) -> Self {
        let base = usize::try_from(font_base_addr).ok().unwrap_or(0);
        if base >= rom.len() {
            return Self::default();
        }
        let mut glyphs = HashMap::new();
        // IQ-7000 uses a 6-byte column glyph table indexed directly by ASCII code.
        for codepoint in 0x20u32..0x80u32 {
            let start = base + (codepoint as usize) * IQ7000_CELL_BYTES;
            if start + IQ7000_CELL_BYTES > rom.len() {
                break;
            }
            let mut pattern = [0u8; IQ7000_CELL_BYTES];
            pattern.copy_from_slice(&rom[start..start + IQ7000_CELL_BYTES]);
            if let Some(ch) = char::from_u32(codepoint) {
                glyphs.entry(pattern).or_insert(ch);
                let mut inverted = [0u8; IQ7000_CELL_BYTES];
                for (dst, src_byte) in inverted.iter_mut().zip(pattern) {
                    *dst = !src_byte;
                }
                glyphs.entry(inverted).or_insert(ch);
            }
        }
        Self { glyphs }
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    fn resolve(&self, pattern: &[u8; IQ7000_CELL_BYTES]) -> char {
        *self.glyphs.get(pattern).unwrap_or(&'?')
    }
}

#[derive(Debug, Default, Clone)]
pub struct Iq7000LargeFontMap {
    glyphs: HashMap<[u8; IQ7000_LARGE_CELL_BYTES], char>,
}

impl Iq7000LargeFontMap {
    pub fn from_rom(rom: &[u8], font_base_addr: u32) -> Self {
        let base = usize::try_from(font_base_addr).ok().unwrap_or(0);
        if base >= rom.len() {
            return Self::default();
        }
        let mut glyphs = HashMap::new();
        // Large IQ-7000 glyphs are stored as two 8-byte halves (top then bottom), 16 bytes per codepoint.
        for codepoint in 0x20u32..0x80u32 {
            let start = base + (codepoint as usize) * IQ7000_LARGE_CELL_BYTES;
            if start + IQ7000_LARGE_CELL_BYTES > rom.len() {
                break;
            }
            let mut pattern = [0u8; IQ7000_LARGE_CELL_BYTES];
            pattern.copy_from_slice(&rom[start..start + IQ7000_LARGE_CELL_BYTES]);
            if let Some(ch) = char::from_u32(codepoint) {
                glyphs.entry(pattern).or_insert(ch);
                let mut inverted = [0u8; IQ7000_LARGE_CELL_BYTES];
                for (dst, src_byte) in inverted.iter_mut().zip(pattern) {
                    *dst = !src_byte;
                }
                glyphs.entry(inverted).or_insert(ch);
            }
        }
        Self { glyphs }
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    fn resolve(&self, pattern: &[u8; IQ7000_LARGE_CELL_BYTES]) -> char {
        *self.glyphs.get(pattern).unwrap_or(&'?')
    }
}

pub fn decode_iq7000_display_text(lcd: &dyn LcdHal, font: &Iq7000FontMap) -> Vec<String> {
    let bytes = lcd.display_vram_bytes();
    let mut out: Vec<String> = (0..IQ7000_TEXT_ROWS)
        .map(|row| decode_iq7000_small_row(&bytes, row, font))
        .collect();
    trim_trailing_empty_lines(&mut out);
    out
}

pub fn decode_iq7000_large_display_text(
    lcd: &dyn LcdHal,
    font: &Iq7000LargeFontMap,
) -> Vec<String> {
    let bytes = lcd.display_vram_bytes();
    let mut out: Vec<String> = (0..IQ7000_LARGE_TEXT_ROWS)
        .map(|row| decode_iq7000_large_row(&bytes, row, font))
        .collect();
    trim_trailing_empty_lines(&mut out);
    out
}

pub fn decode_iq7000_display_text_auto(
    lcd: &dyn LcdHal,
    small_font: &Iq7000FontMap,
    large_font: &Iq7000LargeFontMap,
) -> Vec<String> {
    let bytes = lcd.display_vram_bytes();
    let mut out = Vec::with_capacity(IQ7000_TEXT_ROWS);

    for row_pair in 0..IQ7000_LARGE_TEXT_ROWS {
        let small_top = decode_iq7000_small_row(&bytes, row_pair * 2, small_font);
        let small_bottom = decode_iq7000_small_row(&bytes, row_pair * 2 + 1, small_font);
        let large = decode_iq7000_large_row(&bytes, row_pair, large_font);

        let small_score = score_text(&small_top).merge(score_text(&small_bottom));
        let large_score = score_text(&large);

        if prefer_iq7000_large_row(small_score, large_score) {
            out.push(large);
            out.push(String::new());
        } else {
            out.push(small_top);
            out.push(small_bottom);
        }
    }

    trim_trailing_empty_lines(&mut out);
    out
}

fn decode_iq7000_small_row(
    bytes: &[[u8; LCD_DISPLAY_COLS]; 8],
    row: usize,
    font: &Iq7000FontMap,
) -> String {
    if row >= bytes.len() {
        return String::new();
    }

    let mut line = String::with_capacity(IQ7000_TEXT_COLS);
    for col in 0..IQ7000_TEXT_COLS {
        let start = col * IQ7000_CELL_BYTES;
        let mut pattern = [0u8; IQ7000_CELL_BYTES];
        pattern.copy_from_slice(&bytes[row][start..start + IQ7000_CELL_BYTES]);
        line.push(font.resolve(&pattern));
    }
    line.trim_end().to_string()
}

fn decode_iq7000_large_row(
    bytes: &[[u8; LCD_DISPLAY_COLS]; 8],
    row: usize,
    font: &Iq7000LargeFontMap,
) -> String {
    let page_top = row * 2;
    let page_bottom = page_top + 1;
    if page_bottom >= bytes.len() {
        return String::new();
    }

    let mut line = String::with_capacity(IQ7000_LARGE_TEXT_COLS);
    for col in 0..IQ7000_LARGE_TEXT_COLS {
        let start = col * IQ7000_LARGE_CELL_HALF_BYTES;
        let mut pattern = [0u8; IQ7000_LARGE_CELL_BYTES];
        pattern[..IQ7000_LARGE_CELL_HALF_BYTES]
            .copy_from_slice(&bytes[page_top][start..start + IQ7000_LARGE_CELL_HALF_BYTES]);
        pattern[IQ7000_LARGE_CELL_HALF_BYTES..]
            .copy_from_slice(&bytes[page_bottom][start..start + IQ7000_LARGE_CELL_HALF_BYTES]);
        line.push(font.resolve(&pattern));
    }
    line.trim_end().to_string()
}

#[derive(Clone, Copy, Debug, Default)]
struct TextScore {
    unknown: usize,
    non_space: usize,
    total: usize,
}

impl TextScore {
    fn merge(mut self, other: Self) -> Self {
        self.unknown += other.unknown;
        self.non_space += other.non_space;
        self.total += other.total;
        self
    }
}

fn score_text(text: &str) -> TextScore {
    let mut score = TextScore::default();
    for ch in text.chars() {
        score.total += 1;
        if ch == '?' {
            score.unknown += 1;
        }
        if ch != ' ' {
            score.non_space += 1;
        }
    }
    score
}

fn prefer_iq7000_large_row(small: TextScore, large: TextScore) -> bool {
    if large.total == 0 {
        return false;
    }
    if large.non_space == 0 {
        return false;
    }
    if small.total == 0 {
        return true;
    }

    let large_unknown_scaled = large.unknown.saturating_mul(small.total);
    let small_unknown_scaled = small.unknown.saturating_mul(large.total);
    if large_unknown_scaled < small_unknown_scaled {
        return true;
    }
    if large_unknown_scaled > small_unknown_scaled {
        return false;
    }

    large.non_space > small.non_space
}

fn trim_trailing_empty_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcd::Iq7000LcdController;

    #[test]
    fn iq7000_text_decoder_ocr_matches_font_map() {
        // Build a tiny synthetic ROM with a 6-byte glyph table indexed directly by ASCII code.
        let mut rom = vec![0u8; 0x80 * IQ7000_CELL_BYTES];
        // Space (0x20) left as all-zero.
        rom[(0x41 * IQ7000_CELL_BYTES)..(0x41 * IQ7000_CELL_BYTES + IQ7000_CELL_BYTES)]
            .copy_from_slice(&[0x3E, 0x48, 0x88, 0x48, 0x3E, 0x00]); // 'A' source order
        rom[(0x42 * IQ7000_CELL_BYTES)..(0x42 * IQ7000_CELL_BYTES + IQ7000_CELL_BYTES)]
            .copy_from_slice(&[0x82, 0xFE, 0x92, 0x92, 0x6C, 0x00]); // 'B' source order
        let font = Iq7000FontMap::from_rom(&rom, 0);
        assert!(!font.is_empty());

        let mut lcd = Iq7000LcdController::new();
        // `display_vram_bytes()` reports columns in IOCS coordinate order (x=0 at the left edge),
        // but the underlying VRAM bytes are stored mirrored. Write the first two cells (AB) at the
        // left edge of row 0 by targeting VRAM columns 95..84.
        for (idx, byte) in [0x3E, 0x48, 0x88, 0x48, 0x3E, 0x00].into_iter().enumerate() {
            lcd.write(0x4000 + (0x5F - idx as u32), byte);
        }
        for (idx, byte) in [0x82, 0xFE, 0x92, 0x92, 0x6C, 0x00].into_iter().enumerate() {
            lcd.write(0x4000 + (0x59 - idx as u32), byte);
        }

        let lines = decode_iq7000_display_text(&lcd, &font);
        assert_eq!(lines, vec!["AB"]);
    }

    #[test]
    fn iq7000_large_text_decoder_ocr_matches_font_map() {
        let mut rom = vec![0u8; 0x80 * IQ7000_LARGE_CELL_BYTES];
        rom[(0x41 * IQ7000_LARGE_CELL_BYTES)
            ..(0x41 * IQ7000_LARGE_CELL_BYTES + IQ7000_LARGE_CELL_BYTES)]
            .copy_from_slice(&[
                0x00, 0x07, 0x38, 0x40, 0x38, 0x07, 0x00, 0x00, // top half
                0x3C, 0xC0, 0x40, 0x40, 0x40, 0xC0, 0x3C, 0x00, // bottom half
            ]);
        rom[(0x42 * IQ7000_LARGE_CELL_BYTES)
            ..(0x42 * IQ7000_LARGE_CELL_BYTES + IQ7000_LARGE_CELL_BYTES)]
            .copy_from_slice(&[
                0x00, 0xFF, 0x91, 0x91, 0x91, 0x91, 0x6E, 0x00, // top half
                0x00, 0xFF, 0x91, 0x91, 0x91, 0x91, 0x6E, 0x00, // bottom half (placeholder)
            ]);
        let font = Iq7000LargeFontMap::from_rom(&rom, 0);
        assert!(!font.is_empty());

        let mut lcd = Iq7000LcdController::new();
        // Place AB in the large-font grid's first row at the left edge. Each large glyph spans
        // 8 columns in the top page (0) and bottom page (1).
        let a_top = [0x00, 0x07, 0x38, 0x40, 0x38, 0x07, 0x00, 0x00];
        let a_bottom = [0x3C, 0xC0, 0x40, 0x40, 0x40, 0xC0, 0x3C, 0x00];
        for (idx, byte) in a_top.into_iter().enumerate() {
            lcd.write(0x4000 + (0x5F - idx as u32), byte);
        }
        for (idx, byte) in a_bottom.into_iter().enumerate() {
            lcd.write(0x4000 + 0x80 + (0x5F - idx as u32), byte);
        }

        let b_top = [0x00, 0xFF, 0x91, 0x91, 0x91, 0x91, 0x6E, 0x00];
        let b_bottom = [0x00, 0xFF, 0x91, 0x91, 0x91, 0x91, 0x6E, 0x00];
        for (idx, byte) in b_top.into_iter().enumerate() {
            lcd.write(0x4000 + (0x57 - idx as u32), byte);
        }
        for (idx, byte) in b_bottom.into_iter().enumerate() {
            lcd.write(0x4000 + 0x80 + (0x57 - idx as u32), byte);
        }

        let lines = decode_iq7000_large_display_text(&lcd, &font);
        assert_eq!(lines, vec!["AB"]);
    }

    #[test]
    fn iq7000_text_decoder_auto_supports_mixed_font_rows() {
        let mut small_rom = vec![0u8; 0x80 * IQ7000_CELL_BYTES];
        small_rom[(0x43 * IQ7000_CELL_BYTES)..(0x43 * IQ7000_CELL_BYTES + IQ7000_CELL_BYTES)]
            .copy_from_slice(&[0x1C, 0x22, 0x41, 0x41, 0x22, 0x1C]); // 'C'
        small_rom[(0x44 * IQ7000_CELL_BYTES)..(0x44 * IQ7000_CELL_BYTES + IQ7000_CELL_BYTES)]
            .copy_from_slice(&[0x7F, 0x41, 0x41, 0x41, 0x22, 0x1C]); // 'D'
        let small_font = Iq7000FontMap::from_rom(&small_rom, 0);
        assert!(!small_font.is_empty());

        let mut large_rom = vec![0u8; 0x80 * IQ7000_LARGE_CELL_BYTES];
        large_rom[(0x41 * IQ7000_LARGE_CELL_BYTES)
            ..(0x41 * IQ7000_LARGE_CELL_BYTES + IQ7000_LARGE_CELL_BYTES)]
            .copy_from_slice(&[
                0x00, 0x07, 0x38, 0x40, 0x38, 0x07, 0x00, 0x00, // top half
                0x3C, 0xC0, 0x40, 0x40, 0x40, 0xC0, 0x3C, 0x00, // bottom half
            ]);
        let large_font = Iq7000LargeFontMap::from_rom(&large_rom, 0);
        assert!(!large_font.is_empty());

        let mut lcd = Iq7000LcdController::new();
        // Draw large 'A' in the top row pair (pages 0+1), then small 'CD' starting at row 2
        // (page 2). This matches how the firmware can mix header-style large glyphs with small
        // body text.
        let a_top = [0x00, 0x07, 0x38, 0x40, 0x38, 0x07, 0x00, 0x00];
        let a_bottom = [0x3C, 0xC0, 0x40, 0x40, 0x40, 0xC0, 0x3C, 0x00];
        for (idx, byte) in a_top.into_iter().enumerate() {
            lcd.write(0x4000 + (0x5F - idx as u32), byte);
        }
        for (idx, byte) in a_bottom.into_iter().enumerate() {
            lcd.write(0x4000 + 0x80 + (0x5F - idx as u32), byte);
        }

        let c_bytes = [0x1C, 0x22, 0x41, 0x41, 0x22, 0x1C];
        let d_bytes = [0x7F, 0x41, 0x41, 0x41, 0x22, 0x1C];
        let row2_base = 0x4000 + 0x80 * 2;
        for (idx, byte) in c_bytes.into_iter().enumerate() {
            lcd.write(row2_base + (0x5F - idx as u32), byte);
        }
        for (idx, byte) in d_bytes.into_iter().enumerate() {
            lcd.write(row2_base + (0x59 - idx as u32), byte);
        }

        let lines = decode_iq7000_display_text_auto(&lcd, &small_font, &large_font);
        assert_eq!(lines, vec!["A", "", "CD"]);
    }
}
