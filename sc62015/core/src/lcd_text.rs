// PY_SOURCE: pce500/display/text_decoder.py:decode_display_text
// PY_SOURCE: pce500/display/font.py

use crate::lcd::{LcdHal, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
use std::collections::{HashMap, VecDeque};

const GLYPH_WIDTH: usize = 5;
const GLYPH_STRIDE: usize = 6; // five data columns + spacer
const GLYPH_COUNT: usize = 96; // ASCII 0x20-0x7F
const ROWS_PER_CELL: usize = 8;
const COLS_PER_CELL: usize = 6;
const PCE500_ARROW_UP_DOWN: [u8; GLYPH_WIDTH] = [0x00, 0x28, 0x6c, 0x6c, 0x28];
const PCE500_LBRACKET: [u8; GLYPH_WIDTH] = [0x00, 0x7f, 0x7f, 0x41, 0x00];
const PCE500_RBRACKET: [u8; GLYPH_WIDTH] = [0x00, 0x41, 0x7f, 0x7f, 0x00];

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
        insert_pce500_special(&mut glyphs, PCE500_ARROW_UP_DOWN, '⇳');
        insert_pce500_special(&mut glyphs, PCE500_LBRACKET, '[');
        insert_pce500_special(&mut glyphs, PCE500_RBRACKET, ']');
        Self { glyphs }
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    fn resolve(&self, pattern: &[u8; GLYPH_WIDTH]) -> char {
        *self.glyphs.get(pattern).unwrap_or(&'?')
    }
}

fn insert_pce500_special(
    glyphs: &mut HashMap<[u8; GLYPH_WIDTH], char>,
    pattern: [u8; GLYPH_WIDTH],
    ch: char,
) {
    glyphs.entry(pattern).or_insert(ch);
    let mut inverted = [0u8; GLYPH_WIDTH];
    for (dest, src) in inverted.iter_mut().zip(pattern) {
        *dest = (!src) & 0x7F;
    }
    glyphs.entry(inverted).or_insert(ch);
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

#[derive(Clone)]
struct LcdCharTrieNode {
    // Child index + 1, keyed by byte value. Using 0 as "no edge" keeps the node compact and fast.
    children: [u16; 256],
    terminal: Option<char>,
}

impl Default for LcdCharTrieNode {
    fn default() -> Self {
        Self {
            children: [0u16; 256],
            terminal: None,
        }
    }
}

#[derive(Clone)]
struct LcdCharTrie {
    nodes: Vec<LcdCharTrieNode>,
    max_len: usize,
}

impl LcdCharTrie {
    fn new() -> Self {
        Self {
            nodes: vec![LcdCharTrieNode::default()],
            max_len: 0,
        }
    }

    fn insert(&mut self, pattern: &[u8], ch: char) {
        self.max_len = self.max_len.max(pattern.len());
        let mut node = 0usize;

        // Insert reversed so a match can be found by walking backward from the newest write.
        for &byte in pattern.iter().rev() {
            let edge = self.nodes[node].children[byte as usize];
            if edge == 0 {
                self.nodes.push(LcdCharTrieNode::default());
                let new_idx = self.nodes.len() - 1;
                self.nodes[node].children[byte as usize] =
                    u16::try_from(new_idx + 1).expect("lcd char trie too large");
                node = new_idx;
            } else {
                node = (edge as usize).saturating_sub(1);
            }
        }

        // If multiple fonts contain an identical byte pattern, keep the first-seen char label.
        self.nodes[node].terminal.get_or_insert(ch);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LcdCharWriteSample {
    pub value: u8,
    pub op_index: u64,
    pub x: u16,
    pub y: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LcdCharMatch {
    pub ch: char,
    pub start_op_index: u64,
    pub end_op_index: u64,
    pub x: u16,
    pub y: u8,
    pub len: usize,
}

/// Streaming glyph detector used by LCD write tracing.
///
/// This matcher keeps the last `max_glyph_len` byte writes and checks if the newest suffix matches
/// any known glyph pattern from the device's font map(s). It emits the **longest** match ending at
/// the current write (by design, to disambiguate overlapping font widths).
#[derive(Clone)]
pub(crate) struct LcdCharMatcher {
    trie: LcdCharTrie,
    window: VecDeque<LcdCharWriteSample>,
}

impl LcdCharMatcher {
    const STREAM_BREAK_INSTRS: u64 = 10;

    pub(crate) fn from_pce500_font_map(font: &Pce500FontMap) -> Option<Self> {
        if font.is_empty() {
            return None;
        }
        let mut trie = LcdCharTrie::new();
        for (pattern, ch) in font.glyphs.iter() {
            trie.insert(pattern, *ch);
        }
        Some(Self {
            trie,
            window: VecDeque::new(),
        })
    }

    pub(crate) fn from_iq7000_font_maps(
        small_font: &Iq7000FontMap,
        large_font: &Iq7000LargeFontMap,
    ) -> Option<Self> {
        if small_font.is_empty() && large_font.is_empty() {
            return None;
        }
        let mut trie = LcdCharTrie::new();
        for (pattern, ch) in small_font.glyphs.iter() {
            trie.insert(pattern, *ch);
        }
        for (pattern, ch) in large_font.glyphs.iter() {
            trie.insert(pattern, *ch);
        }
        Some(Self {
            trie,
            window: VecDeque::new(),
        })
    }

    pub(crate) fn reset_stream(&mut self) {
        self.window.clear();
    }

    pub(crate) fn push(&mut self, sample: LcdCharWriteSample) -> Option<LcdCharMatch> {
        let max_len = self.trie.max_len;
        if max_len == 0 {
            return None;
        }

        if let Some(prev) = self.window.back() {
            if sample.op_index.saturating_sub(prev.op_index) >= Self::STREAM_BREAK_INSTRS {
                self.window.clear();
            }
        }

        self.window.push_back(sample);
        while self.window.len() > max_len {
            self.window.pop_front();
        }

        let mut node = 0usize;
        let mut depth = 0usize;
        let mut best: Option<(char, usize)> = None;

        for ev in self.window.iter().rev() {
            let edge = self.trie.nodes[node].children[ev.value as usize];
            if edge == 0 {
                break;
            }
            node = (edge as usize).saturating_sub(1);
            depth = depth.saturating_add(1);

            if let Some(ch) = self.trie.nodes[node].terminal {
                // Longest-match policy: keep updating as we walk deeper.
                best = Some((ch, depth));
            }
        }

        let (ch, len) = best?;
        let start_idx = self.window.len().saturating_sub(len);
        let start = *self.window.get(start_idx)?;
        let end = *self.window.back()?;
        Some(LcdCharMatch {
            ch,
            start_op_index: start.op_index,
            end_op_index: end.op_index,
            x: start.x,
            y: start.y,
            len,
        })
    }
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
    use crate::lcd::LcdController;

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

    #[test]
    fn lcd_char_matcher_emits_longest_match() {
        // Small font has 'A' = 6 bytes.
        let mut small_rom = vec![0u8; 0x80 * IQ7000_CELL_BYTES];
        let small_a = [1u8, 2, 3, 4, 5, 6];
        small_rom[(0x41 * IQ7000_CELL_BYTES)..(0x41 * IQ7000_CELL_BYTES + IQ7000_CELL_BYTES)]
            .copy_from_slice(&small_a);
        let small_font = Iq7000FontMap::from_rom(&small_rom, 0);

        // Large font has 'B' = 16 bytes and ends with the same 6-byte suffix as the small 'A'.
        let mut large_rom = vec![0u8; 0x80 * IQ7000_LARGE_CELL_BYTES];
        let mut large_b = [0u8; IQ7000_LARGE_CELL_BYTES];
        large_b[(IQ7000_LARGE_CELL_BYTES - small_a.len())..].copy_from_slice(&small_a);
        large_rom[(0x42 * IQ7000_LARGE_CELL_BYTES)
            ..(0x42 * IQ7000_LARGE_CELL_BYTES + IQ7000_LARGE_CELL_BYTES)]
            .copy_from_slice(&large_b);
        let large_font = Iq7000LargeFontMap::from_rom(&large_rom, 0);

        let mut matcher =
            LcdCharMatcher::from_iq7000_font_maps(&small_font, &large_font).expect("matcher");

        let mut last = None;
        for (idx, byte) in large_b.into_iter().enumerate() {
            last = matcher.push(LcdCharWriteSample {
                value: byte,
                op_index: idx as u64,
                x: 7,
                y: 2,
            });
        }

        assert_eq!(
            last,
            Some(LcdCharMatch {
                ch: 'B',
                start_op_index: 0,
                end_op_index: (IQ7000_LARGE_CELL_BYTES - 1) as u64,
                x: 7,
                y: 2,
                len: IQ7000_LARGE_CELL_BYTES,
            })
        );
    }

    #[test]
    fn lcd_char_matcher_detects_pce500_glyphs() {
        let mut rom = vec![0u8; GLYPH_COUNT * GLYPH_STRIDE];
        let a_index = (0x41u32 - 0x20u32) as usize;
        let start = a_index * GLYPH_STRIDE;
        let a_pattern = [0x01u8, 0x02, 0x04, 0x08, 0x10];
        rom[start..start + GLYPH_WIDTH].copy_from_slice(&a_pattern);
        let font = Pce500FontMap::from_rom(&rom, 0, 0);

        let mut matcher = LcdCharMatcher::from_pce500_font_map(&font).expect("matcher");
        let mut out = None;
        for (idx, byte) in a_pattern.into_iter().enumerate() {
            out = matcher.push(LcdCharWriteSample {
                value: byte,
                op_index: idx as u64,
                x: 11,
                y: 3,
            });
        }
        assert_eq!(
            out,
            Some(LcdCharMatch {
                ch: 'A',
                start_op_index: 0,
                end_op_index: (GLYPH_WIDTH - 1) as u64,
                x: 11,
                y: 3,
                len: GLYPH_WIDTH,
            })
        );

        // Sanity: pushing an unrelated byte should not keep re-emitting the same glyph.
        let next = matcher.push(LcdCharWriteSample {
            value: 0xFF,
            op_index: 99,
            x: 12,
            y: 3,
        });
        assert!(next.is_none());

        // Smoke: matcher can coexist with actual LCD types (no trait coupling).
        let _lcd = LcdController::new();
    }

    #[test]
    fn lcd_char_matcher_breaks_stream_on_instruction_gap() {
        let mut large_rom = vec![0u8; 0x80 * IQ7000_LARGE_CELL_BYTES];
        let pattern: [u8; IQ7000_LARGE_CELL_BYTES] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
            0x17, 0x18,
        ];
        large_rom[(0x41 * IQ7000_LARGE_CELL_BYTES)
            ..(0x41 * IQ7000_LARGE_CELL_BYTES + IQ7000_LARGE_CELL_BYTES)]
            .copy_from_slice(&pattern);
        let large_font = Iq7000LargeFontMap::from_rom(&large_rom, 0);
        let small_font = Iq7000FontMap::default();

        let mut matcher =
            LcdCharMatcher::from_iq7000_font_maps(&small_font, &large_font).expect("matcher");

        for (idx, byte) in pattern[..8].iter().copied().enumerate() {
            assert!(
                matcher
                    .push(LcdCharWriteSample {
                        value: byte,
                        op_index: idx as u64,
                        x: 0,
                        y: 0,
                    })
                    .is_none(),
                "partial glyph should not match"
            );
        }

        // A sufficiently large instruction gap implies a new character write stream, so the
        // matcher must not match across this boundary.
        let gap_start = 7u64 + LcdCharMatcher::STREAM_BREAK_INSTRS;
        let mut last = None;
        for (idx, byte) in pattern[8..].iter().copied().enumerate() {
            last = matcher.push(LcdCharWriteSample {
                value: byte,
                op_index: gap_start + idx as u64,
                x: 0,
                y: 0,
            });
        }
        assert!(
            last.is_none(),
            "expected no match when glyph bytes are split by an instruction gap"
        );
    }
}
