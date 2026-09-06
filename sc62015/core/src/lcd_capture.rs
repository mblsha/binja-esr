// PY_SOURCE: pce500/display/lcd_visualization.py
//! Common logical LCD capture. IQ-7000 fixed segments have no Python counterpart.

use crate::iq7000_annunciators::Iq7000Annunciators;
use crate::lcd_render::{render_lcd, IQ_DISPLAY_SCALE};
use crate::{LcdHal, LcdKind, MemoryImage, LCD_DISPLAY_COLS, LCD_DISPLAY_ROWS};
use serde::Serialize;

/// Final-resolution grayscale pixels. Matrix pixels remain crisp integer
/// squares; fixed glass segments are anti-aliased at this output resolution.
#[derive(Debug, Serialize)]
pub struct LcdCapture {
    pub kind: LcdKind,
    pub cols: usize,
    pub rows: usize,
    pub pixel_format: &'static str,
    pub pixel_scale: usize,
    pub pixels: Vec<u8>,
    pub annunciators: Option<Iq7000Annunciators>,
}

impl LcdCapture {
    pub fn read(lcd: Option<&dyn LcdHal>, memory: &MemoryImage) -> Self {
        let scale = if lcd.is_some_and(|lcd| lcd.kind() == LcdKind::Iq7000Vram) {
            IQ_DISPLAY_SCALE
        } else {
            1
        };
        Self::read_at_scale(lcd, memory, scale).expect("fixed LCD geometry")
    }

    pub fn read_at_scale(
        lcd: Option<&dyn LcdHal>,
        memory: &MemoryImage,
        pixel_scale: usize,
    ) -> Result<Self, &'static str> {
        let kind = lcd.map_or(LcdKind::Unknown, LcdHal::kind);
        let matrix = lcd.map_or_else(
            || vec![vec![0; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS],
            lcd_matrix_pixels,
        );
        let annunciators = (kind == LcdKind::Iq7000Vram).then(|| Iq7000Annunciators::read(memory));
        let pixels = render_lcd(&matrix, annunciators.as_ref(), pixel_scale)?;
        Ok(Self {
            kind,
            cols: pixels.first().map_or(0, Vec::len),
            rows: pixels.len(),
            pixel_format: "gray8",
            pixel_scale,
            pixels: pixels.into_iter().flatten().collect(),
            annunciators,
        })
    }
}

pub fn lcd_matrix_pixels(lcd: &dyn LcdHal) -> Vec<Vec<u8>> {
    if lcd.kind() == LcdKind::Iq7000Vram {
        const IQ7000_COLS: usize = 96;
        const IQ7000_ROWS: usize = 64;
        const IQ7000_PAGES: usize = IQ7000_ROWS / 8;
        let bytes = lcd.display_vram_bytes();
        let mut out = vec![vec![0u8; IQ7000_COLS]; IQ7000_ROWS];
        for page in 0..IQ7000_PAGES {
            for col in 0..IQ7000_COLS {
                let byte = bytes[page][col];
                for dy in 0..8usize {
                    let bit = 7usize.saturating_sub(dy);
                    out[(page * 8) + dy][col] = (byte >> bit) & 1;
                }
            }
        }
        return out;
    }
    lcd.display_buffer()
        .iter()
        .map(|row| row.iter().map(|px| u8::from(*px != 0)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_lcd;

    #[test]
    fn full_iq_display_preserves_vram_and_adds_only_fixed_segments() {
        let mut lcd = create_lcd(LcdKind::Iq7000Vram);
        let mut memory = MemoryImage::new();
        lcd.write(0x405A, 0x80);
        lcd.write(0x605A, 0x40);
        memory.store(0x6160, 8, 0x10).unwrap();
        let reads = memory.memory_read_count();
        let capture = LcdCapture::read(Some(lcd.as_ref()), &memory);
        assert_eq!((capture.cols, capture.rows), (488, 256));
        assert_eq!(capture.pixel_scale, 4);
        assert_eq!(capture.pixel_format, "gray8");
        assert_eq!(capture.pixels.len(), 488 * 256);
        assert_eq!(capture.pixels[5 * 4], 0);
        assert_eq!(capture.pixels[33 * 4 * 488 + 5 * 4], 0);
        assert!(capture.annunciators.as_ref().unwrap().shift);
        assert_eq!(memory.memory_read_count(), reads);
        let matrix = lcd_matrix_pixels(lcd.as_ref());
        for (y, row) in matrix.iter().enumerate() {
            for (x, bit) in row.iter().enumerate() {
                for dy in 0..4 {
                    for dx in 0..4 {
                        assert_eq!(
                            capture.pixels[(y * 4 + dy) * 488 + x * 4 + dx],
                            if *bit == 0 { 192 } else { 0 }
                        );
                    }
                }
            }
        }
        memory.store(0x6160, 8, 0).unwrap();
        let cleared = LcdCapture::read(Some(lcd.as_ref()), &memory);
        assert_ne!(cleared.pixels, capture.pixels); // capture owns its arrays
        let (rows, remainder) = cleared.pixels.as_chunks::<488>();
        assert!(remainder.is_empty());
        for row in rows {
            assert!(row[384..].iter().all(|&shade| shade == 192));
        }
    }

    #[test]
    fn pc_e500_does_not_gain_iq_symbols_even_if_ram_contains_the_bits() {
        let lcd = create_lcd(LcdKind::Hd61202);
        let mut memory = MemoryImage::new();
        memory.store(0x6160, 8, 0xff).unwrap();
        let capture = LcdCapture::read(Some(lcd.as_ref()), &memory);
        assert_eq!((capture.cols, capture.rows), (240, 32));
        assert!(capture.annunciators.is_none());
    }
}
