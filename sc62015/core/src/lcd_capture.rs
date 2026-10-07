// PY_SOURCE: pce500/display/lcd_visualization.py
//! Common logical LCD capture. IQ-7000 fixed segments have no Python counterpart.

use crate::iq7000_annunciators::Iq7000Annunciators;
use crate::lcd_frame::LcdFrame;
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
        let frame = lcd.map_or_else(
            || {
                LcdFrame::new(
                    LCD_DISPLAY_COLS,
                    LCD_DISPLAY_ROWS,
                    vec![0; LCD_DISPLAY_COLS * LCD_DISPLAY_ROWS],
                )
                .expect("blank LCD geometry")
            },
            LcdHal::matrix_frame,
        );
        let annunciators = (kind == LcdKind::Iq7000Vram).then(|| Iq7000Annunciators::read(memory));
        Self::from_frame(kind, &frame, annunciators, pixel_scale)
    }

    /// Render an owned controller observation with its full geometry. This
    /// path does not need a memory bus or a device-specific fixed-array view.
    pub fn from_frame(
        kind: LcdKind,
        frame: &LcdFrame,
        annunciators: Option<Iq7000Annunciators>,
        pixel_scale: usize,
    ) -> Result<Self, &'static str> {
        let pixels = render_lcd(&frame.to_rows(), annunciators.as_ref(), pixel_scale)?;
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
    lcd.matrix_frame().to_rows()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_lcd;

    // This fixture supplies only a dynamic framebuffer. Accessing any legacy
    // array or bus method fails, exposing accidental 240x32/96x64 fallback.
    struct GeometryOnlyLcd(LcdFrame);

    impl LcdHal for GeometryOnlyLcd {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn kind(&self) -> LcdKind {
            LcdKind::Unknown
        }
        fn matrix_frame(&self) -> LcdFrame {
            self.0.clone()
        }
        fn reset(&mut self) {
            panic!("capture must not reset the controller")
        }
        fn handles(&self, _: u32) -> bool {
            false
        }
        fn read(&mut self, _: u32) -> Option<u8> {
            panic!("capture must not read the bus")
        }
        fn write(&mut self, _: u32, _: u8) {
            panic!("capture must not write the bus")
        }
        fn read_placeholder(&self, _: u32) -> u32 {
            panic!("capture must not peek the bus")
        }
        fn begin_display_write_capture(&mut self) {
            panic!("not a bus capture")
        }
        fn take_display_write_capture(&mut self) -> Vec<crate::lcd::LcdDisplayWrite> {
            panic!("not a bus capture")
        }
        fn display_buffer(&self) -> [[u8; LCD_DISPLAY_COLS]; LCD_DISPLAY_ROWS] {
            panic!("legacy array crops a large LCD")
        }
        fn chip_display_buffer(
            &self,
            _: usize,
        ) -> [[u8; crate::LCD_CHIP_COLS]; crate::LCD_CHIP_ROWS] {
            panic!("not a legacy chip")
        }
        fn display_vram_bytes(&self) -> [[u8; LCD_DISPLAY_COLS]; 8] {
            panic!("legacy page array crops a large LCD")
        }
        fn display_trace_buffer(&self) -> [[crate::lcd::LcdWriteTrace; LCD_DISPLAY_COLS]; 8] {
            panic!("not a trace capture")
        }
        fn stats(&self) -> crate::lcd::LcdStats {
            panic!("not a statistics capture")
        }
        fn snapshot_state(&self) -> (crate::lcd_snapshot::LcdSnapshotMetadata, Vec<u8>) {
            panic!("not a running snapshot")
        }
        fn restore_state(
            &mut self,
            _: &crate::lcd_snapshot::LcdSnapshotMetadata,
            _: &[u8],
        ) -> Result<(), String> {
            panic!("capture must not restore state")
        }
    }

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

    #[test]
    fn capture_preserves_full_large_frame_instead_of_legacy_array_crop() {
        let mut pixels = vec![0; 336 * 240];
        pixels[239 * 336 + 335] = 1;
        let frame = LcdFrame::new(336, 240, pixels).unwrap();
        let source = GeometryOnlyLcd(frame);
        let memory = MemoryImage::new();
        let reads = memory.memory_read_count();
        let capture = LcdCapture::read_at_scale(Some(&source), &memory, 3).unwrap();
        assert_eq!((capture.cols, capture.rows), (1008, 720));
        assert_eq!(capture.pixels.len(), 1008 * 720);
        for y in 717..720 {
            for x in 1005..1008 {
                assert_eq!(capture.pixels[y * 1008 + x], 0);
            }
        }
        assert!(capture.pixels[..717 * 1008]
            .iter()
            .all(|&shade| shade == 192));
        assert_eq!(source.0.pixels()[239 * 336 + 335], 1);
        assert_eq!(memory.memory_read_count(), reads);
        let matrix = lcd_matrix_pixels(&source);
        assert_eq!((matrix[0].len(), matrix.len()), (336, 240));
        assert_eq!(matrix[239][335], 1);
        assert!(capture.annunciators.is_none());
    }
}
