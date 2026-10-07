// PY_SOURCE: pce500/display/lcd_frame.py:LcdFrame
//! Owned logical pixels with device-supplied geometry. No bezel artwork,
//! bus reads, controller state or display scaling are part of this value.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LcdFrame {
    cols: usize,
    rows: usize,
    pixels: Vec<u8>,
}

impl LcdFrame {
    /// Row-major logical pixels: zero is clear and one is lit.
    pub fn new(cols: usize, rows: usize, pixels: Vec<u8>) -> Result<Self, &'static str> {
        if cols == 0 || rows == 0 || cols.checked_mul(rows) != Some(pixels.len()) {
            return Err("LCD frame requires nonzero geometry matching its pixel count");
        }
        if pixels.iter().any(|&pixel| pixel > 1) {
            return Err("LCD frame pixels must be zero or one");
        }
        Ok(Self { cols, rows, pixels })
    }

    pub fn geometry(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }

    pub fn to_rows(&self) -> Vec<Vec<u8>> {
        self.pixels
            .chunks_exact(self.cols)
            .map(<[u8]>::to_vec)
            .collect()
    }

    /// Binary PBM packs each row independently, most-significant bit first.
    /// Unused bits at the end of a row remain clear.
    pub fn pbm(&self) -> Vec<u8> {
        let stride = self.cols.div_ceil(8);
        let mut output = format!("P4\n{} {}\n", self.cols, self.rows).into_bytes();
        let header = output.len();
        output.resize(header + stride * self.rows, 0);
        for (y, row) in self.pixels.chunks_exact(self.cols).enumerate() {
            for (x, &pixel) in row.iter().enumerate() {
                output[header + y * stride + x / 8] |= pixel << (7 - x % 8);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbm_keeps_row_padding_separate_and_uses_logical_polarity() {
        // Two rows cross byte boundaries. The second must not start in the
        // first row's seven unused bits; 1 denotes a black PBM pixel.
        let frame = LcdFrame::new(
            9,
            2,
            vec![1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        )
        .unwrap();
        assert_eq!(frame.pbm(), b"P4\n9 2\n\x81\x80\x40\x80");
        assert_eq!(frame.geometry(), (9, 2));
        assert_eq!(frame.to_rows().len(), 2);
    }

    #[test]
    fn full_oz9600_frame_preserves_pixels_beyond_legacy_dimensions() {
        let mut pixels = vec![0; 336 * 240];
        for (x, y) in [(0, 0), (239, 31), (319, 239), (335, 239)] {
            pixels[y * 336 + x] = 1;
        }
        let frame = LcdFrame::new(336, 240, pixels).unwrap();
        let pbm = frame.pbm();
        let header = b"P4\n336 240\n";
        assert_eq!(pbm.len(), header.len() + 42 * 240);
        assert_eq!(&pbm[..header.len()], header);
        assert_eq!(pbm[header.len()], 0x80);
        assert_eq!(pbm[header.len() + 31 * 42 + 29], 1);
        assert_eq!(pbm[header.len() + 239 * 42 + 39], 1);
        assert_eq!(pbm[header.len() + 239 * 42 + 41], 1);
        assert_eq!(frame.pixels().iter().map(|&p| u32::from(p)).sum::<u32>(), 4);
    }

    #[test]
    fn invalid_geometry_overflow_and_nonbinary_pixels_are_rejected() {
        for (cols, rows, pixels) in [
            (0, 1, vec![]),
            (1, 0, vec![]),
            (2, 2, vec![0; 3]),
            (usize::MAX, 2, vec![]),
            (1, 1, vec![2]),
        ] {
            assert!(LcdFrame::new(cols, rows, pixels).is_err());
        }
    }
}
