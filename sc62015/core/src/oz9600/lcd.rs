// PY_SOURCE: pce500/oz9600/lcd.py
//! LH1553F exploration driven by the fixed ROM's 084xx accesses.
//!
//! Register names and timing are still being qualified. These operations cover
//! Pixel streams, masked blocks, independent read/write coordinates, and
//! the one-byte read pipeline used by F73A2/F767B. SET (0), XOR (1) and
//! OR (3) follow firmware behavior. Physical LH1553F measurements establish
//! AND (2) and bit-2 source inversion for a vertical eight-pixel group.
//! Sixteen window descriptors and register readback are checked against the
//! F0 diagnostic loops. Physical interior/window-edge controls establish
//! inclusive read/write clipping. The extra Y-end bit preserves readback but
//! relaxes neither tested bound. Its other roles and reset geometry remain open.
//! Access-ready acknowledgment and data-buffer writes follow the paired ROM
//! guards and VRAM self-test. Their silicon timing/behavior remain provisional.

pub const STORAGE_WIDTH: usize = 336; // F2CC0 inclusive X limit 014F
pub const HEIGHT: usize = 240; // F2CC4 inclusive Y limit 00EF
pub const MAIN_WIDTH: usize = 320; // manual cover; physical crop unresolved
                                   // Firmware-derived field names. Access-ready acknowledges the control request
                                   // synchronously in this prototype; electrical latency is still unqualified.
pub const ACCESS_REQUEST: u8 = 1;
pub const ACCESS_READY: u8 = 2;
pub const SAMPLE_SYNC: u8 = 4;
pub const FRAME_PHASE: u8 = 8;
pub const FRAME_HALF_PERIOD: u64 = 32768;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct LcdController {
    pub registers: [u8; 32],
    #[serde(default = "legacy_storage_width")]
    width: usize,
    #[serde(default = "legacy_storage_height")]
    height: usize,
    windows: [[u8; 8]; 16],
    pixels: Vec<u8>,
    read_latch: u8,
    pub data_writes: u64,
    pub data_reads: u64,
    pub block_operations: u64,
    /// Provisional write-side priority. The physical To Do overlap measurement
    /// separately establishes read clipping for overlapping windows 2 and 3.
    #[serde(default)]
    pub ordered_window_clipping_candidate: bool,
    /// Related firmware uses extended storage and an extra-bit full-area view.
    /// Preserve that unmeasured window contract explicitly; OZ uses the
    /// physically measured inclusive read/write contract by default.
    #[serde(default)]
    pub firmware_window_candidate: bool,
}

// Older OZRUN01 images predate configurable controller geometry. Their backing
// was always336x240 and their controller had no ordered-window experiment.
const fn legacy_storage_width() -> usize {
    STORAGE_WIDTH
}
const fn legacy_storage_height() -> usize {
    HEIGHT
}

impl Default for LcdController {
    fn default() -> Self {
        Self::with_geometry(STORAGE_WIDTH, HEIGHT)
    }
}

impl LcdController {
    #[cfg(feature = "pvf1")]
    pub(crate) fn validate_live_geometry(&self, width: usize, height: usize) -> Result<(), String> {
        if self.width != width
            || self.height != height
            || self.pixels.len() != width * height
            || self.pixels.iter().any(|p| *p > 1)
        {
            return Err("invalid live LCD geometry/pixels".into());
        }
        Ok(())
    }
    /// Explicit controller storage for related boards; panel cropping is separate.
    pub fn with_geometry(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0 && width <= 1024 && height <= 512);
        // Diagnostic C62C0/C62DE require these unimplemented high bits to
        // read as ones. This is a zero descriptor, not a full-screen reset
        // window; the physical reset geometry has not been qualified.
        let descriptor = [0, 0x7c, 0, 0x7e, 0, 0x7c, 0, 0x7c];
        let mut registers = [0; 32];
        registers[0x18..].copy_from_slice(&descriptor);
        Self {
            registers,
            windows: [descriptor; 16],
            pixels: vec![0; width * height],
            width,
            height,
            read_latch: 0,
            data_writes: 0,
            data_reads: 0,
            block_operations: 0,
            ordered_window_clipping_candidate: false,
            firmware_window_candidate: false,
        }
    }
}

impl LcdController {
    pub(super) fn validate_session(&self) -> Result<(), String> {
        if self.width != STORAGE_WIDTH
            || self.height != HEIGHT
            || self.pixels.len() != STORAGE_WIDTH * HEIGHT
            || self.pixels.iter().any(|v| *v > 1)
        {
            return Err("Invalid saved LCD state".into());
        }
        Ok(())
    }

    /// Readback bytes, including the descriptor's fixed high bits.
    pub fn window_descriptor(&self, index: usize) -> [u8; 8] {
        self.windows[index]
    }

    // F0 C6244/C6282 specify sign extension and an overflow indication in
    // the first bit above the coordinate field. Preserve the written word
    // for arithmetic; the normalized representation is a read-port view.
    fn coordinate_high_readback(value: u8, field_mask: u8) -> u8 {
        let upper_mask = 0x7f & !field_mask;
        let overflow = field_mask + 1;
        if value & 0x80 == 0 {
            (value & field_mask) | if value & upper_mask != 0 { overflow } else { 0 }
        } else if value & upper_mask == upper_mask {
            value | !field_mask
        } else {
            (value & !overflow) | !(field_mask | overflow)
        }
    }

    fn word(&self, offset: usize) -> u16 {
        u16::from_le_bytes([self.registers[offset], self.registers[offset + 1]])
    }

    fn set_word(&mut self, offset: usize, value: u16) {
        [self.registers[offset], self.registers[offset + 1]] = value.to_le_bytes();
    }

    pub fn pixel(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && self.pixels[y * self.width + x] != 0
    }

    fn bit_location(&self, x: u16, y: u16, bit: usize) -> Option<(usize, usize)> {
        // Coordinate words are signed, including spans which begin outside
        // backing and cross zero. Do not discard their entire first byte.
        let (x, y) = (i32::from(x as i16), i32::from(y as i16));
        let (x, y) = if self.registers[8] & 0x80 != 0 {
            (x, y + bit as i32)
        } else {
            (x + bit as i32, y)
        };
        Some((usize::try_from(x).ok()?, usize::try_from(y).ok()?))
    }

    fn read_pixels(&self, x: u16, y: u16) -> u8 {
        let mut value = 0;
        for bit in 0..8 {
            if self.bit_location(x, y, bit).is_some_and(|(px, py)| {
                (self.firmware_window_candidate || self.permits_pixel(px, py, true))
                    && self.pixel(px, py)
            }) {
                value |= 0x80 >> bit;
            }
        }
        value & self.registers[3]
    }

    fn window_coordinate(low: u8, high: u8, mask: u8) -> i32 {
        // Descriptor high bytes include fixed readback ones. The ROM's
        // FA3CD helper strips those bits for positive coordinates and retains
        // sign extension for negative ones. Y-end bit 1 is a separate flag.
        let high = if high & 0x80 != 0 {
            high | !mask
        } else {
            high & mask
        };
        i32::from(i16::from_le_bytes([low, high]))
    }

    fn permits_pixel(&self, x: usize, y: usize, read: bool) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let window = self.window_descriptor(usize::from(self.registers[9]));
        if self.firmware_window_candidate && window[7] & 2 != 0 {
            return true;
        }
        // Physical OZ interior controls retain this bit on readback without
        // bypassing the measured X/Y bounds; PV geometry remains model-owned.
        let xmin = Self::window_coordinate(window[0], window[1], 3);
        let ymin = Self::window_coordinate(window[2], window[3], 1);
        let xmax = Self::window_coordinate(window[4], window[5], 3);
        let ymax = Self::window_coordinate(window[6], window[7], 1);
        if !(xmin..=xmax).contains(&(x as i32)) || !(ymin..=ymax).contains(&(y as i32)) {
            return false;
        }
        // The physical To Do overlap probe established window-2/3 read clipping.
        // Other layouts remain provisional; write priority is opt-in.
        if read || self.ordered_window_clipping_candidate {
            for previous in &self.windows[..usize::from(self.registers[9])] {
                if previous[7] & 2 != 0 {
                    continue;
                }
                let xmin = Self::window_coordinate(previous[0], previous[1], 3);
                let ymin = Self::window_coordinate(previous[2], previous[3], 1);
                let xmax = Self::window_coordinate(previous[4], previous[5], 3);
                let ymax = Self::window_coordinate(previous[6], previous[7], 1);
                if (xmin..=xmax).contains(&(x as i32)) && (ymin..=ymax).contains(&(y as i32)) {
                    return false;
                }
            }
        }
        true
    }

    fn write_pixels(&mut self, x: u16, y: u16, value: u8) {
        for bit in 0..8 {
            let mask = 0x80 >> bit;
            let Some((px, py)) = self.bit_location(x, y, bit) else {
                continue;
            };
            if self.registers[3] & mask != 0 && self.permits_pixel(px, py, false) {
                let destination = &mut self.pixels[py * self.width + px];
                let source = u8::from(value & mask != 0) ^ ((self.registers[8] >> 2) & 1);
                let combined = match self.registers[8] & 3 {
                    0 => source,
                    1 => *destination ^ source,
                    2 => *destination & source,
                    // Scrapbook issues live mode 3 at its native line entry,
                    // then ORs the same stroke into its saved software bitmap.
                    // AND hid live strokes until a later SET bitmap redraw.
                    3 => *destination | source,
                    _ => unreachable!("two-bit raster operation"),
                };
                // Physical AA-destination/3C-source cases distinguish
                // mode 2 (28), inverted AND (82), and inverted OR (EB).
                // Invert the source before the Boolean operation, rather
                // than complementing the combined destination afterward.
                *destination = combined;
            }
        }
    }

    fn increment_axis(&mut self, coordinate: usize, nibble: u8) -> Result<(), String> {
        if nibble == 0 {
            return Ok(());
        }
        // Positive/negative are bits 0/1; F76B4 undoes the last prefetch using
        // bit 1 and selects a step of eight using bit 3. The role of bit 2
        // beyond the observed 5/6 controls remains a qualification item.
        let step = if nibble & 8 != 0 { 8 } else { 1 };
        let value = self.word(coordinate);
        let next = match nibble & 3 {
            1 => value.wrapping_add(step),
            2 => value.wrapping_sub(step),
            _ => return Err(format!("unqualified LCD increment nibble {nibble:X}")),
        };
        self.set_word(coordinate, next);
        Ok(())
    }

    fn advance(&mut self, read: bool) -> Result<(), String> {
        let increments = self.registers[if read { 4 } else { 5 }];
        let coordinate = if read { 0x14 } else { 0x10 };
        self.increment_axis(coordinate, increments >> 4)?;
        self.increment_axis(coordinate + 2, increments & 15)
    }

    fn check_increments(&self, read: bool) -> Result<(), String> {
        let increments = self.registers[if read { 4 } else { 5 }];
        for nibble in [increments >> 4, increments & 15] {
            if nibble != 0 && !matches!(nibble & 3, 1 | 2) {
                return Err(format!("unqualified LCD increment nibble {nibble:X}"));
            }
        }
        Ok(())
    }

    /// A side-effect-free preflight view, including the existing read latch.
    pub fn peek(&self, offset: usize, cycle: u64) -> u8 {
        match offset {
            1 => {
                let ready = if self.registers[0x0e] & ACCESS_REQUEST != 0 {
                    ACCESS_READY
                } else {
                    0
                };
                // F0 D9E73/D9E7A and its ADC helpers require a low-to-high
                // status-bit-2 edge. The manual establishes 240 scan outputs,
                // but not this bit's name or timing. This provisional policy
                // supplies one square-wave sampling phase per scan line;
                // neither phase depends on read count. Silicon qualification
                // of the association, duty cycle, and timebase is pending.
                let sample_phase =
                    (cycle % FRAME_HALF_PERIOD * self.height as u64) % FRAME_HALF_PERIOD;
                ready
                    | if sample_phase < FRAME_HALF_PERIOD / 2 {
                        SAMPLE_SYNC
                    } else {
                        0
                    }
                    | if (cycle / FRAME_HALF_PERIOD) & 1 != 0 {
                        FRAME_PHASE
                    } else {
                        0
                    }
            }
            2 => self.read_latch,
            0x11 | 0x15 => Self::coordinate_high_readback(self.registers[offset], 3),
            0x13 | 0x17 => Self::coordinate_high_readback(self.registers[offset], 1),
            _ => self.registers[offset],
        }
    }

    pub fn read(&mut self, offset: usize, cycle: u64) -> Result<u8, String> {
        match offset {
            0 => {
                self.check_increments(true)?;
                self.check_increments(false)?;
                for _ in 0..self.word(6) {
                    let value = self.read_pixels(self.word(0x14), self.word(0x16));
                    self.write_pixels(self.word(0x10), self.word(0x12), value);
                    self.advance(true)?;
                    self.advance(false)?;
                }
                // Completed OZ-9600 blocks consume the count. A following
                // trigger without reloading it performs no pixel transfers.
                self.set_word(6, 0);
                self.block_operations += 1;
                Ok(0)
            }
            2 => {
                self.check_increments(true)?;
                let previous = self.read_latch;
                self.read_latch = self.read_pixels(self.word(0x14), self.word(0x16));
                self.advance(true)?;
                self.data_reads += 1;
                Ok(previous)
            }
            _ => Ok(self.peek(offset, cycle)),
        }
    }

    pub fn write(&mut self, offset: usize, value: u8) -> Result<(), String> {
        match offset {
            0 => {
                self.check_increments(false)?;
                for _ in 0..self.word(6) {
                    self.write_pixels(self.word(0x10), self.word(0x12), value);
                    self.advance(false)?;
                }
                self.set_word(6, 0);
                self.block_operations += 1;
            }
            2 => {
                self.check_increments(false)?;
                self.write_pixels(self.word(0x10), self.word(0x12), value);
                self.advance(false)?;
                // C65A3 writes a uniform pattern then reads without a dummy
                // fetch. Its first comparison requires the last written
                // byte in the CPU data buffer. Ordinary rectangle reads still
                // prefetch before consuming a potentially different pixel.
                self.read_latch = value;
                self.data_writes += 1;
            }
            7 => self.registers[offset] = value & 3,
            9 => {
                self.registers[9] = value & 15;
                self.registers[0x18..].copy_from_slice(&self.windows[usize::from(value & 15)]);
            }
            0x18..=0x1f => {
                let fixed_bits = match offset {
                    0x19 | 0x1d | 0x1f => 0x7c,
                    0x1b => 0x7e,
                    _ => 0,
                };
                let readback = value | fixed_bits;
                self.windows[usize::from(self.registers[9])][offset - 0x18] = readback;
                self.registers[offset] = readback;
            }
            _ => self.registers[offset] = value,
        }
        Ok(())
    }

    /// The same owned geometry/pixel contract used by shared native and WASM
    /// display exporters. It observes controller RAM without bus side effects.
    pub fn matrix_frame(&self) -> crate::lcd_frame::LcdFrame {
        crate::lcd_frame::LcdFrame::new(self.width, self.height, self.pixels.clone())
            .expect("OZ-9600 controller geometry and logical pixel invariants")
    }

    /// Controller RAM in portable bitmap format. No host fonts or UI drawing.
    /// Output is the full 336-column working area, not a qualified panel crop.
    pub fn pbm(&self) -> Vec<u8> {
        self.matrix_frame().pbm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured_window(lcd: &mut LcdController, bounds: [u16; 4]) {
        for (n, word) in bounds.into_iter().enumerate() {
            let [low, high] = word.to_le_bytes();
            lcd.write(24 + n * 2, low).unwrap();
            lcd.write(25 + n * 2, high).unwrap();
        }
    }

    fn measured_group_read(lcd: &mut LcdController) -> u8 {
        lcd.read(2, 0).unwrap();
        lcd.read(2, 0).unwrap()
    }

    #[test]
    fn physical_block_count_is_consumed_and_requires_reload() {
        // OZ-9600, 13 Oct 2026: original fill/copy sequence versus explicit
        // reloads, and native F74D8 completion reads count zero. Intermediate
        // busy timing remains unqualified; this models completed operations.
        for increment in [5, 6] {
            let mut lcd = LcdController::default();
            measured_window(&mut lcd, [0, 0, 335, 239]);
            lcd.write(3, 255).unwrap();
            lcd.write(4, increment).unwrap();
            lcd.write(5, increment).unwrap();
            let (source, destination, next) = if increment == 5 {
                (208, 212, 1)
            } else {
                (209, 213, -1)
            };
            for (y, value) in [(source, 0xa5), (source + next, 0x5a)] {
                lcd.set_word(16, 300);
                lcd.set_word(18, y as u16);
                lcd.write(2, value).unwrap();
            }
            lcd.set_word(16, 300);
            lcd.set_word(18, destination as u16);
            lcd.set_word(6, 2);
            lcd.write(0, 0x3c).unwrap();
            assert_eq!(lcd.word(6), 0);
            assert_eq!(lcd.read_pixels(300, destination as u16), 0x3c);
            lcd.set_word(20, 300);
            lcd.set_word(22, source as u16);
            lcd.set_word(18, destination as u16);
            lcd.read(0, 0).unwrap(); // No reload after fill: no transfer.
            assert_eq!(lcd.word(22), source as u16);
            assert_eq!(lcd.word(18), destination as u16);
            assert_eq!(lcd.read_pixels(300, destination as u16), 0x3c);
            lcd.set_word(6, 2);
            lcd.read(0, 0).unwrap();
            assert_eq!(lcd.word(6), 0);
            assert_eq!(lcd.read_pixels(300, destination as u16), 0xa5);
            assert_eq!(lcd.read_pixels(300, (destination + next) as u16), 0x5a);
            let coordinates = lcd.registers[16..24].to_vec();
            lcd.read(0, 0).unwrap(); // Completed copy also requires reload.
            assert_eq!(&lcd.registers[16..24], coordinates);
        }
    }

    #[test]
    fn physical_window_readback_preserves_excluded_backing_pixels() {
        // Physical original-window control: each baseline FF, then SET 00.
        // The clipped read was 00; restoring the descriptor revealed FF/C3/FF.
        for (bounds, source, backing) in [
            ([305, 208, 306, 215], 0x00, 0xff),
            ([304, 210, 304, 213], 0x00, 0xc3),
            ([302, 208, 303, 0x200], 0x00, 0xff),
            ([304, 216, 304, 223], 0x3c, 0xff),
            ([304, 216, 304, 735], 0x3c, 0xff),
            ([304, 0, 304, 207], 0x3c, 0xff),
            ([304, 0, 304, 719], 0x3c, 0xff),
        ] {
            let mut lcd = LcdController::default();
            lcd.write(9, 2).unwrap();
            measured_window(&mut lcd, [0, 0, 319, 239]);
            lcd.write(3, 255).unwrap();
            lcd.write(8, 0x80).unwrap();
            for (offset, word) in [(16, 304u16), (18, 208), (20, 304), (22, 208)] {
                let [low, high] = word.to_le_bytes();
                lcd.write(offset, low).unwrap();
                lcd.write(offset + 1, high).unwrap();
            }
            lcd.write(2, 0xff).unwrap();
            assert_eq!(measured_group_read(&mut lcd), 0xff);
            measured_window(&mut lcd, bounds);
            lcd.write(2, source).unwrap();
            assert_eq!(measured_group_read(&mut lcd), 0, "clipped {bounds:?}");
            measured_window(&mut lcd, [0, 0, 319, 239]);
            assert_eq!(measured_group_read(&mut lcd), backing, "backing {bounds:?}");
        }
    }

    #[test]
    fn physical_vertical_raster_modes_match_aa_destination_3c_source() {
        // Physical OZ-9600/LH1553F, 13 Oct 2026: each baseline AA was
        // read back before applying source 3C. Modes 2/6/7 distinguish
        // AND from OR and source inversion from result inversion.
        let observed = [0x3c, 0x96, 0x28, 0xbe, 0xc3, 0x69, 0x82, 0xeb];
        for (mode, expected) in observed.into_iter().enumerate() {
            let mut lcd = LcdController::default();
            measured_window(&mut lcd, [0, 0, 335, 239]);
            lcd.write(3, 255).unwrap();
            lcd.write(16, 48).unwrap();
            lcd.write(18, 88).unwrap();
            lcd.write(20, 48).unwrap();
            lcd.write(22, 88).unwrap();
            lcd.write(8, 0x80).unwrap();
            lcd.write(2, 0xaa).unwrap();
            lcd.read(2, 0).unwrap(); // Prime the read latch.
            assert_eq!(lcd.read(2, 0).unwrap(), 0xaa);
            lcd.write(8, 0x80 | mode as u8).unwrap();
            lcd.write(2, 0x3c).unwrap();
            lcd.read(2, 0).unwrap();
            assert_eq!(lcd.read(2, 0).unwrap(), expected, "mode {mode}");
        }
    }

    fn full_window(lcd: &mut LcdController) {
        let bounds = [0, 0, lcd.width as u16 - 1, lcd.height as u16 - 1];
        measured_window(lcd, bounds);
    }

    #[test]
    fn firmware_window_candidate_keeps_extended_storage_separate_from_oz_bounds() {
        // PV native full-area context has Y end 239 plus the extra bit,
        // while its independent rectangle fixture reaches working row 288.
        for candidate in [false, true] {
            let mut lcd = LcdController::with_geometry(336, 292);
            lcd.firmware_window_candidate = candidate;
            measured_window(&mut lcd, [0, 0, 335, 0x200 | 239]);
            lcd.write(3, 255).unwrap();
            lcd.write(8, 0x80).unwrap();
            for (offset, value) in [(16, 304), (18, 288), (20, 304), (22, 288)] {
                lcd.set_word(offset, value);
            }
            lcd.write(2, 255).unwrap();
            assert_eq!(lcd.pixel(304, 288), candidate);
            measured_window(&mut lcd, [305, 288, 306, 291]);
            assert_eq!(
                measured_group_read(&mut lcd),
                if candidate { 0xf0 } else { 0 }
            );
        }
    }

    #[test]
    fn ordered_window_candidate_preserves_earlier_region_and_advances_masked_blocks() {
        let mut lcd = LcdController::with_geometry(16, 8);
        let define = |lcd: &mut LcdController, index, bounds: [u16; 4], bypass| {
            lcd.write(9, index).unwrap();
            for (n, value) in bounds.iter().enumerate() {
                let [lo, hi] = value.to_le_bytes();
                lcd.write(24 + n * 2, lo).unwrap();
                lcd.write(25 + n * 2, hi | if n == 3 { bypass } else { 0 })
                    .unwrap();
            }
        };
        define(&mut lcd, 1, [0, 0, 15, 7], 2);
        lcd.write(3, 255).unwrap();
        lcd.set_word(16, 2);
        lcd.set_word(18, 2);
        lcd.write(2, 255).unwrap();
        define(&mut lcd, 2, [4, 2, 6, 3], 0);
        define(&mut lcd, 4, [8, 2, 9, 3], 0); // Later windows do not occlude.
        define(&mut lcd, 3, [2, 1, 9, 5], 0);
        lcd.ordered_window_clipping_candidate = true;
        lcd.write(5, 0x10).unwrap();
        lcd.write(2, 0).unwrap();
        for x in 2..10 {
            assert_eq!(lcd.pixel(x, 2), (4..=6).contains(&x));
        }
        assert_eq!(lcd.word(16), 3);
        lcd.write(3, 128).unwrap();
        lcd.set_word(16, 4);
        lcd.set_word(18, 3);
        lcd.set_word(6, 2);
        lcd.write(0, 255).unwrap();
        assert!(!lcd.pixel(4, 3) && !lcd.pixel(5, 3));
        assert_eq!(lcd.word(16), 6); // Clipped transfers still advance.
        lcd.write(9, 2).unwrap();
        lcd.write(2, 255).unwrap();
        assert!(lcd.pixel(6, 3)); // Selected earlier window can draw its edge.
        lcd.write(9, 1).unwrap();
        lcd.set_word(16, 4);
        lcd.set_word(18, 2);
        lcd.write(2, 0).unwrap();
        assert!(!lcd.pixel(4, 2)); // Selected background remains directly writable.
    }

    #[test]
    fn physical_window_overlap_clips_reads_without_enabling_write_candidate() {
        let mut lcd = LcdController::with_geometry(16, 8);
        lcd.registers[3] = 0xff;
        lcd.registers[9] = 3;
        lcd.windows[0] = [8, 0, 0, 0, 15, 0, 7, 0];
        lcd.windows[1] = [0, 0, 0, 0, 15, 0, 7, 2];
        for index in [2, 3] {
            lcd.windows[index] = [0, 0, 0, 0, 7, 0, 7, 0];
        }
        lcd.pixels[..8].fill(1);

        assert!(!lcd.ordered_window_clipping_candidate);
        assert_eq!(lcd.read_pixels(0, 0), 0);
        lcd.windows[2][0] = 8;
        assert_eq!(lcd.read_pixels(0, 0), 0xff);
        lcd.windows[2][0] = 0;
        assert_eq!(lcd.read_pixels(0, 0), 0);
        lcd.write_pixels(0, 0, 0);
        assert_eq!(&lcd.pixels[..8], &[0; 8]);
    }

    #[test]
    fn signed_pixel_stream_crosses_zero_on_both_axes() {
        // Clock's picture save/restore starts at (-1,-1). Pixels outside
        // backing are blank; each remaining bit still has its own position.
        for vertical in [0, 128] {
            for missing in [1_u16, 2] {
                let mut lcd = LcdController::default();
                measured_window(&mut lcd, [0, 0, 335, 239]);
                lcd.write(31, 2).unwrap();
                lcd.write(3, 255).unwrap();
                lcd.write(8, vertical).unwrap();
                let (axis, other) = if vertical == 0 { (16, 18) } else { (18, 16) };
                lcd.set_word(axis, 0);
                lcd.set_word(other, 10);
                lcd.write(2, 0xa5).unwrap();
                lcd.set_word(axis + 4, 0_u16.wrapping_sub(missing));
                lcd.set_word(other + 4, 10);
                lcd.read(2, 0).unwrap(); // documented prefetch dummy
                assert_eq!(lcd.read(2, 0).unwrap(), 0xa5 >> missing);
                lcd.set_word(axis, 0_u16.wrapping_sub(missing));
                lcd.write(2, 0x53).unwrap();
                lcd.set_word(axis + 4, 0);
                lcd.read(2, 0).unwrap();
                let expected = (0x53_u8 << missing) | (0xa5 & ((1 << missing) - 1));
                assert_eq!(lcd.read(2, 0).unwrap(), expected);
            }
        }
    }

    #[test]
    fn scrapbook_mode_three_adds_stroke_without_erasing_existing_ink() {
        // Live mode 3 must agree with the ROM's software OR bitmap path.
        for vertical in [0, 128] {
            let mut lcd = LcdController::default();
            full_window(&mut lcd);
            lcd.write(3, 255).unwrap();
            lcd.write(16, 48).unwrap();
            lcd.write(18, 88).unwrap();
            lcd.write(8, vertical).unwrap();
            lcd.write(2, 0xa5).unwrap();
            lcd.write(8, vertical | 3).unwrap();
            lcd.write(2, 0x42).unwrap();
            lcd.write(2, 0).unwrap();
            for bit in 0..8 {
                let (x, y) = if vertical == 0 {
                    (48 + bit, 88)
                } else {
                    (48, 88 + bit)
                };
                assert_eq!(lcd.pixel(x, y), 0xe7 & (128 >> bit) != 0);
            }
        }
    }

    #[test]
    fn sampling_sync_advances_with_cycles_and_status_reads_do_not_advance_it() {
        let mut lcd = LcdController::default();
        let before = (lcd.registers, lcd.pbm());
        let mut seen_high = false;
        let mut seen_low = false;
        for cycle in 0..300 {
            let status = lcd.peek(1, cycle);
            for _ in 0..8 {
                assert_eq!(lcd.read(1, cycle).unwrap(), status);
            }
            seen_high |= status & SAMPLE_SYNC != 0;
            seen_low |= status & SAMPLE_SYNC == 0;
            assert_eq!(status & FRAME_PHASE, 0);
        }
        assert!(seen_high && seen_low);
        assert_ne!(lcd.peek(1, FRAME_HALF_PERIOD) & FRAME_PHASE, 0);
        assert_eq!(before, (lcd.registers, lcd.pbm()));
    }

    #[test]
    fn unqualified_increment_rejects_before_pixels_latch_coordinates_or_counts_change() {
        // A valid X increment followed by an invalid Y increment used to
        // mutate pixels/latch/X before returning an error. Cover both data
        // ports and both block-transfer directions.
        for (read, port, bad_read) in [
            (false, 2, false),
            (false, 0, false),
            (true, 2, true),
            (true, 0, true),
            (true, 0, false),
        ] {
            let mut lcd = LcdController::default();
            lcd.write(3, 255).unwrap();
            lcd.write(2, 0x81).unwrap();
            lcd.write(0x10, 16).unwrap();
            lcd.write(6, 2).unwrap();
            lcd.write(4, if bad_read { 0x53 } else { 0x50 }).unwrap();
            lcd.write(5, if bad_read { 0x50 } else { 0x53 }).unwrap();
            let before = (
                lcd.registers,
                lcd.pbm(),
                lcd.read_latch,
                lcd.data_writes,
                lcd.data_reads,
                lcd.block_operations,
            );
            let result = if read {
                lcd.read(port, 0).map(|_| ())
            } else {
                lcd.write(port, 255)
            };
            assert!(result.is_err());
            assert_eq!(
                before,
                (
                    lcd.registers,
                    lcd.pbm(),
                    lcd.read_latch,
                    lcd.data_writes,
                    lcd.data_reads,
                    lcd.block_operations
                )
            );
        }
    }
}
