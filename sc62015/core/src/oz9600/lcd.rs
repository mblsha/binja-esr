// PY_SOURCE: pce500/oz9600/lcd.py
//! LH1553F exploration driven by the fixed ROM's 084xx accesses.
//!
//! Register names and timing are still being qualified. These operations cover
//! Pixel streams, masked blocks, independent read/write coordinates, and
//! the one-byte read pipeline used by F73A2/F767B. SET (0), XOR (1) and
//! Scrapbook's OR (3) follow firmware behavior. Mode 2 and inversion of the
//! combined result remain hypotheses requiring silicon qualification.
//! Sixteen window descriptors and register readback are checked against the
//! F0 diagnostic loops. Ordinary writes clip to inclusive signed window bounds;
//! the Y-end bit-9 bypass, read clipping, and reset geometry are provisional.
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

pub struct LcdController {
    pub registers: [u8; 32],
    windows: [[u8; 8]; 16],
    pixels: Vec<u8>,
    read_latch: u8,
    pub data_writes: u64,
    pub data_reads: u64,
    pub block_operations: u64,
}

impl Default for LcdController {
    fn default() -> Self {
        // Diagnostic C62C0/C62DE require these unimplemented high bits to
        // read as ones. This is a zero descriptor, not a full-screen reset
        // window; the physical reset geometry has not been qualified.
        let descriptor = [0, 0x7c, 0, 0x7e, 0, 0x7c, 0, 0x7c];
        let mut registers = [0; 32];
        registers[0x18..].copy_from_slice(&descriptor);
        Self {
            registers,
            windows: [descriptor; 16],
            pixels: vec![0; STORAGE_WIDTH * HEIGHT],
            read_latch: 0,
            data_writes: 0,
            data_reads: 0,
            block_operations: 0,
        }
    }
}

impl LcdController {
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
        x < STORAGE_WIDTH && y < HEIGHT && self.pixels[y * STORAGE_WIDTH + x] != 0
    }

    fn bit_location(&self, x: u16, y: u16, bit: usize) -> (usize, usize) {
        if self.registers[8] & 0x80 != 0 {
            (usize::from(x), usize::from(y) + bit)
        } else {
            (usize::from(x) + bit, usize::from(y))
        }
    }

    fn read_pixels(&self, x: u16, y: u16) -> u8 {
        let mut value = 0;
        for bit in 0..8 {
            let (px, py) = self.bit_location(x, y, bit);
            if self.pixel(px, py) {
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

    fn permits_write(&self, x: usize, y: usize) -> bool {
        if x >= STORAGE_WIDTH || y >= HEIGHT {
            return false;
        }
        let window = self.window_descriptor(usize::from(self.registers[9]));
        // FA242/FA2CF set this flag for the background/full-area context;
        // FA3A9 clears it for normal clipping windows. Bypass is inferred
        // from those paired uses, not established by a controller datasheet.
        if window[7] & 2 != 0 {
            return true;
        }
        let xmin = Self::window_coordinate(window[0], window[1], 3);
        let ymin = Self::window_coordinate(window[2], window[3], 1);
        let xmax = Self::window_coordinate(window[4], window[5], 3);
        let ymax = Self::window_coordinate(window[6], window[7], 1);
        (xmin..=xmax).contains(&(x as i32)) && (ymin..=ymax).contains(&(y as i32))
    }

    fn write_pixels(&mut self, x: u16, y: u16, value: u8) {
        for bit in 0..8 {
            let mask = 0x80 >> bit;
            let (px, py) = self.bit_location(x, y, bit);
            if self.registers[3] & mask != 0 && self.permits_write(px, py) {
                let destination = &mut self.pixels[py * STORAGE_WIDTH + px];
                let source = u8::from(value & mask != 0);
                let combined = match self.registers[8] & 3 {
                    0 => source,
                    1 => *destination ^ source,
                    2 => *destination | source,
                    // Scrapbook issues live mode 3 at its native line entry,
                    // then ORs the same stroke into its saved software bitmap.
                    // AND hid live strokes until a later SET bitmap redraw.
                    3 => *destination | source,
                    _ => unreachable!("two-bit raster operation"),
                };
                // Calendar's F6E79 solid pattern uses mode 1 over an existing
                // date: XOR preserves inverse glyphs. Notebook's F7157 uses
                // mode 5 with zero data to complement its existing title.
                // Mode 2 retains its unqualified OR hypothesis. Bit-2 result
                // inversion still needs physical-controller qualification.
                *destination = combined ^ ((self.registers[8] >> 2) & 1);
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
                let sample_phase = (cycle % FRAME_HALF_PERIOD * HEIGHT as u64) % FRAME_HALF_PERIOD;
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
        crate::lcd_frame::LcdFrame::new(STORAGE_WIDTH, HEIGHT, self.pixels.clone())
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

    #[test]
    fn scrapbook_mode_three_adds_stroke_without_erasing_existing_ink() {
        // Live mode 3 must agree with the ROM's software OR bitmap path.
        for vertical in [0, 128] {
            let mut lcd = LcdController::default();
            lcd.write(31, 2).unwrap();
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
