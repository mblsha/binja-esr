// PY_SOURCE: pce500/oz9600/tablet.py
//! Physical tablet samples. ADC routing is derived from the unmodified F0
//! firmware; coordinate calibration and electrical timing remain provisional.

#[derive(Default)]
pub struct Tablet {
    pub x: u16,
    pub y: u16,
    pub pressed: bool,
    pub conversion_control: u8,
    pub drive_control: u8,
    pub data_reads: u64,
    low_part_next: bool,
    latched_sample: u16,
}

impl Tablet {
    /// Raw ten-bit inputs, not LCD pixel coordinates or translated UI events.
    /// Returns a press edge for the provisional GA tablet interrupt latch.
    pub fn set_contact(&mut self, x: u16, y: u16, pressed: bool) -> Result<bool, String> {
        if x > 1023 || y > 1023 {
            return Err("tablet ADC samples must be ten-bit values (0..1023)".into());
        }
        let edge = pressed && !self.pressed;
        self.x = x;
        self.y = y;
        self.pressed = pressed;
        Ok(edge)
    }

    pub fn write_conversion_control(&mut self, value: u8) -> Result<(), String> {
        if !matches!(value, 6 | 7) {
            return Err(format!(
                "unimplemented tablet ADC conversion control {value:02X}"
            ));
        }
        // The service manual identifies CHS as the VIN0/VIN1 selector. F0
        // writes 06/07 and then reads bits 9..2 followed by bits 1..0.
        // BUSC/OUTC modes beyond this firmware contract remain unsupported.
        self.conversion_control = value;
        self.low_part_next = false;
        Ok(())
    }

    fn analog_sample(&self) -> u16 {
        if !self.pressed {
            return 0;
        }
        match (self.drive_control, self.conversion_control & 1) {
            (0xa8, 0) => self.x,
            (0x8a, 1) => self.y,
            // The ROM's presence checks require a high byte >= C8. This
            // binary contact source models a firm press at full scale, not
            // measured panel pressure, switch voltages, or settling time.
            _ => 1023,
        }
    }

    pub fn peek_data(&self) -> u8 {
        if self.low_part_next {
            (self.latched_sample & 3) as u8
        } else {
            (self.analog_sample() >> 2) as u8
        }
    }

    pub fn read_data(&mut self) -> Result<u8, String> {
        if !matches!(self.conversion_control, 6 | 7) {
            return Err("tablet ADC read before supported conversion selection".into());
        }
        // Explicit provisional policy: conversion completes at the first
        // architectural read after the ROM has configured its drive switches.
        // Hold that sample across the two bus reads. Peeks cannot complete a
        // conversion or change its read phase; ADC clock latency is pending.
        let value = self.peek_data();
        if !self.low_part_next {
            self.latched_sample = self.analog_sample();
        }
        self.low_part_next = !self.low_part_next;
        self.data_reads += 1;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_split_holds_sample_and_preflight_peeks_do_not_advance_it() {
        let mut tablet = Tablet::default();
        tablet.set_contact(403, 601, true).unwrap();
        tablet.drive_control = 0xa8;
        tablet.write_conversion_control(6).unwrap();
        for _ in 0..8 {
            assert_eq!(tablet.peek_data(), 100);
        }
        assert_eq!(tablet.data_reads, 0);
        assert_eq!(tablet.read_data().unwrap(), 100);
        // A physical movement between byte reads must not tear a conversion.
        tablet.set_contact(404, 602, true).unwrap();
        for _ in 0..8 {
            assert_eq!(tablet.peek_data(), 3);
        }
        assert_eq!(tablet.read_data().unwrap(), 3);
        tablet.drive_control = 0x8a;
        tablet.write_conversion_control(7).unwrap();
        assert_eq!(tablet.read_data().unwrap(), 150);
        assert_eq!(tablet.read_data().unwrap(), 2);
        assert_eq!(tablet.data_reads, 4);
    }

    #[test]
    fn invalid_contact_preserves_state_and_edges_do_not_repeat_while_held() {
        let mut tablet = Tablet::default();
        assert!(tablet.set_contact(1023, 0, true).unwrap());
        assert!(!tablet.set_contact(0, 1023, true).unwrap());
        assert!(tablet.set_contact(1024, 2, false).is_err());
        assert_eq!((tablet.x, tablet.y, tablet.pressed), (0, 1023, true));
        assert!(!tablet.set_contact(0, 1023, false).unwrap());
        tablet.write_conversion_control(6).unwrap();
        assert_eq!(tablet.read_data().unwrap(), 0);
        assert!(tablet.set_contact(0, 1023, true).unwrap());
    }
}
