// PY_SOURCE: pce500/oz9600/tablet.py
//! Physical tablet samples. ADC routing is derived from the unmodified F0
//! firmware; coordinate calibration and electrical timing remain provisional.

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Tablet {
    pub x: u16,
    pub y: u16,
    pub pressed: bool,
    pub conversion_control: u8,
    pub drive_control: u8,
    pub data_reads: u64,
    low_part_next: bool,
    latched_sample: u16,
    // Observation only: assistance can wait for a complete ROM X/Y sample.
    #[serde(default)]
    sample_x_seen: bool,
    #[serde(default)]
    contact_sampled: bool,
    #[serde(default = "default_contact_sample")]
    contact_sample: u16,
}

fn default_contact_sample() -> u16 {
    1023
}

impl Default for Tablet {
    fn default() -> Self {
        Self::with_contact_sample(1023).expect("default ten-bit contact sample")
    }
}

impl Tablet {
    pub(super) fn validate_session(&self) -> Result<(), String> {
        if self.x > 1023
            || self.y > 1023
            || self.latched_sample > 1023
            || self.contact_sample > 1023
        {
            return Err("Invalid saved tablet state".into());
        }
        Ok(())
    }

    #[cfg(feature = "pvf1")]
    pub(crate) fn validate_live_state(&self) -> Result<(), String> {
        if [self.x, self.y, self.latched_sample, self.contact_sample]
            .iter()
            .any(|v| *v > 1023)
        {
            return Err("invalid live ten-bit ADC state".into());
        }
        Ok(())
    }
    /// Explicit model pressure hypothesis. The existing boards retain their
    /// full-scale default. Coordinate routing and held conversion are shared;
    /// voltage interpretation and pressure thresholds belong to each board.
    pub fn with_contact_sample(contact_sample: u16) -> Result<Self, String> {
        if contact_sample > 1023 {
            return Err("contact pressure sample must be ten-bit".into());
        }
        Ok(Self {
            x: 0,
            y: 0,
            pressed: false,
            conversion_control: 0,
            drive_control: 0,
            data_reads: 0,
            low_part_next: false,
            latched_sample: 0,
            sample_x_seen: false,
            contact_sampled: false,
            contact_sample,
        })
    }
    /// Raw ten-bit inputs, not LCD pixel coordinates or translated UI events.
    /// Returns a press edge for the provisional GA tablet interrupt latch.
    pub fn set_contact(&mut self, x: u16, y: u16, pressed: bool) -> Result<bool, String> {
        if x > 1023 || y > 1023 {
            return Err("tablet ADC samples must be ten-bit values (0..1023)".into());
        }
        let edge = pressed && !self.pressed;
        if edge || !pressed {
            self.sample_x_seen = false;
            self.contact_sampled = false;
        }
        self.x = x;
        self.y = y;
        self.pressed = pressed;
        Ok(edge)
    }

    /// Host-assistance observation: both coordinates have been read and the ROM
    /// has returned the panel drive to 10. A later acquisition makes this false
    /// again; a previous X/Y pair cannot make an in-flight release safe.
    /// This does not acknowledge guest input or qualify physical ADC timing.
    pub fn contact_release_ready(&self) -> bool {
        self.pressed && self.contact_sampled && self.drive_control == 0x10
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
            // Explicit board pressure policy, independent of coordinate
            // routing and the held two-read conversion mechanism.
            _ => self.contact_sample,
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
        if self.low_part_next && self.pressed {
            match (self.drive_control, self.conversion_control & 1) {
                (0xa8, 0) => self.sample_x_seen = true,
                (0x8a, 1) if self.sample_x_seen => self.contact_sampled = true,
                _ => {}
            }
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
    fn contact_acknowledgement_requires_both_axes_and_is_observation_only() {
        let mut t = Tablet::default();
        t.set_contact(403, 601, true).unwrap();
        for (drive, control, high, low) in [
            (0xa8, 1, 255, 3),
            (0x8a, 1, 150, 1),
            (0xa8, 0, 100, 3),
            (0x8a, 1, 150, 1),
        ] {
            t.drive_control = drive;
            t.write_conversion_control(6 | control).unwrap();
            assert!(!t.contact_release_ready());
            for _ in 0..3 {
                assert_eq!(t.peek_data(), high);
            }
            assert_eq!(t.read_data().unwrap(), high);
            assert!(!t.contact_release_ready());
            assert_eq!(t.read_data().unwrap(), low);
        }
        assert!(!t.contact_release_ready()); // Y acquisition is still active.
        t.drive_control = 0x10;
        assert!(t.contact_release_ready());
        t.drive_control = 0x8a;
        assert!(!t.contact_release_ready()); // Another acquisition after a full pair.
        t.drive_control = 0x10;
        t.set_contact(404, 602, true).unwrap();
        assert!(t.contact_release_ready()); // Dragging is the same electrical press.
        t.set_contact(404, 602, false).unwrap();
        assert!(!t.contact_release_ready());
        t.set_contact(404, 602, true).unwrap();
        assert!(!t.contact_release_ready());
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
    #[test]
    fn model_pressure_keeps_coordinate_routes_and_holds_both_pressure_parts() {
        let mut t = Tablet::with_contact_sample(513).unwrap();
        t.set_contact(403, 601, true).unwrap();
        t.drive_control = 0x80;
        t.write_conversion_control(6).unwrap();
        assert_eq!(t.peek_data(), 128);
        assert_eq!(t.data_reads, 0);
        assert_eq!(t.read_data().unwrap(), 128);
        t.set_contact(404, 602, false).unwrap();
        assert_eq!(t.read_data().unwrap(), 1);
        t.set_contact(403, 601, true).unwrap();
        t.drive_control = 0xa8;
        t.write_conversion_control(6).unwrap();
        assert_eq!(t.read_data().unwrap(), 100);
        assert_eq!(t.read_data().unwrap(), 3);
        assert!(Tablet::with_contact_sample(1024).is_err());
    }
}
