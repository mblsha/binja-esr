// PY_SOURCE: pce500/oz9600/rtc.py
//! RTC register behavior established by the OZ-9600 fixed/F0 ROM.
//!
//! Calendar fields are packed binary, not BCD. Experimental cause generation
//! supports ordinary deadlines and the observed Schedule seconds-3F form.
//! Other wildcard forms, silicon matching, hold and physical timing remain
//! unqualified and are never inferred from model selection.

pub const MASK_A: usize = 8;
pub const MASK_B: usize = 9;
pub const CAUSE_A: usize = 10;
pub const CAUSE_B: usize = 11;
pub const CONTROL: usize = 12;
pub const STATUS: usize = 13;
pub const HOLD: u8 = 0x80;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clock {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// Numeric RTC weekday. Its correspondence to weekday names is pending.
    pub weekday: u8,
}

impl Clock {
    fn time_key(self) -> (u16, u8, u8, u8, u8, u8) {
        (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )
    }
    /// F0 D81FD/D8227/D8238/D824F/D826A/D827B establish these fields.
    pub fn decode(bytes: [u8; 5]) -> Result<Self, String> {
        let clock = Self {
            second: bytes[0] & 0x3f,
            minute: bytes[1] & 0x3f,
            hour: (bytes[1] >> 6) | ((bytes[2] & 7) << 2),
            day: bytes[2] >> 3,
            month: bytes[3] & 15,
            year: 1900 + u16::from((bytes[3] >> 4) | ((bytes[4] & 15) << 4)),
            weekday: (bytes[4] >> 4) & 7,
        };
        clock.validate()?;
        Ok(clock)
    }

    fn month_length(self) -> u8 {
        match self.month {
            4 | 6 | 9 | 11 => 30,
            // F0 D888B tests only the low two year bits. Its 1900 result
            // differs from a Gregorian century-exception rule.
            2 if self.year.is_multiple_of(4) => 29,
            2 => 28,
            _ => 31,
        }
    }

    fn validate(self) -> Result<(), String> {
        if !(1900..=2099).contains(&self.year)
            || !(1..=12).contains(&self.month)
            || self.day == 0
            || self.day > self.month_length()
            || self.hour >= 24
            || self.minute >= 60
            || self.second >= 60
            || self.weekday >= 7
        {
            return Err(format!("invalid ordinary RTC date/time: {self:?}"));
        }
        Ok(())
    }

    /// Preserve the unassigned high bits in seconds and the fifth byte.
    pub fn encode(self, previous: [u8; 5]) -> Result<[u8; 5], String> {
        self.validate()?;
        let year = (self.year - 1900) as u8;
        Ok([
            (previous[0] & 0xc0) | self.second,
            self.minute | ((self.hour & 3) << 6),
            (self.hour >> 2) | (self.day << 3),
            self.month | ((year & 15) << 4),
            (previous[4] & 0x80) | (self.weekday << 4) | (year >> 4),
        ])
    }

    fn advance_seconds(mut self, seconds: u64) -> Result<Self, String> {
        self.validate()?;
        let current =
            u64::from(self.hour) * 3600 + u64::from(self.minute) * 60 + u64::from(self.second);
        let total = current
            .checked_add(seconds)
            .ok_or("RTC elapsed time overflow")?;
        self.hour = ((total % 86400) / 3600) as u8;
        self.minute = ((total % 3600) / 60) as u8;
        self.second = (total % 60) as u8;
        for _ in 0..(total / 86400) {
            self.weekday = (self.weekday + 1) % 7;
            self.day += 1;
            if self.day > self.month_length() {
                self.day = 1;
                self.month += 1;
                if self.month > 12 {
                    self.month = 1;
                    self.year += 1;
                    // The behavior beyond the ROM getter's range is unknown.
                    if self.year > 2099 {
                        return Err("RTC year rollover beyond 2099 is not qualified".into());
                    }
                }
            }
        }
        Ok(self)
    }
}

#[derive(Default)]
pub struct Rtc {
    registers: [u8; 32],
}

impl Rtc {
    pub(crate) fn retained_registers(&self) -> [u8; 32] {
        self.registers
    }

    pub(crate) fn restore_retained_registers(&mut self, registers: [u8; 32]) {
        self.registers = registers;
    }

    /// Reads and peeks do not acknowledge causes or advance time.
    pub fn read(&self, offset: usize) -> u8 {
        let offset = offset & 31;
        if offset == STATUS {
            (self.registers[STATUS] & !HOLD) | (self.registers[CONTROL] & HOLD)
        } else {
            self.registers[offset]
        }
    }

    pub fn write(&mut self, offset: usize, value: u8) {
        self.registers[offset & 31] = value;
    }

    pub fn registers(&self) -> Vec<u8> {
        (0..32).map(|offset| self.read(offset)).collect()
    }

    pub fn clock(&self) -> Result<Clock, String> {
        Clock::decode(self.registers[..5].try_into().unwrap())
    }

    /// Explicit elapsed-time experiment. CPU/crystal coupling and advancement
    /// while held are pending; this function makes neither assumption.
    /// Periodic and alarm causes are deliberately not generated here yet.
    pub fn advance_seconds(&mut self, seconds: u64) -> Result<(), String> {
        if self.registers[CONTROL] & HOLD != 0 {
            return Err("RTC time advancement during hold is not qualified".into());
        }
        let previous = self.registers[..5].try_into().unwrap();
        let next = self.clock()?.advance_seconds(seconds)?.encode(previous)?;
        self.registers[..5].copy_from_slice(&next);
        Ok(())
    }

    /// Explicit experimental source policy: cyclic A bits 3/4/5/6 correspond
    /// to second/minute/hour/day, inferred from the manual and ROM callbacks.
    /// A bits 0/1 are ordinary alarm deadlines. D0355 establishes the first
    /// due cause; the second channel and silicon comparison remain hypotheses.
    /// Only the observed sixth-byte 07 form and valid complete date/times
    /// participate. Schedule's seconds-3F form matches any crossed second in
    /// that minute (F0 D8003 and a normal-ROM programmed deadline). Exact silicon
    /// reassert timing and other wildcard fields remain unqualified.
    /// All validation precedes mutation. Masked events still latch/coalesce.
    pub fn advance_experimental_seconds(&mut self, seconds: u64) -> Result<(), String> {
        if self.registers[CONTROL] & HOLD != 0 {
            return Err("RTC time advancement during hold is not qualified".into());
        }
        let old = self.clock()?;
        let new = old.advance_seconds(seconds)?;
        let previous = self.registers[..5].try_into().unwrap();
        let encoded = new.encode(previous)?;
        let mut causes = 0;
        if seconds != 0 {
            causes |= 8;
            if seconds >= 60 - u64::from(old.second) {
                causes |= 16;
            }
            if seconds >= 3600 - (u64::from(old.minute) * 60 + u64::from(old.second)) {
                causes |= 32;
            }
            if seconds
                >= 86400
                    - (u64::from(old.hour) * 3600
                        + u64::from(old.minute) * 60
                        + u64::from(old.second))
            {
                causes |= 64;
            }
            for (channel, offset) in [16, 24].into_iter().enumerate() {
                if self.registers[offset + 5] != 7 {
                    continue;
                }
                let mut bytes: [u8; 5] = self.registers[offset..offset + 5].try_into().unwrap();
                let any_second = bytes[0] & 0x3f == 0x3f;
                if any_second {
                    bytes[0] &= 0xc0;
                }
                if let Ok(alarm) = Clock::decode(bytes) {
                    let matched = if any_second {
                        // Intersection of (old,new] with the closed match minute.
                        // After acknowledgement a subsequent second can reassert;
                        // the unmodified ROM is responsible for rearming the block.
                        let last = Clock {
                            second: 59,
                            ..alarm
                        };
                        old.time_key() < last.time_key() && alarm.time_key() <= new.time_key()
                    } else {
                        old.time_key() < alarm.time_key() && alarm.time_key() <= new.time_key()
                    };
                    if matched {
                        causes |= 1 << channel;
                    }
                }
            }
        }
        self.registers[..5].copy_from_slice(&encoded);
        self.raise_causes(causes, 0);
        Ok(())
    }

    /// Device causes latch independently of masks. The ROM clears causes with
    /// AND writes and may also restore causes with OR writes (F9134/F914A).
    pub fn raise_causes(&mut self, a: u8, b: u8) {
        self.registers[CAUSE_A] |= a;
        self.registers[CAUSE_B] |= b;
    }

    /// The fixed-ROM handler consumes bits 0..6 of both mask/cause pairs.
    /// Bit 7's hardware role is not established by that handler.
    pub fn interrupt_asserted(&self) -> bool {
        ((self.registers[MASK_A] & self.registers[CAUSE_A])
            | (self.registers[MASK_B] & self.registers[CAUSE_B]))
            & 0x7f
            != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_clock(rtc: &mut Rtc, offset: usize, clock: Clock) {
        for (index, byte) in clock.encode([0; 5]).unwrap().into_iter().enumerate() {
            rtc.write(offset + index, byte);
        }
    }

    #[test]
    fn experimental_clock_crosses_deadlines_latches_masked_causes_and_does_not_repeat() {
        let mut rtc = Rtc::default();
        let before = Clock {
            year: 1993,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 58,
            weekday: 5,
        };
        set_clock(&mut rtc, 0, before);
        set_clock(
            &mut rtc,
            16,
            Clock {
                second: 59,
                weekday: 0,
                ..before
            },
        );
        set_clock(
            &mut rtc,
            24,
            Clock {
                year: 1994,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                weekday: 0,
            },
        );
        rtc.write(21, 7);
        rtc.write(29, 7);
        rtc.advance_experimental_seconds(3).unwrap();
        assert_eq!(
            rtc.clock().unwrap(),
            Clock {
                year: 1994,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 1,
                weekday: 6
            }
        );
        // Both crossed deadlines plus all four cyclic boundaries coalesce.
        assert_eq!(rtc.read(CAUSE_A), 0x7b);
        assert!(!rtc.interrupt_asserted());
        rtc.write(MASK_A, 1);
        assert!(rtc.interrupt_asserted());
        for _ in 0..100 {
            assert_eq!(rtc.read(CAUSE_A), 0x7b);
        }
        rtc.write(CAUSE_A, 0);
        rtc.advance_experimental_seconds(1).unwrap();
        assert_eq!(rtc.read(CAUSE_A), 8);
        assert!(!rtc.interrupt_asserted());
        let saved = rtc.registers();
        rtc.advance_experimental_seconds(0).unwrap();
        assert_eq!(rtc.registers(), saved);
    }

    #[test]
    fn experimental_alarm_support_is_limited_and_errors_preserve_clock_and_latches() {
        let mut rtc = Rtc::default();
        let start = Clock {
            year: 2000,
            month: 2,
            day: 28,
            hour: 23,
            minute: 59,
            second: 59,
            weekday: 1,
        };
        set_clock(&mut rtc, 0, start);
        set_clock(
            &mut rtc,
            16,
            Clock {
                month: 2,
                day: 29,
                hour: 0,
                minute: 0,
                second: 0,
                ..start
            },
        );
        rtc.write(21, 0); // Unknown sixth-byte form must not be treated as ordinary.
        for (offset, byte) in [0x3c, 0xfb, 0xfd, 0x7c, 0x0c, 7].into_iter().enumerate() {
            rtc.write(24 + offset, byte);
        }
        rtc.advance_experimental_seconds(1).unwrap();
        assert_eq!(rtc.clock().unwrap().day, 29);
        assert_eq!(rtc.read(CAUSE_A) & 3, 0);
        rtc.write(CONTROL, HOLD);
        let saved = rtc.registers();
        assert!(rtc.advance_experimental_seconds(1).is_err());
        assert_eq!(rtc.registers(), saved);
        rtc.write(CONTROL, 0);
        set_clock(
            &mut rtc,
            0,
            Clock {
                year: 2099,
                month: 12,
                day: 31,
                ..start
            },
        );
        let saved = rtc.registers();
        assert!(rtc.advance_experimental_seconds(1).is_err());
        assert_eq!(rtc.registers(), saved);
    }

    #[test]
    fn schedule_special_seconds_match_each_crossed_second_in_the_programmed_minute() {
        let mut rtc = Rtc::default();
        // Actual normal-ROM Schedule deadline, device basis 1993-02-15 02:15.
        for (offset, byte) in [0x3f, 0x8f, 0x78, 0xd2, 5, 7].into_iter().enumerate() {
            rtc.write(16 + offset, byte);
        }
        let start = Clock {
            year: 1993,
            month: 2,
            day: 15,
            hour: 2,
            minute: 14,
            second: 58,
            weekday: 0,
        };
        set_clock(&mut rtc, 0, start);
        for (seconds, alarm) in [(1, false), (1, true), (1, true), (58, true), (1, false)] {
            rtc.write(CAUSE_A, 0);
            rtc.advance_experimental_seconds(seconds).unwrap();
            assert_eq!(rtc.read(CAUSE_A) & 1 != 0, alarm);
        }
        // A single interval can skip the entire match minute and still latch.
        set_clock(&mut rtc, 0, start);
        rtc.write(CAUSE_A, 0);
        rtc.advance_experimental_seconds(64).unwrap();
        assert_eq!(rtc.read(CAUSE_A) & 1, 1);
        assert!(!rtc.interrupt_asserted());
        rtc.write(MASK_A, 1);
        assert!(rtc.interrupt_asserted());
        // The ordinary clock getter remains strict; 3F is an alarm form only.
        assert!(Clock::decode([0x3f, 0x8f, 0x78, 0xd2, 5]).is_err());
        for second in [0x3c, 0x3d, 0x3e] {
            rtc.write(16, second);
            set_clock(&mut rtc, 0, start);
            rtc.write(CAUSE_A, 0);
            rtc.advance_experimental_seconds(64).unwrap();
            assert_eq!(rtc.read(CAUSE_A) & 1, 0);
        }
    }

    #[test]
    fn elapsed_time_matches_rom_leap_rule_and_preserves_unassigned_bits() {
        for (year, expected_month, expected_day) in [(1900, 2, 29), (2000, 2, 29), (2001, 3, 1)] {
            let start = Clock {
                year,
                month: 2,
                day: 28,
                hour: 23,
                minute: 59,
                second: 59,
                weekday: 6,
            };
            let bytes = start.encode([0xc0, 0, 0, 0, 0x80]).unwrap();
            let mut rtc = Rtc::default();
            for (offset, byte) in bytes.into_iter().enumerate() {
                rtc.write(offset, byte);
            }
            for _ in 0..100 {
                assert_eq!(rtc.clock().unwrap(), start);
            }
            rtc.advance_seconds(1).unwrap();
            assert_eq!(
                rtc.clock().unwrap(),
                Clock {
                    year,
                    month: expected_month,
                    day: expected_day,
                    hour: 0,
                    minute: 0,
                    second: 0,
                    weekday: 0,
                }
            );
            assert_eq!(rtc.read(0) & 0xc0, 0xc0);
            assert_eq!(rtc.read(4) & 0x80, 0x80);
            assert_eq!(rtc.read(CAUSE_A), 0); // source-to-period mapping pending
        }
    }

    #[test]
    fn invalid_clock_and_unqualified_hold_or_year_rollover_fail_atomically() {
        let mut rtc = Rtc::default();
        assert!(rtc.advance_seconds(1).is_err());
        assert_eq!(rtc.registers(), vec![0; 32]);
        let end = Clock {
            year: 2099,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            weekday: 3,
        }
        .encode([0; 5])
        .unwrap();
        for (offset, byte) in end.into_iter().enumerate() {
            rtc.write(offset, byte);
        }
        let previous = rtc.registers();
        assert!(rtc.advance_seconds(1).is_err());
        assert_eq!(rtc.registers(), previous);
        rtc.write(CONTROL, HOLD);
        let previous = rtc.registers();
        assert!(rtc.advance_seconds(1).is_err());
        assert_eq!(rtc.registers(), previous);
    }
}
