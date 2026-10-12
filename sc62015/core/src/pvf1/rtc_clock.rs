// PY_SOURCE: pce500/pvf1/rtc_clock.py
//! PV-F1 RTC A2 clock candidate. Native F1BEB dispatches BFE15=F210A;
//! that callback owns BFE32. The oscillator period is uncalibrated.
use crate::oz9600::rtc::{Rtc, CAUSE_A, CONTROL, HOLD, MASK_A};
use serde_json::{json, Value};

/// Native FAB25..FABCB retains alarm sources A0/A1 across OFF, while
/// discarding periodic causes. Their connection to power wake is a candidate;
/// periodic A2..A6 never assert this inferred ON/power level.
pub fn alarm_wake_level(rtc: &Rtc) -> bool {
    rtc.read(MASK_A) & rtc.read(CAUSE_A) & 3 != 0
}

/// Calendar progression at a nominal model period. PV-F1 FC C01F6/C0228
/// copies these packed fields; CA856..CA8D4 independently extract them.
/// HOLD freezes calendar phase without replay; silicon timing is unqualified.
#[derive(Default)]
pub struct CalendarClock {
    period: Option<u64>,
    phase: u64,
    seconds: u64,
    held: u64,
    uninitialized: u64,
    elapsed: u64,
}
impl CalendarClock {
    pub fn configure(&mut self, period: Option<u64>) -> Result<(), String> {
        if period == Some(0) {
            return Err("PV-F1 calendar candidate requires a positive period".into());
        }
        self.period = period;
        self.phase = 0;
        Ok(())
    }
    pub fn advance(&mut self, rtc: &mut Rtc, elapsed: u64) -> Result<(), String> {
        let Some(period) = self.period else {
            return Ok(());
        };
        if rtc.read(CONTROL) & HOLD != 0 {
            self.held = self.held.wrapping_add(elapsed);
        } else if rtc.clock().is_err() {
            // A fresh board has no valid clock until the ROM initializes it.
            // Do not synthesize a date or replay time from invalid backing.
            self.uninitialized = self.uninitialized.wrapping_add(elapsed);
        } else {
            let total = u128::from(self.phase) + u128::from(elapsed);
            let seconds = (total / u128::from(period)) as u64;
            if seconds != 0 {
                rtc.advance_experimental_seconds(seconds)?;
            }
            self.phase = (total % u128::from(period)) as u64;
            self.seconds = self.seconds.wrapping_add(seconds);
        }
        self.elapsed = self.elapsed.wrapping_add(elapsed);
        Ok(())
    }
    pub fn snapshot(&self) -> Value {
        json!({"second_period_candidate":self.period,"calendar_progression":self.period.is_some(),"phase_units":self.phase,"elapsed_seconds":self.seconds,"held_units":self.held,"uninitialized_units":self.uninitialized,"elapsed_timing_units":self.elapsed,"physical_verified":false,"HOLD_freeze_candidate":true})
    }
}

#[derive(Default)]
pub struct HalfSecondClock {
    period: Option<u64>,
    phase: u64,
    ticks: u64,
    elapsed: u64,
}

impl HalfSecondClock {
    pub fn configure(&mut self, period: Option<u64>) -> Result<(), String> {
        if period == Some(0) {
            return Err("PV-F1 RTC candidate requires a positive period".into());
        }
        self.period = period;
        self.phase = 0;
        Ok(())
    }
    /// Masking/acknowledgement belong to the shared RTC, not the divider.
    pub fn advance(&mut self, elapsed: u64) -> bool {
        let Some(period) = self.period else {
            return false;
        };
        let total = u128::from(self.phase) + u128::from(elapsed);
        let ticks = (total / u128::from(period)) as u64;
        self.phase = (total % u128::from(period)) as u64;
        self.ticks = self.ticks.wrapping_add(ticks);
        self.elapsed = self.elapsed.wrapping_add(elapsed);
        ticks != 0
    }
    pub fn snapshot(&self) -> Value {
        json!({"half_second_period_candidate":self.period,"phase_units":self.phase,"ticks":self.ticks,"elapsed_timing_units":self.elapsed,"cause_A":4,"physical_verified":false})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calendar_rollover_uses_packed_fields_and_latches_masked_sources() {
        let mut clock = CalendarClock::default();
        let mut rtc = Rtc::default();
        clock.configure(Some(4)).unwrap();
        clock.advance(&mut rtc, 100).unwrap(); // ROM has not initialized its date.
        assert_eq!(clock.snapshot()["uninitialized_units"], 100);
        for (i, v) in [0xfb, 0xfb, 0xed, 0xc2, 0xc7].into_iter().enumerate() {
            rtc.write(i, v);
        }
        clock.advance(&mut rtc, 3).unwrap();
        assert!(clock.configure(Some(0)).is_err());
        clock.advance(&mut rtc, 1).unwrap();
        assert_eq!(&rtc.registers()[..5], &[0xc0, 0, 8, 0xc3, 0xd7]);
        assert_eq!(rtc.read(crate::oz9600::rtc::CAUSE_A), 0x78);
        assert!(!rtc.interrupt_asserted());
        rtc.write(crate::oz9600::rtc::MASK_A, 0x10);
        assert!(rtc.interrupt_asserted());
        assert_eq!(clock.snapshot()["elapsed_seconds"], 1);
        assert_eq!(clock.snapshot()["elapsed_timing_units"], 104);
    }
    #[test]
    fn calendar_hold_freezes_partial_phase_and_disabled_time_is_not_replayed() {
        let mut clock = CalendarClock::default();
        let mut rtc = Rtc::default();
        for (i, v) in [0, 0xc0, 8, 0xd1, 5].into_iter().enumerate() {
            rtc.write(i, v);
        }
        clock.configure(Some(4)).unwrap();
        clock.advance(&mut rtc, 2).unwrap();
        rtc.write(CONTROL, HOLD);
        let before = rtc.registers();
        clock.advance(&mut rtc, 12).unwrap();
        assert_eq!(rtc.registers(), before);
        rtc.write(CONTROL, 0);
        clock.advance(&mut rtc, 2).unwrap();
        assert_eq!(rtc.read(0), 1);
        assert_eq!(clock.snapshot()["held_units"], 12);
        clock.configure(None).unwrap();
        clock.advance(&mut rtc, 100).unwrap();
        clock.configure(Some(4)).unwrap();
        clock.advance(&mut rtc, 3).unwrap();
        assert_eq!(rtc.read(0), 1);
        clock.advance(&mut rtc, 1).unwrap();
        assert_eq!(rtc.read(0), 2);
    }
    #[test]
    fn invalid_configuration_preserves_a_partial_divider_phase() {
        let mut clock = HalfSecondClock::default();
        assert!(!clock.advance(u64::MAX));
        clock.configure(Some(4)).unwrap();
        assert!(!clock.advance(3));
        assert!(clock.configure(Some(0)).is_err());
        assert!(clock.advance(1));
        assert_eq!(clock.snapshot()["ticks"], 1);
        assert!(!clock.advance(0));
    }
    #[test]
    fn elapsed_pulses_coalesce_and_disabled_time_is_not_replayed() {
        let mut clock = HalfSecondClock::default();
        clock.configure(Some(4)).unwrap();
        assert!(clock.advance(11));
        assert_eq!(clock.snapshot()["ticks"], 2);
        assert_eq!(clock.snapshot()["phase_units"], 3);
        clock.configure(None).unwrap();
        assert!(!clock.advance(100));
        clock.configure(Some(4)).unwrap();
        assert!(!clock.advance(3));
        assert!(clock.advance(1));
        assert_eq!(clock.snapshot()["ticks"], 3);
    }
}
