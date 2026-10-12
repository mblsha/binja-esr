// PY_SOURCE: pce500/pvf1/rtc_clock.py
//! PV-F1 RTC A2 clock candidate. Native F1BEB dispatches BFE15=F210A;
//! that callback owns BFE32. The oscillator period is uncalibrated.
use serde_json::{json, Value};

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
        json!({"half_second_period_candidate":self.period,"phase_units":self.phase,"ticks":self.ticks,"elapsed_timing_units":self.elapsed,"cause_A":4,"physical_verified":false,"calendar_progression":false})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
