// PY_SOURCE: pce500/digital_audio.py
//! Opt-in digital capture of SCR software and provisional clocked CO modes.
//! This is nominal-clock PCM, not a calibrated model of the physical speaker.
use std::collections::VecDeque;

pub const SAMPLE_RATE: u32 = 48_000;
pub const TIMEBASE_HZ: u64 = 1_024_000;
pub const QUEUE_CAPACITY: usize = SAMPLE_RATE as usize * 2;
const HIGH_LEVEL: i64 = 8192;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AudioModePolicy {
    #[default]
    ProvisionalClocked,
    SoftwareOnly,
    PvF1Firmware,
}

#[derive(Debug, serde::Serialize)]
pub struct AudioChunk {
    pub sample_rate: u32,
    pub first_sample: u64,
    pub total_samples: u64,
    pub dropped_samples: u64,
    pub samples: Vec<i16>,
}

#[derive(Default, serde::Serialize, serde::Deserialize, Clone)]
pub struct AudioCapture {
    policy: AudioModePolicy,
    mode_units: [u64; 8],
    enabled: bool,
    phase: u64,
    area: i64,
    samples: VecDeque<i16>,
    total_samples: u64,
    dropped_samples: u64,
    elapsed_units: u64,
    unsupported_units: u64,
    provisional_units: u64,
    mode: Option<u8>,
    oscillator_phase: u64,
    oscillator_high: bool,
}

impl AudioCapture {
    pub fn with_policy(policy: AudioModePolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Starting or stopping capture discards host backlog and partial samples.
    /// It never writes SCR, CPU state, RAM, RTC or input contacts.
    pub fn set_enabled(&mut self, enabled: bool) {
        *self = Self {
            enabled,
            policy: self.policy,
            ..Self::default()
        };
    }

    pub fn reset(&mut self) {
        self.set_enabled(self.enabled);
    }

    pub fn status(&self) -> serde_json::Value {
        let mut status = serde_json::json!({
            "enabled":self.enabled,"sample_rate":SAMPLE_RATE,"timebase_hz":TIMEBASE_HZ,
            "elapsed_units":self.elapsed_units,"total_samples":self.total_samples,
            "queued_samples":self.samples.len(),"dropped_samples":self.dropped_samples,
            "unsupported_units":self.unsupported_units,
            "provisional_units":self.provisional_units,
            "qualification":match self.policy {
                AudioModePolicy::ProvisionalClocked => "Nominal digital CO: 000/001 low/high, provisional 010/011 2/4 kHz, 100/101 low/high, 110/111 explicit CI input. CI wiring, physical mode table, pitch, amplitude and speaker remain unqualified.",
                AudioModePolicy::SoftwareOnly => "Nominal software-driven CO: BZ2/BZ1 clear, BZ0 low/high. Other modes are silent and counted unsupported; physical pitch, amplitude and speaker remain unqualified.",
                AudioModePolicy::PvF1Firmware => "Nominal CO: software BZ0 and provisional BZ1 2/4 kHz. Modes 100..111 are silent and counted unsupported; physical pitch, amplitude and speaker remain unqualified."
            }
        });
        if self.policy != AudioModePolicy::ProvisionalClocked {
            status["mode_units"] = serde_json::json!(self.mode_units);
        }
        status
    }

    /// Integrate the level present BEFORE the boundary over its elapsed time.
    /// The instruction's new SCR value begins at the following boundary.
    /// Rational sample integration preserves phase across slices and drains.
    pub fn advance(&mut self, units: u64, scr: u8, off: bool, ci: bool) {
        if !self.enabled {
            return;
        }
        let mode = (scr >> 4) & 7; // ISE (bit 7) is not a buzzer mode bit.
        if self.policy != AudioModePolicy::ProvisionalClocked {
            if !off {
                self.mode_units[mode as usize] = self.mode_units[mode as usize].wrapping_add(units);
            }
            let limit = if self.policy == AudioModePolicy::SoftwareOnly {
                1
            } else {
                3
            };
            if mode > limit {
                self.mode = None;
                self.elapsed_units = self.elapsed_units.wrapping_add(units);
                if !off {
                    self.unsupported_units = self.unsupported_units.wrapping_add(units);
                }
                self.integrate(units, 0);
                return;
            }
        }
        if !off && mode >= 6 {
            // The CI source is explicit hardware backing, not a guessed SSR
            // bit alias. Its physical connection remains unavailable.
            self.unsupported_units = self.unsupported_units.wrapping_add(units);
        }
        if !off && mode > 1 {
            self.provisional_units = self.provisional_units.wrapping_add(units);
        }
        self.elapsed_units = self.elapsed_units.wrapping_add(units);
        let active_mode = (!off).then_some(mode);
        if active_mode != self.mode {
            self.mode = active_mode;
            self.oscillator_phase = 0;
            self.oscillator_high = true;
        }
        if !off && matches!(mode, 2 | 3) {
            let half_period = TIMEBASE_HZ / if mode == 2 { 4000 } else { 8000 };
            let mut remaining = units;
            while remaining != 0 {
                let part = remaining.min(half_period - self.oscillator_phase);
                self.integrate(part, if self.oscillator_high { HIGH_LEVEL } else { 0 });
                self.oscillator_phase += part;
                remaining -= part;
                if self.oscillator_phase == half_period {
                    self.oscillator_phase = 0;
                    self.oscillator_high = !self.oscillator_high;
                }
            }
        } else {
            let high = !off && (matches!(mode, 1 | 5) || (mode >= 6 && ci));
            self.integrate(units, if high { HIGH_LEVEL } else { 0 });
        }
    }

    fn integrate(&mut self, units: u64, level: i64) {
        let mut remaining = u128::from(units) * u128::from(SAMPLE_RATE);
        while remaining != 0 {
            let part = remaining.min(u128::from(TIMEBASE_HZ - self.phase)) as u64;
            self.area += level * part as i64;
            self.phase += part;
            remaining -= u128::from(part);
            if self.phase == TIMEBASE_HZ {
                if self.samples.len() == QUEUE_CAPACITY {
                    self.samples.pop_front();
                    self.dropped_samples = self.dropped_samples.wrapping_add(1);
                }
                self.samples
                    .push_back((self.area / TIMEBASE_HZ as i64) as i16);
                self.total_samples = self.total_samples.wrapping_add(1);
                self.phase = 0;
                self.area = 0;
            }
        }
    }

    /// Samples own their storage. Draining does not change integration phase.
    pub fn take(&mut self) -> AudioChunk {
        AudioChunk {
            sample_rate: SAMPLE_RATE,
            first_sample: self.total_samples.wrapping_sub(self.samples.len() as u64),
            total_samples: self.total_samples,
            dropped_samples: self.dropped_samples,
            samples: self.samples.drain(..).collect(),
        }
    }
}

impl crate::CoreRuntime {
    /// Opt-in observer routed through the configured model's capture policy.
    pub fn set_audio_enabled(&mut self, enabled: bool) -> crate::Result<()> {
        #[cfg(feature = "oz9600")]
        if let Some(hardware) = &self.oz9600_hardware {
            hardware.borrow_mut().audio.set_enabled(enabled);
            return Ok(());
        }
        #[cfg(feature = "pvf1")]
        if let Some(hardware) = &self.pvf1_hardware {
            hardware.lock().unwrap().audio.set_enabled(enabled);
            return Ok(());
        }
        let _ = enabled;
        Err(crate::CoreError::Other(
            "Audio is unavailable for this model".into(),
        ))
    }

    pub fn take_audio(&mut self) -> crate::Result<AudioChunk> {
        #[cfg(feature = "oz9600")]
        if let Some(hardware) = &self.oz9600_hardware {
            return Ok(hardware.borrow_mut().audio.take());
        }
        #[cfg(feature = "pvf1")]
        if let Some(hardware) = &self.pvf1_hardware {
            return Ok(hardware.lock().unwrap().audio.take());
        }
        Err(crate::CoreError::Other(
            "Audio is unavailable for this model".into(),
        ))
    }

    pub fn audio_status(&self) -> crate::Result<serde_json::Value> {
        #[cfg(feature = "oz9600")]
        if let Some(hardware) = &self.oz9600_hardware {
            return Ok(hardware.borrow().audio.status());
        }
        #[cfg(feature = "pvf1")]
        if let Some(hardware) = &self.pvf1_hardware {
            return Ok(hardware.lock().unwrap().audio.status());
        }
        Err(crate::CoreError::Other(
            "Audio is unavailable for this model".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_profile_keeps_unknown_modes_silent_and_survives_reset() {
        let mut capture = AudioCapture::with_policy(AudioModePolicy::SoftwareOnly);
        capture.set_enabled(true);
        capture.advance(TIMEBASE_HZ / 100, 0x90, false, false); // ISE | BZ0.
        let high = capture.take();
        assert_eq!(high.samples, vec![8192; 480]);
        capture.advance(TIMEBASE_HZ / 100, 0x20, false, false); // BZ1, unsupported.
        assert_eq!(capture.take().samples, vec![0; 480]);
        assert_eq!(capture.status()["unsupported_units"], TIMEBASE_HZ / 100);
        capture.reset();
        capture.advance(TIMEBASE_HZ / 100, 0x70, false, true); // Unknown CI mode.
        assert_eq!(capture.take().samples, vec![0; 480]);
        capture.advance(TIMEBASE_HZ / 100, 0x10, true, false); // OFF is quiet.
        assert_eq!(capture.take().samples, vec![0; 480]);
        capture.set_enabled(false);
        capture.advance(TIMEBASE_HZ, 0x10, false, false);
        assert!(capture.take().samples.is_empty());
    }

    #[test]
    fn pv_profile_renders_native_clocked_modes_but_leaves_ci_unsupported() {
        let mut capture = AudioCapture::with_policy(AudioModePolicy::PvF1Firmware);
        capture.set_enabled(true);
        capture.advance(TIMEBASE_HZ / 100, 0x30, false, false); // BZ1 | BZ0.
        let samples = capture.take().samples;
        assert_eq!(samples.len(), 480);
        assert!(samples.iter().any(|s| *s > 0));
        assert!(samples.contains(&0));
        assert_eq!(capture.status()["unsupported_units"], 0);
        assert_eq!(capture.status()["provisional_units"], TIMEBASE_HZ / 100);
        capture.advance(TIMEBASE_HZ / 100, 0x70, false, true);
        assert_eq!(capture.take().samples, vec![0; 480]);
        assert_eq!(capture.status()["unsupported_units"], TIMEBASE_HZ / 100);
    }
}
