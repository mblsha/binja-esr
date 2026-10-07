// PY_SOURCE: pce500/oz9600/audio.py
//! Opt-in digital capture of SCR software and provisional clocked CO modes.
//! This is nominal-clock PCM, not a calibrated model of the physical speaker.
use crate::{CoreError, CoreRuntime, Result};
use std::collections::VecDeque;

pub const SAMPLE_RATE: u32 = 48_000;
pub const TIMEBASE_HZ: u64 = 1_024_000;
pub const QUEUE_CAPACITY: usize = SAMPLE_RATE as usize * 2;
const HIGH_LEVEL: i64 = 8192;

#[derive(Debug, serde::Serialize)]
pub struct AudioChunk {
    pub sample_rate: u32,
    pub first_sample: u64,
    pub total_samples: u64,
    pub dropped_samples: u64,
    pub samples: Vec<i16>,
}

#[derive(Default)]
pub struct AudioCapture {
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
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Starting or stopping capture discards host backlog and partial samples.
    /// It never writes SCR, CPU state, RAM, RTC or input contacts.
    pub fn set_enabled(&mut self, enabled: bool) {
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    pub fn reset(&mut self) {
        self.set_enabled(self.enabled);
    }

    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled":self.enabled,"sample_rate":SAMPLE_RATE,"timebase_hz":TIMEBASE_HZ,
            "elapsed_units":self.elapsed_units,"total_samples":self.total_samples,
            "queued_samples":self.samples.len(),"dropped_samples":self.dropped_samples,
            "unsupported_units":self.unsupported_units,
            "provisional_units":self.provisional_units,
            "qualification":"Nominal digital CO: 000/001 low/high, provisional 010/011 2/4 kHz, 100/101 low/high, 110/111 explicit CI input. CI wiring, physical mode table, pitch, amplitude and speaker remain unqualified."
        })
    }

    /// Integrate the level present BEFORE the boundary over its elapsed time.
    /// The instruction's new SCR value begins at the following boundary.
    /// Rational sample integration preserves phase across slices and drains.
    pub fn advance(&mut self, units: u64, scr: u8, off: bool, ci: bool) {
        if !self.enabled {
            return;
        }
        let mode = (scr >> 4) & 7; // ISE (bit 7) is not a buzzer mode bit.
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

impl CoreRuntime {
    pub fn set_oz9600_audio_enabled(&mut self, enabled: bool) -> Result<()> {
        let hardware = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ audio requires OZ hardware".into()))?;
        hardware.borrow_mut().audio.set_enabled(enabled);
        Ok(())
    }

    pub fn take_oz9600_audio(&mut self) -> Result<AudioChunk> {
        let hardware = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ audio requires OZ hardware".into()))?;
        Ok(hardware.borrow_mut().audio.take())
    }
}
