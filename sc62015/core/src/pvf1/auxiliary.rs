// PY_SOURCE: pce500/pvf1/auxiliary.py
//! PV-F1 FF406/FF40E control transport. Absent mode exposes the acquired idle
//! status and lets ROM time out. The opt-in control candidate reuses the shared
//! handshake engine, gated to PV-F1-confirmed request/response contracts only.
use crate::pi3000::auxiliary::Auxiliary as ControlEngine;
use serde_json::{json, Value};

/// Provisional sample cadence, independent of the acquired register/IRQ ABI.
pub const CAPTURE_SAMPLE_PERIOD_CANDIDATE: u64 = 4096;

#[derive(Default)]
pub struct CaptureClock {
    period: Option<u64>,
    next: Option<u64>,
    ready: bool,
    events: u64,
}
impl CaptureClock {
    fn advance(&mut self, active: bool, cycle: u64) -> bool {
        if !active || self.period.is_none() {
            self.next = None;
            self.ready = false;
            return false;
        }
        if self.ready {
            return false;
        }
        let deadline = *self
            .next
            .get_or_insert_with(|| cycle.saturating_add(self.period.unwrap()));
        if cycle < deadline {
            return false;
        }
        self.next = None;
        self.ready = true;
        self.events += 1;
        true
    }
}

pub enum Auxiliary {
    Absent {
        request: [u8; 32],
        controls: [u8; 3],
        enabled: bool,
        starts: u64,
    },
    Control {
        request: [u8; 32],
        controls: [u8; 3],
        enabled: bool,
        engine: ControlEngine,
        clock: CaptureClock,
    },
}
impl Auxiliary {
    pub fn absent() -> Self {
        Self::Absent {
            request: [0; 32],
            controls: [0; 3],
            enabled: true,
            starts: 0,
        }
    }
    pub fn control() -> Self {
        Self::Control {
            request: [0; 32],
            controls: [0; 3],
            enabled: true,
            engine: ControlEngine::default(),
            clock: CaptureClock::default(),
        }
    }
    pub fn configure_sample_period(&mut self, period: Option<u64>) -> Result<(), String> {
        if period == Some(0) {
            return Err("PV-F1 capture sample period must be positive".into());
        }
        let Self::Control { clock, .. } = self else {
            return Err("capture clock requires the auxiliary control candidate".into());
        };
        clock.period = period;
        clock.next = None;
        clock.ready = false;
        Ok(())
    }
    pub fn snapshot(&self) -> Value {
        match self {
            Self::Absent {
                controls,
                enabled,
                starts,
                ..
            } => {
                json!({"control_candidate":false,"controls":controls.to_vec(),"enabled":enabled,"starts":starts})
            }
            Self::Control {
                controls,
                enabled,
                engine,
                clock,
                ..
            } => {
                json!({"control_candidate":true,"controls":controls.to_vec(),"enabled":enabled,"capture_active":engine.capture_active,"accepted_controls":engine.accepted,"accepted_count":engine.accepted_count,"sample_period_candidate":clock.period,"sample_ready":clock.ready,"sample_events":clock.events})
            }
        }
    }
    pub fn peek(&self, address: u32) -> Option<u8> {
        match (address, self) {
            (0x8a00, Self::Absent { enabled, .. }) => Some(if *enabled { 2 } else { 0 }),
            (
                0x8a00,
                Self::Control {
                    engine, enabled, ..
                },
            ) => Some((engine.status() & !2) | if *enabled { 2 } else { 0 }),
            // FE DE39B/DE3AB explicitly compare writes to06 with reads at04.
            (0x8a04, Self::Absent { controls, .. } | Self::Control { controls, .. }) => {
                Some(controls[0])
            }
            (0x8a05, Self::Control { clock, .. }) => Some(if clock.ready { 0x80 } else { 0 }),
            (0x8a05 | 0x8a09, _) => Some(0),
            (0x8a40, Self::Control { engine, .. }) => engine.response(0),
            _ => None,
        }
    }
    /// One fresh sample-ready edge. ROM acknowledges the80 status through07;
    /// peeks and the GA pending latch do not consume or repeatedly assert it.
    pub fn advance(&mut self, cycle: u64) -> bool {
        if let Self::Control {
            engine,
            clock,
            controls,
            enabled,
            ..
        } = self
        {
            engine.advance(cycle);
            return clock.advance(
                *enabled && engine.capture_active && controls[0] & 0x80 != 0,
                cycle,
            );
        }
        false
    }
    pub fn write(&mut self, address: u32, value: u8, cycle: u64) -> Result<(), String> {
        match address {
            // FE DD3AE/DD3BD drives FD, delay, 02, delay. Control bit1 feedback
            // and reset cancellation remain explicit controller hypotheses.
            0x8a01 if matches!(value, 2 | 0xfd) => {
                let period = match self {
                    Self::Control { clock, .. } => clock.period,
                    Self::Absent { .. } => None,
                };
                if matches!(self, Self::Absent { .. }) {
                    *self = Self::absent();
                } else {
                    *self = Self::control();
                    self.configure_sample_period(period)?;
                }
                let enabled = match self {
                    Self::Absent { enabled, .. } | Self::Control { enabled, .. } => enabled,
                };
                *enabled = value == 2;
                Ok(())
            }
            0x8a20..=0x8a3f => {
                let offset = (address - 0x8a20) as usize;
                match self {
                    Self::Absent { request, .. } => request[offset] = value,
                    Self::Control {
                        request, engine, ..
                    } => {
                        request[offset] = value;
                        engine.mark_written(offset);
                    }
                }
                Ok(())
            }
            // Bank FE DB947/DB94D/DB953 initializes these latches.
            // Interrupt production from them is not yet recovered.
            0x8a06..=0x8a08 => {
                let controls = match self {
                    Self::Absent { controls, .. } | Self::Control { controls, .. } => controls,
                };
                controls[(address - 0x8a06) as usize] = value;
                if let Self::Control { clock, .. } = self {
                    if address == 0x8a07 && value & 0x80 == 0 {
                        clock.ready = false;
                        clock.next = None;
                    }
                }
                Ok(())
            }
            0x8a02 if value == 0x80 => match self {
                Self::Absent { starts, .. } => {
                    *starts += 1;
                    Ok(())
                }
                Self::Control {
                    request, engine, ..
                } => {
                    let command = u16::from_le_bytes([request[0], request[1]]);
                    // FE DD77B..DD7B3 sends two-byte controls; DD55E sends
                    // 04FE plus one parameter (3 bytes); DD589 sends 03FE plus geometry
                    // (8 bytes). These callers consume only status 01 at 08A40
                    // and acknowledge BF; recognition/query bytes remain absent.
                    // Their successful status is an explicit candidate.
                    if !matches!(
                        command,
                        0x03fc
                            | 0x05fe
                            | 0x00fe
                            | 0x01fe
                            | 0x02fe
                            | 0x30fb
                            | 0x00f9
                            | 0x01f9
                            | 0x02f9
                            | 0x04fe
                            | 0x03fe
                    ) {
                        return Err(format!("unrecovered PV-F1 auxiliary command {command:04X}"));
                    }
                    engine.start(request, cycle)
                }
            },
            0x8a03 if value == 0xbf => match self {
                Self::Absent { .. } => Ok(()),
                Self::Control { engine, .. } => engine.acknowledge(value),
            },
            _ => Err(format!(
                "unrecovered PV-F1 auxiliary write {address:05X}={value:02X}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(a: &mut Auxiliary, header: u16, cycle: u64) {
        for (i, b) in header.to_le_bytes().into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, cycle).unwrap();
        }
        a.write(0x8a02, 0x80, cycle).unwrap();
    }

    #[test]
    fn capture_requires_an_explicit_cadence_and_preserves_ready_until_ack() {
        let mut a = Auxiliary::control();
        start(&mut a, 0x03fc, 0);
        assert!(!a.advance(32));
        a.write(0x8a03, 0xbf, 32).unwrap();
        a.write(0x8a06, 0x80, 32).unwrap();
        assert!(!a.advance(10000));
        assert_eq!(a.peek(0x8a05), Some(0));
        a.configure_sample_period(Some(4096)).unwrap();
        assert!(!a.advance(100));
        assert!(!a.advance(4195));
        assert!(a.advance(4196));
        for _ in 0..3 {
            assert_eq!(a.peek(0x8a04), Some(0x80));
            assert_eq!(a.peek(0x8a05), Some(0x80));
            assert_eq!(a.peek(0x8a41), None);
        }
        assert!(!a.advance(100000));
        a.write(0x8a07, 0xff, 100000).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0x80));
        a.write(0x8a07, 0x7f, 100000).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0));
        assert!(!a.advance(100000));
        assert!(a.advance(104096));
        assert_eq!(a.snapshot()["sample_events"], 2);
        start(&mut a, 0x05fe, 104097);
        assert!(!a.advance(104129));
        assert_eq!(a.peek(0x8a05), Some(0));
        assert_eq!(a.peek(0x8a40), Some(1));
        assert_eq!(a.peek(0x8a41), None);
    }

    #[test]
    fn native_control_readback_and_reset_keep_configuration_but_cancel_capture() {
        let mut a = Auxiliary::control();
        a.configure_sample_period(Some(4096)).unwrap();
        for value in [0xaa, 0x55] {
            a.write(0x8a06, value, 0).unwrap();
            assert_eq!(a.peek(0x8a04), Some(value));
        }
        assert!(a.configure_sample_period(Some(0)).is_err());
        assert_eq!(a.snapshot()["sample_period_candidate"], 4096);
        start(&mut a, 0x03fc, 0);
        a.advance(32);
        a.write(0x8a01, 0xfd, 32).unwrap();
        a.write(0x8a01, 2, 33).unwrap();
        assert_eq!(a.snapshot()["sample_period_candidate"], 4096);
        assert_eq!(a.snapshot()["capture_active"], false);
        assert!(!a.advance(500000));
        a.configure_sample_period(None).unwrap();
        assert_eq!(a.snapshot()["sample_period_candidate"], Value::Null);
    }
}
