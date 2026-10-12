// PY_SOURCE: pce500/pvf1/auxiliary.py
//! PV-F1 FF406/FF40E control transport. Absent mode exposes the acquired idle
//! status and lets ROM time out. The opt-in control candidate reuses the shared
//! two-byte handshake engine, gated to PV-F1-confirmed commands only.
use crate::pi3000::auxiliary::Auxiliary as ControlEngine;

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
            (0x8a04 | 0x8a05 | 0x8a09, _) => Some(0),
            (0x8a40, Self::Control { engine, .. }) => engine.response(0),
            _ => None,
        }
    }
    pub fn advance(&mut self, cycle: u64) {
        if let Self::Control { engine, .. } = self {
            engine.advance(cycle);
        }
    }
    pub fn write(&mut self, address: u32, value: u8, cycle: u64) -> Result<(), String> {
        match address {
            // FE DD3AE/DD3BD drives FD, delay, 02, delay. Control bit1 feedback
            // and reset cancellation remain explicit controller hypotheses.
            0x8a01 if matches!(value, 2 | 0xfd) => {
                if matches!(self, Self::Absent { .. }) {
                    *self = Self::absent();
                } else {
                    *self = Self::control();
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
                    // FE DD77B..DD7B3 sends these two-byte controls,
                    // consumes only status 01 at 08A40, and acknowledges BF.
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
