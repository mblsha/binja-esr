// PY_SOURCE: pce500/pi3000/auxiliary.py
//! Provisional firmware-derived auxiliary status control protocol.
//! No recognition/query data is synthesized. Unknown commands remain errors.

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct Auxiliary {
    written: u32,
    ready_at: Option<u64>,
    complete: bool,
    response: Option<u8>,
    acknowledgment_required: bool,
    pub capture_active: bool,
    pub accepted: Vec<u16>,
    pub accepted_count: u64,
}
impl Auxiliary {
    pub fn mark_written(&mut self, offset: usize) {
        self.written |= 1 << offset;
    }
    pub fn start(&mut self, request: &[u8; 32], cycle: u64) -> Result<(), String> {
        if self.ready_at.is_some() || self.complete {
            return Err("PI auxiliary start before prior completion acknowledgment".into());
        }
        let header = u16::from_le_bytes([request[0], request[1]]);
        let request_bytes = match header {
            0x00fe | 0x01fe | 0x30fb | 0x02fe | 0x00f9 | 0x01f9 | 0x02f9 | 0x03fc | 0x05fe => 2,
            // FE:DDFDB and DE006 send parameter/geometry bytes, then read
            // only DE219-compatible status01 and acknowledge BF.
            0x04fe => 3,
            0x03fe => 8,
            _ => {
                return Err(format!(
                    "unimplemented PI auxiliary command {header:04X}, written mask {:08X}",
                    self.written
                ))
            }
        };
        self.start_packet(
            request,
            cycle,
            request_bytes,
            if header == 0x03fc { None } else { Some(1) },
            true,
        )?;
        if header == 0x03fc {
            self.capture_active = true;
        }
        if header == 0x05fe {
            self.capture_active = false;
        }
        Ok(())
    }
    /// Transport only: the caller owns command, length and response contracts.
    pub fn start_packet(
        &mut self,
        request: &[u8; 32],
        cycle: u64,
        request_bytes: usize,
        response: Option<u8>,
        acknowledgment_required: bool,
    ) -> Result<(), String> {
        if self.ready_at.is_some() || self.complete {
            return Err("PI auxiliary start before prior completion acknowledgment".into());
        }
        if !(2..=32).contains(&request_bytes) {
            return Err("auxiliary packet length must be 2..32".into());
        }
        let expected = u32::MAX >> (32 - request_bytes);
        if self.written != expected {
            let count = if request_bytes == 2 {
                "two"
            } else if request_bytes == 3 {
                "three"
            } else if request_bytes == 8 {
                "eight"
            } else {
                return Err(format!(
                    "auxiliary packet requires exactly {request_bytes} written bytes"
                ));
            };
            return Err(format!(
                "PI auxiliary control requires exactly {count} written bytes"
            ));
        }
        self.written = 0;
        // FD6EA sets IL=1; FD722 decrements it to zero and acknowledges
        // 03FC at FD726 without reading any response byte.
        self.response = response;
        self.acknowledgment_required = acknowledgment_required;
        self.ready_at = Some(cycle.saturating_add(32));
        self.accepted_count += 1;
        if self.accepted.len() < 256 {
            self.accepted
                .push(u16::from_le_bytes([request[0], request[1]]));
        }
        Ok(())
    }
    pub fn advance(&mut self, cycle: u64) {
        if self.ready_at.is_some_and(|ready| cycle >= ready) {
            self.ready_at = None;
            self.complete = self.acknowledgment_required;
        }
    }
    pub fn status(&self) -> u8 {
        2 | if self.ready_at.is_some() {
            0x80
        } else if self.complete {
            0x40
        } else {
            0
        }
    }
    pub fn response(&self, offset: usize) -> Option<u8> {
        if self.complete && offset == 0 {
            self.response
        } else {
            None
        }
    }
    pub fn acknowledge(&mut self, value: u8) -> Result<(), String> {
        if value != 0xbf {
            return Err(format!(
                "unimplemented PI auxiliary acknowledgment {value:02X}"
            ));
        }
        self.complete = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completion_requires_time_and_ack_and_never_exposes_query_bytes() {
        let mut a = Auxiliary::default();
        let mut request = [0; 32];
        request[..2].copy_from_slice(&[0xfe, 1]);
        assert_eq!(a.status(), 2);
        a.mark_written(0);
        a.mark_written(1);
        a.start(&request, 100).unwrap();
        assert_eq!(a.status(), 0x82);
        assert_eq!(a.response(0), None);
        a.advance(131);
        assert_eq!(a.status(), 0x82);
        a.advance(132);
        assert_eq!(a.status(), 0x42);
        assert_eq!(a.response(0), Some(1));
        assert_eq!(a.response(1), None);
        assert!(a.start(&request, 132).is_err());
        assert!(a.acknowledge(255).is_err());
        assert_eq!(a.status(), 0x42);
        a.acknowledge(0xbf).unwrap();
        assert_eq!(a.status(), 2);
        assert_eq!(a.response(0), None);
        a.mark_written(0);
        assert!(a.start(&request, 133).is_err());
        a.mark_written(1);
        request[..2].copy_from_slice(&[0xfb, 0x14]);
        assert!(a.start(&request, 133).unwrap_err().contains("14FB"));
        assert_eq!(a.status(), 2);
        assert_eq!(a.accepted_count, 1);
    }

    #[test]
    fn initialization_and_editor_controls_use_single_byte_acknowledgment() {
        // FE:DE202/DE207/DE20C (00..02F9) and DE211 enter DE219/DE21E.
        for header in [0x00fe_u16, 0x00f9, 0x01f9, 0x02f9] {
            let mut a = Auxiliary::default();
            let mut request = [0; 32];
            request[..2].copy_from_slice(&header.to_le_bytes());
            a.mark_written(0);
            a.mark_written(1);
            a.start(&request, 40).unwrap();
            assert_eq!(a.status(), 0x82);
            a.advance(72);
            assert_eq!(a.response(0), Some(1));
            assert_eq!(a.response(1), None);
            a.acknowledge(0xbf).unwrap();
            assert_eq!(a.status(), 2);
            assert_eq!(a.accepted, [header]);
        }
    }

    #[test]
    fn parameter_controls_require_every_request_byte_and_expose_only_status() {
        for (header, count) in [(0x04fe_u16, 3), (0x03fe, 8)] {
            let mut a = Auxiliary::default();
            let mut request = [0x7f; 32];
            request[..2].copy_from_slice(&header.to_le_bytes());
            for offset in 0..count - 1 {
                a.mark_written(offset);
            }
            assert!(a.start(&request, 40).is_err());
            assert_eq!(a.status(), 2);
            a.mark_written(count - 1);
            a.start(&request, 40).unwrap();
            a.advance(72);
            assert_eq!(a.response(0), Some(1));
            assert_eq!(a.response(1), None);
            a.acknowledge(0xbf).unwrap();
            a.mark_written(count);
            for offset in 0..count {
                a.mark_written(offset);
            }
            assert!(a.start(&request, 80).is_err());
        }
    }

    #[test]
    fn contact_handshake_completes_without_any_response_bytes() {
        let mut a = Auxiliary::default();
        let mut request = [0; 32];
        request[..2].copy_from_slice(&0x03fc_u16.to_le_bytes());
        a.mark_written(0);
        a.mark_written(1);
        a.start(&request, 40).unwrap();
        assert!(a.capture_active);
        assert_eq!(a.status(), 0x82);
        a.advance(72);
        assert_eq!(a.status(), 0x42);
        assert_eq!(a.response(0), None);
        assert_eq!(a.response(1), None);
        a.acknowledge(0xbf).unwrap();
        assert_eq!(a.status(), 2);
        assert_eq!(a.accepted, [0x03fc]);
        request[..2].copy_from_slice(&0x05fe_u16.to_le_bytes());
        a.mark_written(0);
        a.mark_written(1);
        a.start(&request, 80).unwrap();
        assert!(!a.capture_active);
        a.advance(112);
        assert_eq!(a.response(0), Some(1));
        a.acknowledge(0xbf).unwrap();
    }
}
