// PY_SOURCE: pce500/pi3000/auxiliary.py
//! Provisional firmware-derived auxiliary startup control protocol.
//! No recognition/query data is synthesized. Unknown commands remain errors.

#[derive(Default)]
pub struct Auxiliary {
    written: u32,
    ready_at: Option<u64>,
    complete: bool,
    response: [u8; 32],
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
        if !matches!(header, 0x00fe | 0x01fe | 0x30fb | 0x02fe) {
            return Err(format!(
                "unimplemented PI auxiliary command {header:04X}, written mask {:08X}",
                self.written
            ));
        }
        if self.written != 3 {
            return Err("PI auxiliary startup control requires exactly two written bytes".into());
        }
        self.written = 0;
        self.response = [0; 32];
        self.response[0] = 1;
        self.ready_at = Some(cycle.saturating_add(32));
        self.accepted_count += 1;
        if self.accepted.len() < 256 {
            self.accepted.push(header);
        }
        Ok(())
    }
    pub fn advance(&mut self, cycle: u64) {
        if self.ready_at.is_some_and(|ready| cycle >= ready) {
            self.ready_at = None;
            self.complete = true;
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
            Some(self.response[0])
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
    fn initialization_control_uses_the_same_single_byte_acknowledgment_contract() {
        // FE:DE211 enters the same DE219/DE21E response path as 01FE.
        let mut a = Auxiliary::default();
        let mut request = [0; 32];
        request[0] = 0xfe;
        a.mark_written(0);
        a.mark_written(1);
        a.start(&request, 40).unwrap();
        assert_eq!(a.status(), 0x82);
        a.advance(72);
        assert_eq!(a.response(0), Some(1));
        assert_eq!(a.response(1), None);
        a.acknowledge(0xbf).unwrap();
        assert_eq!(a.status(), 2);
        assert_eq!(a.accepted, [0x00fe]);
    }
}
