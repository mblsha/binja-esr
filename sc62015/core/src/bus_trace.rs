// PY_SOURCE: pce500/emulator.py:PCE500Emulator
//! Versioned fixed-record bus trace. No JSON work is performed by this codec.
use serde::Serialize;
use std::io::{self, Read, Write};

pub const HEADER: &[u8; 8] = b"SCBUS\0\x01\0";
const REGIONS: [&str; 7] = [
    "lcd_primary",
    "lcd_mirror",
    "ce6_rom",
    "ce1_slot",
    "system_ram",
    "main_rom",
    "low_rom",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BusTraceEvent {
    pub index: u64,
    pub kind: &'static str,
    pub region: &'static str,
    pub addr: u32,
    pub value: u8,
    pub bits: u8,
    pub byte_offset: u8,
    pub pc: u32,
    pub instr_index: u64,
    pub cycle: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl BusTraceEvent {
    pub fn write_binary(&self, writer: &mut impl Write) -> io::Result<()> {
        let kind = match self.kind {
            "read" => 0,
            "write" => 1,
            _ => return Err(invalid("invalid bus event kind")),
        };
        let region = REGIONS
            .iter()
            .position(|s| *s == self.region)
            .ok_or_else(|| invalid("invalid bus event region"))? as u8;
        let mut bytes = [0u8; 37];
        bytes[..8].copy_from_slice(&self.index.to_le_bytes());
        bytes[8] = kind;
        bytes[9] = region;
        bytes[10..14].copy_from_slice(&self.addr.to_le_bytes());
        bytes[14] = self.value;
        bytes[15] = self.bits;
        bytes[16] = self.byte_offset;
        bytes[17..21].copy_from_slice(&self.pc.to_le_bytes());
        bytes[21..29].copy_from_slice(&self.instr_index.to_le_bytes());
        bytes[29..37].copy_from_slice(&self.cycle.to_le_bytes());
        writer.write_all(&bytes)
    }

    pub fn read_binary(reader: &mut impl Read) -> io::Result<Option<Self>> {
        let mut bytes = [0u8; 37];
        // Distinguish a clean end of stream from a truncated event.
        match reader.read_exact(&mut bytes[..1]) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            result => result?,
        }
        reader.read_exact(&mut bytes[1..])?;
        let kind = match bytes[8] {
            0 => "read",
            1 => "write",
            _ => return Err(invalid("invalid bus event kind")),
        };
        let region = *REGIONS
            .get(bytes[9] as usize)
            .ok_or_else(|| invalid("invalid bus event region"))?;
        Ok(Some(Self {
            index: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            kind,
            region,
            addr: u32::from_le_bytes(bytes[10..14].try_into().unwrap()),
            value: bytes[14],
            bits: bytes[15],
            byte_offset: bytes[16],
            pc: u32::from_le_bytes(bytes[17..21].try_into().unwrap()),
            instr_index: u64::from_le_bytes(bytes[21..29].try_into().unwrap()),
            cycle: u64::from_le_bytes(bytes[29..37].try_into().unwrap()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_records_roundtrip_and_reject_truncation() {
        for kind in ["read", "write"] {
            for region in REGIONS {
                let event = BusTraceEvent {
                    index: u64::MAX,
                    kind,
                    region,
                    addr: 0xfffff,
                    value: 255,
                    bits: 24,
                    byte_offset: 2,
                    pc: 0xabced,
                    instr_index: u64::MAX - 1,
                    cycle: 123456,
                };
                let mut bytes = Vec::new();
                event.write_binary(&mut bytes).unwrap();
                assert_eq!(
                    BusTraceEvent::read_binary(&mut bytes.as_slice()).unwrap(),
                    Some(event)
                );
                for end in 1..bytes.len() {
                    assert!(BusTraceEvent::read_binary(&mut &bytes[..end]).is_err());
                }
            }
        }
        assert_eq!(BusTraceEvent::read_binary(&mut &[][..]).unwrap(), None);
    }
}
