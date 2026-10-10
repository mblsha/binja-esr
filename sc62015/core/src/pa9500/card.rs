// PY_SOURCE: pce500/pa9500/card.py
//! Explicit PA-9C3 logical view. ROM identity and mirror are acquired evidence;
//! SRAM read alias and SSR bit1 presence remain a provisional OZ-style profile.
use sha2::{Digest, Sha256};
pub const ROM_SHA256: &str = "55f3c327abce50a1d44b0076f3bb6384ef64e4d7af3edd4ad9b53dc84ae87332";
pub struct Pa9c3Card {
    rom: Vec<u8>,
    pub sram: Vec<u8>,
}
impl Pa9c3Card {
    pub fn rom(&self) -> &[u8] {
        &self.rom
    }
    pub fn from_images(rom: &[u8], sram: &[u8]) -> Result<Self, String> {
        if rom.len() != 0x20000
            || sram.len() != 0x8000
            || format!("{:x}", Sha256::digest(rom)) != ROM_SHA256
        {
            return Err("PA-9C3 requires the verified 128 KiB ROM and a 32 KiB SRAM image".into());
        }
        Ok(Self {
            rom: rom.to_vec(),
            sram: sram.to_vec(),
        })
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0x30000..=0x3ffff => Some(self.sram[(address as usize - 0x30000) % 0x8000]),
            0x40000..=0x7ffff => Some(self.rom[(address as usize - 0x40000) % 0x20000]),
            _ => None,
        }
    }
    pub fn write(&mut self, address: u32, value: u8) {
        if (0x38000..=0x3ffff).contains(&address) {
            self.sram[(address - 0x38000) as usize] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_guard_and_logical_mirrors() {
        assert!(Pa9c3Card::from_images(&[], &[]).is_err());
        assert!(Pa9c3Card::from_images(&vec![0; 0x20000], &vec![0; 0x8000]).is_err());
        let mut card = Pa9c3Card {
            rom: vec![0x12; 0x20000],
            sram: vec![0x34; 0x8000],
        };
        card.rom[0x111] = 0x56;
        assert_eq!(
            (card.read(0x40111), card.read(0x60111)),
            (Some(0x56), Some(0x56))
        );
        card.write(0x38013, 0xa5);
        assert_eq!(
            (card.read(0x30013), card.read(0x38013)),
            (Some(0xa5), Some(0xa5))
        );
        card.write(0x30013, 0x77);
        card.write(0x40111, 0x77);
        assert_eq!(
            (card.read(0x30013), card.read(0x40111)),
            (Some(0xa5), Some(0x56))
        );
        assert_eq!(card.read(0x20000), None);
    }
}
