// PY_SOURCE: pce500/oz9600/card.py
//! Explicit OZ-707 logical view, qualified by normal ROM launch/input.
//! SRAM aliases, mapper modes and physical card-present polarity are provisional.
use crate::{memory::MemoryOverlay, CoreError, CoreRuntime, Result};
use sha2::{Digest, Sha256};

pub const ROM_SIZE: usize = 0x20000;
pub const SRAM_SIZE: usize = 0x8000;
pub const ROM_SHA256: &str = "a8a1afb91bf39f07f528a9690a60d45c6cc61232a126fb0fe1ccff95ec112d9d";

pub struct Oz707Card {
    rom: Vec<u8>,
    sram: Vec<u8>,
}
impl Oz707Card {
    pub fn from_images(rom: &[u8], sram: &[u8]) -> std::result::Result<Self, String> {
        if rom.len() != ROM_SIZE || sram.len() != SRAM_SIZE {
            return Err("OZ-707 requires a 128 KiB ROM and 32 KiB SRAM image".into());
        }
        if format!("{:x}", Sha256::digest(rom)) != ROM_SHA256 {
            return Err("OZ-707 ROM identity does not match the verified capture".into());
        }
        Ok(Self {
            rom: rom.to_vec(),
            sram: sram.to_vec(),
        })
    }
    pub fn rom(&self) -> &[u8] {
        &self.rom
    }
    pub fn sram(&self) -> &[u8] {
        &self.sram
    }
    pub fn read(&self, address: u32) -> Option<u8> {
        match address {
            0x30000..=0x3ffff => Some(self.sram[(address as usize - 0x30000) % SRAM_SIZE]),
            0x40000..=0x7ffff => Some(self.rom[(address as usize - 0x40000) % ROM_SIZE]),
            _ => None,
        }
    }
    pub fn write(&mut self, address: u32, value: u8) {
        // Canonical upper SRAM is writable. The lower read-only mirror forces
        // the native card's probe to choose 38000, as in the IQ card fixture.
        // This deliberately does not assert the lower physical write alias.
        if (0x38000..=0x3ffff).contains(&address) {
            self.sram[(address - 0x38000) as usize] = value;
        }
    }
    pub fn presence_ssr(&self, ssr: u8) -> u8 {
        // F9BDD samples SSR bit 1 and EIL bit 6. Only the former is asserted
        // for this ROM-qualified logical card view; hardware wiring is open.
        ssr | 2
    }
}

impl CoreRuntime {
    /// Attach a verified OZ-707 before execution; no captured organizer RAM is
    /// imported. The separate card SRAM remains ordinary guest-writable media.
    pub fn install_oz9600_oz707_card(&mut self, rom: &[u8], sram: &[u8]) -> Result<()> {
        if self.instruction_count() != 0 || self.cycle_count() != 0 {
            return Err(CoreError::Other(
                "attach the OZ card before the first CPU boundary".into(),
            ));
        }
        let hw = self
            .oz9600_hardware
            .clone()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        let card = Oz707Card::from_images(rom, sram).map_err(CoreError::Other)?;
        if hw.borrow().card.is_some() {
            return Err(CoreError::Other("OZ card already attached".into()));
        }
        // Executable card bytes are a static, epoch-bound view, just like the
        // banked system ROM. Callback-backed fetches cannot safely cross the
        // shared scheduler's decode/execution boundary.
        self.memory
            .add_rom_overlay(0x40000, &rom.repeat(2), "oz9600_card_rom");
        let reader = hw.clone();
        let peeker = hw.clone();
        let writer = hw.clone();
        self.memory.add_overlay(MemoryOverlay {
            start: 0x30000,
            end: 0x3ffff,
            name: "oz9600_card_sram".into(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(move |address, pc| {
                Some(reader.borrow_mut().architectural_read(address, pc))
            })),
            preflight_read_handler: Some(Box::new(move |address, _| peeker.borrow().read(address))),
            write_handler: Some(Box::new(move |address, value, pc| {
                writer.borrow_mut().write(address, value, pc);
                true
            })),
            perfetto_thread: None,
        });
        hw.borrow_mut().card = Some(card);
        Ok(())
    }

    /// Export actual card backing without CPU execution or bus side effects.
    /// Main retained RAM/RTC format intentionally does not contain this media.
    pub fn oz9600_card_sram(&self) -> Result<Option<Vec<u8>>> {
        let hw = self
            .oz9600_hardware
            .as_ref()
            .ok_or_else(|| CoreError::Other("OZ hardware unavailable".into()))?;
        Ok(hw.borrow().card.as_ref().map(|card| card.sram().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_sram_and_rom_mirrors_share_one_backing() {
        let mut card = Oz707Card {
            rom: vec![0x12; ROM_SIZE],
            sram: vec![0x34; SRAM_SIZE],
        };
        card.rom[0x111] = 0x56;
        assert_eq!(card.read(0x40111), Some(0x56));
        assert_eq!(card.read(0x60111), Some(0x56));
        card.write(0x38013, 0xa5);
        assert_eq!(card.read(0x30013), Some(0xa5));
        assert_eq!(card.read(0x38013), Some(0xa5));
        card.write(0x30013, 0x77);
        card.write(0x40111, 0x77);
        assert_eq!(card.read(0x30013), Some(0xa5));
        assert_eq!(card.read(0x40111), Some(0x56));
        assert_eq!(card.read(0x80000), None);
        assert_eq!(card.presence_ssr(0xa4), 0xa6);
    }
    #[test]
    fn malformed_media_and_wrong_machine_are_rejected_before_mutation() {
        assert!(Oz707Card::from_images(&[], &[]).is_err());
        assert!(Oz707Card::from_images(&vec![0; ROM_SIZE], &vec![0; SRAM_SIZE]).is_err());
        let mut rt = CoreRuntime::new();
        assert!(rt.install_oz9600_oz707_card(&[], &[]).is_err());
        assert!(rt.oz9600_card_sram().is_err());
    }
}
