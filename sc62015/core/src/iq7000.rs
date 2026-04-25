// PY_SOURCE: iq7000/emulator.py:IQ7000Emulator (placeholder)

use crate::memory::MemoryImage;
use crate::{CoreRuntime, Result};

pub const ROM_WINDOW_START: usize = 0x0C0000;
pub const ROM_WINDOW_LEN: usize = 0x40000;
pub const ROM_READONLY_START: u32 = ROM_WINDOW_START as u32;
pub const ROM_READONLY_END: u32 = (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32;
pub const CLOCK_WORKSPACE_START: u32 = 0x01FD20;
pub const CLOCK_WORKSPACE_LEN: usize = 13;
pub const CLOCK_INITIALIZED_FLAG: u32 = 0x01FE72;

fn mask_for_bits(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else if bits == 0 {
        0
    } else {
        (1u32 << bits) - 1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iq7000ClockSeed {
    bytes: [u8; CLOCK_WORKSPACE_LEN],
}

impl Iq7000ClockSeed {
    pub fn from_yyyymmddhhmm(raw: &str) -> std::result::Result<Self, String> {
        if raw.len() != 12 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "IQ-7000 RTC seed must be YYYYMMDDHHMM, got '{raw}'"
            ));
        }
        let month: u32 = raw[4..6]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC month in '{raw}'"))?;
        let day: u32 = raw[6..8]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC day in '{raw}'"))?;
        let hour: u32 = raw[8..10]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC hour in '{raw}'"))?;
        let minute: u32 = raw[10..12]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC minute in '{raw}'"))?;
        if !(1..=12).contains(&month) {
            return Err(format!("invalid IQ-7000 RTC month in '{raw}'"));
        }
        if !(1..=31).contains(&day) {
            return Err(format!("invalid IQ-7000 RTC day in '{raw}'"));
        }
        if hour > 23 {
            return Err(format!("invalid IQ-7000 RTC hour in '{raw}'"));
        }
        if minute > 59 {
            return Err(format!("invalid IQ-7000 RTC minute in '{raw}'"));
        }

        let mut bytes = [0u8; CLOCK_WORKSPACE_LEN];
        bytes[..12].copy_from_slice(raw.as_bytes());
        Ok(Self { bytes })
    }

    pub fn read(&self, addr: u32, bits: u8) -> Option<u32> {
        if bits == 0 {
            return Some(0);
        }
        let width = usize::from(bits.div_ceil(8).clamp(1, 4));
        let start = addr.checked_sub(CLOCK_WORKSPACE_START)? as usize;
        if start >= self.bytes.len() || start + width > self.bytes.len() {
            return None;
        }
        let mut value = 0u32;
        for idx in 0..width {
            value |= (self.bytes[start + idx] as u32) << (idx * 8);
        }
        Some(value & mask_for_bits(bits))
    }

    pub fn apply_to_memory(&self, memory: &mut MemoryImage) {
        for (idx, byte) in self.bytes.iter().copied().enumerate() {
            memory.write_external_byte(CLOCK_WORKSPACE_START + idx as u32, byte);
        }
        memory.write_external_byte(CLOCK_INITIALIZED_FLAG, 1);
    }

    pub fn as_ascii(&self) -> &str {
        std::str::from_utf8(&self.bytes[..12]).unwrap_or("")
    }
}

pub fn load_iq7000_rom_image(rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
    load_iq7000_rom_image_into_memory(&mut rt.memory, rom);
    rt.memory
        .set_readonly_ranges(vec![(ROM_READONLY_START, ROM_READONLY_END)]);
    rt.memory.set_keyboard_bridge(false);
    Ok(())
}

pub fn load_iq7000_rom_image_into_memory(memory: &mut MemoryImage, rom: &[u8]) {
    let src_start = rom.len().saturating_sub(ROM_WINDOW_LEN);
    let slice = &rom[src_start..];
    let copy_len = slice.len().min(ROM_WINDOW_LEN);
    let start_in_slice = slice.len().saturating_sub(copy_len);
    memory.write_external_slice(
        ROM_WINDOW_START,
        &slice[start_in_slice..start_in_slice + copy_len],
    );
}
