// PY_SOURCE: iq7000/emulator.py:IQ7000Emulator (placeholder)

use crate::llama::opcodes::RegName;
use crate::llama::state::{mask_for, LlamaState};
use crate::memory::MemoryImage;
use crate::{CoreRuntime, Result};

pub const ROM_WINDOW_START: usize = 0x0C0000;
pub const ROM_WINDOW_LEN: usize = 0x40000;
pub const ROM_READONLY_START: u32 = ROM_WINDOW_START as u32;
pub const ROM_READONLY_END: u32 = (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32;
pub const CLOCK_WORKSPACE_START: u32 = 0x01FD20;
pub const CLOCK_WORKSPACE_LEN: usize = 13;
pub const CLOCK_INITIALIZED_FLAG: u32 = 0x01FE72;
const RTC_IOCS_HANDLER_ADDR: u32 = 0x0F31EF;
const RTC_GET_DATETIME_SUBCMD: u8 = 0x44;

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

    fn write_ascii_to(&self, memory: &mut MemoryImage, addr: u32) {
        for (idx, byte) in self.bytes.iter().copied().enumerate() {
            let _ = memory.store(addr.wrapping_add(idx as u32), 8, byte as u32);
        }
    }
}

pub fn maybe_short_circuit_rtc_iocs(
    seed: &Iq7000ClockSeed,
    pc: u32,
    state: &mut LlamaState,
    memory: &mut MemoryImage,
) -> bool {
    if (pc & 0x000F_FFFF) != RTC_IOCS_HANDLER_ADDR {
        return false;
    }
    let subcmd = (state.get_reg(RegName::I) & 0xFF) as u8;
    if subcmd != RTC_GET_DATETIME_SUBCMD {
        return false;
    }

    let dst = state.get_reg(RegName::X);
    seed.write_ascii_to(memory, dst);
    state.set_reg(RegName::FC, 0);
    force_retf(state, memory);
    true
}

fn force_retf(state: &mut LlamaState, memory: &mut MemoryImage) {
    let ret = pop_stack_value(state, memory, 24);
    state.set_pc(ret & mask_for(RegName::PC));
    state.call_depth_dec();
    let _ = state.pop_call_frame();
}

fn pop_stack_value(state: &mut LlamaState, memory: &mut MemoryImage, bits: u8) -> u32 {
    let bytes = bits.div_ceil(8);
    let mask = mask_for(RegName::S);
    let mut value = 0u32;
    let mut sp = state.get_reg(RegName::S);
    for i in 0..bytes {
        let byte = memory.load_with_pc(sp, 8, Some(state.pc())).unwrap_or(0) & 0xFF;
        value |= byte << (8 * i);
        sp = sp.wrapping_add(1) & mask;
    }
    state.set_reg(RegName::S, sp);
    value
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtc_iocs_44_short_circuit_returns_seeded_datetime() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut memory = MemoryImage::new();
        let mut state = LlamaState::new();
        state.set_pc(RTC_IOCS_HANDLER_ADDR);
        state.set_reg(RegName::I, RTC_GET_DATETIME_SUBCMD as u32);
        state.set_reg(RegName::X, 0x002000);
        state.set_reg(RegName::S, 0x000100);
        let _ = memory.store(0x000100, 8, 0x23);
        let _ = memory.store(0x000101, 8, 0x01);
        let _ = memory.store(0x000102, 8, 0x0E);

        assert!(maybe_short_circuit_rtc_iocs(
            &seed,
            RTC_IOCS_HANDLER_ADDR,
            &mut state,
            &mut memory,
        ));

        let actual: Vec<u8> = (0..CLOCK_WORKSPACE_LEN)
            .map(|idx| memory.load(0x002000 + idx as u32, 8).unwrap_or(0) as u8)
            .collect();
        assert_eq!(&actual[..12], b"202604252119");
        assert_eq!(actual[12], 0);
        assert_eq!(state.pc(), 0x0E0123);
        assert_eq!(state.get_reg(RegName::S), 0x000103);
        assert_eq!(state.get_reg(RegName::FC), 0);
    }

    #[test]
    fn rtc_iocs_short_circuit_ignores_unmodeled_subcommands() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut memory = MemoryImage::new();
        let mut state = LlamaState::new();
        state.set_pc(RTC_IOCS_HANDLER_ADDR);
        state.set_reg(RegName::I, 0x45);

        assert!(!maybe_short_circuit_rtc_iocs(
            &seed,
            RTC_IOCS_HANDLER_ADDR,
            &mut state,
            &mut memory,
        ));
    }
}
