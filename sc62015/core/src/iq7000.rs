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
pub const IMEM_EOL_OFFSET: u32 = 0xF3;
pub const IMEM_EIL_OFFSET: u32 = 0xF5;
const EOL_STROBE: u8 = 0x01;
const EOL_OUT_DATA: u8 = 0x02;
const EIL_IN_DATA: u8 = 0x08;
const EIL_READY: u8 = 0x10;
const RTC_COMMAND_CURRENT_DATETIME: u8 = 0xF4;

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

    pub fn rtc_datetime_bcd(&self) -> [u8; 6] {
        let digits = &self.bytes[..12];
        [
            packed_bcd(digits[0], digits[1]),
            packed_bcd(digits[2], digits[3]),
            packed_bcd(digits[4], digits[5]),
            packed_bcd(digits[6], digits[7]),
            packed_bcd(digits[8], digits[9]),
            packed_bcd(digits[10], digits[11]),
        ]
    }
}

fn packed_bcd(tens: u8, ones: u8) -> u8 {
    ((tens - b'0') << 4) | (ones - b'0')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtcWritePhase {
    Idle,
    ReadyHigh,
    AwaitData,
    ReadyLow,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtcReadPhase {
    Idle,
    ReadyHigh,
    ReadyLow,
    Sample,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iq7000RtcPeripheral {
    seed: Iq7000ClockSeed,
    eol: u8,
    write_phase: RtcWritePhase,
    write_acc: u8,
    write_bits: u8,
    read_phase: RtcReadPhase,
    response_bits: Vec<bool>,
    response_index: usize,
    payload_remaining: u8,
    response_after_payload: Vec<u8>,
    current_read_data: u8,
    last_command: Option<u8>,
}

impl Iq7000RtcPeripheral {
    pub fn new(seed: Iq7000ClockSeed) -> Self {
        Self {
            seed,
            eol: 0,
            write_phase: RtcWritePhase::Idle,
            write_acc: 0,
            write_bits: 0,
            read_phase: RtcReadPhase::Idle,
            response_bits: Vec::new(),
            response_index: 0,
            payload_remaining: 0,
            response_after_payload: Vec::new(),
            current_read_data: 0,
            last_command: None,
        }
    }

    pub fn set_seed(&mut self, seed: Iq7000ClockSeed) {
        self.seed = seed;
        self.reset_protocol();
    }

    pub fn seed(&self) -> &Iq7000ClockSeed {
        &self.seed
    }

    pub fn handle_eol_write(&mut self, value: u8) {
        let strobe_was_low = (self.eol & EOL_STROBE) == 0;
        self.eol = value;

        if (value & EOL_STROBE) == 0 {
            self.write_phase = RtcWritePhase::Idle;
            self.read_phase = RtcReadPhase::Idle;
            return;
        }

        if self.has_pending_response() {
            if strobe_was_low || self.read_phase == RtcReadPhase::Idle {
                self.read_phase = RtcReadPhase::ReadyHigh;
            }
            return;
        }

        if strobe_was_low
            || matches!(
                self.write_phase,
                RtcWritePhase::Idle | RtcWritePhase::Complete
            )
        {
            self.write_phase = RtcWritePhase::ReadyHigh;
        } else if self.write_phase == RtcWritePhase::AwaitData {
            self.latch_host_bit((value & EOL_OUT_DATA) != 0);
            self.write_phase = RtcWritePhase::ReadyLow;
        }
    }

    pub fn handle_eil_read(&mut self) -> u8 {
        if self.write_phase == RtcWritePhase::ReadyLow {
            self.write_phase = RtcWritePhase::Complete;
            return 0;
        }

        if self.has_pending_response() && (self.eol & EOL_STROBE) != 0 {
            return self.next_response_eil();
        }

        match self.write_phase {
            RtcWritePhase::ReadyHigh => {
                self.write_phase = RtcWritePhase::AwaitData;
                EIL_READY
            }
            _ => 0,
        }
    }

    fn has_pending_response(&self) -> bool {
        self.response_index < self.response_bits.len()
    }

    fn reset_protocol(&mut self) {
        self.write_phase = RtcWritePhase::Idle;
        self.write_acc = 0;
        self.write_bits = 0;
        self.read_phase = RtcReadPhase::Idle;
        self.response_bits.clear();
        self.response_index = 0;
        self.payload_remaining = 0;
        self.response_after_payload.clear();
        self.current_read_data = 0;
        self.last_command = None;
    }

    fn latch_host_bit(&mut self, bit: bool) {
        if bit {
            self.write_acc |= 1 << self.write_bits;
        }
        self.write_bits += 1;
        if self.write_bits == 8 {
            let byte = self.write_acc;
            self.write_acc = 0;
            self.write_bits = 0;
            if self.payload_remaining > 0 {
                self.payload_remaining -= 1;
                if self.payload_remaining == 0 && !self.response_after_payload.is_empty() {
                    let response = std::mem::take(&mut self.response_after_payload);
                    self.queue_response_bytes(&response);
                }
            } else {
                self.accept_command(byte);
            }
        }
    }

    fn accept_command(&mut self, command: u8) {
        self.last_command = Some(command);
        self.response_bits.clear();
        self.response_index = 0;
        self.payload_remaining = 0;
        self.response_after_payload.clear();
        match command {
            0xF0 | 0xF1 => {
                self.payload_remaining = 6;
                self.response_after_payload.push(0);
            }
            0xF2 => {
                self.payload_remaining = 2;
                self.response_after_payload.push(0);
            }
            RTC_COMMAND_CURRENT_DATETIME | 0xF5 => {
                self.queue_response_bytes(&self.seed.rtc_datetime_bcd());
            }
            0xF6 => self.queue_response_bytes(&[0, 0]),
            0xF7 => self.queue_response_bytes(&[0, 0, 0, 0]),
            0xF8 | 0xFD => self.queue_response_bytes(&[0]),
            _ => {}
        }
    }

    fn queue_response_bytes(&mut self, bytes: &[u8]) {
        self.response_bits.clear();
        self.response_index = 0;
        for byte in bytes {
            // The ROM read helper XORs the assembled byte with 0xFF before returning it.
            let wire_byte = !byte;
            for bit in 0..8 {
                self.response_bits.push(((wire_byte >> bit) & 1) != 0);
            }
        }
    }

    fn next_response_eil(&mut self) -> u8 {
        match self.read_phase {
            RtcReadPhase::Idle | RtcReadPhase::ReadyHigh => {
                self.read_phase = RtcReadPhase::ReadyLow;
                EIL_READY
            }
            RtcReadPhase::ReadyLow => {
                self.current_read_data = if self.response_bits[self.response_index] {
                    EIL_IN_DATA
                } else {
                    0
                };
                self.read_phase = RtcReadPhase::Sample;
                self.current_read_data
            }
            RtcReadPhase::Sample => {
                let value = self.current_read_data;
                self.response_index += 1;
                self.read_phase = if self.has_pending_response() {
                    RtcReadPhase::ReadyHigh
                } else {
                    RtcReadPhase::Idle
                };
                value
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn host_write_byte(peripheral: &mut Iq7000RtcPeripheral, byte: u8) {
        for bit in 0..8 {
            peripheral.handle_eol_write(EOL_STROBE);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, EIL_READY);
            let data = if ((byte >> bit) & 1) != 0 {
                EOL_STROBE | EOL_OUT_DATA
            } else {
                EOL_STROBE
            };
            peripheral.handle_eol_write(data);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, 0);
        }
        peripheral.handle_eol_write(0);
    }

    fn host_read_byte_like_rom(peripheral: &mut Iq7000RtcPeripheral) -> u8 {
        peripheral.handle_eol_write(EOL_STROBE);
        let mut assembled = 0u8;
        for _ in 0..8 {
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, EIL_READY);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, 0);
            let sample = peripheral.handle_eil_read();
            let carry = (sample & EIL_IN_DATA) != 0;
            assembled >>= 1;
            if carry {
                assembled |= 0x80;
            }
        }
        peripheral.handle_eol_write(0);
        assembled ^ 0xFF
    }

    #[test]
    fn clock_seed_converts_to_rtc_bcd() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        assert_eq!(
            seed.rtc_datetime_bcd(),
            [0x20, 0x26, 0x04, 0x25, 0x21, 0x19]
        );
    }

    #[test]
    fn rtc_peripheral_accepts_lsb_first_command_bits() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, RTC_COMMAND_CURRENT_DATETIME);

        assert_eq!(peripheral.last_command, Some(RTC_COMMAND_CURRENT_DATETIME));
        assert!(peripheral.has_pending_response());
    }

    #[test]
    fn rtc_peripheral_streams_current_datetime_like_rom_read_helper() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, RTC_COMMAND_CURRENT_DATETIME);
        let actual = [
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
        ];

        assert_eq!(actual, [0x20, 0x26, 0x04, 0x25, 0x21, 0x19]);
        assert!(!peripheral.has_pending_response());
    }

    #[test]
    fn rtc_peripheral_waits_for_write_payload_before_status_response() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, 0xF1);
        assert_eq!(peripheral.last_command, Some(0xF1));
        assert!(!peripheral.has_pending_response());

        for byte in [0x20, 0x26, 0x04, 0x25, 0x21] {
            host_write_byte(&mut peripheral, byte);
            assert!(!peripheral.has_pending_response());
        }
        host_write_byte(&mut peripheral, 0x19);

        assert!(peripheral.has_pending_response());
        assert_eq!(host_read_byte_like_rom(&mut peripheral), 0x00);
    }
}
