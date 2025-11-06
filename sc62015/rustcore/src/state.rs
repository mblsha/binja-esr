use std::fmt;
use std::str::FromStr;

use crate::constants::{NUM_TEMP_REGISTERS, PC_MASK};

/// Architectural registers recognised by the SC62015.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Register {
    // 8-bit
    A,
    B,
    IL,
    IH,
    // 16-bit
    I,
    BA,
    // 24-bit (stored in 3 bytes)
    X,
    Y,
    U,
    S,
    // 20-bit (stored in 3 bytes)
    PC,
    // Flags
    FC,
    FZ,
    F,
    // Temporary registers (3 bytes each)
    Temp(u8),
}

impl Register {
    /// Return the size of the register in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            Register::A
            | Register::B
            | Register::IL
            | Register::IH
            | Register::FC
            | Register::FZ
            | Register::F => 1,
            Register::I | Register::BA => 2,
            Register::X | Register::Y | Register::U | Register::S | Register::Temp(_) => 3,
            Register::PC => 3,
        }
    }

    fn temp(index: u8) -> Result<Self, RegisterError> {
        if (index as usize) < NUM_TEMP_REGISTERS {
            Ok(Register::Temp(index))
        } else {
            Err(RegisterError::InvalidTempIndex(index))
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Register::A => write!(f, "A"),
            Register::B => write!(f, "B"),
            Register::IL => write!(f, "IL"),
            Register::IH => write!(f, "IH"),
            Register::I => write!(f, "I"),
            Register::BA => write!(f, "BA"),
            Register::X => write!(f, "X"),
            Register::Y => write!(f, "Y"),
            Register::U => write!(f, "U"),
            Register::S => write!(f, "S"),
            Register::PC => write!(f, "PC"),
            Register::FC => write!(f, "FC"),
            Register::FZ => write!(f, "FZ"),
            Register::F => write!(f, "F"),
            Register::Temp(idx) => write!(f, "TEMP{idx}"),
        }
    }
}

impl FromStr for Register {
    type Err = RegisterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "A" => Ok(Register::A),
            "B" => Ok(Register::B),
            "IL" => Ok(Register::IL),
            "IH" => Ok(Register::IH),
            "I" => Ok(Register::I),
            "BA" => Ok(Register::BA),
            "X" => Ok(Register::X),
            "Y" => Ok(Register::Y),
            "U" => Ok(Register::U),
            "S" => Ok(Register::S),
            "PC" => Ok(Register::PC),
            "FC" => Ok(Register::FC),
            "FZ" => Ok(Register::FZ),
            "F" => Ok(Register::F),
            other => {
                if let Some(rest) = other.strip_prefix("TEMP") {
                    let index: u8 = rest
                        .parse()
                        .map_err(|_| RegisterError::UnknownRegister(other.into()))?;
                    Register::temp(index)
                } else {
                    Err(RegisterError::UnknownRegister(other.into()))
                }
            }
        }
    }
}

/// Flag identifiers accepted by `get_flag`/`set_flag`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Flag {
    Carry,
    Zero,
}

impl Flag {
    pub fn from_char(value: char) -> Result<Self, RegisterError> {
        match value.to_ascii_uppercase() {
            'C' => Ok(Flag::Carry),
            'Z' => Ok(Flag::Zero),
            other => Err(RegisterError::UnknownFlag(other)),
        }
    }
}

/// Errors surfaced while parsing or manipulating register names.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum RegisterError {
    #[error("unknown register '{0}'")]
    UnknownRegister(String),
    #[error("TEMP register index {0} is out of range")]
    InvalidTempIndex(u8),
    #[error("unknown flag '{0}'")]
    UnknownFlag(char),
}

/// Mutable SC62015 register file with masking semantics equivalent to the Python implementation.
#[derive(Clone, Debug)]
pub struct Registers {
    ba: u16,
    i: u16,
    x: u32,
    y: u32,
    u: u32,
    s: u32,
    pc: u32,
    f: u8,
    temps: [u32; NUM_TEMP_REGISTERS],
    call_sub_level: u32,
}

impl Default for Registers {
    fn default() -> Self {
        Self {
            ba: 0,
            i: 0,
            x: 0,
            y: 0,
            u: 0,
            s: 0,
            pc: 0,
            f: 0,
            temps: [0; NUM_TEMP_REGISTERS],
            call_sub_level: 0,
        }
    }
}

impl Registers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the value of a register (masked to the register width).
    pub fn get(&self, reg: Register) -> u32 {
        match reg {
            Register::A => (self.ba & 0x00FF) as u32,
            Register::B => ((self.ba >> 8) & 0x00FF) as u32,
            Register::IL => (self.i & 0x00FF) as u32,
            Register::IH => ((self.i >> 8) & 0x00FF) as u32,
            Register::I => self.i as u32,
            Register::BA => self.ba as u32,
            Register::X => self.x & 0x00FF_FFFF,
            Register::Y => self.y & 0x00FF_FFFF,
            Register::U => self.u & 0x00FF_FFFF,
            Register::S => self.s & 0x00FF_FFFF,
            Register::PC => self.pc & PC_MASK,
            Register::FC => (self.f & 0x01) as u32,
            Register::FZ => ((self.f >> 1) & 0x01) as u32,
            Register::F => self.f as u32,
            Register::Temp(index) => {
                let idx = index as usize;
                debug_assert!(idx < NUM_TEMP_REGISTERS);
                self.temps[idx] & 0x00FF_FFFF
            }
        }
    }

    /// Set the value of a register (value is masked to the register width).
    pub fn set(&mut self, reg: Register, value: u32) {
        match reg {
            Register::A => {
                let masked = (value & 0xFF) as u16;
                self.ba = (self.ba & 0xFF00) | masked;
            }
            Register::B => {
                let masked = ((value & 0xFF) as u16) << 8;
                self.ba = (self.ba & 0x00FF) | masked;
            }
            Register::IL => {
                let masked = (value & 0xFF) as u16;
                self.i = (self.i & 0xFF00) | masked;
            }
            Register::IH => {
                let masked = ((value & 0xFF) as u16) << 8;
                self.i = (self.i & 0x00FF) | masked;
            }
            Register::I => {
                self.i = (value & 0xFFFF) as u16;
            }
            Register::BA => {
                self.ba = (value & 0xFFFF) as u16;
            }
            Register::X => {
                self.x = value & 0x00FF_FFFF;
            }
            Register::Y => {
                self.y = value & 0x00FF_FFFF;
            }
            Register::U => {
                self.u = value & 0x00FF_FFFF;
            }
            Register::S => {
                self.s = value & 0x00FF_FFFF;
            }
            Register::PC => {
                self.pc = value & PC_MASK;
            }
            Register::FC => {
                let bit = (value & 0x01) as u8;
                self.f = (self.f & !0x01) | bit;
            }
            Register::FZ => {
                let bit = ((value & 0x01) as u8) << 1;
                self.f = (self.f & !0x02) | bit;
            }
            Register::F => {
                self.f = (value & 0xFF) as u8;
            }
            Register::Temp(index) => {
                let idx = index as usize;
                debug_assert!(idx < NUM_TEMP_REGISTERS);
                self.temps[idx] = value & 0x00FF_FFFF;
            }
        }
    }

    /// Convenience wrapper that parses the register name before reading.
    pub fn get_by_name(&self, name: &str) -> Result<u32, RegisterError> {
        Register::from_str(name).map(|reg| self.get(reg))
    }

    /// Convenience wrapper that parses the register name before writing.
    pub fn set_by_name(&mut self, name: &str, value: u32) -> Result<(), RegisterError> {
        let reg = Register::from_str(name)?;
        self.set(reg, value);
        Ok(())
    }

    /// Read a flag by symbolic name ("C"/"Z").
    pub fn get_flag(&self, flag: Flag) -> u8 {
        match flag {
            Flag::Carry => self.get(Register::FC) as u8,
            Flag::Zero => self.get(Register::FZ) as u8,
        }
    }

    /// Write a flag by symbolic name ("C"/"Z").
    pub fn set_flag(&mut self, flag: Flag, value: u8) {
        match flag {
            Flag::Carry => self.set(Register::FC, value.into()),
            Flag::Zero => self.set(Register::FZ, value.into()),
        }
    }

    /// Parse-and-set helper for flags expressed as characters.
    pub fn set_flag_from_char(&mut self, flag: char, value: u8) -> Result<(), RegisterError> {
        let flag = Flag::from_char(flag)?;
        self.set_flag(flag, value);
        Ok(())
    }

    /// Return the current call-subroutine depth tracker.
    pub fn call_sub_level(&self) -> u32 {
        self.call_sub_level
    }

    pub fn set_call_sub_level(&mut self, level: u32) {
        self.call_sub_level = level;
    }

    pub fn increment_call_sub_level(&mut self) {
        self.call_sub_level = self.call_sub_level.saturating_add(1);
    }

    pub fn decrement_call_sub_level(&mut self) {
        self.call_sub_level = self.call_sub_level.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registers_are_zeroed() {
        let regs = Registers::new();
        assert_eq!(regs.get(Register::A), 0);
        assert_eq!(regs.get(Register::BA), 0);
        assert_eq!(regs.get(Register::PC), 0);
        assert_eq!(regs.call_sub_level(), 0);
    }

    #[test]
    fn composite_registers_update_sub_components() {
        let mut regs = Registers::new();
        regs.set(Register::BA, 0x1234);
        assert_eq!(regs.get(Register::A), 0x34);
        assert_eq!(regs.get(Register::B), 0x12);

        regs.set(Register::A, 0xAA);
        assert_eq!(regs.get(Register::BA), 0x12AA);

        regs.set(Register::B, 0x55);
        assert_eq!(regs.get(Register::BA), 0x55AA);
    }

    #[test]
    fn pc_is_masked_to_20_bits() {
        let mut regs = Registers::new();
        regs.set(Register::PC, 0xABCDE0);
        assert_eq!(regs.get(Register::PC), 0x0BCDE0 & PC_MASK);
    }

    #[test]
    fn flags_map_into_f_register() {
        let mut regs = Registers::new();
        regs.set_flag(Flag::Carry, 1);
        regs.set_flag(Flag::Zero, 1);
        assert_eq!(regs.get(Register::F), 0b11);
        regs.set(Register::F, 0);
        assert_eq!(regs.get_flag(Flag::Carry), 0);
        assert_eq!(regs.get_flag(Flag::Zero), 0);
    }

    #[test]
    fn temps_are_masked_to_24_bits() {
        let mut regs = Registers::new();
        regs.set(Register::Temp(3), 0xFF00_FFEE);
        assert_eq!(regs.get(Register::Temp(3)), 0x00_00FFEE & 0x00FF_FFFF);
    }

    #[test]
    fn register_name_parsing() {
        assert_eq!(Register::from_str("A").unwrap(), Register::A);
        assert_eq!(Register::from_str("temp5").unwrap(), Register::Temp(5));
        assert!(Register::from_str("temp99").is_err());
    }
}
