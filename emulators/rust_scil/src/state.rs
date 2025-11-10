use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const PC_MASK: u32 = 0xF_FFFF;

fn mask(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    regs: HashMap<String, u32>,
    #[serde(default)]
    flags: HashMap<String, u8>,
    #[serde(default)]
    pub pc: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            regs: HashMap::new(),
            flags: HashMap::new(),
            pc: 0,
        }
    }
}

impl State {
    pub fn get_reg(&self, name: &str, bits: u8) -> u32 {
        if name == "PC" {
            return self.pc & PC_MASK;
        }
        *self.regs.get(name).unwrap_or(&0) & mask(bits)
    }

    pub fn set_reg(&mut self, name: &str, value: u32, bits: u8) {
        if name == "PC" {
            self.pc = value & PC_MASK;
            return;
        }
        let masked = value & mask(bits);
        match name {
            "A" => {
                self.regs.insert("A".into(), masked & 0xFF);
                let mut base = self.regs.get("BA").copied().unwrap_or(0);
                base = (base & 0xFF00) | (masked & 0xFF);
                self.regs.insert("BA".into(), base & 0xFFFF);
                return;
            }
            "B" => {
                self.regs.insert("B".into(), masked & 0xFF);
                let mut base = self.regs.get("BA").copied().unwrap_or(0);
                base = (base & 0x00FF) | ((masked & 0xFF) << 8);
                self.regs.insert("BA".into(), base & 0xFFFF);
                return;
            }
            "BA" => {
                let base = masked & 0xFFFF;
                self.regs.insert("BA".into(), base);
                self.regs.insert("A".into(), base & 0xFF);
                self.regs.insert("B".into(), (base >> 8) & 0xFF);
                return;
            }
            "IL" => {
                self.regs.insert("IL".into(), masked & 0xFF);
                let mut base = self.regs.get("I").copied().unwrap_or(0);
                base = (base & 0xFF00) | (masked & 0xFF);
                self.regs.insert("I".into(), base & 0xFFFF);
                return;
            }
            "IH" => {
                self.regs.insert("IH".into(), masked & 0xFF);
                let mut base = self.regs.get("I").copied().unwrap_or(0);
                base = (base & 0x00FF) | ((masked & 0xFF) << 8);
                self.regs.insert("I".into(), base & 0xFFFF);
                return;
            }
            "I" => {
                let base = masked & 0xFFFF;
                self.regs.insert("I".into(), base);
                self.regs.insert("IL".into(), base & 0xFF);
                self.regs.insert("IH".into(), (base >> 8) & 0xFF);
                return;
            }
            "F" => {
                self.regs.insert("F".into(), masked & 0xFF);
                self.flags.insert("C".into(), (masked & 1) as u8);
                self.flags.insert("Z".into(), ((masked >> 1) & 1) as u8);
                return;
            }
            _ => {}
        }
        self.regs.insert(name.to_string(), masked);
    }

    pub fn get_flag(&self, name: &str) -> u32 {
        self.flags.get(name).copied().unwrap_or(0) as u32
    }

    pub fn set_flag(&mut self, name: &str, value: u32) {
        let bit = (value & 1) as u8;
        self.flags.insert(name.to_string(), bit);
        let mut f = self.regs.get("F").copied().unwrap_or(0);
        match name {
            "C" => {
                if bit == 1 {
                    f |= 1;
                } else {
                    f &= !1;
                }
            }
            "Z" => {
                if bit == 1 {
                    f |= 1 << 1;
                } else {
                    f &= !(1 << 1);
                }
            }
            _ => {}
        }
        self.regs.insert("F".into(), f & 0xFF);
    }
}
