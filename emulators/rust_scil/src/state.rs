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

fn alias_trace_enabled() -> bool {
    std::env::var("LCD_LOOP_TRACE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn trace_alias(state: &State, name: &str, value: u32, bits: u8) {
    if !alias_trace_enabled() {
        return;
    }
    if matches!(name, "A" | "B" | "BA" | "IL" | "IH" | "I") {
        eprintln!(
            "[state-reg] pc=0x{pc:06X} reg={name} value=0x{val:06X} bits={bits}",
            pc = state.pc & PC_MASK,
            name = name,
            val = value & mask(bits),
            bits = bits
        );
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
    #[serde(default)]
    pub halted: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            regs: HashMap::new(),
            flags: HashMap::new(),
            pc: 0,
            halted: false,
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
                trace_alias(self, "A", masked & 0xFF, 8);
                trace_alias(self, "BA", base & 0xFFFF, 16);
                return;
            }
            "B" => {
                self.regs.insert("B".into(), masked & 0xFF);
                let mut base = self.regs.get("BA").copied().unwrap_or(0);
                base = (base & 0x00FF) | ((masked & 0xFF) << 8);
                self.regs.insert("BA".into(), base & 0xFFFF);
                trace_alias(self, "B", masked & 0xFF, 8);
                trace_alias(self, "BA", base & 0xFFFF, 16);
                return;
            }
            "BA" => {
                let base = masked & 0xFFFF;
                self.regs.insert("BA".into(), base);
                self.regs.insert("A".into(), base & 0xFF);
                self.regs.insert("B".into(), (base >> 8) & 0xFF);
                trace_alias(self, "BA", base, 16);
                trace_alias(self, "A", base & 0xFF, 8);
                trace_alias(self, "B", (base >> 8) & 0xFF, 8);
                return;
            }
            "IL" => {
                let low = masked & 0xFF;
                self.regs.insert("IL".into(), low);
                // Match the Python backend semantics: writing IL zeroes the upper byte.
                self.regs.insert("IH".into(), 0);
                self.regs.insert("I".into(), low);
                trace_alias(self, "IL", low, 8);
                trace_alias(self, "IH", 0, 8);
                trace_alias(self, "I", low, 16);
                return;
            }
            "IH" => {
                self.regs.insert("IH".into(), masked & 0xFF);
                let mut base = self.regs.get("I").copied().unwrap_or(0);
                base = (base & 0x00FF) | ((masked & 0xFF) << 8);
                self.regs.insert("I".into(), base & 0xFFFF);
                trace_alias(self, "IH", masked & 0xFF, 8);
                trace_alias(self, "I", base & 0xFFFF, 16);
                return;
            }
            "I" => {
                let base = masked & 0xFFFF;
                self.regs.insert("I".into(), base);
                self.regs.insert("IL".into(), base & 0xFF);
                self.regs.insert("IH".into(), (base >> 8) & 0xFF);
                trace_alias(self, "I", base, 16);
                trace_alias(self, "IL", base & 0xFF, 8);
                trace_alias(self, "IH", (base >> 8) & 0xFF, 8);
                return;
            }
            "F" => {
                self.regs.insert("F".into(), masked & 0xFF);
                let c = (masked & 1) as u8;
                let z = ((masked >> 1) & 1) as u8;
                self.flags.insert("C".into(), c);
                self.flags.insert("Z".into(), z);
                self.regs.insert("FC".into(), c as u32);
                self.regs.insert("FZ".into(), z as u32);
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
        match name {
            "C" => {
                self.regs.insert("FC".into(), bit as u32);
            }
            "Z" => {
                self.regs.insert("FZ".into(), bit as u32);
            }
            _ => {
                self.regs.insert(name.to_string(), bit as u32);
            }
        }
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
