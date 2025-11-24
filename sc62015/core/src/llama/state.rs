//! Minimal LLAMA state scaffold.
//!
//! Holds register values keyed by `RegName`. This will grow to mirror the
//! Python emulator’s masking/aliasing rules once the evaluator lands.

use std::collections::HashMap;

use super::opcodes::RegName;

pub fn mask_for(name: RegName) -> u32 {
    match name {
        RegName::A | RegName::B | RegName::IL | RegName::IH => 0xFF,
        RegName::BA | RegName::I => 0xFFFF,
        RegName::X | RegName::Y | RegName::U | RegName::S => 0xFFFFFF,
        RegName::PC => 0x0F_FFFF,
        RegName::F | RegName::IMR => 0xFF,
        RegName::FC | RegName::FZ => 0x1,
        RegName::Temp(_) => 0xFFFFFF,
        RegName::Unknown(_) => 0xFFFF_FFFF,
    }
}

#[derive(Default)]
pub struct LlamaState {
    regs: HashMap<RegName, u32>,
    halted: bool,
    call_depth: u32,
}

impl LlamaState {
    pub fn new() -> Self {
        Self {
            regs: HashMap::new(),
            halted: false,
            call_depth: 0,
        }
    }

    pub fn set_reg(&mut self, name: RegName, value: u32) {
        let masked = value & mask_for(name);
        match name {
            RegName::BA => {
                self.regs.insert(RegName::BA, masked);
            }
            RegName::A => {
                let b = (self.get_reg(RegName::BA) >> 8) & 0xFF;
                let ba = ((b << 8) | (masked & 0xFF)) & mask_for(RegName::BA);
                self.regs.insert(RegName::BA, ba);
            }
            RegName::B => {
                let a = self.get_reg(RegName::BA) & 0xFF;
                let ba = (((masked & 0xFF) << 8) | a) & mask_for(RegName::BA);
                self.regs.insert(RegName::BA, ba);
            }
            RegName::I => {
                self.regs.insert(RegName::I, masked);
            }
            RegName::IL => {
                // IL writes clear the high byte, matching the Python emulator semantics.
                self.regs.insert(RegName::I, masked & 0xFF);
            }
            RegName::IH => {
                let low = self.get_reg(RegName::IL);
                let i = ((masked & 0xFF) << 8) | (low & 0xFF);
                self.regs.insert(RegName::I, i & mask_for(RegName::I));
            }
            RegName::F => {
                // F is derived from FC/FZ; only keep those bits.
                self.regs.insert(RegName::FC, masked & 0x1);
                self.regs.insert(RegName::FZ, (masked >> 1) & 0x1);
                self.regs.insert(RegName::F, masked & 0x3);
            }
            RegName::FC => {
                self.regs.insert(RegName::FC, masked & 0x1);
            }
            RegName::FZ => {
                self.regs.insert(RegName::FZ, masked & 0x1);
            }
            RegName::Temp(_) => {
                self.regs.insert(name, masked);
            }
            _ => {
                self.regs.insert(name, masked);
            }
        }
    }

    pub fn get_reg(&self, name: RegName) -> u32 {
        match name {
            RegName::BA => *self.regs.get(&RegName::BA).unwrap_or(&0) & mask_for(RegName::BA),
            RegName::A => self.get_reg(RegName::BA) & 0xFF,
            RegName::B => (self.get_reg(RegName::BA) >> 8) & 0xFF,
            RegName::I => *self.regs.get(&RegName::I).unwrap_or(&0) & mask_for(RegName::I),
            RegName::IL => self.get_reg(RegName::I) & 0xFF,
            RegName::IH => (self.get_reg(RegName::I) >> 8) & 0xFF,
            RegName::F => {
                let fc = self.get_reg(RegName::FC) & 0x1;
                let fz = self.get_reg(RegName::FZ) & 0x1;
                (fc | (fz << 1)) & mask_for(RegName::F)
            }
            RegName::FC => *self.regs.get(&RegName::FC).unwrap_or(&0) & 0x1,
            RegName::FZ => *self.regs.get(&RegName::FZ).unwrap_or(&0) & 0x1,
            RegName::Temp(_) => *self.regs.get(&name).unwrap_or(&0) & mask_for(name),
            _ => *self.regs.get(&name).unwrap_or(&0) & mask_for(name),
        }
    }

    pub fn pc(&self) -> u32 {
        self.get_reg(RegName::PC)
    }

    pub fn set_pc(&mut self, value: u32) {
        self.set_reg(RegName::PC, value);
    }

    pub fn halt(&mut self) {
        self.halted = true;
    }

    pub fn set_halted(&mut self, value: bool) {
        self.halted = value;
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn reset(&mut self) {
        self.regs.clear();
        self.halted = false;
        self.call_depth = 0;
    }

    pub fn call_depth_inc(&mut self) {
        self.call_depth = self.call_depth.saturating_add(1);
    }

    pub fn call_depth_dec(&mut self) {
        if self.call_depth > 0 {
            self.call_depth -= 1;
        }
    }

    pub fn call_depth(&self) -> u32 {
        self.call_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;

    #[test]
    fn il_write_clears_high_byte_and_updates_aliases() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 0xABCD);

        state.set_reg(RegName::IL, 0x34);

        assert_eq!(state.get_reg(RegName::IL), 0x34);
        assert_eq!(state.get_reg(RegName::IH), 0x00);
        assert_eq!(state.get_reg(RegName::I), 0x0034);
    }

    #[test]
    fn f_facade_on_fc_fz() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::F, 0b0000_0011);

        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
        assert_eq!(state.get_reg(RegName::F), 0b0000_0011);

        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::FZ, 0);
        assert_eq!(state.get_reg(RegName::F), 0b0000_0000);
    }
}
