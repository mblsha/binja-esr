//! Minimal LLAMA state scaffold.
//!
//! Holds register values keyed by `RegName`. This will grow to mirror the
//! Python emulator’s masking/aliasing rules once the evaluator lands.
// PY_SOURCE: sc62015/pysc62015/emulator.py:Registers

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
    call_sub_level: u32,
}

impl LlamaState {
    pub fn new() -> Self {
        Self {
            regs: HashMap::new(),
            halted: false,
            call_depth: 0,
            call_sub_level: 0,
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
                // Preserve IH; only replace the low byte for parity with Python registers.
                let high = self.get_reg(RegName::I) & 0xFF00;
                self.regs.insert(RegName::I, high | (masked & 0xFF));
            }
            RegName::IH => {
                let low = self.get_reg(RegName::IL);
                let i = ((masked & 0xFF) << 8) | (low & 0xFF);
                self.regs.insert(RegName::I, i & mask_for(RegName::I));
            }
            RegName::F => {
                // Preserve full F byte; sync FC/FZ aliases from bits 0/1.
                self.regs.insert(RegName::F, masked & mask_for(RegName::F));
                self.regs.insert(RegName::FC, masked & 0x1);
                self.regs.insert(RegName::FZ, (masked >> 1) & 0x1);
            }
            RegName::FC => {
                let bit = masked & 0x1;
                let f = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFF;
                let new_f = (f & !0x1) | bit;
                self.regs.insert(RegName::F, new_f);
                self.regs.insert(RegName::FC, bit);
            }
            RegName::FZ => {
                let bit = masked & 0x1;
                let f = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFF;
                let new_f = (f & !0x2) | (bit << 1);
                self.regs.insert(RegName::F, new_f);
                self.regs.insert(RegName::FZ, bit);
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
                let raw = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFF;
                let fc = self.regs.get(&RegName::FC).copied().unwrap_or(raw & 0x1) & 0x1;
                let fz = self
                    .regs
                    .get(&RegName::FZ)
                    .copied()
                    .unwrap_or((raw >> 1) & 0x1)
                    & 0x1;
                (raw & !0x3) | fc | (fz << 1)
            }
            RegName::FC => {
                let raw = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFF;
                *self.regs.get(&RegName::FC).unwrap_or(&raw) & 0x1
            }
            RegName::FZ => {
                let raw = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFF;
                (*self.regs.get(&RegName::FZ).unwrap_or(&((raw >> 1) & 0x1))) & 0x1
            }
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
        self.call_sub_level = 0;
    }

    pub fn call_depth_inc(&mut self) {
        self.call_depth = self.call_depth.saturating_add(1);
        self.call_sub_level = self.call_depth;
    }

    pub fn call_depth_dec(&mut self) {
        if self.call_depth > 0 {
            self.call_depth -= 1;
        }
        self.call_sub_level = self.call_depth;
    }

    pub fn call_depth(&self) -> u32 {
        self.call_depth
    }

    pub fn call_sub_level(&self) -> u32 {
        self.call_sub_level
    }

    pub fn set_call_sub_level(&mut self, value: u32) {
        self.call_sub_level = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;

    #[test]
    fn il_write_preserves_high_byte_and_updates_aliases() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 0xABCD);

        state.set_reg(RegName::IL, 0x34);

        assert_eq!(state.get_reg(RegName::IL), 0x34);
        assert_eq!(state.get_reg(RegName::IH), 0xAB);
        assert_eq!(state.get_reg(RegName::I), 0xAB34);
    }

    #[test]
    fn f_facade_on_fc_fz() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::F, 0b1010_0011);

        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
        assert_eq!(state.get_reg(RegName::F), 0b1010_0011);

        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::FZ, 0);
        assert_eq!(state.get_reg(RegName::F), 0b1010_0000);
    }

    #[test]
    fn fc_fz_updates_preserve_upper_bits() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::F, 0b1111_1111);
        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::FZ, 1);

        assert_eq!(state.get_reg(RegName::F), 0b1111_1110);
        assert_eq!(state.get_reg(RegName::FC), 0);
        assert_eq!(state.get_reg(RegName::FZ), 1);
    }
}
