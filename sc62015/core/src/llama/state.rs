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
                self.regs.insert(RegName::A, (masked & 0xFF) as u32);
                self.regs.insert(RegName::B, ((masked >> 8) & 0xFF) as u32);
            }
            RegName::A => {
                self.regs.insert(RegName::A, masked);
                let b = self.get_reg(RegName::B);
                self.regs
                    .insert(RegName::BA, ((b & 0xFF) << 8) | (masked & 0xFF));
            }
            RegName::B => {
                self.regs.insert(RegName::B, masked);
                let a = self.get_reg(RegName::A);
                self.regs
                    .insert(RegName::BA, ((masked & 0xFF) << 8) | (a & 0xFF));
            }
            RegName::I => {
                self.regs.insert(RegName::I, masked);
                self.regs.insert(RegName::IL, (masked & 0xFF) as u32);
                self.regs.insert(RegName::IH, ((masked >> 8) & 0xFF) as u32);
            }
            RegName::IL => {
                self.regs.insert(RegName::IL, masked);
                let ih = self.get_reg(RegName::IH);
                self.regs
                    .insert(RegName::I, ((ih & 0xFF) << 8) | (masked & 0xFF));
            }
            RegName::IH => {
                self.regs.insert(RegName::IH, masked);
                let il = self.get_reg(RegName::IL);
                self.regs
                    .insert(RegName::I, ((masked & 0xFF) << 8) | (il & 0xFF));
            }
            RegName::F => {
                self.regs.insert(RegName::F, masked);
                self.regs.insert(RegName::FC, masked & 0x1);
                self.regs.insert(RegName::FZ, (masked >> 1) & 0x1);
            }
            RegName::FC => {
                self.regs.insert(RegName::FC, masked & 0x1);
                let f = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFE;
                self.regs.insert(RegName::F, f | (masked & 0x1));
            }
            RegName::FZ => {
                self.regs.insert(RegName::FZ, masked & 0x1);
                let mut f = self.regs.get(&RegName::F).copied().unwrap_or(0) & 0xFD;
                f |= (masked & 0x1) << 1;
                self.regs.insert(RegName::F, f);
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
            RegName::BA => {
                let a = *self.regs.get(&RegName::A).unwrap_or(&0);
                let b = *self.regs.get(&RegName::B).unwrap_or(&0);
                (((b & 0xFF) << 8) | (a & 0xFF)) & mask_for(RegName::BA)
            }
            RegName::I => {
                let il = *self.regs.get(&RegName::IL).unwrap_or(&0);
                let ih = *self.regs.get(&RegName::IH).unwrap_or(&0);
                (((ih & 0xFF) << 8) | (il & 0xFF)) & mask_for(RegName::I)
            }
            RegName::F => {
                let f = *self.regs.get(&RegName::F).unwrap_or(&0) & 0xFC;
                let fc = *self.regs.get(&RegName::FC).unwrap_or(&0) & 0x1;
                let fz = *self.regs.get(&RegName::FZ).unwrap_or(&0) & 0x1;
                (f | fc | (fz << 1)) & mask_for(RegName::F)
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
