//! Minimal LLAMA state scaffold.
//!
//! Holds register values keyed by `RegName`. This will grow to mirror the
//! Python emulator’s masking/aliasing rules once the evaluator lands.
// PY_SOURCE: sc62015/pysc62015/emulator.py:Registers

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::opcodes::RegName;

pub const MODELED_F_MASK: u32 = 0x03;
const FAST_TEMP_REGISTERS: usize = 16;

pub fn validate_f_image(value: u32) -> Result<u32, &'static str> {
    if value & !MODELED_F_MASK != 0 {
        Err("unsupported SC62015 F image: bits 2-7 require real-hardware tracing")
    } else {
        Ok(value & MODELED_F_MASK)
    }
}

pub fn mask_for(name: RegName) -> u32 {
    match name {
        RegName::A | RegName::B | RegName::IL | RegName::IH => 0xFF,
        RegName::BA | RegName::I => 0xFFFF,
        RegName::X | RegName::Y | RegName::U | RegName::S => 0x0F_FFFF,
        RegName::PC => 0x0F_FFFF,
        // F is transferred as a byte by stack/interrupt instructions, but the
        // available SC62015 evidence identifies only carry and zero. Do not
        // preserve invented upper flag state pending a silicon round-trip.
        RegName::F => MODELED_F_MASK,
        RegName::IMR => 0xFF,
        RegName::FC | RegName::FZ => 0x1,
        RegName::Temp(_) => 0xFFFFFF,
        RegName::Unknown(_) => 0xFFFF_FFFF,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerState {
    #[default]
    Running,
    Halted,
    Off,
}

#[derive(Clone, Default)]
pub struct LlamaState {
    // Architectural registers are hot enough that hashing every access (and
    // cloning a hash table for fail-closed preflight) dominates ROM runs.
    // Aliases A/B and IL/IH are derived from BA/I; FC/FZ are derived from F.
    ba: u32,
    i: u32,
    x: u32,
    y: u32,
    u: u32,
    s: u32,
    f: u32,
    pc: u32,
    imr: u32,
    temps: [u32; FAST_TEMP_REGISTERS],
    // Retain the public RegName behavior for malformed/out-of-range helper
    // inputs without putting a HashMap on normal execution paths.
    extra_regs: HashMap<RegName, u32>,
    power_state: PowerState,
    last_off_pc: Option<u32>,
    last_off_call_stack: Vec<u32>,
    call_depth: u32,
    call_sub_level: u32,
    call_page_stack: Vec<u32>,
    call_return_widths: Vec<u8>,
    call_stack: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct CallMetricsSnapshot {
    pub call_stack: Vec<u32>,
    pub call_depth: u32,
    pub call_sub_level: u32,
    pub call_page_stack: Vec<u32>,
    pub call_return_widths: Vec<u8>,
}

impl LlamaState {
    pub fn new() -> Self {
        Self {
            ba: 0,
            i: 0,
            x: 0,
            y: 0,
            u: 0,
            s: 0,
            f: 0,
            pc: 0,
            imr: 0,
            temps: [0; FAST_TEMP_REGISTERS],
            extra_regs: HashMap::new(),
            power_state: PowerState::Running,
            last_off_pc: None,
            last_off_call_stack: Vec::new(),
            call_depth: 0,
            call_sub_level: 0,
            call_page_stack: Vec::new(),
            call_return_widths: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    /// Copy the architectural register image for silent operand/vector
    /// validation without cloning trace-only call stacks or OFF diagnostics.
    /// Preflight may mutate pointer registers on this candidate while proving
    /// addressing modes, but it never observes call bookkeeping.
    pub(crate) fn clone_for_decode_preflight(&self) -> Self {
        Self {
            ba: self.ba,
            i: self.i,
            x: self.x,
            y: self.y,
            u: self.u,
            s: self.s,
            f: self.f,
            pc: self.pc,
            imr: self.imr,
            temps: self.temps,
            extra_regs: self.extra_regs.clone(),
            power_state: self.power_state,
            last_off_pc: None,
            last_off_call_stack: Vec::new(),
            call_depth: 0,
            call_sub_level: 0,
            call_page_stack: Vec::new(),
            call_return_widths: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    pub fn set_reg(&mut self, name: RegName, value: u32) {
        let masked = value & mask_for(name);
        match name {
            RegName::BA => self.ba = masked,
            RegName::A => {
                let b = (self.get_reg(RegName::BA) >> 8) & 0xFF;
                self.ba = ((b << 8) | (masked & 0xFF)) & mask_for(RegName::BA);
            }
            RegName::B => {
                let a = self.get_reg(RegName::BA) & 0xFF;
                self.ba = (((masked & 0xFF) << 8) | a) & mask_for(RegName::BA);
            }
            RegName::I => self.i = masked,
            RegName::IL => {
                // Hardware behaviour: writing IL updates the low byte and clears IH.
                self.i = masked & mask_for(RegName::I);
            }
            RegName::IH => {
                let low = self.get_reg(RegName::IL);
                self.i = (((masked & 0xFF) << 8) | (low & 0xFF)) & mask_for(RegName::I);
            }
            RegName::X => self.x = masked,
            RegName::Y => self.y = masked,
            RegName::U => self.u = masked,
            RegName::S => self.s = masked,
            // Raw callers validate unless an instruction explicitly
            // normalizes its input. POPU F and POPS F are measured exceptions.
            RegName::F => self.f = masked,
            RegName::PC => self.pc = masked,
            RegName::IMR => self.imr = masked,
            RegName::FC => {
                self.f = (self.f & !0x1) | (masked & 0x1);
            }
            RegName::FZ => {
                self.f = (self.f & !0x2) | ((masked & 0x1) << 1);
            }
            RegName::Temp(index) if usize::from(index) < self.temps.len() => {
                self.temps[usize::from(index)] = masked;
            }
            RegName::Temp(_) | RegName::Unknown(_) => {
                self.extra_regs.insert(name, masked);
            }
        }
    }

    pub fn get_reg(&self, name: RegName) -> u32 {
        match name {
            RegName::BA => self.ba,
            RegName::A => self.get_reg(RegName::BA) & 0xFF,
            RegName::B => (self.get_reg(RegName::BA) >> 8) & 0xFF,
            RegName::I => self.i,
            RegName::IL => self.get_reg(RegName::I) & 0xFF,
            RegName::IH => (self.get_reg(RegName::I) >> 8) & 0xFF,
            RegName::X => self.x,
            RegName::Y => self.y,
            RegName::U => self.u,
            RegName::S => self.s,
            RegName::F => self.f,
            RegName::PC => self.pc,
            RegName::FC => self.f & 0x1,
            RegName::FZ => (self.f >> 1) & 0x1,
            RegName::IMR => self.imr,
            RegName::Temp(index) if usize::from(index) < self.temps.len() => {
                self.temps[usize::from(index)]
            }
            RegName::Temp(_) | RegName::Unknown(_) => {
                self.extra_regs.get(&name).copied().unwrap_or(0) & mask_for(name)
            }
        }
    }

    pub fn pc(&self) -> u32 {
        self.get_reg(RegName::PC)
    }

    pub fn set_pc(&mut self, value: u32) {
        self.set_reg(RegName::PC, value);
    }

    pub fn halt(&mut self) {
        self.power_state = PowerState::Halted;
    }

    pub fn power_off(&mut self) {
        self.record_off_transition(self.pc());
        self.power_state = PowerState::Off;
    }

    pub fn set_power_state(&mut self, state: PowerState) {
        self.power_state = state;
    }

    pub fn record_off_transition(&mut self, pc: u32) {
        self.last_off_pc = Some(pc & mask_for(RegName::PC));
        self.last_off_call_stack = self.call_stack.clone();
    }

    pub fn last_off_pc(&self) -> Option<u32> {
        self.last_off_pc
    }

    pub fn last_off_call_stack(&self) -> &[u32] {
        &self.last_off_call_stack
    }

    pub fn set_halted(&mut self, value: bool) {
        if value {
            self.power_state = PowerState::Halted;
        } else {
            self.power_state = PowerState::Running;
        }
    }

    pub fn is_halted(&self) -> bool {
        !matches!(self.power_state, PowerState::Running)
    }

    pub fn is_off(&self) -> bool {
        matches!(self.power_state, PowerState::Off)
    }

    pub fn power_state(&self) -> PowerState {
        self.power_state
    }

    pub fn reset(&mut self) {
        self.ba = 0;
        self.i = 0;
        self.x = 0;
        self.y = 0;
        self.u = 0;
        self.s = 0;
        self.f = 0;
        self.pc = 0;
        self.imr = 0;
        self.temps = [0; FAST_TEMP_REGISTERS];
        self.extra_regs.clear();
        self.power_state = PowerState::Running;
        self.last_off_pc = None;
        self.last_off_call_stack.clear();
        self.call_depth = 0;
        self.call_sub_level = 0;
        self.call_page_stack.clear();
        self.call_return_widths.clear();
        self.call_stack.clear();
    }

    pub fn call_depth_inc(&mut self) {
        self.call_depth = self.call_depth.saturating_add(1);
        self.call_sub_level = self.call_sub_level.saturating_add(1);
    }

    pub fn call_depth_dec(&mut self) {
        if self.call_depth > 0 {
            self.call_depth -= 1;
        }
        if self.call_sub_level > 0 {
            self.call_sub_level -= 1;
        }
    }

    pub fn call_depth(&self) -> u32 {
        self.call_depth
    }

    pub fn call_sub_level(&self) -> u32 {
        self.call_sub_level
    }

    pub fn call_stack(&self) -> &[u32] {
        &self.call_stack
    }

    pub fn call_page_depth(&self) -> usize {
        self.call_page_stack.len()
    }

    pub fn push_call_stack(&mut self, dest: u32) {
        self.push_call_frame(dest, 0);
    }

    pub fn pop_call_stack(&mut self) -> Option<u32> {
        self.pop_call_frame()
    }

    pub fn push_call_frame(&mut self, dest: u32, ret_bits: u8) {
        self.call_stack.push(dest & mask_for(RegName::PC));
        self.call_return_widths.push(ret_bits);
    }

    pub fn pop_call_frame(&mut self) -> Option<u32> {
        let _ = self.call_return_widths.pop();
        self.call_stack.pop()
    }

    pub fn peek_call_return_width(&self) -> Option<u8> {
        self.call_return_widths.last().copied()
    }

    pub fn set_call_depth(&mut self, value: u32) {
        self.call_depth = value;
    }

    pub fn set_call_sub_level(&mut self, value: u32) {
        self.call_sub_level = value;
    }

    /// Drop any saved call-page context (used for 16-bit CALL/RET page reconstruction).
    pub fn clear_call_page_stack(&mut self) {
        self.call_page_stack.clear();
    }

    /// Track the 20-bit page of near CALL sites so RET can reconstruct the full return PC.
    pub fn push_call_page(&mut self, page: u32) {
        self.call_page_stack.push(page & 0xFF_0000);
    }

    pub fn pop_call_page(&mut self) -> Option<u32> {
        self.call_page_stack.pop()
    }

    /// Peek the most recent call page without popping; helps RET when stack was manipulated.
    pub fn peek_call_page(&self) -> Option<u32> {
        self.call_page_stack.last().copied()
    }

    /// Clear call-depth bookkeeping used only for tracing/metrics; does not alter registers.
    pub fn reset_call_metrics(&mut self) {
        self.call_depth = 0;
        self.call_sub_level = 0;
        self.call_page_stack.clear();
        self.call_return_widths.clear();
        self.call_stack.clear();
    }

    pub fn snapshot_call_metrics(&self) -> CallMetricsSnapshot {
        CallMetricsSnapshot {
            call_stack: self.call_stack.clone(),
            call_depth: self.call_depth,
            call_sub_level: self.call_sub_level,
            call_page_stack: self.call_page_stack.clone(),
            call_return_widths: self.call_return_widths.clone(),
        }
    }

    pub fn restore_call_metrics(&mut self, snapshot: CallMetricsSnapshot) {
        self.call_stack = snapshot.call_stack;
        self.call_depth = snapshot.call_depth;
        self.call_sub_level = snapshot.call_sub_level;
        self.call_page_stack = snapshot.call_page_stack;
        self.call_return_widths = snapshot.call_return_widths;
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
        state.set_reg(RegName::F, 0b11);

        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
        assert_eq!(state.get_reg(RegName::F), 0b11);

        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::FZ, 0);
        assert_eq!(state.get_reg(RegName::F), 0);
    }

    #[test]
    fn fc_fz_updates_stay_within_modeled_image() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::F, 0b11);
        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::FZ, 1);

        assert_eq!(state.get_reg(RegName::F), 0b10);
        assert_eq!(state.get_reg(RegName::FC), 0);
        assert_eq!(state.get_reg(RegName::FZ), 1);
    }

    #[test]
    fn raw_f_validator_quarantines_unverified_upper_bits() {
        for valid in 0..=MODELED_F_MASK {
            assert_eq!(validate_f_image(valid), Ok(valid));
        }
        for invalid in [0x04, 0x80, 0xA4, 0xFC, 0xFF] {
            assert!(validate_f_image(invalid).is_err());
        }
    }

    #[test]
    fn x_register_masks_to_20_bits() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x9F9F9F);

        assert_eq!(state.get_reg(RegName::X), 0x0F_9F9F);
    }

    #[test]
    fn call_sub_level_is_cumulative() {
        let mut state = LlamaState::new();
        state.call_depth_inc();
        state.call_depth_inc();
        assert_eq!(state.call_depth(), 2);
        assert_eq!(state.call_sub_level(), 2);
        state.call_depth_dec();
        // Python parity: call_sub_level tracks current depth and should decrement on returns.
        assert_eq!(state.call_depth(), 1);
        assert_eq!(state.call_sub_level(), 1);
    }

    #[test]
    fn call_page_stack_tracks_pages() {
        let mut state = LlamaState::new();
        state.push_call_page(0x120000);
        state.push_call_page(0xAB0000);
        assert_eq!(state.pop_call_page(), Some(0xAB0000));
        assert_eq!(state.pop_call_page(), Some(0x120000));
        assert_eq!(state.pop_call_page(), None);
    }
}
