// PY_SOURCE: sc62015/pysc62015/instr/opcodes.py

use std::collections::VecDeque;

use crate::llama::opcodes::RegName;
use crate::llama::state::mask_for;
use crate::memory::{
    MemoryImage, IMEM_RXD_OFFSET, IMEM_TXD_OFFSET, IMEM_UCR_OFFSET, IMEM_USR_OFFSET,
};

const IMEM_BH_OFFSET: u32 = 0xD5;
const USR_RX_READY: u8 = 0x20;
const USR_TX_EMPTY: u8 = 0x10;
const USR_TX_READY: u8 = 0x08;
const USR_ERROR_MASK: u8 = 0x07;
const SIO_CMD42_DIRECT_INPUT_ADDR: u32 = 0x00EB030;
const SIO_TX_WAIT_READY_ADDR: u32 = 0x00EB31C;
const SIO_CMD41_DIRECT_OUTPUT_ADDR: u32 = 0x00EB33D;

#[derive(Debug, Default)]
pub struct SioStub {
    rx_queue: VecDeque<u8>,
    tx_queue: VecDeque<u8>,
    auto_response: u8,
}

impl SioStub {
    pub fn new() -> Self {
        Self {
            rx_queue: VecDeque::new(),
            tx_queue: VecDeque::new(),
            auto_response: 0x41,
        }
    }

    pub fn init(&mut self, memory: &mut MemoryImage) {
        self.apply_status(memory);
    }

    pub fn maybe_short_circuit(
        &mut self,
        pc: u32,
        state: &mut crate::llama::state::LlamaState,
        memory: &mut MemoryImage,
    ) -> bool {
        let pc = pc & 0x000f_ffff;
        if !matches!(
            pc,
            SIO_CMD42_DIRECT_INPUT_ADDR | SIO_TX_WAIT_READY_ADDR | SIO_CMD41_DIRECT_OUTPUT_ADDR
        ) {
            return false;
        }
        let response = self.auto_response;
        memory.write_internal_byte(IMEM_RXD_OFFSET, response);
        memory.write_internal_byte(IMEM_BH_OFFSET, response);
        state.set_reg(RegName::FC, 0);
        self.force_return_auto(state, memory);
        true
    }

    pub fn handle_read(&mut self, offset: u32, memory: &mut MemoryImage) -> Option<u8> {
        match offset {
            IMEM_USR_OFFSET => {
                self.apply_status(memory);
                memory.read_internal_byte(IMEM_USR_OFFSET)
            }
            IMEM_RXD_OFFSET => {
                let value = self.consume_rx(memory);
                Some(value)
            }
            IMEM_UCR_OFFSET | IMEM_TXD_OFFSET => memory.read_internal_byte(offset),
            _ => None,
        }
    }

    pub fn handle_write(&mut self, offset: u32, value: u8, memory: &mut MemoryImage) -> bool {
        match offset {
            IMEM_UCR_OFFSET => {
                memory.write_internal_byte(offset, value);
                true
            }
            IMEM_TXD_OFFSET => {
                memory.write_internal_byte(offset, value);
                self.tx_queue.push_back(value);
                self.queue_auto_response(memory);
                self.apply_status(memory);
                true
            }
            IMEM_USR_OFFSET | IMEM_RXD_OFFSET => {
                memory.write_internal_byte(offset, value);
                true
            }
            _ => false,
        }
    }

    fn apply_status(&self, memory: &mut MemoryImage) {
        let mut usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        usr |= USR_TX_READY | USR_TX_EMPTY;
        if self.rx_queue.is_empty() {
            usr &= !USR_RX_READY;
        } else {
            usr |= USR_RX_READY;
        }
        usr &= !USR_ERROR_MASK;
        memory.write_internal_byte(IMEM_USR_OFFSET, usr);
    }

    fn queue_auto_response(&mut self, memory: &mut MemoryImage) {
        let response = self.auto_response;
        self.rx_queue.push_back(response);
        memory.write_internal_byte(IMEM_RXD_OFFSET, response);
        memory.write_internal_byte(IMEM_BH_OFFSET, response);
    }

    fn consume_rx(&mut self, memory: &mut MemoryImage) -> u8 {
        let value = self.rx_queue.pop_front().unwrap_or(0);
        if let Some(next) = self.rx_queue.front().copied() {
            memory.write_internal_byte(IMEM_RXD_OFFSET, next);
        }
        self.apply_status(memory);
        value
    }

    fn force_return_auto(
        &mut self,
        state: &mut crate::llama::state::LlamaState,
        memory: &mut MemoryImage,
    ) {
        let ret_bits = match state.peek_call_return_width() {
            Some(16) => 16,
            Some(24) => 24,
            _ => {
                let call_depth = state.call_stack().len();
                let page_depth = state.call_page_depth();
                if page_depth < call_depth {
                    24
                } else {
                    16
                }
            }
        };
        if ret_bits == 24 {
            let ret = Self::pop_stack_value(state, memory, 24);
            let dest = ret & 0xFFFFF;
            state.set_pc(dest);
        } else {
            let pc_before = state.pc();
            let ret = Self::pop_stack_value(state, memory, 16);
            let current_page = pc_before & 0xFF0000;
            let dest = (current_page | (ret & 0xFFFF)) & 0xFFFFF;
            let _ = state.pop_call_page();
            state.set_pc(dest);
        }
        state.call_depth_dec();
        let _ = state.pop_call_frame();
    }

    fn pop_stack_value(
        state: &mut crate::llama::state::LlamaState,
        memory: &mut MemoryImage,
        bits: u8,
    ) -> u32 {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_write_enqueues_rx_and_sets_status() {
        let mut memory = MemoryImage::new();
        let mut stub = SioStub::new();
        stub.init(&mut memory);

        assert_eq!(
            memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0) & USR_RX_READY,
            0
        );

        assert!(stub.handle_write(IMEM_TXD_OFFSET, 0x55, &mut memory));
        let usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        assert!(usr & USR_TX_READY != 0);
        assert!(usr & USR_TX_EMPTY != 0);
        assert!(usr & USR_RX_READY != 0);

        let value = stub.handle_read(IMEM_RXD_OFFSET, &mut memory).unwrap_or(0);
        assert_eq!(value, 0x41);
        let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        assert_eq!(usr_after & USR_RX_READY, 0);
    }
}
