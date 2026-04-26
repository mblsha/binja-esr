// PY_SOURCE: sc62015/pysc62015/instr/opcodes.py

use std::collections::VecDeque;

use crate::llama::opcodes::RegName;
use crate::llama::state::mask_for;
use crate::memory::{
    MemoryImage, IMEM_EIL_OFFSET, IMEM_RXD_OFFSET, IMEM_TXD_OFFSET, IMEM_UCR_OFFSET,
    IMEM_USR_OFFSET,
};

const IMEM_BH_OFFSET: u32 = 0xD5;
const USR_RX_READY: u8 = 0x20;
const USR_TX_EMPTY: u8 = 0x10;
const USR_TX_READY: u8 = 0x08;
const USR_ERROR_MASK: u8 = 0x07;
const USR_FRAMING_ERROR: u8 = 0x04;
const USR_OVERRUN_ERROR: u8 = 0x02;
const USR_PARITY_ERROR: u8 = 0x01;
const SERIAL_WORKSPACE_START: u32 = 0x00BFE40;
const SERIAL_WORKSPACE_END: u32 = 0x00BFE48;
const SERIAL_HANDSHAKE_ADDR: u32 = 0x00BFE46;
const SERIAL_EIL_CS_MASK: u8 = 0x04;
const SERIAL_EIL_CD_MASK: u8 = 0x02;
const SIO_CMD42_DIRECT_INPUT_ADDR: u32 = 0x00EB030;
const SIO_TX_WAIT_READY_ADDR: u32 = 0x00EB31C;
const SIO_CMD41_DIRECT_OUTPUT_ADDR: u32 = 0x00EB33D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SioQueuedByte {
    pub value: u8,
    pub parity_error: bool,
    pub overrun_error: bool,
    pub framing_error: bool,
}

impl SioQueuedByte {
    pub fn new(value: u8) -> Self {
        Self {
            value,
            parity_error: false,
            overrun_error: false,
            framing_error: false,
        }
    }

    fn error_bits(self) -> u8 {
        let mut bits = 0;
        if self.framing_error {
            bits |= USR_FRAMING_ERROR;
        }
        if self.overrun_error {
            bits |= USR_OVERRUN_ERROR;
        }
        if self.parity_error {
            bits |= USR_PARITY_ERROR;
        }
        bits
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SioSnapshot {
    pub ucr: u8,
    pub usr: u8,
    pub eil: u8,
    pub handshake: u8,
    pub workspace: Vec<(u32, u8)>,
    pub rx_queue: Vec<SioQueuedByte>,
    pub tx_queue: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SioInputLines {
    pub cs: bool,
    pub cd: bool,
}

#[derive(Debug, Default)]
pub struct SioStub {
    rx_queue: VecDeque<SioQueuedByte>,
    tx_queue: VecDeque<u8>,
    auto_response: u8,
    direct_input_timeout: bool,
}

impl SioStub {
    pub fn new() -> Self {
        Self {
            rx_queue: VecDeque::new(),
            tx_queue: VecDeque::new(),
            auto_response: 0x41,
            direct_input_timeout: false,
        }
    }

    pub fn init(&mut self, memory: &mut MemoryImage) {
        self.apply_status(memory);
    }

    pub fn set_auto_response(&mut self, value: u8) {
        self.auto_response = value;
    }

    pub fn set_direct_input_timeout(&mut self, enabled: bool) {
        self.direct_input_timeout = enabled;
    }

    pub fn snapshot(&self, memory: &MemoryImage) -> SioSnapshot {
        SioSnapshot {
            ucr: memory.read_internal_byte(IMEM_UCR_OFFSET).unwrap_or(0),
            usr: memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0),
            eil: memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0),
            handshake: self.get_handshake(memory),
            workspace: serial_workspace(memory),
            rx_queue: self.rx_queue.iter().copied().collect(),
            tx_queue: self.tx_queue.iter().copied().collect(),
        }
    }

    pub fn restore(&mut self, snapshot: SioSnapshot, memory: &mut MemoryImage) {
        memory.write_internal_byte(IMEM_UCR_OFFSET, snapshot.ucr);
        memory.write_internal_byte(IMEM_USR_OFFSET, snapshot.usr);
        memory.write_internal_byte(IMEM_EIL_OFFSET, snapshot.eil);
        self.set_handshake(memory, snapshot.handshake);
        for (addr, value) in snapshot.workspace {
            let _ = memory.store(addr, 8, u32::from(value));
        }
        self.rx_queue = VecDeque::from(snapshot.rx_queue);
        self.tx_queue = VecDeque::from(snapshot.tx_queue);
        self.latch_next_received(memory);
        self.apply_status(memory);
    }

    pub fn queue_receive(
        &mut self,
        value: u8,
        parity_error: bool,
        overrun_error: bool,
        framing_error: bool,
        memory: &mut MemoryImage,
    ) {
        self.rx_queue.push_back(SioQueuedByte {
            value,
            parity_error,
            overrun_error,
            framing_error,
        });
        self.latch_next_received(memory);
        self.apply_status(memory);
    }

    pub fn queue_receive_byte(&mut self, value: u8, memory: &mut MemoryImage) {
        self.queue_receive(value, false, false, false, memory);
    }

    pub fn consume_received(&mut self, memory: &mut MemoryImage) -> Option<SioQueuedByte> {
        if self.rx_queue.is_empty() {
            return None;
        }
        let entry = self.rx_queue.pop_front();
        self.latch_next_received(memory);
        self.apply_status(memory);
        entry
    }

    pub fn pending_receive(&self) -> Vec<SioQueuedByte> {
        self.rx_queue.iter().copied().collect()
    }

    pub fn pending_transmit(&self) -> Vec<u8> {
        self.tx_queue.iter().copied().collect()
    }

    pub fn queue_transmit(&mut self, value: u8, memory: &mut MemoryImage) {
        self.tx_queue.push_back(value);
        self.apply_status(memory);
    }

    pub fn complete_transmit(&mut self, memory: &mut MemoryImage) -> Option<u8> {
        let value = self.tx_queue.pop_front();
        self.apply_status(memory);
        value
    }

    pub fn set_handshake(&self, memory: &mut MemoryImage, value: u8) {
        let _ = memory.store(SERIAL_HANDSHAKE_ADDR, 8, u32::from(value));
    }

    pub fn get_handshake(&self, memory: &MemoryImage) -> u8 {
        memory
            .load(SERIAL_HANDSHAKE_ADDR, 8)
            .map(|value| value as u8)
            .unwrap_or(0)
    }

    pub fn set_input_lines(&self, memory: &mut MemoryImage, cs: Option<bool>, cd: Option<bool>) {
        let mut value = memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0);
        if let Some(enabled) = cs {
            value = set_mask(value, SERIAL_EIL_CS_MASK, enabled);
        }
        if let Some(enabled) = cd {
            value = set_mask(value, SERIAL_EIL_CD_MASK, enabled);
        }
        memory.write_internal_byte(IMEM_EIL_OFFSET, value);
    }

    pub fn input_lines(&self, memory: &MemoryImage) -> SioInputLines {
        let value = memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0);
        SioInputLines {
            cs: value & SERIAL_EIL_CS_MASK != 0,
            cd: value & SERIAL_EIL_CD_MASK != 0,
        }
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
        if pc == SIO_CMD42_DIRECT_INPUT_ADDR && self.direct_input_timeout {
            state.set_reg(RegName::A, 0x00);
            state.set_reg(RegName::FC, 1);
        } else {
            let response = self.auto_response;
            memory.write_internal_byte(IMEM_RXD_OFFSET, response);
            memory.write_internal_byte(IMEM_BH_OFFSET, response);
            state.set_reg(RegName::A, u32::from(response));
            state.set_reg(RegName::FC, 0);
        }
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
                self.queue_transmit(value, memory);
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
        if self.tx_queue.is_empty() {
            usr |= USR_TX_READY | USR_TX_EMPTY;
        } else {
            usr &= !(USR_TX_READY | USR_TX_EMPTY);
        }
        if self.rx_queue.is_empty() {
            usr &= !USR_RX_READY;
        } else {
            usr |= USR_RX_READY;
        }
        usr &= !USR_ERROR_MASK;
        if let Some(entry) = self.rx_queue.front().copied() {
            usr |= entry.error_bits();
        }
        memory.write_internal_byte(IMEM_USR_OFFSET, usr);
    }

    fn queue_auto_response(&mut self, memory: &mut MemoryImage) {
        let response = self.auto_response;
        self.rx_queue.push_back(SioQueuedByte::new(response));
        self.latch_next_received(memory);
    }

    fn consume_rx(&mut self, memory: &mut MemoryImage) -> u8 {
        let value = self.consume_received(memory).map_or(0, |entry| entry.value);
        self.apply_status(memory);
        value
    }

    fn latch_next_received(&self, memory: &mut MemoryImage) {
        if let Some(next) = self.rx_queue.front().copied() {
            memory.write_internal_byte(IMEM_RXD_OFFSET, next.value);
            memory.write_internal_byte(IMEM_BH_OFFSET, next.value);
        }
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

fn set_mask(value: u8, mask: u8, enabled: bool) -> u8 {
    if enabled {
        value | mask
    } else {
        value & !mask
    }
}

fn serial_workspace(memory: &MemoryImage) -> Vec<(u32, u8)> {
    (SERIAL_WORKSPACE_START..SERIAL_WORKSPACE_END)
        .map(|addr| {
            let value = memory.load(addr, 8).map(|value| value as u8).unwrap_or(0);
            (addr, value)
        })
        .collect()
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
        assert_eq!(stub.pending_transmit(), vec![0x55]);
        assert_eq!(usr & USR_TX_READY, 0);
        assert_eq!(usr & USR_TX_EMPTY, 0);
        assert!(usr & USR_RX_READY != 0);

        let value = stub.handle_read(IMEM_RXD_OFFSET, &mut memory).unwrap_or(0);
        assert_eq!(value, 0x41);
        let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        assert_eq!(usr_after & USR_RX_READY, 0);
    }

    #[test]
    fn receive_queue_sets_status_and_error_bits() {
        let mut memory = MemoryImage::new();
        let mut stub = SioStub::new();

        stub.queue_receive(0x41, true, false, true, &mut memory);

        assert_eq!(memory.read_internal_byte(IMEM_RXD_OFFSET), Some(0x41));
        let usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        assert!(usr & USR_RX_READY != 0);
        assert!(usr & USR_FRAMING_ERROR != 0);
        assert!(usr & USR_PARITY_ERROR != 0);

        let consumed = stub.consume_received(&mut memory).unwrap();
        assert_eq!(consumed.value, 0x41);
        assert!(consumed.framing_error);
        assert!(consumed.parity_error);

        let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
        assert_eq!(usr_after & USR_RX_READY, 0);
        assert_eq!(usr_after & USR_ERROR_MASK, 0);
    }

    #[test]
    fn input_lines_and_snapshot_restore_roundtrip() {
        let mut memory = MemoryImage::new();
        let mut stub = SioStub::new();

        stub.queue_receive_byte(0x41, &mut memory);
        stub.queue_transmit(0x55, &mut memory);
        stub.set_input_lines(&mut memory, Some(true), Some(true));
        stub.set_handshake(&mut memory, 0xA5);

        let snap = stub.snapshot(&memory);

        assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x41);
        assert_eq!(stub.complete_transmit(&mut memory), Some(0x55));
        stub.set_input_lines(&mut memory, Some(false), Some(false));
        stub.set_handshake(&mut memory, 0x00);

        stub.restore(snap, &mut memory);

        assert_eq!(
            stub.pending_receive()
                .iter()
                .map(|entry| entry.value)
                .collect::<Vec<_>>(),
            vec![0x41]
        );
        assert_eq!(stub.pending_transmit(), vec![0x55]);
        assert_eq!(
            stub.input_lines(&memory),
            SioInputLines { cs: true, cd: true }
        );
        assert_eq!(stub.get_handshake(&memory), 0xA5);
    }
}
