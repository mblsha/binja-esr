//! Register-only byte UART: no IOCS/workspace/BH or software flow-control writes.
//! Timing is an explicit functional clock hypothesis, not silicon qualification.
// PY_SOURCE: pce500/peripherals/uart.py
use crate::sio::SioQueuedByte;
#[cfg(feature = "json-compat")]
use serde_json::{json, Value};
use std::collections::VecDeque;

const QUEUE_LIMIT: usize = 4096;
const BAUD: [u64; 8] = [0, 300, 600, 1200, 2400, 4800, 9600, 19200];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartEvent {
    RxReady(u8),
    TxReady(u8),
    TxComplete(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uart {
    pub timebase_hz: u64,
    pub baud_divisor: u64,
    pub control: u8,
    pub last_tx: u8,
    pub rx_data: u8,
    pub rx_errors: u8,
    pub rx_latch: Option<SioQueuedByte>,
    pub pending_rx: VecDeque<SioQueuedByte>,
    pub rx_remaining: Option<u64>,
    pub tx_holding: Option<u8>,
    pub tx_shift: Option<u8>,
    pub tx_load_remaining: Option<u64>,
    pub tx_remaining: Option<u64>,
    pub shift_break: bool,
    pub completed_tx: VecDeque<u8>,
    pub rejected_rx: u64,
    pub rejected_tx: u64,
    pub dropped_tx: u64,
    pub suppressed_break_frames: u64,
}

impl Uart {
    pub fn new(timebase_hz: u64, baud_divisor: u64) -> Self {
        assert!(timebase_hz > 0 && baud_divisor > 0);
        Self {
            timebase_hz,
            baud_divisor,
            control: 0,
            last_tx: 0,
            rx_data: 0,
            rx_errors: 0,
            rx_latch: None,
            pending_rx: VecDeque::new(),
            rx_remaining: None,
            tx_holding: None,
            tx_shift: None,
            tx_load_remaining: None,
            tx_remaining: None,
            shift_break: false,
            completed_tx: VecDeque::new(),
            rejected_rx: 0,
            rejected_tx: 0,
            dropped_tx: 0,
            suppressed_break_frames: 0,
        }
    }

    pub fn baud(&self) -> u64 {
        BAUD[((self.control >> 4) & 7) as usize] / self.baud_divisor
    }
    pub fn bit_units(&self) -> u64 {
        self.timebase_hz.div_ceil(self.baud().max(1))
    }
    pub fn frame_units(&self) -> u64 {
        let data = if self.control & 2 == 0 { 8 } else { 7 };
        let parity = u64::from(self.control & 8 == 0);
        let stop = if self.control & 1 == 0 { 1 } else { 2 };
        (1 + data + parity + stop) * self.bit_units()
    }
    fn data_mask(&self) -> u8 {
        if self.control & 2 == 0 {
            255
        } else {
            127
        }
    }

    pub fn status(&self) -> u8 {
        (u8::from(self.rx_latch.is_some()) * 0x20)
            | (u8::from(self.tx_shift.is_none()) * 0x10)
            | (u8::from(self.tx_holding.is_none()) * 8)
            | self.rx_errors
    }

    pub fn write_control(&mut self, value: u8) {
        self.control = value;
        if value & 0x70 == 0 {
            // UART reset aborts in-flight work; already emitted host bytes
            // remain available to drain. No software RAM is touched.
            self.rx_latch = None;
            self.rx_data = 0;
            self.rx_errors = 0;
            self.pending_rx.clear();
            self.rx_remaining = None;
            self.tx_holding = None;
            self.tx_shift = None;
            self.tx_load_remaining = None;
            self.tx_remaining = None;
            self.shift_break = false;
        } else if value & 0x80 != 0 && self.tx_shift.is_some() {
            self.shift_break = true;
        }
    }

    pub fn write_tx(&mut self, value: u8) -> bool {
        self.last_tx = value;
        if self.baud() == 0 || self.tx_holding.is_some() {
            self.rejected_tx += 1;
            return false;
        }
        self.tx_holding = Some(value & self.data_mask());
        if self.tx_shift.is_none() {
            // Related manual specifies one/two bit-times. Use two explicitly
            // until physical load latency is measured; TXE is still idle here.
            self.tx_load_remaining = Some(2 * self.bit_units());
        }
        true
    }

    pub fn queue_rx(&mut self, entry: SioQueuedByte) -> bool {
        if self.baud() == 0 || self.pending_rx.len() >= QUEUE_LIMIT {
            self.rejected_rx += 1;
            return false;
        }
        self.pending_rx.push_back(entry);
        if self.rx_remaining.is_none() {
            self.rx_remaining = Some(self.frame_units());
        }
        true
    }

    pub fn consume_rx(&mut self) -> Option<SioQueuedByte> {
        self.rx_latch.take()
    }
    pub fn read_rx(&mut self) -> u8 {
        self.rx_latch = None;
        self.rx_data
    }
    pub fn take_tx(&mut self) -> Option<u8> {
        self.completed_tx.pop_front()
    }
    pub fn pending_tx(&self) -> Vec<u8> {
        self.tx_shift.into_iter().chain(self.tx_holding).collect()
    }

    fn load_shift(&mut self, events: &mut Vec<UartEvent>) {
        let value = self.tx_holding.take().expect("armed UART load has a byte");
        self.tx_shift = Some(value);
        self.tx_remaining = Some(self.frame_units());
        self.tx_load_remaining = None;
        self.shift_break = self.control & 0x80 != 0;
        events.push(UartEvent::TxReady(value));
    }

    pub fn advance(&mut self, units: u64) -> Vec<UartEvent> {
        let mut events = Vec::new();
        let mut remaining = units;
        while remaining != 0 {
            let Some(deadline) = [self.rx_remaining, self.tx_load_remaining, self.tx_remaining]
                .into_iter()
                .flatten()
                .min()
            else {
                break;
            };
            let step = remaining.min(deadline);
            for countdown in [
                &mut self.rx_remaining,
                &mut self.tx_load_remaining,
                &mut self.tx_remaining,
            ] {
                if let Some(value) = countdown.as_mut() {
                    *value -= step;
                }
            }
            remaining -= step;
            if self.rx_remaining == Some(0) {
                let mut entry = self
                    .pending_rx
                    .pop_front()
                    .expect("armed RX has a host byte");
                entry.value &= self.data_mask();
                entry.overrun_error |= self.rx_latch.is_some();
                self.rx_errors = u8::from(entry.parity_error)
                    | (u8::from(entry.overrun_error) * 2)
                    | (u8::from(entry.framing_error) * 4);
                self.rx_data = entry.value;
                self.rx_latch = Some(entry);
                self.rx_remaining = (!self.pending_rx.is_empty()).then(|| self.frame_units());
                events.push(UartEvent::RxReady(entry.value));
            }
            if self.tx_load_remaining == Some(0) {
                self.load_shift(&mut events);
            }
            if self.tx_remaining == Some(0) {
                let byte = self.tx_shift.take().expect("armed TX has a shift byte");
                self.tx_remaining = None;
                if self.shift_break {
                    self.suppressed_break_frames += 1;
                } else {
                    if self.completed_tx.len() == QUEUE_LIMIT {
                        self.completed_tx.pop_front();
                        self.dropped_tx += 1;
                    }
                    self.completed_tx.push_back(byte);
                    events.push(UartEvent::TxComplete(byte));
                }
                if self.tx_holding.is_some() {
                    self.load_shift(&mut events);
                }
            }
        }
        events
    }

    #[cfg(feature = "json-compat")]
    pub fn report(&self) -> Value {
        json!({"control":self.control,"status":self.status(),"rx_data":self.rx_data,
            "last_tx":self.last_tx,"baud":self.baud(),"timebase_hz":self.timebase_hz,
            "baud_divisor":self.baud_divisor,"bit_units":self.bit_units(),
            "frame_units":self.frame_units(),"tx_holding":self.tx_holding,
            "tx_shift":self.tx_shift,"rx_remaining":self.rx_remaining,
            "tx_load_remaining":self.tx_load_remaining,"tx_remaining":self.tx_remaining,
            "pending_rx":self.pending_rx.len(),"rx_ready":self.rx_latch.is_some(),
            "completed_tx":self.completed_tx.len(),"rejected_rx":self.rejected_rx,
            "rejected_tx":self.rejected_tx,"dropped_tx":self.dropped_tx,
            "suppressed_break_frames":self.suppressed_break_frames})
    }
}
