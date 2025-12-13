// PY_SOURCE: pce500/keyboard_matrix.py:KeyboardMatrix

use crate::memory::MemoryImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const FIFO_SIZE: usize = 8;
const COLUMN_COUNT: usize = 11; // 8 from KOL, 3 from KOH
const FIFO_BASE_ADDR: u32 = 0x00BFC96;
const FIFO_HEAD_ADDR: u32 = 0x00BFC9D;
const FIFO_TAIL_ADDR: u32 = 0x00BFC9E;
const DEFAULT_PRESS_TICKS: u8 = 6;
const DEFAULT_RELEASE_TICKS: u8 = 6;
const DEFAULT_REPEAT_DELAY: u8 = 24;
const DEFAULT_REPEAT_INTERVAL: u8 = 6;

#[derive(Copy, Clone, Debug, Default)]
struct KeyLocation {
    column: u8,
    row: u8,
}

#[derive(Copy, Clone, Debug, Default)]
struct KeyState {
    location: KeyLocation,
    pressed: bool,
    debounced: bool,
    press_ticks: u8,
    release_ticks: u8,
    repeat_ticks: u8,
}

impl KeyState {
    fn matrix_code(&self) -> u8 {
        (self.location.column << 3) | (self.location.row & 0x07)
    }
}

// Matrix layout (row-major) copied from the Python keyboard map.
const KEY_NAMES: [Option<&str>; 88] = [
    // row 0
    Some("KEY_TRIANGLE_UP_DOWN"),
    Some("KEY_W"),
    Some("KEY_R"),
    Some("KEY_Y"),
    Some("KEY_I"),
    Some("KEY_RCL"),
    Some("KEY_STO"),
    Some("KEY_C_CE"),
    Some("KEY_UP_DOWN"),
    Some("KEY_RPAREN"),
    Some("KEY_P"),
    // row 1
    Some("KEY_Q"),
    Some("KEY_E"),
    Some("KEY_T"),
    Some("KEY_U"),
    Some("KEY_O"),
    Some("KEY_HYP"),
    Some("KEY_SIN"),
    Some("KEY_COS"),
    Some("KEY_TAN"),
    Some("KEY_FSE"),
    Some("KEY_2NDF"),
    // row 2
    Some("KEY_MENU"),
    Some("KEY_S"),
    Some("KEY_F"),
    Some("KEY_H"),
    Some("KEY_K"),
    Some("KEY_TO_HEX"),
    Some("KEY_TO_DEG"),
    Some("KEY_LN"),
    Some("KEY_LOG"),
    Some("KEY_1_X"),
    Some("KEY_F5"),
    // row 3
    Some("KEY_A"),
    Some("KEY_D"),
    Some("KEY_G"),
    Some("KEY_J"),
    Some("KEY_L"),
    Some("KEY_EXP"),
    Some("KEY_Y_X"),
    Some("KEY_SQRT"),
    Some("KEY_X2"),
    Some("KEY_LPAREN"),
    Some("KEY_F4"),
    // row 4
    Some("KEY_BASIC"),
    Some("KEY_X"),
    Some("KEY_V"),
    Some("KEY_N"),
    Some("KEY_COMMA"),
    Some("KEY_7"),
    Some("KEY_8"),
    Some("KEY_9"),
    Some("KEY_DIVIDE"),
    Some("KEY_DELETE"),
    Some("KEY_F3"),
    // row 5
    Some("KEY_Z"),
    Some("KEY_C"),
    Some("KEY_B"),
    Some("KEY_M"),
    Some("KEY_SEMICOLON"),
    Some("KEY_4"),
    Some("KEY_5"),
    Some("KEY_6"),
    Some("KEY_MULTIPLY"),
    Some("KEY_BACKSPACE"),
    Some("KEY_F2"),
    // row 6
    Some("KEY_SHIFT"),
    Some("KEY_CAPS"),
    Some("KEY_SPACE"),
    Some("KEY_UP"),
    Some("KEY_RIGHT"),
    Some("KEY_1"),
    Some("KEY_2"),
    Some("KEY_3"),
    Some("KEY_MINUS"),
    Some("KEY_INSERT"),
    Some("KEY_F1"),
    // row 7
    Some("KEY_CTRL"),
    Some("KEY_ANS"),
    Some("KEY_DOWN"),
    Some("KEY_DOWN_TRIANGLE"),
    Some("KEY_LEFT"),
    Some("KEY_0"),
    Some("KEY_PLUSMINUS"),
    Some("KEY_PERIOD"),
    Some("KEY_PLUS"),
    Some("KEY_EQUALS"),
    None,
];

#[derive(Default)]
pub struct KeyboardMatrix {
    kol: u8,
    koh: u8,
    kil_latch: u8,
    columns_active_high: bool,
    press_threshold: u8,
    release_threshold: u8,
    repeat_delay: u8,
    repeat_interval: u8,
    repeat_enabled: bool,
    scan_enabled: bool,
    states: Vec<KeyState>,
    fifo_storage: [u8; FIFO_SIZE],
    fifo_head: usize,
    fifo_tail: usize,
    fifo_count: usize,
    strobe_count: u32,
    irq_count: u32,
    column_histogram: [u32; COLUMN_COUNT],
    keyi_latch: bool,
    kil_read_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct KeyboardTelemetry {
    pub pressed: usize,
    pub strobe_count: u32,
    pub kol: u8,
    pub koh: u8,
    pub active_columns: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardSnapshot {
    pub kol: u8,
    pub koh: u8,
    pub kil_latch: u8,
    pub fifo_len: usize,
    pub fifo: Vec<u8>,
    pub head: usize,
    pub tail: usize,
    pub irq_count: u32,
    pub strobe_count: u32,
    pub active_columns: Vec<u8>,
    pub pressed_keys: Vec<String>,
    pub key_states: HashMap<String, KeyStateSnapshot>,
    pub column_histogram: Vec<u32>,
    pub press_threshold: u8,
    pub release_threshold: u8,
    pub repeat_delay: u8,
    pub repeat_interval: u8,
    pub columns_active_high: bool,
    pub scan_enabled: bool,
    #[serde(default)]
    pub kil_read_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyStateSnapshot {
    pub pressed: bool,
    pub debounced: bool,
    pub press_ticks: u8,
    pub release_ticks: u8,
    pub repeat_ticks: u8,
}

impl KeyboardMatrix {
    pub fn new() -> Self {
        let mut matrix = Self {
            kol: 0x00,
            koh: 0x00,
            kil_latch: 0x00,
            columns_active_high: true,
            press_threshold: DEFAULT_PRESS_TICKS,
            release_threshold: DEFAULT_RELEASE_TICKS,
            repeat_delay: DEFAULT_REPEAT_DELAY,
            repeat_interval: DEFAULT_REPEAT_INTERVAL,
            repeat_enabled: true,
            scan_enabled: true,
            states: Vec::new(),
            fifo_storage: [0; FIFO_SIZE],
            fifo_head: 0,
            fifo_tail: 0,
            fifo_count: 0,
            strobe_count: 0,
            irq_count: 0,
            column_histogram: [0; COLUMN_COUNT],
            keyi_latch: false,
            kil_read_count: 0,
        };
        for matrix_code in 0..(11 * 8) {
            let column = (matrix_code >> 3) as u8;
            let row = (matrix_code & 0x07) as u8;
            matrix.states.push(KeyState {
                location: KeyLocation { column, row },
                ..Default::default()
            });
        }
        matrix
    }

    pub fn active_columns(&self) -> Vec<u8> {
        let mut cols = Vec::new();
        for col in 0..8 {
            let bit = (self.kol >> col) & 1;
            let active = if self.columns_active_high {
                bit == 1
            } else {
                bit == 0
            };
            if active {
                cols.push(col);
            }
        }
        for col in 0..3 {
            let bit = (self.koh >> col) & 1;
            let active = if self.columns_active_high {
                bit == 1
            } else {
                bit == 0
            };
            if active {
                cols.push(col + 8);
            }
        }
        cols
    }

    pub fn compute_kil(&self, allow_pending: bool) -> u8 {
        let mut value = 0u8;
        let active = self.active_columns();
        for state in &self.states {
            let col_active = active.contains(&state.location.column);
            if !col_active && !allow_pending && !state.pressed {
                continue;
            }
            if state.debounced
                || (allow_pending
                    && state.pressed
                    && state.press_ticks.saturating_add(1) >= self.press_threshold)
            {
                value |= 1 << (state.location.row & 0x07);
            }
        }
        value
    }

    fn enqueue_event(&mut self, code: u8, release: bool, count_irq: bool) -> usize {
        let mut value = code & 0x7F;
        if release {
            value |= 0x80;
        }
        if self.fifo_count == FIFO_SIZE {
            self.fifo_head = (self.fifo_head + 1) % FIFO_SIZE;
            self.fifo_count -= 1;
        }
        self.fifo_storage[self.fifo_tail] = value;
        self.fifo_tail = (self.fifo_tail + 1) % FIFO_SIZE;
        self.fifo_count += 1;
        if count_irq {
            self.irq_count = self.irq_count.wrapping_add(1);
            self.keyi_latch = true;
        }
        1
    }

    /// Reconcile FIFO head/tail with firmware-written pointers in RAM so drained entries
    /// do not linger in the Rust model (avoids reasserting KEYI after the host consumes data).
    pub fn sync_fifo_from_memory(&mut self, memory: &MemoryImage) {
        if let Some(head) = memory.load(FIFO_HEAD_ADDR, 8) {
            let new_head = (head as usize) % FIFO_SIZE;
            if new_head != self.fifo_head {
                self.fifo_head = new_head;
            }
        }
        // Tail is owned by the producer (this model); keep the in-memory value as advisory only.
        // Recompute count based on reconciled head/tail.
        let mut count = if self.fifo_tail >= self.fifo_head {
            self.fifo_tail.saturating_sub(self.fifo_head)
        } else {
            FIFO_SIZE.saturating_sub(self.fifo_head.saturating_sub(self.fifo_tail))
        };
        if self.fifo_head == self.fifo_tail {
            count = 0;
        }
        self.fifo_count = count.min(FIFO_SIZE);
        if self.fifo_count == 0 {
            self.keyi_latch = false;
        }
    }

    /// Inject a matrix event immediately, bypassing debounce, and mirror state into memory.
    pub fn inject_matrix_event(
        &mut self,
        code: u8,
        release: bool,
        memory: &mut MemoryImage,
        kb_irq_enabled: bool,
    ) -> usize {
        self.sync_fifo_from_memory(memory);
        let events = self.enqueue_event(code & 0x7F, release, true);
        // Update KIL latch directly for bridge calls (row is low 3 bits).
        let row_mask = 1u8 << (code & 0x07);
        if release {
            self.kil_latch &= !row_mask;
        } else {
            self.kil_latch |= row_mask;
        }
        memory.write_internal_byte(0xF2, self.kil_latch);
        if events > 0 {
            self.write_fifo_to_memory(memory, kb_irq_enabled);
        }
        events
    }

    fn log_fifo_write(addr: u32, value: u8) {
        let mut guard = crate::PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let (seq, pc) = crate::llama::eval::perfetto_instr_context().unwrap_or_else(|| {
                (
                    crate::llama::eval::perfetto_last_instr_index(),
                    crate::llama::eval::perfetto_last_pc(),
                )
            });
            let substep = crate::llama::eval::perfetto_next_substep();
            tracer.record_mem_write_with_substep(
                seq,
                pc,
                addr,
                value as u32,
                "external",
                8,
                substep,
            );
        });
    }

    pub fn write_fifo_to_memory(&mut self, memory: &mut MemoryImage, kb_irq_enabled: bool) {
        self.sync_fifo_from_memory(memory);
        let mut idx = self.fifo_head;
        for slot in 0..FIFO_SIZE {
            let value = if slot < self.fifo_count {
                let byte = self.fifo_storage[idx];
                idx = (idx + 1) % FIFO_SIZE;
                byte
            } else {
                0
            };
            let addr = FIFO_BASE_ADDR + slot as u32;
            memory.write_external_byte(addr, value);
            Self::log_fifo_write(addr, value);
        }
        memory.write_external_byte(FIFO_HEAD_ADDR, self.fifo_head as u8);
        Self::log_fifo_write(FIFO_HEAD_ADDR, self.fifo_head as u8);
        memory.write_external_byte(FIFO_TAIL_ADDR, self.fifo_tail as u8);
        Self::log_fifo_write(FIFO_TAIL_ADDR, self.fifo_tail as u8);
        // Assert KEYI only if keyboard IRQs are enabled, matching Python gating.
        if self.keyi_latch && self.fifo_count > 0 && kb_irq_enabled {
            if let Some(isr) = memory.read_internal_byte(0xFC) {
                if (isr & 0x04) == 0 {
                    memory.write_internal_byte(0xFC, isr | 0x04);
                    let mut guard = crate::PERFETTO_TRACER.enter();
                    guard.with_some(|tracer| {
                        let (seq, pc) = crate::llama::eval::perfetto_instr_context()
                            .unwrap_or_else(|| {
                                (
                                    crate::llama::eval::perfetto_last_instr_index(),
                                    crate::llama::eval::perfetto_last_pc(),
                                )
                            });
                        let substep = crate::llama::eval::perfetto_next_substep();
                        tracer.record_mem_write_with_substep(
                            seq,
                            pc,
                            crate::INTERNAL_MEMORY_START + 0xFC,
                            (isr | 0x04) as u32,
                            "internal",
                            8,
                            substep,
                        );
                    });
                }
            }
        }
    }

    pub fn reset(&mut self, memory: &mut MemoryImage) {
        self.kol = 0;
        self.koh = 0;
        self.kil_latch = 0;
        self.fifo_storage = [0; FIFO_SIZE];
        self.fifo_head = 0;
        self.fifo_tail = 0;
        self.fifo_count = 0;
        self.strobe_count = 0;
        self.irq_count = 0;
        self.column_histogram = [0; COLUMN_COUNT];
        self.scan_enabled = true;
        self.keyi_latch = false;
        self.kil_read_count = 0;
        for state in &mut self.states {
            state.pressed = false;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = 0;
        }
        self.write_fifo_to_memory(memory, true);
    }

    pub fn fifo_snapshot(&self) -> Vec<u8> {
        let mut snapshot = Vec::with_capacity(self.fifo_count);
        let mut idx = self.fifo_head;
        for _ in 0..self.fifo_count {
            snapshot.push(self.fifo_storage[idx]);
            idx = (idx + 1) % FIFO_SIZE;
        }
        snapshot
    }

    fn key_name_for(matrix_code: u8) -> Option<&'static str> {
        KEY_NAMES.get(matrix_code as usize).and_then(|entry| *entry)
    }

    pub fn irq_count(&self) -> u32 {
        self.irq_count
    }

    pub fn strobe_count(&self) -> u32 {
        self.strobe_count
    }

    fn bump_column_histogram(&mut self) {
        let active = self.active_columns();
        for col in &active {
            if (*col as usize) < self.column_histogram.len() {
                self.column_histogram[*col as usize] =
                    self.column_histogram[*col as usize].wrapping_add(1);
            }
        }
    }

    pub fn fifo_len(&self) -> usize {
        self.fifo_count
    }

    pub fn telemetry(&self) -> KeyboardTelemetry {
        let pressed = self.states.iter().filter(|s| s.pressed).count();
        KeyboardTelemetry {
            pressed,
            strobe_count: self.strobe_count,
            kol: self.kol,
            koh: self.koh,
            active_columns: self.active_columns(),
        }
    }

    pub fn snapshot_state(&self) -> KeyboardSnapshot {
        let mut key_states: HashMap<String, KeyStateSnapshot> = HashMap::new();
        let mut pressed_keys: Vec<String> = Vec::new();
        for (idx, state) in self.states.iter().enumerate() {
            if let Some(name) = Self::key_name_for(idx as u8) {
                if state.pressed {
                    pressed_keys.push(name.to_string());
                }
                key_states.insert(
                    name.to_string(),
                    KeyStateSnapshot {
                        pressed: state.pressed,
                        debounced: state.debounced,
                        press_ticks: state.press_ticks,
                        release_ticks: state.release_ticks,
                        repeat_ticks: state.repeat_ticks,
                    },
                );
            }
        }
        KeyboardSnapshot {
            kol: self.kol,
            koh: self.koh,
            kil_latch: self.kil_latch,
            fifo_len: self.fifo_count,
            fifo: self.fifo_storage.to_vec(),
            head: self.fifo_head,
            tail: self.fifo_tail,
            irq_count: self.irq_count,
            strobe_count: self.strobe_count,
            active_columns: self.active_columns(),
            pressed_keys,
            key_states,
            column_histogram: self.column_histogram.to_vec(),
            press_threshold: self.press_threshold,
            release_threshold: self.release_threshold,
            repeat_delay: self.repeat_delay,
            repeat_interval: self.repeat_interval,
            columns_active_high: self.columns_active_high,
            scan_enabled: self.scan_enabled,
            kil_read_count: self.kil_read_count,
        }
    }

    pub fn load_snapshot_state(&mut self, snapshot: &KeyboardSnapshot) {
        self.kol = snapshot.kol;
        self.koh = snapshot.koh & 0x0F;
        self.kil_latch = snapshot.kil_latch;
        self.fifo_storage = [0; FIFO_SIZE];
        for (idx, byte) in snapshot.fifo.iter().enumerate().take(FIFO_SIZE) {
            self.fifo_storage[idx] = *byte;
        }
        self.fifo_head = snapshot.head.min(FIFO_SIZE.saturating_sub(1));
        self.fifo_tail = snapshot.tail.min(FIFO_SIZE.saturating_sub(1));
        self.fifo_count = snapshot.fifo_len.min(FIFO_SIZE);
        self.irq_count = snapshot.irq_count;
        self.strobe_count = snapshot.strobe_count;
        self.column_histogram = {
            let mut hist = [0u32; COLUMN_COUNT];
            for (idx, val) in snapshot
                .column_histogram
                .iter()
                .enumerate()
                .take(COLUMN_COUNT)
            {
                hist[idx] = *val;
            }
            hist
        };
        self.press_threshold = snapshot.press_threshold;
        self.release_threshold = snapshot.release_threshold;
        self.repeat_delay = snapshot.repeat_delay;
        self.repeat_interval = snapshot.repeat_interval;
        self.columns_active_high = snapshot.columns_active_high;
        self.scan_enabled = snapshot.scan_enabled;
        self.kil_read_count = snapshot.kil_read_count;
        // Reset per-key state and repopulate from snapshot where names match.
        for state in &mut self.states {
            state.pressed = false;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = 0;
        }
        for (name, saved) in &snapshot.key_states {
            if let Some(idx) = KEY_NAMES
                .iter()
                .position(|entry| entry.map(|s| s == name).unwrap_or(false))
            {
                if let Some(state) = self.states.get_mut(idx) {
                    state.pressed = saved.pressed;
                    state.debounced = saved.debounced;
                    state.press_ticks = saved.press_ticks;
                    state.release_ticks = saved.release_ticks;
                    state.repeat_ticks = saved.repeat_ticks;
                }
            }
        }
        self.keyi_latch = snapshot.fifo_len > 0;
        self.kil_latch = self.compute_kil(false);
    }

    pub fn set_repeat_enabled(&mut self, enabled: bool) {
        self.repeat_enabled = enabled;
    }

    pub fn press_matrix_code(&mut self, code: u8, memory: &mut MemoryImage) {
        let _ = memory;
        if let Some(state) = self.states.get_mut(code as usize) {
            state.pressed = true;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = self.repeat_delay;
            self.kil_latch = self.compute_kil(false);
            // Parity: defer event enqueue/KEYI to timer-driven scan_tick; do not push KIL to IMEM here.
        }
    }

    pub fn release_matrix_code(&mut self, code: u8, memory: &mut MemoryImage) {
        let _ = memory;
        if let Some(state) = self.states.get_mut(code as usize) {
            state.pressed = false;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = 0;
            self.kil_latch = self.compute_kil(false);
            // Parity: defer event enqueue/KEYI to timer-driven scan_tick.
        }
    }

    pub fn handle_read(&mut self, offset: u32, _memory: &mut MemoryImage) -> Option<u8> {
        match offset {
            0xF0 => Some(self.kol),
            0xF1 => Some(self.koh),
            0xF2 => {
                // Parity: Python KIL reads expose pending (nearly debounced) presses.
                self.kil_latch = self.compute_kil(true);
                self.kil_read_count = self.kil_read_count.wrapping_add(1);
                Some(self.kil_latch)
            }
            _ => None,
        }
    }

    pub fn handle_write(&mut self, offset: u32, value: u8, memory: &mut MemoryImage) -> bool {
        match offset {
            0xF0 => {
                self.kol = value;
                self.kil_latch = self.compute_kil(false);
                self.strobe_count = self.strobe_count.wrapping_add(1);
                self.bump_column_histogram();
                true
            }
            0xF1 => {
                self.koh = value & 0x0F;
                self.kil_latch = self.compute_kil(false);
                self.strobe_count = self.strobe_count.wrapping_add(1);
                self.bump_column_histogram();
                true
            }
            0xF2 => {
                memory.write_internal_byte(offset, self.kil_latch);
                true
            }
            _ => false,
        }
    }

    pub fn scan_tick(&mut self, memory: &mut MemoryImage, count_irq: bool) -> usize {
        // Parity: respect firmware-updated head/tail before enqueuing new events.
        self.sync_fifo_from_memory(memory);
        if !self.scan_enabled {
            return 0;
        }
        let active = self.active_columns();
        let mut events = 0usize;
        for idx in 0..self.states.len() {
            let mut enqueue: Option<(u8, bool)> = None;
            {
                let state = &mut self.states[idx];
                let strobed = active.contains(&state.location.column);
                if state.pressed && strobed {
                    if !state.debounced {
                        state.press_ticks = state.press_ticks.saturating_add(1);
                        if state.press_ticks >= self.press_threshold {
                            state.debounced = true;
                            enqueue = Some((state.matrix_code(), false));
                        }
                    } else if self.repeat_enabled {
                        state.repeat_ticks = state.repeat_ticks.saturating_sub(1);
                        if state.repeat_ticks == 0 {
                            state.repeat_ticks = self.repeat_interval;
                            enqueue = Some((state.matrix_code(), false));
                        }
                    }
                } else if state.debounced && !strobed {
                    state.release_ticks = state.release_ticks.saturating_add(1);
                    if state.release_ticks >= self.release_threshold {
                        state.debounced = false;
                        enqueue = Some((state.matrix_code(), true));
                    }
                }
            }
            if let Some((code, release)) = enqueue {
                events += self.enqueue_event(code, release, count_irq);
            }
        }
        self.kil_latch = self.compute_kil(false);
        if self.fifo_count == 0 {
            self.keyi_latch = false;
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryImage;

    #[test]
    fn kil_read_includes_pending_press() {
        let mut kb = KeyboardMatrix::new();
        let mut mem = MemoryImage::new();
        // Activate column 0 (row 0 belongs to matrix code 0).
        kb.handle_write(0xF0, 0x01, &mut mem);
        // Mark a key as physically pressed but not yet debounced.
        if let Some(state) = kb.states.get_mut(0) {
            state.pressed = true;
            state.debounced = false;
            state.press_ticks = DEFAULT_PRESS_TICKS.saturating_sub(1);
        }
        let kil = kb.handle_read(0xF2, &mut mem).unwrap();
        assert_ne!(kil & 0x01, 0, "row 0 should be set for pending press");
        assert_eq!(kb.kil_read_count, 1);
    }

    #[test]
    fn scan_tick_respects_irq_disable() {
        let mut kb = KeyboardMatrix::new();
        kb.press_threshold = 1;
        let mut mem = MemoryImage::new();
        kb.handle_write(0xF0, 0x01, &mut mem);
        kb.press_matrix_code(0, &mut mem);
        let events = kb.scan_tick(&mut mem, false);
        assert!(events > 0, "expected a debounced event");
        kb.write_fifo_to_memory(&mut mem, false);
        let isr = mem
            .read_internal_byte(crate::memory::IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_eq!(isr & 0x04, 0, "KEYI should remain clear when IRQs disabled");
        assert!(kb.fifo_len() > 0, "FIFO should still capture the event");
    }

    #[test]
    fn scan_tick_syncs_head_from_memory() {
        let mut kb = KeyboardMatrix::new();
        kb.press_threshold = 1;
        let mut mem = MemoryImage::new();
        kb.handle_write(0xF0, 0x01, &mut mem);
        kb.press_matrix_code(0, &mut mem);
        let events = kb.scan_tick(&mut mem, true);
        assert!(events > 0, "expected a debounced event");
        kb.write_fifo_to_memory(&mut mem, true);
        assert_eq!(kb.fifo_len(), 1);
        // Simulate firmware consuming one entry by advancing head in RAM.
        mem.write_external_byte(FIFO_HEAD_ADDR, 1);
        kb.release_matrix_code(0, &mut mem);
        let _ = kb.scan_tick(&mut mem, true);
        assert_eq!(kb.fifo_len(), 0, "head sync should drop consumed entries");
    }
}
