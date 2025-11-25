use crate::memory::MemoryImage;

const FIFO_SIZE: usize = 8;
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
    states: Vec<KeyState>,
    fifo_storage: [u8; FIFO_SIZE],
    fifo_head: usize,
    fifo_tail: usize,
    fifo_count: usize,
    strobe_count: u32,
    irq_count: u32,
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
            states: Vec::new(),
            fifo_storage: [0; FIFO_SIZE],
            fifo_head: 0,
            fifo_tail: 0,
            fifo_count: 0,
            strobe_count: 0,
            irq_count: 0,
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

    fn enqueue_event(&mut self, code: u8, release: bool) -> usize {
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
        self.irq_count = self.irq_count.wrapping_add(1);
        1
    }

    pub fn write_fifo_to_memory(&self, memory: &mut MemoryImage) {
        let mut idx = self.fifo_head;
        for slot in 0..FIFO_SIZE {
            let value = if slot < self.fifo_count {
                let byte = self.fifo_storage[idx];
                idx = (idx + 1) % FIFO_SIZE;
                byte
            } else {
                0
            };
            memory.write_external_byte(FIFO_BASE_ADDR + slot as u32, value);
        }
        memory.write_external_byte(FIFO_HEAD_ADDR, self.fifo_head as u8);
        memory.write_external_byte(FIFO_TAIL_ADDR, self.fifo_tail as u8);
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
        for state in &mut self.states {
            state.pressed = false;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = 0;
        }
        self.write_fifo_to_memory(memory);
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

    pub fn irq_count(&self) -> u32 {
        self.irq_count
    }

    pub fn strobe_count(&self) -> u32 {
        self.strobe_count
    }

    pub fn set_repeat_enabled(&mut self, enabled: bool) {
        self.repeat_enabled = enabled;
    }

    pub fn press_matrix_code(&mut self, code: u8, memory: &mut MemoryImage) {
        if let Some(state) = self.states.get_mut(code as usize) {
            state.pressed = true;
            state.debounced = true;
            state.press_ticks = self.press_threshold;
            state.release_ticks = 0;
            state.repeat_ticks = self.repeat_delay;
            self.kil_latch = self.compute_kil(true);
            self.enqueue_event(code, false);
            self.write_fifo_to_memory(memory);
        }
    }

    pub fn release_matrix_code(&mut self, code: u8, memory: &mut MemoryImage) {
        if let Some(state) = self.states.get_mut(code as usize) {
            state.pressed = false;
            state.debounced = false;
            state.press_ticks = 0;
            state.release_ticks = 0;
            state.repeat_ticks = 0;
            self.kil_latch = self.compute_kil(false);
            self.enqueue_event(code, true);
            self.write_fifo_to_memory(memory);
        }
    }

    pub fn handle_read(&mut self, offset: u32, memory: &mut MemoryImage) -> Option<u8> {
        match offset {
            0xF0 => Some(self.kol),
            0xF1 => Some(self.koh),
            0xF2 => {
                if self.scan_tick() > 0 {
                    self.write_fifo_to_memory(memory);
                }
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
                memory.write_internal_byte(offset, value);
                true
            }
            0xF1 => {
                self.koh = value & 0x0F;
                self.kil_latch = self.compute_kil(false);
                self.strobe_count = self.strobe_count.wrapping_add(1);
                memory.write_internal_byte(offset, self.koh);
                true
            }
            0xF2 => {
                memory.write_internal_byte(offset, self.kil_latch);
                true
            }
            _ => false,
        }
    }

    pub fn scan_tick(&mut self) -> usize {
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
                events += self.enqueue_event(code, release);
            }
        }
        self.kil_latch = self.compute_kil(false);
        events
    }
}
