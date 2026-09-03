// PY_SOURCE: pce500/scheduler.py:TimerScheduler
// PY_SOURCE: pce500/emulator.py:PCE500Emulator._tick_timers

use crate::keyboard::KeyboardTelemetry;
use crate::llama::eval::perfetto_last_pc;
use crate::memory::MemoryImage;
use crate::perfetto::AnnotationValue;
use crate::PERFETTO_TRACER;
use crate::{InterruptInfo, TimerInfo};
use serde_json::json;
use std::collections::HashMap;

const ISR_OFFSET: u32 = 0xFC;
const SCR_MTS: u8 = 0x02;
const SCR_STS: u8 = 0x04;

#[derive(Clone, Debug)]
pub struct TimerContext {
    pub enabled: bool,
    pub mti_period: u64,
    pub sti_period: u64,
    pub next_mti: u64,
    pub next_sti: u64,
    pub kb_irq_enabled: bool,
    pub irq_pending: bool,
    pub irq_source: Option<String>,
    pub irq_imr: u8,
    pub irq_isr: u8,
    pub in_interrupt: bool,
    pub interrupt_stack: Vec<u32>,
    pub next_interrupt_id: u32,
    pub last_fired: Option<String>,
    pub key_irq_latched: bool,
    pub irq_total: u32,
    pub irq_key: u32,
    pub irq_mti: u32,
    pub irq_sti: u32,
    pub last_irq_src: Option<String>,
    pub last_irq_pc: Option<u32>,
    pub last_irq_vector: Option<u32>,
    pub irq_bit_watch: Option<serde_json::Map<String, serde_json::Value>>,
    pub delivered_masks: Vec<u8>,
    instruction_start_cycle: u64,
    last_mti_fire_cycle: Option<u64>,
    last_sti_fire_cycle: Option<u64>,
    fired_mti_since_boundary: bool,
    fired_sti_since_boundary: bool,
    timer_scale: f64,
    preserve_phase: bool,
    mti_short_period: u64,
    mti_long_period: u64,
    sti_short_period: u64,
    sti_long_period: u64,
    scr_selector: u8,
}

fn default_bit_watch_table() -> serde_json::Map<String, serde_json::Value> {
    let mut table = serde_json::Map::new();
    for reg in ["IMR", "ISR"] {
        let mut reg_map = serde_json::Map::new();
        for bit in 0..8u8 {
            reg_map.insert(
                bit.to_string(),
                json!({
                    "set": [],
                    "clear": [],
                }),
            );
        }
        table.insert(reg.to_string(), serde_json::Value::Object(reg_map));
    }
    table
}

fn normalize_bit_watch(table: &mut serde_json::Map<String, serde_json::Value>) {
    for reg in ["IMR", "ISR"] {
        if !table.get(reg).map(|v| v.is_object()).unwrap_or(false) {
            table.insert(
                reg.to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
        let reg_entry = table.get_mut(reg).expect("bit watch reg must exist");
        if !reg_entry.is_object() {
            *reg_entry = serde_json::Value::Object(serde_json::Map::new());
        }
        let reg_obj = reg_entry.as_object_mut().expect("bit watch reg is object");
        for bit in 0..8u8 {
            if !reg_obj
                .get(&bit.to_string())
                .map(|v| v.is_object())
                .unwrap_or(false)
            {
                reg_obj.insert(bit.to_string(), json!({ "set": [], "clear": [] }));
            }
            let bit_obj = reg_obj
                .get_mut(&bit.to_string())
                .and_then(|v| v.as_object_mut())
                .expect("bit watch bit entry should be object");
            bit_obj
                .entry("set".to_string())
                .or_insert_with(|| json!([]));
            bit_obj
                .entry("clear".to_string())
                .or_insert_with(|| json!([]));
        }
    }
}

impl TimerContext {
    pub fn new(enabled: bool, mti_period: i32, sti_period: i32) -> Self {
        let mut ctx = Self {
            enabled,
            mti_period: mti_period.max(0) as u64,
            sti_period: sti_period.max(0) as u64,
            next_mti: 0,
            next_sti: 0,
            kb_irq_enabled: true,
            irq_pending: false,
            irq_source: None,
            irq_imr: 0,
            irq_isr: 0,
            in_interrupt: false,
            interrupt_stack: Vec::new(),
            next_interrupt_id: 0,
            last_fired: None,
            key_irq_latched: false,
            irq_total: 0,
            irq_key: 0,
            irq_mti: 0,
            irq_sti: 0,
            last_irq_src: None,
            last_irq_pc: None,
            last_irq_vector: None,
            irq_bit_watch: None,
            delivered_masks: Vec::new(),
            instruction_start_cycle: 0,
            last_mti_fire_cycle: None,
            last_sti_fire_cycle: None,
            fired_mti_since_boundary: false,
            fired_sti_since_boundary: false,
            timer_scale: 1.0,
            preserve_phase: true,
            mti_short_period: mti_period.max(0) as u64,
            mti_long_period: mti_period.max(0) as u64,
            sti_short_period: sti_period.max(0) as u64,
            sti_long_period: sti_period.max(0) as u64,
            scr_selector: 0,
        };
        ctx.reset(0);
        ctx
    }

    pub fn set_timer_scale(&mut self, scale: f64) {
        self.timer_scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
    }

    pub fn set_preserve_phase(&mut self, preserve: bool) {
        self.preserve_phase = preserve;
    }

    pub fn set_instruction_start_cycle(&mut self, cycle: u64) {
        self.instruction_start_cycle = cycle;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn configure_scr_periods(
        &mut self,
        mti_short_period: u64,
        mti_long_period: u64,
        sti_short_period: u64,
        sti_long_period: u64,
        scr: u8,
        current_cycle: u64,
    ) {
        self.mti_short_period = mti_short_period;
        self.mti_long_period = mti_long_period;
        self.sti_short_period = sti_short_period;
        self.sti_long_period = sti_long_period;
        self.scr_selector = scr & (SCR_MTS | SCR_STS);
        self.mti_period = if scr & SCR_MTS != 0 {
            mti_long_period
        } else {
            mti_short_period
        };
        self.sti_period = if scr & SCR_STS != 0 {
            sti_long_period
        } else {
            sti_short_period
        };
        self.next_mti = if self.enabled && self.mti_period > 0 {
            current_cycle.wrapping_add(self.mti_period)
        } else {
            0
        };
        self.next_sti = if self.enabled && self.sti_period > 0 {
            current_cycle.wrapping_add(self.sti_period)
        } else {
            0
        };
    }

    /// Apply `SCR.MTS`/`SCR.STS` to the active periods.  The exact silicon
    /// divider phase when a selector changes is not captured, so a changed
    /// selection starts a fresh compatibility period at this boundary.
    pub fn sync_scr_selection(&mut self, scr: u8, current_cycle: u64) {
        let old_selector = self.scr_selector;
        let new_selector = scr & (SCR_MTS | SCR_STS);
        let mti_period = if scr & SCR_MTS != 0 {
            self.mti_long_period
        } else {
            self.mti_short_period
        };
        let sti_period = if scr & SCR_STS != 0 {
            self.sti_long_period
        } else {
            self.sti_short_period
        };
        if old_selector & SCR_MTS != new_selector & SCR_MTS {
            self.mti_period = mti_period;
            self.next_mti = if self.enabled && mti_period > 0 {
                current_cycle.wrapping_add(mti_period)
            } else {
                0
            };
        }
        if old_selector & SCR_STS != new_selector & SCR_STS {
            self.sti_period = sti_period;
            self.next_sti = if self.enabled && sti_period > 0 {
                current_cycle.wrapping_add(sti_period)
            } else {
                0
            };
        }
        self.scr_selector = new_selector;
    }

    /// Restore the selector latch without restarting either divider. Snapshot
    /// metadata already carries the exact active periods and deadlines.
    pub(crate) fn restore_scr_selector(&mut self, scr: u8) {
        self.scr_selector = scr & (SCR_MTS | SCR_STS);
    }

    pub fn tick_counts(&self, cycle_count: u64) -> (u64, u64) {
        let mti = Self::tick_count_for(cycle_count, self.mti_period, self.next_mti);
        let sti = Self::tick_count_for(cycle_count, self.sti_period, self.next_sti);
        (mti, sti)
    }

    fn tick_count_for(cycle_count: u64, period: u64, next: u64) -> u64 {
        if period == 0 {
            return 0;
        }
        let last = next.saturating_sub(period);
        let mut elapsed = cycle_count.saturating_sub(last);
        if elapsed >= period {
            elapsed %= period;
        }
        elapsed
    }

    /// Compare wrapping cycle counters using the conventional half-range
    /// ordering. Timer periods are far below half the u64 range, so this keeps
    /// a deadline that wrapped to zero in the future while the current cycle
    /// is still near `u64::MAX`.
    fn deadline_reached(cycle_count: u64, deadline: u64) -> bool {
        cycle_count.wrapping_sub(deadline) < (1_u64 << 63)
    }

    /// Return the next cycle in `(after_cycle, end_cycle]` on which at least
    /// one enabled timer can fire. Non-deadline cycles have no architectural
    /// timer or keyboard-scan effect and can therefore be skipped safely.
    pub fn next_fire_cycle_in_span(&self, after_cycle: u64, end_cycle: u64) -> Option<u64> {
        self.next_fire_cycle_in_span_selected(after_cycle, end_cycle, true, true)
    }

    pub fn next_fire_cycle_in_span_selected(
        &self,
        after_cycle: u64,
        end_cycle: u64,
        run_mti: bool,
        run_sti: bool,
    ) -> Option<u64> {
        if !self.enabled {
            return None;
        }
        let span = end_cycle.wrapping_sub(after_cycle);
        if span == 0 || span >= (1_u64 << 63) {
            return None;
        }
        let first_cycle = after_cycle.wrapping_add(1);
        let candidate = |period: u64, deadline: u64| {
            if period == 0 {
                return None;
            }
            let cycle = if Self::deadline_reached(first_cycle, deadline) {
                first_cycle
            } else {
                deadline
            };
            let distance = cycle.wrapping_sub(after_cycle);
            (distance <= span).then_some((distance, cycle))
        };
        let mti = run_mti
            .then(|| candidate(self.mti_period, self.next_mti))
            .flatten();
        let sti = run_sti
            .then(|| candidate(self.sti_period, self.next_sti))
            .flatten();
        match (mti, sti) {
            (Some(mti), Some(sti)) => Some(if mti.0 <= sti.0 { mti.1 } else { sti.1 }),
            (Some((_, cycle)), None) | (None, Some((_, cycle))) => Some(cycle),
            (None, None) => None,
        }
    }

    /// Keep the main-timer phase stationary while the system clock is stopped.
    pub fn defer_mti(&mut self, timing_units: u64) {
        if self.enabled && self.mti_period > 0 {
            self.next_mti = self.next_mti.wrapping_add(timing_units);
        }
    }

    /// Advance a due deadline to the first phase-aligned target after the
    /// current cycle in constant time. The u128 intermediate preserves the
    /// exact result before the u64 cycle clock wraps.
    fn advance_deadline(deadline: u64, cycle_count: u64, period: u64) -> u64 {
        debug_assert!(period > 0);
        let elapsed = u128::from(cycle_count.wrapping_sub(deadline));
        let period = u128::from(period);
        let intervals = elapsed / period + 1;
        (u128::from(deadline) + intervals * period) as u64
    }

    pub fn reset(&mut self, current_cycle: u64) {
        self.irq_pending = false;
        self.irq_source = None;
        self.irq_imr = 0;
        self.irq_isr = 0;
        self.fired_mti_since_boundary = false;
        self.fired_sti_since_boundary = false;
        self.instruction_start_cycle = current_cycle;
        self.last_mti_fire_cycle = None;
        self.last_sti_fire_cycle = None;
        self.next_mti = if self.enabled && self.mti_period > 0 {
            current_cycle.wrapping_add(self.mti_period)
        } else {
            0
        };
        self.next_sti = if self.enabled && self.sti_period > 0 {
            current_cycle.wrapping_add(self.sti_period)
        } else {
            0
        };
        self.in_interrupt = false;
        self.interrupt_stack.clear();
        self.delivered_masks.clear();
        self.next_interrupt_id = 0;
        self.key_irq_latched = false;
    }

    /// Full reset matching power-on behavior: clear pending/stack/bit-watch and counters.
    pub fn reset_full(&mut self, current_cycle: u64) {
        self.reset(current_cycle);
        self.in_interrupt = false;
        self.delivered_masks.clear();
        self.next_interrupt_id = 0;
        self.last_fired = None;
        self.key_irq_latched = false;
        self.irq_total = 0;
        self.irq_key = 0;
        self.irq_mti = 0;
        self.irq_sti = 0;
        self.last_irq_src = None;
        self.last_irq_pc = None;
        self.last_irq_vector = None;
        self.irq_bit_watch = None;
        self.fired_mti_since_boundary = false;
        self.fired_sti_since_boundary = false;
        self.last_mti_fire_cycle = None;
        self.last_sti_fire_cycle = None;
    }

    /// Clear pending/active interrupt bookkeeping after a RESET intrinsic so mirrors reflect
    /// the cleared IMEM state (ISR/SCR/UCR/USR/SSR) instead of stale latched values.
    pub fn clear_pending_for_reset(&mut self) {
        self.irq_pending = false;
        self.in_interrupt = false;
        self.irq_source = None;
        self.last_fired = None;
        self.fired_mti_since_boundary = false;
        self.fired_sti_since_boundary = false;
        self.last_mti_fire_cycle = None;
        self.last_sti_fire_cycle = None;
        self.key_irq_latched = false;
        self.delivered_masks.clear();
        self.interrupt_stack.clear();
        self.next_interrupt_id = 0;
        self.last_irq_src = None;
        self.last_irq_pc = None;
        self.last_irq_vector = None;
    }

    pub fn snapshot_info(&self) -> (TimerInfo, InterruptInfo) {
        let timer = TimerInfo {
            enabled: self.enabled,
            mti_period: self.mti_period,
            sti_period: self.sti_period,
            next_mti: self.next_mti,
            next_sti: self.next_sti,
            kb_irq_enabled: self.kb_irq_enabled,
            instruction_start_cycle: self.instruction_start_cycle,
            last_mti_fire_cycle: self.last_mti_fire_cycle,
            last_sti_fire_cycle: self.last_sti_fire_cycle,
            fired_mti_since_boundary: self.fired_mti_since_boundary,
            fired_sti_since_boundary: self.fired_sti_since_boundary,
            preserve_phase: self.preserve_phase,
        };
        let mut watch = self
            .irq_bit_watch
            .clone()
            .unwrap_or_else(default_bit_watch_table);
        normalize_bit_watch(&mut watch);
        let interrupts = InterruptInfo {
            pending: self.irq_pending,
            in_interrupt: self.in_interrupt,
            key_irq_latched: self.key_irq_latched,
            source: self.irq_source.clone(),
            last_fired: self.last_fired.clone(),
            stack: self.interrupt_stack.clone(),
            next_id: self.next_interrupt_id,
            imr: self.irq_imr,
            isr: self.irq_isr,
            irq_counts: Some(json!({
                "total": self.irq_total,
                "KEY": self.irq_key,
                "MTI": self.irq_mti,
                "STI": self.irq_sti,
            })),
            last_irq: Some(json!({
                "src": self.last_irq_src,
                "pc": self.last_irq_pc,
                "vector": self.last_irq_vector,
            })),
            irq_bit_watch: Some(json!(watch)),
            delivered_masks: self.delivered_masks.clone(),
        };
        (timer, interrupts)
    }

    pub fn apply_snapshot_info(
        &mut self,
        timer: &TimerInfo,
        interrupts: &InterruptInfo,
        _current_cycle: u64,
    ) {
        self.enabled = timer.enabled;
        self.mti_period = timer.mti_period;
        self.sti_period = timer.sti_period;
        self.next_mti = timer.next_mti;
        self.next_sti = timer.next_sti;
        // Python stores absolute targets; do not rebase forward. Allow immediate fire if targets are in the past.
        self.kb_irq_enabled = timer.kb_irq_enabled;
        self.instruction_start_cycle = timer.instruction_start_cycle;
        self.last_mti_fire_cycle = timer.last_mti_fire_cycle;
        self.last_sti_fire_cycle = timer.last_sti_fire_cycle;
        self.fired_mti_since_boundary = timer.fired_mti_since_boundary;
        self.fired_sti_since_boundary = timer.fired_sti_since_boundary;
        self.preserve_phase = timer.preserve_phase;

        self.irq_pending = interrupts.pending;
        self.in_interrupt = interrupts.in_interrupt;
        self.key_irq_latched = interrupts.key_irq_latched;
        self.irq_source = interrupts.source.clone();
        self.interrupt_stack = interrupts.stack.clone();
        self.next_interrupt_id = interrupts.next_id;
        self.irq_imr = interrupts.imr;
        self.irq_isr = interrupts.isr;
        self.irq_bit_watch = interrupts
            .irq_bit_watch
            .as_ref()
            .and_then(|value| value.as_object())
            .cloned();
        self.delivered_masks = interrupts.delivered_masks.clone();
        self.last_fired = interrupts.last_fired.clone();
        // Restore IRQ counters/last info if present; otherwise zero them.
        self.irq_total = 0;
        self.irq_key = 0;
        self.irq_mti = 0;
        self.irq_sti = 0;
        self.last_irq_src = None;
        self.last_irq_pc = None;
        self.last_irq_vector = None;
        if let Some(counts) = interrupts.irq_counts.as_ref() {
            self.irq_total = counts.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            self.irq_key = counts.get("KEY").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            self.irq_mti = counts.get("MTI").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            self.irq_sti = counts.get("STI").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        }
        if let Some(last) = interrupts.last_irq.as_ref().and_then(|v| v.as_object()) {
            self.last_irq_src = last
                .get("src")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            self.last_irq_pc = last.get("pc").and_then(|v| v.as_u64()).map(|v| v as u32);
            self.last_irq_vector = last
                .get("vector")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_interrupt_state(
        &mut self,
        pending: bool,
        imr: u8,
        isr: u8,
        next_mti: i32,
        next_sti: i32,
        source: Option<String>,
        in_interrupt: bool,
        interrupt_stack: Option<Vec<u32>>,
        next_interrupt_id: u32,
        irq_bit_watch: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.irq_pending = pending;
        self.irq_source = source;
        self.irq_imr = imr;
        self.irq_isr = isr;
        self.next_mti = next_mti.max(0) as u64;
        self.next_sti = next_sti.max(0) as u64;
        self.in_interrupt = in_interrupt;
        self.interrupt_stack = interrupt_stack.unwrap_or_default();
        self.next_interrupt_id = next_interrupt_id;
        self.last_fired = None;
        self.key_irq_latched = false;
        self.irq_bit_watch = irq_bit_watch;
        self.delivered_masks.clear();
    }

    /// Record IMR/ISR bit transitions (set/clear) keyed by bit number and PC, mirroring Python.
    pub fn record_bit_watch_transition(
        &mut self,
        reg_name: &str,
        prev_val: u8,
        new_val: u8,
        pc: u32,
    ) {
        let table = self
            .irq_bit_watch
            .get_or_insert_with(default_bit_watch_table);
        normalize_bit_watch(table);
        let Some(reg_entry) = table.get_mut(reg_name) else {
            return;
        };
        let Some(reg_obj) = reg_entry.as_object_mut() else {
            return;
        };
        if prev_val == new_val {
            return;
        }
        for bit in 0..8u8 {
            let prev = (prev_val >> bit) & 1;
            let new = (new_val >> bit) & 1;
            if prev == new {
                continue;
            }
            let key = bit.to_string();
            let Some(bit_entry) = reg_obj.get_mut(&key) else {
                continue;
            };
            let Some(bit_obj) = bit_entry.as_object_mut() else {
                continue;
            };
            let action = if new == 1 { "set" } else { "clear" };
            if !bit_obj.contains_key(action) {
                bit_obj.insert(action.to_string(), json!([]));
            }
            if bit_obj.get(action).and_then(|v| v.as_array()).is_none() {
                bit_obj.insert(action.to_string(), json!([]));
            }
            let arr = bit_obj
                .get_mut(action)
                .and_then(|v| v.as_array_mut())
                .expect("array exists");
            if arr.last().and_then(|v| v.as_u64()).map(|v| v as u32) == Some(pc) {
                continue;
            }
            arr.push(json!(pc));
            if arr.len() > 10 {
                arr.remove(0);
            }
        }
    }

    pub fn set_keyboard_irq_enabled(&mut self, enabled: bool) {
        self.kb_irq_enabled = enabled;
    }

    pub fn keyboard_irq_enabled(&self) -> bool {
        self.kb_irq_enabled
    }

    pub fn tick_timers(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        pc_hint: Option<u32>,
    ) -> (bool, bool) {
        self.tick_timers_selected(memory, cycle_count, pc_hint, true, true)
    }

    pub fn tick_timers_selected(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        pc_hint: Option<u32>,
        run_mti: bool,
        run_sti: bool,
    ) -> (bool, bool) {
        if !self.enabled {
            return (false, false);
        }

        let mut fired_mti = false;
        let mut fired_sti = false;

        if run_mti && self.mti_period > 0 && Self::deadline_reached(cycle_count, self.next_mti) {
            fired_mti = true;
            if self.preserve_phase {
                self.next_mti = Self::advance_deadline(self.next_mti, cycle_count, self.mti_period);
            } else {
                self.next_mti = cycle_count.wrapping_add(self.mti_period);
            }
            self.last_mti_fire_cycle = Some(cycle_count);
        }
        if run_sti && self.sti_period > 0 && Self::deadline_reached(cycle_count, self.next_sti) {
            fired_sti = true;
            if self.preserve_phase {
                self.next_sti = Self::advance_deadline(self.next_sti, cycle_count, self.sti_period);
            } else {
                self.next_sti = cycle_count.wrapping_add(self.sti_period);
            }
            self.last_sti_fire_cycle = Some(cycle_count);
        }

        if fired_mti || fired_sti {
            if fired_mti {
                self.fired_mti_since_boundary = true;
            }
            if fired_sti {
                self.fired_sti_since_boundary = true;
            }
            if let Some(current_isr) = memory.read_internal_byte(ISR_OFFSET) {
                let mut new_isr = current_isr;
                if fired_mti {
                    new_isr |= 0x01;
                }
                if fired_sti {
                    new_isr |= 0x02;
                }
                if new_isr != current_isr {
                    memory.write_internal_byte(ISR_OFFSET, new_isr);
                    let pc_trace = crate::llama::eval::perfetto_instr_context()
                        .map(|(_, pc)| pc)
                        .or(pc_hint)
                        .unwrap_or_else(perfetto_last_pc);
                    self.record_bit_watch_transition("ISR", current_isr, new_isr, pc_trace);
                }
            }
            // Match Python: when both fire, the later source wins (STI overwrites MTI).
            if fired_mti {
                self.irq_source = Some("MTI".to_string());
                self.last_fired = self.irq_source.clone();
            }
            if fired_sti {
                self.irq_source = Some("STI".to_string());
                self.last_fired = self.irq_source.clone();
            }
            // Keep mirror fields in sync with the actual IMEM values for snapshots/tracing parity.
            self.irq_imr = memory.read_internal_byte(0xFB).unwrap_or(self.irq_imr);
            self.irq_isr = memory
                .read_internal_byte(ISR_OFFSET)
                .unwrap_or(self.irq_isr);
            // Parity: mark IRQ pending whenever a timer fires, regardless of IMR gating.
            // IMR masking is honored later during delivery.
            self.irq_pending = fired_mti || fired_sti;
            if self.irq_pending {
                self.emit_irq_trace(fired_mti, fired_sti, cycle_count, memory, pc_hint);
            }
            // Record IMR/ISR transitions for parity bit-watch metadata.
        }
        (fired_mti, fired_sti)
    }

    pub fn finalize_instruction_with_clamp(&mut self, cycle_count: u64, clamp: bool) {
        if self.fired_mti_since_boundary {
            if clamp && self.enabled && self.mti_period > 0 {
                let fire_cycle = self.last_mti_fire_cycle.unwrap_or(cycle_count);
                if fire_cycle < cycle_count {
                    self.next_mti = cycle_count.wrapping_add(self.mti_period);
                }
            }
            self.fired_mti_since_boundary = false;
            self.last_mti_fire_cycle = None;
        }
        if self.fired_sti_since_boundary {
            if clamp && self.enabled && self.sti_period > 0 {
                let fire_cycle = self.last_sti_fire_cycle.unwrap_or(cycle_count);
                if fire_cycle < cycle_count {
                    self.next_sti = cycle_count.wrapping_add(self.sti_period);
                }
            }
            self.fired_sti_since_boundary = false;
            self.last_sti_fire_cycle = None;
        }
    }

    pub fn finalize_instruction(&mut self, cycle_count: u64) {
        self.finalize_instruction_with_clamp(cycle_count, true);
    }

    /// Tick timers and optionally run a keyboard scan when MTI fires, mirroring Python's _tick_timers.
    /// Returns (mti, sti, key_events).
    pub fn tick_timers_with_keyboard<F>(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        keyboard_scan: F,
        _y_reg: Option<u32>,
        pc_hint: Option<u32>,
    ) -> (bool, bool, usize, Option<KeyboardTelemetry>)
    where
        F: FnMut(&mut MemoryImage) -> (usize, bool, Option<KeyboardTelemetry>),
    {
        self.tick_timers_with_keyboard_selected(
            memory,
            cycle_count,
            keyboard_scan,
            _y_reg,
            pc_hint,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick_timers_with_keyboard_selected<F>(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        mut keyboard_scan: F,
        _y_reg: Option<u32>,
        pc_hint: Option<u32>,
        run_mti: bool,
        run_sti: bool,
    ) -> (bool, bool, usize, Option<KeyboardTelemetry>)
    where
        F: FnMut(&mut MemoryImage) -> (usize, bool, Option<KeyboardTelemetry>),
    {
        let (mti, sti) = self.tick_timers_selected(memory, cycle_count, pc_hint, run_mti, run_sti);
        let mut key_events = 0usize;
        // Preserve any existing latch even if keyboard IRQs are disabled; Python keeps KEYI latched
        // across IRQ gating so firmware can re-enable later without losing events.
        let had_latch = self.key_irq_latched;
        let mut kb_stats: Option<KeyboardTelemetry> = None;
        if mti {
            let (events, _has_data, stats) = keyboard_scan(memory);
            kb_stats = stats;
            key_events = events;
        }
        let pc_trace = crate::llama::eval::perfetto_instr_context()
            .map(|(_, pc)| pc)
            .or(pc_hint)
            .unwrap_or_else(perfetto_last_pc);
        // Preserve the host-event latch for snapshot/FIFO bookkeeping only.
        // It is not an electrical KEYI source; the runtimes sample the raw,
        // selected KIL matrix independently at scheduling boundaries.
        let new_latch = key_events > 0;
        let latch_active = had_latch || new_latch;
        if key_events == 0 {
            if let Some(stats) = kb_stats.as_ref() {
                if stats.pressed > 0 {
                    let mut guard = PERFETTO_TRACER.enter();
                    guard.with_some(|tracer| {
                        let mut payload = HashMap::new();
                        payload.insert(
                            "pressed".to_string(),
                            AnnotationValue::UInt(stats.pressed as u64),
                        );
                        payload.insert(
                            "strobe_count".to_string(),
                            AnnotationValue::UInt(stats.strobe_count as u64),
                        );
                        payload.insert("kol".to_string(), AnnotationValue::UInt(stats.kol as u64));
                        payload.insert("koh".to_string(), AnnotationValue::UInt(stats.koh as u64));
                        payload.insert(
                            "active_cols".to_string(),
                            AnnotationValue::Str(format!("{:?}", stats.active_columns)),
                        );
                        payload.insert("pc".to_string(), AnnotationValue::Pointer(pc_trace as u64));
                        payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
                        tracer.record_irq_event("KeyScanEmpty", payload);
                    });
                }
            }
        }
        // Track only whether host events remain represented in the FIFO.
        self.key_irq_latched = latch_active;
        // When IRQs are disabled, keep the existing latch state but avoid creating a new one.
        if mti {
            let mut guard = PERFETTO_TRACER.enter();
            guard.with_some(|tracer| {
                let mut payload = HashMap::new();
                payload.insert(
                    "events".to_string(),
                    AnnotationValue::UInt(key_events as u64),
                );
                payload.insert(
                    "imr".to_string(),
                    AnnotationValue::UInt(self.irq_imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    AnnotationValue::UInt(self.irq_isr as u64),
                );
                payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
                if let Some(stats) = kb_stats.as_ref() {
                    payload.insert(
                        "pressed".to_string(),
                        AnnotationValue::UInt(stats.pressed as u64),
                    );
                    payload.insert(
                        "strobe_count".to_string(),
                        AnnotationValue::UInt(stats.strobe_count as u64),
                    );
                    payload.insert("kol".to_string(), AnnotationValue::UInt(stats.kol as u64));
                    payload.insert("koh".to_string(), AnnotationValue::UInt(stats.koh as u64));
                    payload.insert(
                        "active_cols".to_string(),
                        AnnotationValue::Str(format!("{:?}", stats.active_columns)),
                    );
                }
                payload.insert("pc".to_string(), AnnotationValue::Pointer(pc_trace as u64));
                tracer.record_irq_event("KeyScanEvent", payload);
            });
        }
        (mti, sti, key_events, kb_stats)
    }

    pub fn drain_pending_irq(&mut self) -> Option<String> {
        if !self.irq_pending {
            return None;
        }
        self.irq_pending = false;
        self.irq_source.take()
    }

    fn emit_irq_trace(
        &self,
        _fired_mti: bool,
        _fired_sti: bool,
        cycle_count: u64,
        _memory: &MemoryImage,
        pc_hint: Option<u32>,
    ) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let mut payload = std::collections::HashMap::new();
            payload.insert(
                "isr".to_string(),
                AnnotationValue::UInt(self.irq_isr as u64),
            );
            payload.insert(
                "imr".to_string(),
                AnnotationValue::UInt(self.irq_imr as u64),
            );
            payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
            if let Some(src) = self.irq_source.as_deref() {
                payload.insert("src".to_string(), AnnotationValue::Str(src.to_string()));
            }
            if let Some((op_idx, pc)) = crate::llama::eval::perfetto_instr_context() {
                payload.insert("op_index".to_string(), AnnotationValue::UInt(op_idx));
                payload.insert("pc".to_string(), AnnotationValue::Pointer(pc as u64));
            } else if let Some(pc) = pc_hint {
                payload.insert("pc".to_string(), AnnotationValue::Pointer(pc as u64));
            }
            // Align event naming with Python tracer ("TimerFired").
            tracer.record_irq_event("TimerFired", payload);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scr_selects_documented_main_and_sub_timer_periods() {
        let mut timer = TimerContext::new(true, 4, 500);
        timer.configure_scr_periods(4, 16, 500, 2_000, 0, 100);
        assert_eq!((timer.mti_period, timer.sti_period), (4, 500));
        assert_eq!((timer.next_mti, timer.next_sti), (104, 600));

        timer.sync_scr_selection(SCR_MTS | SCR_STS, 103);
        assert_eq!((timer.mti_period, timer.sti_period), (16, 2_000));
        assert_eq!((timer.next_mti, timer.next_sti), (119, 2_103));

        timer.sync_scr_selection(SCR_MTS | SCR_STS, 110);
        assert_eq!(
            (timer.next_mti, timer.next_sti),
            (119, 2_103),
            "an unchanged selector must not restart either divider"
        );
    }

    #[test]
    fn snapshot_selector_restore_preserves_exact_deadlines() {
        let mut timer = TimerContext::new(true, 4, 500);
        timer.configure_scr_periods(4, 16, 500, 2_000, SCR_MTS | SCR_STS, 100);
        timer.next_mti = 777;
        timer.next_sti = 888;
        let (timer_info, interrupt_info) = timer.snapshot_info();

        let mut restored = TimerContext::new(true, 4, 500);
        restored.configure_scr_periods(4, 16, 500, 2_000, 0, 0);
        restored.apply_snapshot_info(&timer_info, &interrupt_info, 100);
        restored.restore_scr_selector(SCR_MTS | SCR_STS);
        restored.sync_scr_selection(SCR_MTS | SCR_STS, 200);

        assert_eq!((restored.mti_period, restored.sti_period), (16, 2_000));
        assert_eq!((restored.next_mti, restored.next_sti), (777, 888));
    }

    #[test]
    fn timers_use_absolute_targets() {
        let mut timer = TimerContext::new(true, 10, 0);
        let mut mem = MemoryImage::new();
        timer.next_mti = 50;
        timer.next_sti = 0;
        let mut cycles = 0u64;
        // Run up to but not including the target
        for _ in 0..49 {
            timer.tick_timers(&mut mem, cycles, None);
            cycles += 1;
            assert!(!timer.irq_pending);
        }
        assert_eq!(cycles, 49);
        // The 50th tick should fire MTI and roll the target forward.
        cycles = 50;
        timer.tick_timers(&mut mem, cycles, None);
        assert!(timer.irq_pending);
        assert!(timer.next_mti > 50);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01);
    }

    #[test]
    fn deadline_span_iterator_visits_only_real_fire_cycles() {
        let mut timer = TimerContext::new(true, 5, 7);
        let mut memory = MemoryImage::new();
        let mut cursor = 0;
        let end = 20;
        let mut fire_cycles = Vec::new();

        while let Some(cycle) = timer.next_fire_cycle_in_span(cursor, end) {
            fire_cycles.push(cycle);
            timer.tick_timers(&mut memory, cycle, None);
            cursor = cycle;
        }

        assert_eq!(fire_cycles, vec![5, 7, 10, 14, 15, 20]);
        assert_eq!(timer.next_mti, 25);
        assert_eq!(timer.next_sti, 21);
    }

    #[test]
    fn deadline_span_iterator_handles_stale_and_wrapped_targets() {
        let mut stale = TimerContext::new(true, 10, 0);
        stale.next_mti = 3;
        assert_eq!(stale.next_fire_cycle_in_span(8, 20), Some(9));

        let mut wrapped = TimerContext::new(true, 4, 0);
        wrapped.next_mti = 1;
        assert_eq!(wrapped.next_fire_cycle_in_span(u64::MAX - 2, 3), Some(1));

        let disabled = TimerContext::new(false, 1, 1);
        assert_eq!(disabled.next_fire_cycle_in_span(0, 100), None);
    }

    #[test]
    fn tick_counts_track_timer_phase() {
        let mut timer = TimerContext::new(true, 10, 20);
        timer.mti_period = 10;
        timer.sti_period = 20;
        timer.next_mti = 40;
        timer.next_sti = 50;

        let (mti, sti) = timer.tick_counts(35);
        assert_eq!(mti, 5);
        assert_eq!(sti, 5);

        let (mti2, sti2) = timer.tick_counts(65);
        assert_eq!(mti2, 5);
        assert_eq!(sti2, 15);
    }

    #[test]
    fn finalize_instruction_discards_remainder() {
        let mut timer = TimerContext::new(true, 10, 0);
        let mut mem = MemoryImage::new();
        timer.reset(0);

        for cycle in 1..=12u64 {
            timer.tick_timers(&mut mem, cycle, None);
        }
        assert!(timer.irq_pending, "timer should have fired by cycle 10");
        assert_eq!(timer.next_mti, 20, "next_mti should track fire cycle");

        timer.finalize_instruction(12);
        assert_eq!(
            timer.next_mti, 22,
            "finalize should drop in-instruction remainder"
        );
        let (mti, _) = timer.tick_counts(12);
        assert_eq!(mti, 0, "ticks reset to 0 at boundary");
    }

    #[test]
    fn finalize_instruction_preserves_remainder_when_unclamped() {
        let mut timer = TimerContext::new(true, 10, 0);
        let mut mem = MemoryImage::new();
        timer.reset(0);

        for cycle in 1..=12u64 {
            timer.tick_timers(&mut mem, cycle, None);
        }
        assert!(timer.irq_pending, "timer should have fired by cycle 10");
        assert_eq!(timer.next_mti, 20, "next_mti should track fire cycle");

        timer.finalize_instruction_with_clamp(12, false);
        assert_eq!(
            timer.next_mti, 20,
            "next_mti should preserve in-instruction remainder"
        );
        let (mti, _) = timer.tick_counts(12);
        assert_eq!(mti, 2, "ticks preserve remainder when unclamped");
    }

    #[test]
    fn snapshot_absolute_targets_match_python_semantics() {
        // Simulate a Python snapshot with absolute next_mti/next_sti values.
        let mut timer = TimerContext::new(true, 20, 30);
        timer.apply_snapshot_info(
            &crate::TimerInfo {
                enabled: true,
                mti_period: 20,
                sti_period: 30,
                next_mti: 150,
                next_sti: 200,
                kb_irq_enabled: true,
                ..Default::default()
            },
            &InterruptInfo::default(),
            100,
        );
        let mut mem = MemoryImage::new();
        let mut cycles = 100u64; // current cycle when snapshot applied

        // Advance to just before first fire
        while cycles < 149 {
            timer.tick_timers(&mut mem, cycles, None);
            cycles += 1;
            assert!(!timer.irq_pending);
        }
        // Fire MTI at cycle 150, then next target moves to 170.
        cycles = 150;
        timer.tick_timers(&mut mem, cycles, None);
        assert!(timer.irq_pending);
        assert!(timer.next_mti > 150);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01);
    }

    #[test]
    fn timer_catch_up_advances_far_stale_targets_in_constant_time() {
        let mut timer = TimerContext::new(true, 1, 7);
        timer.next_mti = 0;
        timer.next_sti = 3;
        let mut mem = MemoryImage::new();

        let (mti, sti) = timer.tick_timers(&mut mem, 1_000_000_000, None);

        assert!(mti);
        assert!(sti);
        assert_eq!(timer.next_mti, 1_000_000_001);
        assert_eq!(timer.next_sti, 1_000_000_004);
    }

    #[test]
    fn timer_deadline_wraps_at_u64_max_without_refiring_same_cycle() {
        let mut timer = TimerContext::new(true, 1, 0);
        timer.next_mti = u64::MAX;
        let mut mem = MemoryImage::new();

        let (first, _) = timer.tick_timers(&mut mem, u64::MAX, None);
        assert!(first);
        assert_eq!(timer.next_mti, 0, "next phase wraps with the cycle clock");

        let (same_cycle, _) = timer.tick_timers(&mut mem, u64::MAX, None);
        assert!(
            !same_cycle,
            "the wrapped deadline is in the future until the cycle clock wraps"
        );

        let (after_wrap, _) = timer.tick_timers(&mut mem, 0, None);
        assert!(after_wrap);
        assert_eq!(timer.next_mti, 1);
    }

    #[test]
    fn apply_snapshot_does_not_silently_normalize_invalid_metadata() {
        let mut timer = TimerContext::new(true, 20, 30);
        let interrupts = InterruptInfo {
            pending: true,
            source: Some("KEY".to_string()),
            isr: 0,
            ..Default::default()
        };

        timer.apply_snapshot_info(
            &crate::TimerInfo {
                enabled: true,
                mti_period: 20,
                sti_period: 30,
                next_mti: 150,
                next_sti: 200,
                kb_irq_enabled: true,
                ..Default::default()
            },
            &interrupts,
            100,
        );

        assert!(timer.irq_pending);
        assert_eq!(timer.irq_source.as_deref(), Some("KEY"));
    }

    #[test]
    fn tick_timers_updates_irq_mirrors() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // Preload IMR so mirror should reflect it.
        mem.write_internal_byte(0xFB, 0xAA);
        // First tick should fire MTI and update ISR/IMR mirrors.
        timer.tick_timers(&mut mem, 1, None);
        assert_eq!(timer.irq_imr, 0xAA);
        assert_eq!(timer.irq_isr & 0x01, 0x01);
    }

    #[test]
    fn tick_timers_sets_pending_even_when_imr_masked() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // IMR master cleared -> still pend like Python; delivery will gate later.
        mem.write_internal_byte(0xFB, 0x00);
        let (_mti, _sti) = timer.tick_timers(&mut mem, 1, None);
        assert!(
            timer.irq_pending,
            "irq_pending should set even when IMR master=0"
        );

        // Enable master but mask out MTI bit -> still pend; gating happens during delivery.
        mem.write_internal_byte(0xFB, 0x80);
        timer.tick_timers(&mut mem, 2, None);
        assert!(
            timer.irq_pending,
            "irq_pending should set even when MTI masked"
        );

        // Enable MTI bit -> should pend on next fire.
        mem.write_internal_byte(0xFB, 0x81);
        timer.tick_timers(&mut mem, 3, None);
        assert!(
            timer.irq_pending,
            "irq_pending should set when master+MTI enabled"
        );
    }

    #[test]
    fn tick_timers_increments_counters_on_fire() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        timer.tick_timers(&mut mem, 1, None);
        assert_eq!(timer.irq_total, 0, "counters should advance on delivery");
        assert_eq!(timer.irq_mti, 0);
        assert_eq!(timer.last_irq_src, None);
    }

    #[test]
    fn key_latch_increments_counters() {
        let mut timer = TimerContext::new(true, 1, 0);
        timer.next_mti = 0;
        let mut mem = MemoryImage::new();
        // Force latch_active path by preloading FIFO state via keyboard scan closure.
        let (_mti, _sti, events, _stats) =
            timer.tick_timers_with_keyboard(&mut mem, 0, |_mem| (1, true, None), None, None);
        assert_eq!(events, 1, "keyboard scan should run on MTI fire");
        // Counters should only increment on delivery; latch alone must not bump them.
        assert_eq!(timer.irq_total, 0);
        assert_eq!(timer.irq_key, 0);
        assert_eq!(timer.last_irq_src, None);
    }

    #[test]
    fn keyboard_scan_runs_even_when_irq_disabled() {
        let mut timer = TimerContext::new(true, 1, 0);
        timer.set_keyboard_irq_enabled(false);
        let mut mem = MemoryImage::new();
        // Force MTI to fire on first tick.
        timer.next_mti = 0;
        let mut scanned = false;
        let (mti, _sti, key_events, _stats) = timer.tick_timers_with_keyboard(
            &mut mem,
            1,
            |_mem| {
                scanned = true;
                // Simulate one key event and non-empty FIFO.
                (1, true, None)
            },
            None,
            None,
        );
        assert!(mti, "MTI should fire");
        assert!(scanned, "keyboard_scan should run even when IRQ disabled");
        assert_eq!(key_events, 1);
        // KEYI should not be asserted when kb_irq_enabled is false.
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0);
        // Timer fires should still mark irq_pending, but KEYI must stay clear when disabled.
        assert!(timer.irq_pending, "timer fire should still pend an IRQ");
    }

    #[test]
    fn bit_watch_tracks_mti_isr_transition() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // First fire at cycle 1 should set ISR bit 0 and record a transition.
        let pc = 0x123u32;
        timer.tick_timers(&mut mem, 1, Some(pc));
        let watch = timer
            .irq_bit_watch
            .as_ref()
            .and_then(|w| w.get("ISR"))
            .and_then(|v| v.as_object())
            .expect("bit watch table should capture ISR");
        let bit0 = watch
            .get("0")
            .and_then(|v| v.as_object())
            .expect("bit 0 entry should exist");
        let set = bit0
            .get("set")
            .and_then(|v| v.as_array())
            .expect("'set' array should exist for bit 0");
        assert!(
            set.iter().any(|entry| entry.as_u64() == Some(pc as u64)),
            "bit watch should record PC for MTI ISR set"
        );
    }

    #[test]
    fn host_event_latch_does_not_emit_raw_keyi_bit_watch() {
        let mut timer = TimerContext::new(true, 0, 0);
        let mut mem = MemoryImage::new();
        // Simulate ISR initially cleared.
        mem.write_internal_byte(ISR_OFFSET, 0x00);
        // Force KEYI assertion via keyboard path.
        timer.key_irq_latched = true;
        let pc = 0x234u32;
        let _ =
            timer.tick_timers_with_keyboard(&mut mem, 0, |_mem| (0, true, None), None, Some(pc));
        assert_eq!(mem.read_internal_byte(ISR_OFFSET).unwrap_or(0) & 0x04, 0);
        assert!(timer.irq_bit_watch.as_ref().is_none_or(|watch| {
            watch
                .get("ISR")
                .and_then(|value| value.as_object())
                .and_then(|bits| bits.get("2"))
                .is_none()
        }));
    }

    #[test]
    fn tick_timers_with_keyboard_keeps_events_separate_from_raw_keyi() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // Simulate keyboard scan emitting one event.
        let (_, _, events, _) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (1, true, None), None, None);
        assert_eq!(events, 1);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0);
        assert_eq!(isr & 0x01, 0x01, "the independent MTI still fires");
        assert_eq!(timer.irq_source, Some("MTI".to_string()));
    }

    #[test]
    fn key_latch_requires_new_events() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        timer.key_irq_latched = false;
        // MTI fires, but keyboard reports only buffered data and no new events.
        let (_mti, _sti, events, _stats) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (0, true, None), None, None);
        assert_eq!(events, 0);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0, "KEYI should not assert without new events");
        assert!(
            !timer.key_irq_latched,
            "latch should remain clear without fresh events"
        );
    }

    #[test]
    fn host_event_latch_sets_without_asserting_raw_keyi() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        timer.key_irq_latched = false;
        let (_mti, _sti, events, _stats) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (1, false, None), None, None);
        assert_eq!(events, 1);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0);
        assert!(timer.key_irq_latched, "latch should set on new events");
        assert_eq!(timer.irq_source, Some("MTI".to_string()));
    }

    #[test]
    fn buffered_key_events_do_not_manufacture_raw_keyi_on_enable() {
        let mut timer = TimerContext::new(true, 1, 0);
        timer.set_keyboard_irq_enabled(false);
        let mut mem = MemoryImage::new();
        let mut kb = crate::keyboard::KeyboardMatrix::new();
        // Inject a debounced event while IRQs are masked.
        let injected = kb.inject_matrix_event(0x10, false, &mut mem, false);
        assert_eq!(injected, 1);
        let snap = kb.snapshot_state();
        assert_eq!(
            snap.irq_count, 1,
            "irq_count should increment even when IRQs are masked"
        );
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0, "KEYI should stay clear while IRQs disabled");

        // Re-enabling host event generation does not turn buffered FIFO data
        // into an electrical raw-matrix level.
        timer.set_keyboard_irq_enabled(true);
        let (_mti2, _sti2, events2, _stats2) = timer.tick_timers_with_keyboard(
            &mut mem,
            2,
            |mem| {
                // No new events, but FIFO holds data; mirror to memory now that IRQs are enabled.
                let ev = kb.scan_tick(mem, true);
                let fifo_pending = kb.fifo_len() > 0;
                if ev > 0 || fifo_pending {
                    kb.write_fifo_to_memory(mem, true);
                }
                (ev, ev > 0 || fifo_pending, Some(kb.telemetry()))
            },
            None,
            None,
        );
        assert_eq!(
            events2, 0,
            "no additional events are needed to surface buffered data"
        );
        let isr_after = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr_after & 0x04, 0);
    }

    #[test]
    fn tick_timers_with_keyboard_does_not_reassert_keyi_from_fifo_latch() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        timer.key_irq_latched = true;
        // No new physical sample is available here; an event/FIFO latch is
        // insufficient to assert raw KEYI.
        let (_mti, _sti, events, _) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (0, true, None), None, None);
        assert_eq!(events, 0);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0);
    }

    #[test]
    fn apply_snapshot_restores_timer_periods_without_scaling() {
        let mut timer = TimerContext::new(true, 100, 200);
        timer.set_timer_scale(0.5);
        timer.apply_snapshot_info(
            &crate::TimerInfo {
                enabled: true,
                mti_period: 100,
                sti_period: 200,
                next_mti: 75,
                next_sti: 125,
                kb_irq_enabled: true,
                ..Default::default()
            },
            &InterruptInfo::default(),
            0,
        );

        assert_eq!(timer.mti_period, 100);
        assert_eq!(timer.sti_period, 200);
    }

    #[test]
    fn snapshot_roundtrip_preserves_key_latch_and_timer_phase_exactly() {
        let mut timer = TimerContext::new(true, 100, 200);
        timer.next_mti = 1_000;
        timer.next_sti = 2_000;
        timer.instruction_start_cycle = 777;
        timer.last_mti_fire_cycle = Some(790);
        timer.last_sti_fire_cycle = Some(791);
        timer.fired_mti_since_boundary = true;
        timer.fired_sti_since_boundary = true;
        timer.preserve_phase = false;
        timer.key_irq_latched = true;
        timer.irq_pending = true;
        timer.irq_source = Some("KEY".to_string());
        timer.irq_isr = 0x04;

        let (timer_info, interrupt_info) = timer.snapshot_info();
        let mut restored = TimerContext::new(false, 1, 1);
        restored.apply_snapshot_info(&timer_info, &interrupt_info, 0);

        assert_eq!(restored.next_mti, 1_000);
        assert_eq!(restored.next_sti, 2_000);
        assert_eq!(restored.instruction_start_cycle, 777);
        assert_eq!(restored.last_mti_fire_cycle, Some(790));
        assert_eq!(restored.last_sti_fire_cycle, Some(791));
        assert!(restored.fired_mti_since_boundary);
        assert!(restored.fired_sti_since_boundary);
        assert!(!restored.preserve_phase);
        assert!(restored.key_irq_latched);
        assert!(restored.irq_pending);
        assert_eq!(restored.irq_source.as_deref(), Some("KEY"));
    }
}
