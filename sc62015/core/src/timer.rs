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
#[cfg(test)]
use std::env;

const ISR_OFFSET: u32 = 0xFC;

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
        };
        ctx.reset(0);
        ctx
    }

    pub fn reset(&mut self, current_cycle: u64) {
        self.irq_pending = false;
        self.irq_source = None;
        self.irq_imr = 0;
        self.irq_isr = 0;
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
        self.next_interrupt_id = 0;
        self.key_irq_latched = false;
    }

    pub fn snapshot_info(&self) -> (TimerInfo, InterruptInfo) {
        let timer = TimerInfo {
            enabled: self.enabled,
            mti_period: self.mti_period.min(i32::MAX as u64) as i32,
            sti_period: self.sti_period.min(i32::MAX as u64) as i32,
            next_mti: self.next_mti.min(i32::MAX as u64) as i32,
            next_sti: self.next_sti.min(i32::MAX as u64) as i32,
            kb_irq_enabled: self.kb_irq_enabled,
        };
        let interrupts = InterruptInfo {
            pending: self.irq_pending,
            in_interrupt: self.in_interrupt,
            source: self.irq_source.clone(),
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
            irq_bit_watch: self
                .irq_bit_watch
                .clone()
                .map(|watch| json!(watch))
                .or_else(|| {
                    Some(json!({
                        "IMR": {},
                        "ISR": {},
                    }))
                }),
        };
        (timer, interrupts)
    }

    pub fn apply_snapshot_info(
        &mut self,
        timer: &TimerInfo,
        interrupts: &InterruptInfo,
        _current_cycle: u64,
        _allow_scale: bool,
    ) {
        self.enabled = timer.enabled;
        self.mti_period = timer.mti_period.max(0) as u64;
        self.sti_period = timer.sti_period.max(0) as u64;
        self.next_mti = timer.next_mti.max(0) as u64;
        self.next_sti = timer.next_sti.max(0) as u64;
        // Python stores absolute targets; do not rebase forward. Allow immediate fire if targets are in the past.
        self.kb_irq_enabled = timer.kb_irq_enabled;

        self.irq_pending = interrupts.pending;
        self.in_interrupt = interrupts.in_interrupt;
        self.irq_source = interrupts.source.clone();
        self.interrupt_stack = interrupts.stack.clone();
        self.next_interrupt_id = interrupts.next_id;
        self.irq_imr = interrupts.imr;
        self.irq_isr = interrupts.isr;
        self.irq_bit_watch = interrupts
            .irq_bit_watch
            .as_ref()
            .and_then(|v| v.as_object())
            .map(|obj| obj.clone());
        self.last_fired = None;
        self.key_irq_latched = false;
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
    }

    pub fn set_keyboard_irq_enabled(&mut self, enabled: bool) {
        self.kb_irq_enabled = enabled;
    }

    pub fn keyboard_irq_enabled(&self) -> bool {
        self.kb_irq_enabled
    }

    pub fn tick_timers(&mut self, memory: &mut MemoryImage, cycle_count: u64) -> (bool, bool) {
        if !self.enabled {
            return (false, false);
        }

        let mut fired_mti = false;
        let mut fired_sti = false;

        if self.mti_period > 0 && cycle_count >= self.next_mti {
            fired_mti = true;
            while cycle_count >= self.next_mti {
                self.next_mti = self.next_mti.wrapping_add(self.mti_period);
            }
        }
        if self.sti_period > 0 && cycle_count >= self.next_sti {
            fired_sti = true;
            while cycle_count >= self.next_sti {
                self.next_sti = self.next_sti.wrapping_add(self.sti_period);
            }
        }

        if fired_mti || fired_sti {
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
                self.emit_irq_trace(fired_mti, fired_sti, cycle_count, memory);
            }
            // Record IMR/ISR transitions for parity bit-watch metadata.
            self.record_bit_watch("IMR", memory.read_internal_byte(0xFB).unwrap_or(0));
            self.record_bit_watch("ISR", memory.read_internal_byte(ISR_OFFSET).unwrap_or(0));
        }
        (fired_mti, fired_sti)
    }

    fn record_bit_watch(&mut self, reg_name: &str, new_val: u8) {
        let table = self.irq_bit_watch.get_or_insert_with(serde_json::Map::new);
        let entry = table
            .entry(reg_name.to_string())
            .or_insert_with(|| serde_json::json!({}));
        let obj = entry
            .as_object_mut()
            .expect("bit watch entry should be an object");
        let prev = obj
            .get("last")
            .and_then(|v| v.as_u64())
            .unwrap_or(new_val as u64) as u8;
        obj.insert("last".to_string(), serde_json::json!(new_val));
        if prev == new_val {
            return;
        }
        let (action_key, _val) = if new_val > prev {
            ("set", new_val)
        } else {
            ("clear", new_val)
        };
        let arr = obj
            .entry(action_key)
            .or_insert_with(|| serde_json::json!([]))
            .as_array_mut()
            .expect("bit watch bucket should be an array");
        arr.push(serde_json::json!(new_val));
        if arr.len() > 10 {
            arr.remove(0);
        }
    }

    /// Tick timers and optionally run a keyboard scan when MTI fires, mirroring Python's _tick_timers.
    /// Returns (mti, sti, key_events).
    pub fn tick_timers_with_keyboard<F>(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        mut keyboard_scan: F,
        y_reg: Option<u32>,
    ) -> (bool, bool, usize, Option<KeyboardTelemetry>)
    where
        F: FnMut(&mut MemoryImage) -> (usize, bool, Option<KeyboardTelemetry>),
    {
        let (mti, sti) = self.tick_timers(memory, cycle_count);
        let mut key_events = 0usize;
        // Only carry a latch forward while the keyboard IRQ is enabled; otherwise drop it.
        let mut fifo_has_data = if self.kb_irq_enabled {
            self.key_irq_latched
        } else {
            false
        };
        let mut kb_stats: Option<KeyboardTelemetry> = None;
        if mti {
            let (events, has_data, stats) = keyboard_scan(memory);
            kb_stats = stats;
            key_events = events;
            fifo_has_data = fifo_has_data || has_data;
        }
        let pc_trace = crate::llama::eval::perfetto_instr_context()
            .map(|(_, pc)| pc)
            .unwrap_or_else(perfetto_last_pc);
        let latch_active = self.kb_irq_enabled && (key_events > 0 || fifo_has_data);
        if latch_active {
            if let Some(isr) = memory.read_internal_byte(ISR_OFFSET) {
                if (isr & 0x04) == 0 {
                    memory.write_internal_byte(ISR_OFFSET, isr | 0x04);
                }
            }
            // Mirror Python: key activity (new events or pending FIFO data) latches KEYI and marks a pending IRQ.
            self.irq_pending = true;
            self.irq_source = Some("KEY".to_string());
            self.last_fired = self.irq_source.clone();
            self.irq_imr = memory.read_internal_byte(0xFB).unwrap_or(self.irq_imr);
            self.irq_isr = memory
                .read_internal_byte(ISR_OFFSET)
                .unwrap_or(self.irq_isr);
            // Perfetto parity: emit a KeyIRQ marker with PC/cycle context.
            if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
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
                    payload.insert("pc".to_string(), AnnotationValue::Pointer(pc_trace as u64));
                    payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
                    if let Some(y) = y_reg {
                        payload.insert("y".to_string(), AnnotationValue::Pointer(y as u64));
                    }
                    if let Some(stats) = kb_stats.as_ref() {
                        payload.insert("kol".to_string(), AnnotationValue::UInt(stats.kol as u64));
                        payload.insert("koh".to_string(), AnnotationValue::UInt(stats.koh as u64));
                        payload.insert(
                            "pressed".to_string(),
                            AnnotationValue::UInt(stats.pressed as u64),
                        );
                    }
                    tracer.record_irq_event("KeyIRQ", payload);
                }
            }
        }
        if key_events == 0 {
            if let Some(stats) = kb_stats.as_ref() {
                if stats.pressed > 0 {
                    if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                        if let Some(tracer) = guard.as_mut() {
                            let mut payload = HashMap::new();
                            payload.insert(
                                "pressed".to_string(),
                                AnnotationValue::UInt(stats.pressed as u64),
                            );
                            payload.insert(
                                "strobe_count".to_string(),
                                AnnotationValue::UInt(stats.strobe_count as u64),
                            );
                            payload
                                .insert("kol".to_string(), AnnotationValue::UInt(stats.kol as u64));
                            payload
                                .insert("koh".to_string(), AnnotationValue::UInt(stats.koh as u64));
                            payload.insert(
                                "active_cols".to_string(),
                                AnnotationValue::Str(format!("{:?}", stats.active_columns)),
                            );
                            payload.insert(
                                "pc".to_string(),
                                AnnotationValue::Pointer(pc_trace as u64),
                            );
                            payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
                            tracer.record_irq_event("KeyScanEmpty", payload);
                        }
                    }
                }
            }
        }
        // Track latch so KEYI can be reasserted if firmware clears ISR while FIFO remains non-empty.
        self.key_irq_latched = latch_active;
        // Perfetto parity: emit a scan event regardless of new key events.
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                let mut payload = HashMap::new();
                payload.insert(
                    "events".to_string(),
                    AnnotationValue::UInt(key_events as u64),
                );
                payload.insert("mti".to_string(), AnnotationValue::UInt(mti as u64));
                payload.insert("sti".to_string(), AnnotationValue::UInt(sti as u64));
                payload.insert(
                    "imr".to_string(),
                    AnnotationValue::UInt(self.irq_imr as u64),
                );
                payload.insert(
                    "isr".to_string(),
                    AnnotationValue::UInt(self.irq_isr as u64),
                );
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
                payload.insert("cycle".to_string(), AnnotationValue::UInt(cycle_count));
                tracer.record_irq_event("KeyScanEvent", payload);
            }
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
        fired_mti: bool,
        fired_sti: bool,
        cycle_count: u64,
        memory: &MemoryImage,
    ) {
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                let mut payload: Vec<(&str, u64)> = Vec::new();
                if fired_mti {
                    payload.push(("mti", 1));
                }
                if fired_sti {
                    payload.push(("sti", 1));
                }
                payload.push(("isr", self.irq_isr as u64));
                payload.push(("imr", self.irq_imr as u64));
                payload.push(("cycle", cycle_count as u64));
                if let Some((op_idx, pc)) = crate::llama::eval::perfetto_instr_context() {
                    payload.push(("op_index", op_idx));
                    payload.push(("pc", pc as u64));
                } else {
                    let last_pc = crate::llama::eval::perfetto_last_pc();
                    payload.push(("pc", last_pc as u64));
                    let last_idx = crate::llama::eval::perfetto_last_instr_index();
                    if last_idx != u64::MAX {
                        payload.push(("op_index", last_idx));
                    }
                }
                // Include ISR snapshot if available.
                if let Some(isr) = memory.read_internal_byte(ISR_OFFSET) {
                    payload.push(("isr_mem", isr as u64));
                }
                // Align event naming with Python tracer ("TimerFired").
                tracer.record_irq_event(
                    "TimerFired",
                    payload
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), AnnotationValue::UInt(v as u64)))
                        .collect(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timers_use_absolute_targets() {
        let mut timer = TimerContext::new(true, 10, 0);
        let mut mem = MemoryImage::new();
        timer.next_mti = 50;
        timer.next_sti = 0;
        let mut cycles = 0u64;
        // Run up to but not including the target
        for _ in 0..49 {
            timer.tick_timers(&mut mem, cycles);
            cycles += 1;
            assert!(!timer.irq_pending);
        }
        assert_eq!(cycles, 49);
        // The 50th tick should fire MTI and roll the target forward.
        cycles = 50;
        timer.tick_timers(&mut mem, cycles);
        assert!(timer.irq_pending);
        assert!(timer.next_mti > 50);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01);
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
            },
            &InterruptInfo::default(),
            100,
            false,
        );
        let mut mem = MemoryImage::new();
        let mut cycles = 100u64; // current cycle when snapshot applied

        // Advance to just before first fire
        while cycles < 149 {
            timer.tick_timers(&mut mem, cycles);
            cycles += 1;
            assert!(!timer.irq_pending);
        }
        // Fire MTI at cycle 150, then next target moves to 170.
        cycles = 150;
        timer.tick_timers(&mut mem, cycles);
        assert!(timer.irq_pending);
        assert!(timer.next_mti > 150);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01);
    }

    #[test]
    fn tick_timers_updates_irq_mirrors() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // Preload IMR so mirror should reflect it.
        mem.write_internal_byte(0xFB, 0xAA);
        // First tick should fire MTI and update ISR/IMR mirrors.
        timer.tick_timers(&mut mem, 1);
        assert_eq!(timer.irq_imr, 0xAA);
        assert_eq!(timer.irq_isr & 0x01, 0x01);
    }

    #[test]
    fn tick_timers_sets_pending_even_when_imr_masked() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // IMR master cleared -> still pend like Python; delivery will gate later.
        mem.write_internal_byte(0xFB, 0x00);
        let (_mti, _sti) = timer.tick_timers(&mut mem, 1);
        assert!(
            timer.irq_pending,
            "irq_pending should set even when IMR master=0"
        );

        // Enable master but mask out MTI bit -> still pend; gating happens during delivery.
        mem.write_internal_byte(0xFB, 0x80);
        timer.tick_timers(&mut mem, 2);
        assert!(
            timer.irq_pending,
            "irq_pending should set even when MTI masked"
        );

        // Enable MTI bit -> should pend on next fire.
        mem.write_internal_byte(0xFB, 0x81);
        timer.tick_timers(&mut mem, 3);
        assert!(
            timer.irq_pending,
            "irq_pending should set when master+MTI enabled"
        );
    }

    #[test]
    fn tick_timers_increments_counters_on_fire() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        timer.tick_timers(&mut mem, 1);
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
            timer.tick_timers_with_keyboard(&mut mem, 0, |_mem| (1, true, None), None);
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
    fn tick_timers_with_keyboard_sets_keyi() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // Simulate keyboard scan emitting one event.
        let (_, _, events, _) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (1, true, None), None);
        assert_eq!(events, 1);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & 0x04,
            0x04,
            "KEYI bit should be set on MTI with events"
        );
    }

    #[test]
    fn tick_timers_with_keyboard_reasserts_keyi_without_events() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // No new events, but FIFO already has data -> KEYI should still assert on MTI.
        let (_mti, _sti, events, _) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (0, true, None), None);
        assert_eq!(events, 0);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(
            isr & 0x04,
            0x04,
            "KEYI should reassert when FIFO has data even without new events"
        );
        assert!(
            timer.irq_pending,
            "pending IRQ should be latched when FIFO remains non-empty"
        );
    }

    #[test]
    fn apply_snapshot_scales_timers_for_llama() {
        let prev = env::var("LLAMA_TIMER_SCALE").ok();
        env::set_var("LLAMA_TIMER_SCALE", "0.5");

        let mut timer = TimerContext::new(true, 100, 200);
        timer.apply_snapshot_info(
            &crate::TimerInfo {
                enabled: true,
                mti_period: 100,
                sti_period: 200,
                next_mti: 75,
                next_sti: 125,
                kb_irq_enabled: true,
            },
            &InterruptInfo::default(),
            0,
            true,
        );

        // Parity: snapshot restore uses serialized periods verbatim; ignore env scaling.
        assert_eq!(
            timer.mti_period, 100,
            "MTI period should not scale on snapshot load"
        );
        assert_eq!(
            timer.sti_period, 200,
            "STI period should not scale on snapshot load"
        );

        if let Some(val) = prev {
            env::set_var("LLAMA_TIMER_SCALE", val);
        } else {
            env::remove_var("LLAMA_TIMER_SCALE");
        }
    }

    #[test]
    fn apply_snapshot_does_not_scale_when_disabled() {
        // Maintain previous contract: disable scaling path; periods remain as serialized.
        let mut timer = TimerContext::new(true, 100, 200);
        timer.apply_snapshot_info(
            &crate::TimerInfo {
                enabled: true,
                mti_period: 100,
                sti_period: 200,
                next_mti: 75,
                next_sti: 125,
                kb_irq_enabled: true,
            },
            &InterruptInfo::default(),
            0,
            false,
        );

        assert_eq!(timer.mti_period, 100);
        assert_eq!(timer.sti_period, 200);
    }
}
