// PY_SOURCE: pce500/scheduler.py:TimerScheduler
// PY_SOURCE: pce500/emulator.py:PCE500Emulator._tick_timers

use crate::memory::MemoryImage;
use crate::PERFETTO_TRACER;
use crate::perfetto::AnnotationValue;
use crate::{InterruptInfo, TimerInfo};
use std::collections::HashMap;

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
            irq_counts: None,
            last_irq: None,
            irq_bit_watch: None,
        };
        (timer, interrupts)
    }

    pub fn apply_snapshot_info(
        &mut self,
        timer: &TimerInfo,
        interrupts: &InterruptInfo,
        current_cycle: u64,
    ) {
        self.enabled = timer.enabled;
        self.mti_period = timer.mti_period.max(0) as u64;
        self.sti_period = timer.sti_period.max(0) as u64;
        self.next_mti = timer.next_mti.max(0) as u64;
        self.next_sti = timer.next_sti.max(0) as u64;
        // Rebase saved absolute targets so they do not land in the past relative to the restored cycle counter.
        if self.mti_period > 0 {
            while self.next_mti != 0 && self.next_mti < current_cycle {
                self.next_mti = self.next_mti.wrapping_add(self.mti_period);
            }
        }
        if self.sti_period > 0 {
            while self.next_sti != 0 && self.next_sti < current_cycle {
                self.next_sti = self.next_sti.wrapping_add(self.sti_period);
            }
        }
        self.kb_irq_enabled = timer.kb_irq_enabled;

        self.irq_pending = interrupts.pending;
        self.in_interrupt = interrupts.in_interrupt;
        self.irq_source = interrupts.source.clone();
        self.interrupt_stack = interrupts.stack.clone();
        self.next_interrupt_id = interrupts.next_id;
        self.irq_imr = interrupts.imr;
        self.irq_isr = interrupts.isr;
        self.last_fired = None;
        self.key_irq_latched = false;
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
            self.irq_source = if fired_mti && fired_sti {
                Some("MTI+STI".to_string())
            } else if fired_mti {
                Some("MTI".to_string())
            } else {
                Some("STI".to_string())
            };
            self.last_fired = self.irq_source.clone();
            // Keep mirror fields in sync with the actual IMEM values for snapshots/tracing parity.
            self.irq_imr = memory
                .read_internal_byte(0xFB)
                .unwrap_or(self.irq_imr);
            self.irq_isr = memory
                .read_internal_byte(ISR_OFFSET)
                .unwrap_or(self.irq_isr);
            // Parity: mark IRQ pending whenever a timer fires, regardless of IMR gating.
            // IMR masking is honored later during delivery.
            self.irq_pending = fired_mti || fired_sti;
            if self.irq_pending {
                self.emit_irq_trace(fired_mti, fired_sti, memory);
            }
        }
        (fired_mti, fired_sti)
    }

    /// Tick timers and optionally run a keyboard scan when MTI fires, mirroring Python's _tick_timers.
    /// Returns (mti, sti, key_events).
    pub fn tick_timers_with_keyboard<F>(
        &mut self,
        memory: &mut MemoryImage,
        cycle_count: u64,
        mut keyboard_scan: F,
    ) -> (bool, bool, usize)
    where
        F: FnMut(&mut MemoryImage) -> (usize, bool),
    {
        let (mti, sti) = self.tick_timers(memory, cycle_count);
        let mut key_events = 0usize;
        let mut fifo_has_data = self.key_irq_latched;
        if mti && self.kb_irq_enabled {
            let (events, has_data) = keyboard_scan(memory);
            key_events = events;
            fifo_has_data = fifo_has_data || has_data;
            if key_events > 0 {
                if let Some(isr) = memory.read_internal_byte(ISR_OFFSET) {
                    if (isr & 0x04) == 0 {
                        memory.write_internal_byte(ISR_OFFSET, isr | 0x04);
                    }
                }
                // Mirror Python: key events latch KEYI and mark a pending IRQ immediately.
                self.irq_pending = true;
                self.irq_source = Some("KEY".to_string());
                self.last_fired = self.irq_source.clone();
                self.irq_imr = memory.read_internal_byte(0xFB).unwrap_or(self.irq_imr);
                self.irq_isr = memory.read_internal_byte(ISR_OFFSET).unwrap_or(self.irq_isr);
                // Perfetto parity: emit a KeyIRQ marker.
                if let Ok(mut guard) = PERFETTO_TRACER.lock() {
                    if let Some(tracer) = guard.as_mut() {
                        let mut payload = HashMap::new();
                        payload.insert("events".to_string(), AnnotationValue::UInt(key_events as u64));
                        payload.insert("imr".to_string(), AnnotationValue::UInt(self.irq_imr as u64));
                        payload.insert("isr".to_string(), AnnotationValue::UInt(self.irq_isr as u64));
                        tracer.record_irq_event("KeyIRQ", payload);
                    }
                }
            }
        }
        // Track latch so KEYI can be reasserted if firmware clears ISR while FIFO remains non-empty.
        if key_events > 0 || fifo_has_data {
            self.key_irq_latched = true;
        } else {
            self.key_irq_latched = false;
        }
        // Perfetto parity: emit a scan event regardless of new key events.
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                let mut payload = HashMap::new();
                payload.insert("events".to_string(), AnnotationValue::UInt(key_events as u64));
                payload.insert("mti".to_string(), AnnotationValue::UInt(mti as u64));
                payload.insert("sti".to_string(), AnnotationValue::UInt(sti as u64));
                payload.insert("imr".to_string(), AnnotationValue::UInt(self.irq_imr as u64));
                payload.insert("isr".to_string(), AnnotationValue::UInt(self.irq_isr as u64));
                tracer.record_irq_event("KeyScanEvent", payload);
            }
        }
        (mti, sti, key_events)
    }

    pub fn drain_pending_irq(&mut self) -> Option<String> {
        if !self.irq_pending {
            return None;
        }
        self.irq_pending = false;
        self.irq_source.take()
    }

    fn emit_irq_trace(&self, fired_mti: bool, fired_sti: bool, memory: &MemoryImage) {
        if let Ok(mut guard) = PERFETTO_TRACER.lock() {
            if let Some(tracer) = guard.as_mut() {
                let mut payload = Vec::new();
                if fired_mti {
                    payload.push(("mti", 1u32));
                }
                if fired_sti {
                    payload.push(("sti", 1u32));
                }
                payload.push(("isr", self.irq_isr as u32));
                payload.push(("imr", self.irq_imr as u32));
                // Include ISR snapshot if available.
                if let Some(isr) = memory.read_internal_byte(ISR_OFFSET) {
                    payload.push(("isr_mem", isr as u32));
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
        assert!(timer.irq_pending, "irq_pending should set even when IMR master=0");

        // Enable master but mask out MTI bit -> still pend; gating happens during delivery.
        mem.write_internal_byte(0xFB, 0x80);
        timer.tick_timers(&mut mem, 2);
        assert!(timer.irq_pending, "irq_pending should set even when MTI masked");

        // Enable MTI bit -> should pend on next fire.
        mem.write_internal_byte(0xFB, 0x81);
        timer.tick_timers(&mut mem, 3);
        assert!(timer.irq_pending, "irq_pending should set when master+MTI enabled");
    }

    #[test]
    fn tick_timers_with_keyboard_sets_keyi() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // Simulate keyboard scan emitting one event.
        let (_, _, events) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (1, true));
        assert_eq!(events, 1);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0x04, "KEYI bit should be set on MTI with events");
    }

    #[test]
    fn tick_timers_with_keyboard_does_not_raise_keyi_without_events() {
        let mut timer = TimerContext::new(true, 1, 0);
        let mut mem = MemoryImage::new();
        // No new events, but FIFO already has data -> KEYI should still assert on MTI.
        let (_mti, _sti, events) =
            timer.tick_timers_with_keyboard(&mut mem, 1, |_mem| (0, true));
        assert_eq!(events, 0);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x04, 0x00, "KEYI should remain clear when no new events");
    }
}
