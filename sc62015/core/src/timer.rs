use crate::memory::MemoryImage;
use crate::{InterruptInfo, TimerInfo};

const ISR_OFFSET: u32 = 0xFC;

#[derive(Clone, Debug)]
pub struct TimerContext {
    pub enabled: bool,
    pub mti_period: u64,
    pub sti_period: u64,
    pub next_mti: u64,
    pub next_sti: u64,
    pub irq_pending: bool,
    pub irq_source: Option<String>,
    pub irq_imr: u8,
    pub irq_isr: u8,
    pub in_interrupt: bool,
    pub interrupt_stack: Vec<u32>,
    pub next_interrupt_id: u32,
}

impl TimerContext {
    pub fn new(enabled: bool, mti_period: i32, sti_period: i32) -> Self {
        let mut ctx = Self {
            enabled,
            mti_period: mti_period.max(0) as u64,
            sti_period: sti_period.max(0) as u64,
            next_mti: 0,
            next_sti: 0,
            irq_pending: false,
            irq_source: None,
            irq_imr: 0,
            irq_isr: 0,
            in_interrupt: false,
            interrupt_stack: Vec::new(),
            next_interrupt_id: 0,
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
    }

    pub fn snapshot_info(&self) -> (TimerInfo, InterruptInfo) {
        let timer = TimerInfo {
            enabled: self.enabled,
            mti_period: self.mti_period.min(i32::MAX as u64) as i32,
            sti_period: self.sti_period.min(i32::MAX as u64) as i32,
            next_mti: self.next_mti.min(i32::MAX as u64) as i32,
            next_sti: self.next_sti.min(i32::MAX as u64) as i32,
        };
        let interrupts = InterruptInfo {
            pending: self.irq_pending,
            in_interrupt: self.in_interrupt,
            source: self.irq_source.clone(),
            stack: self.interrupt_stack.clone(),
            next_id: self.next_interrupt_id,
            imr: self.irq_imr,
            isr: self.irq_isr,
        };
        (timer, interrupts)
    }

    pub fn apply_snapshot_info(&mut self, timer: &TimerInfo, interrupts: &InterruptInfo) {
        self.enabled = timer.enabled;
        self.mti_period = timer.mti_period.max(0) as u64;
        self.sti_period = timer.sti_period.max(0) as u64;
        self.next_mti = timer.next_mti.max(0) as u64;
        self.next_sti = timer.next_sti.max(0) as u64;

        self.irq_pending = interrupts.pending;
        self.in_interrupt = interrupts.in_interrupt;
        self.irq_source = interrupts.source.clone();
        self.interrupt_stack = interrupts.stack.clone();
        self.next_interrupt_id = interrupts.next_id;
        self.irq_imr = interrupts.imr;
        self.irq_isr = interrupts.isr;
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
    }

    pub fn tick_timers(&mut self, memory: &mut MemoryImage, cycle_count: &mut u64) {
        *cycle_count = cycle_count.wrapping_add(1);
        if !self.enabled {
            return;
        }

        let mut fired_mti = false;
        let mut fired_sti = false;

        if self.mti_period > 0 && *cycle_count >= self.next_mti {
            fired_mti = true;
            while *cycle_count >= self.next_mti {
                self.next_mti = self.next_mti.wrapping_add(self.mti_period);
            }
        }
        if self.sti_period > 0 && *cycle_count >= self.next_sti {
            fired_sti = true;
            while *cycle_count >= self.next_sti {
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
            self.irq_pending = true;
            self.irq_source = if fired_mti && fired_sti {
                Some("MTI+STI".to_string())
            } else if fired_mti {
                Some("MTI".to_string())
            } else {
                Some("STI".to_string())
            };
        }
    }

    pub fn drain_pending_irq(&mut self) -> Option<String> {
        if !self.irq_pending {
            return None;
        }
        self.irq_pending = false;
        self.irq_source.take()
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
            timer.tick_timers(&mut mem, &mut cycles);
            assert_eq!(timer.irq_pending, false);
        }
        assert_eq!(cycles, 49);
        // The 50th tick should fire MTI and roll the target forward.
        timer.tick_timers(&mut mem, &mut cycles);
        assert_eq!(cycles, 50);
        assert!(timer.irq_pending);
        assert!(timer.next_mti > 50);
        let isr = mem.read_internal_byte(ISR_OFFSET).unwrap_or(0);
        assert_eq!(isr & 0x01, 0x01);
    }
}
