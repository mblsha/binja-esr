use crate::async_driver::{current_cycle, emit_event, sleep_cycles, DriverEvent};
use crate::CoreRuntime;
use std::cell::RefCell;
use std::rc::Rc;

pub struct AsyncTimerKeyboardTask {
    runtime: Rc<RefCell<CoreRuntime>>,
}

impl AsyncTimerKeyboardTask {
    pub fn new(runtime: Rc<RefCell<CoreRuntime>>) -> Self {
        Self { runtime }
    }

    pub async fn run(&self) {
        loop {
            sleep_cycles(1).await;
            let cycle = current_cycle();
            let mut rt = self.runtime.borrow_mut();
            if rt.state.is_off() {
                continue;
            }
            rt.tick_timers_and_keyboard(cycle);
        }
    }

    pub async fn run_for(&self, cycles: u64) {
        for _ in 0..cycles {
            sleep_cycles(1).await;
            let cycle = current_cycle();
            self.runtime.borrow_mut().tick_timers_and_keyboard(cycle);
        }
    }
}

pub struct AsyncDisplayTask {
    frame_period: u64,
    event: DriverEvent,
}

impl AsyncDisplayTask {
    pub fn new(frame_period: u64, event: DriverEvent) -> Self {
        Self {
            frame_period,
            event,
        }
    }

    pub async fn run_frames(&self, frames: u64) {
        let period = self.frame_period.max(1);
        for _ in 0..frames {
            sleep_cycles(period).await;
            emit_event(self.event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_driver::AsyncDriver;
    use crate::{IMEM_ISR_OFFSET, ISR_MTI};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn async_timer_task_fires_mti() {
        let runtime = Rc::new(RefCell::new(CoreRuntime::new()));
        {
            let mut rt = runtime.borrow_mut();
            rt.timer.enabled = true;
            rt.timer.mti_period = 3;
            rt.timer.sti_period = 0;
            rt.timer.reset(0);
        }
        let task = AsyncTimerKeyboardTask::new(runtime.clone());
        let mut driver = AsyncDriver::new();

        driver.spawn(async move {
            task.run_for(4).await;
        });

        let result = driver.run_for(4);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        let isr = runtime
            .borrow()
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_ne!(isr & ISR_MTI, 0, "MTI should assert on schedule");
    }

    #[test]
    fn async_display_task_emits_event() {
        let display = AsyncDisplayTask::new(3, DriverEvent::User(42));
        let mut driver = AsyncDriver::new();

        driver.spawn(async move {
            display.run_frames(1).await;
        });

        let result = driver.run_for(2);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        let result = driver.run_for(4);
        assert_eq!(result.event, DriverEvent::User(42));
    }
}
