use super::eval::{LlamaBus, LlamaExecutor};
use super::state::LlamaState;
use crate::async_driver::sleep_cycles;

pub struct AsyncLlamaExecutor {
    inner: LlamaExecutor,
}

impl AsyncLlamaExecutor {
    pub fn new() -> Self {
        Self {
            inner: LlamaExecutor::new(),
        }
    }

    pub async fn execute<B: LlamaBus>(
        &mut self,
        opcode: u8,
        state: &mut LlamaState,
        bus: &mut B,
        _ticker: &mut TickHelper<'_>,
    ) -> Result<u8, &'static str> {
        self.inner.execute(opcode, state, bus)
    }
}

impl Default for AsyncLlamaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TickHelper<'a> {
    cycle_count: &'a mut u64,
    run_timer_cycles: bool,
    tick_timers: Option<&'a mut dyn FnMut(u64)>,
}

impl<'a> TickHelper<'a> {
    pub fn new(
        cycle_count: &'a mut u64,
        run_timer_cycles: bool,
        tick_timers: Option<&'a mut dyn FnMut(u64)>,
    ) -> Self {
        Self {
            cycle_count,
            run_timer_cycles,
            tick_timers,
        }
    }

    pub async fn tick_once(&mut self) {
        let next = self.cycle_count.wrapping_add(1);
        if self.run_timer_cycles {
            if let Some(cb) = self.tick_timers.as_deref_mut() {
                cb(next);
            }
        }
        *self.cycle_count = next;
        sleep_cycles(1).await;
    }
}

#[allow(unused_macros)]
macro_rules! tick {
    ($ticker:expr) => {
        $ticker.tick_once().await;
    };
    ($ticker:expr, $count:expr) => {
        for _ in 0..$count {
            $ticker.tick_once().await;
        }
    };
}

#[allow(unused_imports)]
pub(crate) use tick;
