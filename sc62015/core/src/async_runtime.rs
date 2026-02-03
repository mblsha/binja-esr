use crate::async_cpu::AsyncCpuHandle;
use crate::async_driver::{emit_event, AsyncDriver, DriverEvent};
use crate::{AsyncCpuStats, CoreError, CoreRuntime};
use std::cell::RefCell;
use std::rc::Rc;

const CPU_DONE_EVENT: u32 = 1;
const DEFAULT_SLICE_CYCLES: u64 = 10_000;

pub struct AsyncRuntimeRunner {
    runtime: Rc<RefCell<CoreRuntime>>,
    driver: AsyncDriver,
    slice_cycles: u64,
}

impl AsyncRuntimeRunner {
    pub fn new(runtime: Rc<RefCell<CoreRuntime>>) -> Self {
        let clock = runtime.borrow().cycle_count();
        Self {
            runtime,
            driver: AsyncDriver::with_clock(clock),
            slice_cycles: DEFAULT_SLICE_CYCLES,
        }
    }

    pub fn with_slice_cycles(mut self, slice_cycles: u64) -> Self {
        self.slice_cycles = slice_cycles.max(1);
        self
    }

    pub fn run_instructions(&mut self, instructions: usize) -> Result<AsyncCpuStats, CoreError> {
        let stats_cell: Rc<RefCell<Option<Result<AsyncCpuStats, CoreError>>>> =
            Rc::new(RefCell::new(None));
        let stats_target = stats_cell.clone();
        let runtime = self.runtime.clone();
        let cpu = AsyncCpuHandle::new(runtime);

        self.driver.spawn(async move {
            let result = cpu.run_instructions(instructions, None).await;
            *stats_target.borrow_mut() = Some(result);
            emit_event(DriverEvent::User(CPU_DONE_EVENT));
        });

        let mut slice_cycles = self.slice_cycles.max(1);
        loop {
            let result = self.driver.run_for(slice_cycles);
            match result.event {
                DriverEvent::MaxCycles => {
                    if result.cycles_executed == 0 {
                        slice_cycles = slice_cycles.saturating_add(1);
                    }
                    continue;
                }
                DriverEvent::User(CPU_DONE_EVENT) => break,
                _ => continue,
            }
        }

        let result = stats_cell.borrow_mut().take().unwrap_or_else(|| {
            Err(CoreError::Other(
                "async runtime missing stats".to_string(),
            ))
        });
        result
    }

    pub fn runtime(&self) -> Rc<RefCell<CoreRuntime>> {
        self.runtime.clone()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;
    use crate::{collect_registers, TimerContext, IMEM_ISR_OFFSET, ISR_MTI};

    #[test]
    fn async_runtime_runs_instructions() {
        let runtime = Rc::new(RefCell::new(CoreRuntime::new()));
        runtime.borrow_mut().state.set_reg(RegName::PC, 0);

        let mut runner = AsyncRuntimeRunner::new(runtime.clone()).with_slice_cycles(1);
        let stats = runner.run_instructions(3).unwrap();

        assert_eq!(stats.instructions_executed, 3);
        assert_eq!(stats.cycles_executed, 3);
        assert_eq!(runtime.borrow().instruction_count(), 3);
        assert_eq!(runtime.borrow().cycle_count(), 3);
        assert_eq!(runtime.borrow().state.pc(), 3);
    }

    #[test]
    fn async_runner_ticks_timers_per_cycle() {
        let program = vec![0x00];
        let mut rt = CoreRuntime::new();
        rt.load_rom(&program, 0);
        rt.state.set_reg(RegName::PC, 0);
        *rt.timer = TimerContext::new(true, 1, 0);
        rt.timer.reset(0);
        rt.timer.next_mti = 1;

        let runtime = Rc::new(RefCell::new(rt));
        let mut runner = AsyncRuntimeRunner::new(runtime.clone()).with_slice_cycles(1);
        runner.run_instructions(1).unwrap();

        let isr = runtime
            .borrow()
            .memory
            .read_internal_byte(IMEM_ISR_OFFSET)
            .unwrap_or(0);
        assert_ne!(isr & ISR_MTI, 0, "async CPU should tick MTI per cycle");
    }

    #[test]
    fn async_runner_matches_sync_for_nops() {
        let program = vec![0x00, 0x00, 0x00, 0x00, 0x00];
        let mut sync_rt = CoreRuntime::new();
        sync_rt.load_rom(&program, 0);
        sync_rt.state.set_reg(RegName::PC, 0);
        *sync_rt.timer = TimerContext::new(false, 0, 0);

        let mut async_rt = CoreRuntime::new();
        async_rt.load_rom(&program, 0);
        async_rt.state.set_reg(RegName::PC, 0);
        *async_rt.timer = TimerContext::new(false, 0, 0);

        sync_rt.step(5).unwrap();
        let async_rc = Rc::new(RefCell::new(async_rt));
        let mut runner = AsyncRuntimeRunner::new(async_rc.clone()).with_slice_cycles(1);
        let stats = runner.run_instructions(5).unwrap();

        let async_rt = async_rc.borrow();
        assert_eq!(stats.instructions_executed, 5);
        assert_eq!(stats.cycles_executed, 5);
        assert_eq!(async_rt.instruction_count(), sync_rt.instruction_count());
        assert_eq!(async_rt.cycle_count(), sync_rt.cycle_count());
        assert_eq!(async_rt.state.pc(), sync_rt.state.pc());
        assert_eq!(
            collect_registers(&async_rt.state),
            collect_registers(&sync_rt.state)
        );
    }

    #[cfg(feature = "perfetto")]
    #[test]
    fn async_runner_matches_sync_perfetto_trace() {
        use crate::llama::eval::reset_perf_counters;
        use crate::perfetto::perfetto_test_guard;
        use crate::perfetto::PerfettoTracer;
        use crate::PERFETTO_TRACER;
        use std::fs;

        let _lock = perfetto_test_guard();

        fn run_and_collect(program: &[u8], use_async: bool) -> Vec<(u32, u8)> {
            let tmp = std::env::temp_dir().join("async_parity_trace.perfetto-trace");
            let _ = fs::remove_file(&tmp);
            reset_perf_counters();
            {
                let mut guard = PERFETTO_TRACER.enter();
                guard.replace(Some(PerfettoTracer::new(tmp.clone())));
            }

            if use_async {
                let mut rt = CoreRuntime::new();
                rt.load_rom(program, 0);
                rt.state.set_reg(RegName::PC, 0);
                *rt.timer = TimerContext::new(false, 0, 0);
                let rc = Rc::new(RefCell::new(rt));
                let mut runner = AsyncRuntimeRunner::new(rc.clone()).with_slice_cycles(1);
                runner.run_instructions(program.len()).unwrap();
            } else {
                let mut rt = CoreRuntime::new();
                rt.load_rom(program, 0);
                rt.state.set_reg(RegName::PC, 0);
                *rt.timer = TimerContext::new(false, 0, 0);
                rt.step(program.len()).unwrap();
            }

            let events = if let Some(tracer) = PERFETTO_TRACER.enter().take() {
                let events = tracer.test_exec_events();
                let _ = tracer.finish();
                events
            } else {
                Vec::new()
            };
            let _ = fs::remove_file(&tmp);
            events
                .into_iter()
                .map(|(pc, opcode, _)| (pc, opcode))
                .collect()
        }

        let program = vec![0x00, 0x00, 0x00, 0x00];
        let sync_events = run_and_collect(&program, false);
        let async_events = run_and_collect(&program, true);
        assert_eq!(sync_events, async_events);
    }
}
