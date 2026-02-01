use crate::async_driver::sleep_cycles;
use crate::{CoreError, CoreRuntime, ADDRESS_MASK};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTraceEvent {
    pub instruction_index: u64,
    pub pc_before: u32,
    pub pc_after: u32,
    pub cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncCpuStats {
    pub instructions_executed: u64,
    pub cycles_executed: u64,
}

pub struct AsyncCpuHandle {
    runtime: Rc<RefCell<CoreRuntime>>,
}

impl AsyncCpuHandle {
    pub fn new(runtime: Rc<RefCell<CoreRuntime>>) -> Self {
        Self { runtime }
    }

    pub async fn run_instructions(
        &self,
        instructions: usize,
        mut on_trace: Option<&mut dyn FnMut(CpuTraceEvent)>,
    ) -> Result<AsyncCpuStats, CoreError> {
        let mut instructions_executed = 0u64;
        let mut cycles_executed = 0u64;

        for _ in 0..instructions {
            let (pc_before, instr_before, cycle_before) = {
                let runtime = self.runtime.borrow();
                (
                    runtime.state.pc() & ADDRESS_MASK,
                    runtime.instruction_count(),
                    runtime.cycle_count(),
                )
            };

            sleep_cycles(1).await;
            {
                let mut runtime = self.runtime.borrow_mut();
                runtime.step(1)?;
            }

            let (pc_after, instr_after, cycle_after) = {
                let runtime = self.runtime.borrow();
                (
                    runtime.state.pc() & ADDRESS_MASK,
                    runtime.instruction_count(),
                    runtime.cycle_count(),
                )
            };

            let instr_delta = instr_after.saturating_sub(instr_before);
            let cycle_delta = cycle_after.saturating_sub(cycle_before);

            cycles_executed = cycles_executed.saturating_add(cycle_delta);

            if instr_delta > 0 {
                instructions_executed = instructions_executed.saturating_add(instr_delta);
                if let Some(cb) = on_trace.as_deref_mut() {
                    cb(CpuTraceEvent {
                        instruction_index: instr_after.saturating_sub(1),
                        pc_before,
                        pc_after,
                        cycles: cycle_delta.max(1),
                    });
                }
            }

        }

        Ok(AsyncCpuStats {
            instructions_executed,
            cycles_executed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_driver::{AsyncDriver, DriverEvent};
    use crate::llama::opcodes::RegName;

    #[test]
    fn async_cpu_respects_cycle_budget_and_emits_trace() {
        let runtime = Rc::new(RefCell::new(CoreRuntime::new()));
        runtime.borrow_mut().state.set_reg(RegName::PC, 0);
        let trace = Rc::new(RefCell::new(Vec::new()));
        let trace_inner = trace.clone();
        let handle = AsyncCpuHandle::new(runtime.clone());
        let mut driver = AsyncDriver::new();

        driver.spawn(async move {
            let mut sink = |event| {
                trace_inner.borrow_mut().push(event);
            };
            handle.run_instructions(3, Some(&mut sink)).await.unwrap();
        });

        let result = driver.run_for(1);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(trace.borrow().len(), 0);

        let result = driver.run_for(2);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(trace.borrow().len(), 1);

        let result = driver.run_for(10);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(runtime.borrow().instruction_count(), 3);
        assert_eq!(trace.borrow().len(), 3);

        let events = trace.borrow();
        assert_eq!(events[0].pc_before, 0);
        assert_eq!(events[0].pc_after, 1);
        assert_eq!(events[0].cycles, 1);
        assert_eq!(events[1].pc_before, 1);
        assert_eq!(events[2].pc_after, 3);
    }
}
