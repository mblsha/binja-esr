// PY_SOURCE: sc62015/pysc62015/emulator.py
//! Rust host-side execution control; not a second machine scheduler.
//!
//! Host deadlines/cancellation must not become emulated timer events. This
//! adapter delegates to the existing architectural scheduler in small batches
//! and only inspects host control between them. It never suspends an instruction
//! partway through its memory operations or manufactures a guest interrupt.

use crate::{CoreRuntime, Result};
use serde::Serialize;

/// Maximum scheduler-boundary budget between host-control checks.
///
/// This is a cooperative bound, not a hard wall-clock guarantee: one counted
/// instruction or a synchronous host callback can take arbitrarily longer.
pub const RUN_CONTROL_POLL_BOUNDARIES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStopReason {
    BoundaryBudget,
    HostYield,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RunProgress {
    /// Boundary budget submitted to successful scheduler calls, NOT retired
    /// instructions. An inert OFF machine can return without consuming time.
    pub boundary_budget_used: usize,
    pub instructions_retired: u64,
    /// Delta of the machine's relative timing counter, not physical cycles or
    /// host time. The independently advancing OFF RTC is not this counter.
    pub timing_units_advanced: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RunSliceResult {
    pub reason: RunStopReason,
    pub progress: RunProgress,
}

impl CoreRuntime {
    /// Execute a cooperatively bounded slice using the existing scheduler.
    ///
    /// `should_yield` is checked before the first batch and between subsequent
    /// batches of at most [`RUN_CONTROL_POLL_BOUNDARIES`]. A native caller can
    /// check a monotonic deadline or an atomic cancellation flag; a WASM caller
    /// must return to the browser event loop between slices to receive messages.
    /// The callback receives counters only, not the machine or bus.
    ///
    /// Returning early leaves a resumable architectural boundary. Execution
    /// faults retain `step_scheduler_boundaries`' existing error/poison contract;
    /// this adapter does not roll back work or silently resume a failed call.
    /// A zero budget or immediate host yield does not access the emulated bus.
    pub fn run_slice(
        &mut self,
        boundary_budget: usize,
        mut should_yield: impl FnMut(&RunProgress) -> bool,
    ) -> Result<RunSliceResult> {
        let initial_instructions = self.instruction_count();
        let initial_timing = self.cycle_count();
        let mut progress = RunProgress::default();
        while progress.boundary_budget_used < boundary_budget {
            if should_yield(&progress) {
                return Ok(RunSliceResult {
                    reason: RunStopReason::HostYield,
                    progress,
                });
            }
            let batch =
                (boundary_budget - progress.boundary_budget_used).min(RUN_CONTROL_POLL_BOUNDARIES);
            self.step_scheduler_boundaries(batch)?;
            progress.boundary_budget_used += batch;
            progress.instructions_retired =
                self.instruction_count().wrapping_sub(initial_instructions);
            progress.timing_units_advanced = self.cycle_count().wrapping_sub(initial_timing);
        }
        Ok(RunSliceResult {
            reason: RunStopReason::BoundaryBudget,
            progress,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::RegName;
    use crate::llama::state::PowerState;
    use crate::memory::{IMEM_IMR_OFFSET, IMEM_ISR_OFFSET};
    use crate::{collect_registers, DeviceModel};
    use std::sync::{Arc, Mutex};

    fn machine(model: DeviceModel, power: PowerState) -> CoreRuntime {
        let mut runtime = CoreRuntime::for_model(model, &[]).expect("model profile");
        runtime.state.set_pc(0x10000);
        runtime.state.set_reg(RegName::S, 0x2000);
        runtime.state.set_power_state(power);
        // Zero-filled RAM executes NOPs. Mask delivery but leave timer status
        // updates/wake enabled, exercising the actual machine scheduler.
        runtime.memory.write_internal_byte(IMEM_IMR_OFFSET, 0);
        runtime.timer.configure_scr_periods(3, 3, 7, 7, 0, 0);
        runtime.timer.next_mti = 3;
        runtime.timer.next_sti = 7;
        if model == DeviceModel::Iq7000 {
            runtime
                .set_iq7000_clock_seed_yyyymmddhhmm("202609060000")
                .expect("deterministic RTC seed");
        }
        runtime
    }

    fn assert_machine_matches(actual: &CoreRuntime, expected: &CoreRuntime) {
        assert_eq!(
            collect_registers(&actual.state),
            collect_registers(&expected.state)
        );
        assert_eq!(actual.state.power_state(), expected.state.power_state());
        assert_eq!(actual.instruction_count(), expected.instruction_count());
        assert_eq!(actual.cycle_count(), expected.cycle_count());
        assert_eq!(
            actual.memory.internal_slice(),
            expected.memory.internal_slice()
        );
        assert_eq!(
            actual.memory.external_slice(),
            expected.memory.external_slice()
        );
        assert_eq!(actual.timer.next_mti, expected.timer.next_mti);
        assert_eq!(actual.timer.next_sti, expected.timer.next_sti);
        assert_eq!(actual.timer.irq_pending, expected.timer.irq_pending);
        assert_eq!(actual.iq7000_rtc, expected.iq7000_rtc);
    }

    #[test]
    fn zero_budget_and_immediate_yield_do_not_touch_machine_or_bus() {
        for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
            let mut runtime = machine(model, PowerState::Running);
            let expected = machine(model, PowerState::Running);
            let reads = runtime.memory.memory_read_count();
            let writes = runtime.memory.memory_write_count();
            let zero = runtime.run_slice(0, |_| panic!("no work to poll")).unwrap();
            assert_eq!(zero.reason, RunStopReason::BoundaryBudget);
            assert_eq!(zero.progress, RunProgress::default());
            let stopped = runtime.run_slice(100_000, |_| true).unwrap();
            assert_eq!(stopped.reason, RunStopReason::HostYield);
            assert_eq!(stopped.progress, RunProgress::default());
            assert_eq!(runtime.memory.memory_read_count(), reads);
            assert_eq!(runtime.memory.memory_write_count(), writes);
            assert_machine_matches(&runtime, &expected);
        }
    }

    #[test]
    fn host_yield_is_observed_at_the_next_small_batch_and_can_resume() {
        let mut runtime = machine(DeviceModel::PcE500, PowerState::Running);
        let mut expected = machine(DeviceModel::PcE500, PowerState::Running);
        let mut polls = Vec::new();
        let stopped = runtime
            .run_slice(100_000, |progress| {
                polls.push(progress.boundary_budget_used);
                progress.boundary_budget_used >= RUN_CONTROL_POLL_BOUNDARIES
            })
            .unwrap();
        assert_eq!(polls, [0, RUN_CONTROL_POLL_BOUNDARIES]);
        assert_eq!(stopped.reason, RunStopReason::HostYield);
        assert_eq!(
            stopped.progress.boundary_budget_used,
            RUN_CONTROL_POLL_BOUNDARIES
        );
        assert_eq!(
            stopped.progress.instructions_retired,
            RUN_CONTROL_POLL_BOUNDARIES as u64
        );
        runtime.run_slice(17, |_| false).unwrap();
        expected
            .step_scheduler_boundaries(RUN_CONTROL_POLL_BOUNDARIES + 17)
            .unwrap();
        assert_machine_matches(&runtime, &expected);
    }

    #[test]
    fn chunking_matches_single_boundaries_for_both_models_and_power_states() {
        for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
            for power in [PowerState::Running, PowerState::Halted, PowerState::Off] {
                let mut expected = machine(model, power);
                for _ in 0..257 {
                    expected.step_scheduler_boundaries(1).unwrap();
                }
                for chunk in [1, 7, 64, 200_000] {
                    let mut runtime = machine(model, power);
                    let mut remaining = 257;
                    while remaining > 0 {
                        let requested = remaining.min(chunk);
                        let result = runtime.run_slice(requested, |_| false).unwrap();
                        assert_eq!(result.reason, RunStopReason::BoundaryBudget);
                        assert_eq!(result.progress.boundary_budget_used, requested);
                        remaining -= result.progress.boundary_budget_used;
                    }
                    assert_machine_matches(&runtime, &expected);
                }
            }
        }
    }

    #[test]
    fn halted_budget_is_not_reported_as_retired_instructions() {
        let mut runtime = machine(DeviceModel::PcE500, PowerState::Halted);
        runtime.timer.enabled = false;
        let result = runtime.run_slice(257, |_| false).unwrap();
        assert_eq!(result.progress.boundary_budget_used, 257);
        assert_eq!(result.progress.instructions_retired, 0);
        assert_eq!(result.progress.timing_units_advanced, 257);
    }

    #[test]
    fn inert_off_budget_is_not_reported_as_emulated_time() {
        let mut runtime = machine(DeviceModel::PcE500, PowerState::Off);
        let result = runtime.run_slice(257, |_| false).unwrap();
        assert_eq!(result.progress.boundary_budget_used, 257);
        assert_eq!(result.progress.instructions_retired, 0);
        assert_eq!(result.progress.timing_units_advanced, 0);
        assert_eq!(runtime.state.power_state(), PowerState::Off);
    }

    #[test]
    fn faults_are_propagated_instead_of_becoming_host_yields() {
        let mut runtime = machine(DeviceModel::PcE500, PowerState::Running);
        runtime.memory.write_external_byte(0x10000, 0x20); // reserved opcode
        let mut direct = machine(DeviceModel::PcE500, PowerState::Running);
        direct.memory.write_external_byte(0x10000, 0x20);
        let expected_error = direct.step_scheduler_boundaries(1).unwrap_err().to_string();
        let actual_error = runtime.run_slice(257, |_| false).unwrap_err().to_string();
        assert_eq!(actual_error, expected_error);
        assert_machine_matches(&runtime, &direct);
        assert_eq!(
            runtime.memory.read_internal_byte_silent(IMEM_ISR_OFFSET),
            Some(0)
        );
    }

    #[test]
    fn sliced_irq_entry_preserves_discarded_callback_fetch_and_frame() {
        type Accesses = Arc<Mutex<Vec<(char, u32, u8)>>>;

        fn irq_machine(model: DeviceModel) -> (CoreRuntime, Accesses) {
            let mut runtime = machine(model, PowerState::Running);
            runtime.timer.enabled = false;
            runtime
                .memory
                .write_internal_byte(IMEM_IMR_OFFSET, crate::IMR_MASTER | crate::IMR_KEY);
            runtime
                .memory
                .write_internal_byte(IMEM_ISR_OFFSET, crate::ISR_KEYI);
            runtime.timer.irq_pending = true;
            runtime.timer.irq_source = Some("KEY".into());
            // Vector/target stability is deliberately backed by plain RAM.
            // Log the discarded callback-backed fallthrough fetch. The IRQ
            // frame uses native memory, compared in assert_machine_matches;
            // these callbacks are not a complete vector/frame bus trace.
            runtime.memory.write_external_slice(0xFFFFA, &[0, 1, 1]);
            runtime.memory.set_python_ranges(vec![(0x10000, 0x10000)]);
            let byte = |address| match address {
                0x10000 => 0x20, // reserved fallthrough must only be fetched
                _ => 0x00,
            };
            runtime.set_host_peek(move |address| Some(byte(address)));
            let accesses = Arc::new(Mutex::new(Vec::new()));
            let reads = accesses.clone();
            runtime.set_host_read(move |address| {
                let value = byte(address);
                reads.lock().unwrap().push(('r', address, value));
                Some(value)
            });
            (runtime, accesses)
        }

        for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
            let (mut sliced, sliced_bus) = irq_machine(model);
            let (mut direct, direct_bus) = irq_machine(model);
            let result = sliced
                .run_slice(100, |progress| progress.boundary_budget_used != 0)
                .unwrap();
            assert_eq!(result.reason, RunStopReason::HostYield);
            sliced
                .run_slice(100 - result.progress.boundary_budget_used, |_| false)
                .unwrap();
            direct.step_scheduler_boundaries(100).unwrap();
            assert_machine_matches(&sliced, &direct);
            let actual = sliced_bus.lock().unwrap();
            assert_eq!(*actual, *direct_bus.lock().unwrap());
            assert_eq!(*actual, [('r', 0x10000, 0x20)]);
            assert_eq!(sliced.state.get_reg(RegName::S), 0x1FFB);
            assert!(sliced.timer.in_interrupt);
        }
    }
}
