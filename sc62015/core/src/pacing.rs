// PY_SOURCE: pce500/run_pce500.py
//! Shared, host-only pacing policy for native and WASM frontends.
//!
//! No clocks are read here and no machine state is mutated. Callers supply a
//! monotonic host timestamp, execute the ordinary bounded scheduler, and charge
//! its *elapsed timing* progress (including OFF idle), never instruction count.
//! Excess host-time backlog is reported and discarded, never injected as guest
//! timer/RTC jumps. One atomic instruction may overshoot; retain that debt.

use crate::run_control::{RunProgress, RunSliceResult};
use crate::CoreRuntime;
use crate::{CoreError, DeviceModel, Result};
use serde::Serialize;

pub const HOST_SLICE_TARGET_US: u64 = 4_000;
pub const MAX_CATCH_UP_HOST_US: u64 = 50_000;
pub const MAX_HOST_SLICE_BOUNDARIES: usize = 200_000;
const NS_PER_SECOND: i128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Nominal compatibility timebase pacing, NOT hardware-calibrated realtime.
    #[default]
    Interactive,
    /// Unthrottled execution, still with bounded host slices/control polling.
    Turbo,
    /// Explicit boundary-budget execution only; no autonomous Run loop.
    /// Reproducibility also requires fixed RTC seed and emulated-boundary inputs.
    Deterministic,
}

impl ExecutionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Turbo => "turbo",
            Self::Deterministic => "deterministic",
        }
    }

    pub fn parse_label(label: &str) -> Result<Self> {
        match label {
            "interactive" => Ok(Self::Interactive),
            "turbo" => Ok(Self::Turbo),
            "deterministic" => Ok(Self::Deterministic),
            _ => Err(CoreError::Other(
                "execution mode must be interactive, turbo, or deterministic".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaceAction {
    Run,
    Wait,
    ExplicitBudgetRequired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PacePlan {
    pub action: PaceAction,
    pub boundary_budget: usize,
    /// Stop at/beyond this elapsed timing amount as well as the host deadline.
    /// Atomic instructions are never split to meet this target.
    pub elapsed_timing_budget: Option<u64>,
    /// Suggested wait; native callers must continue servicing controls.
    /// Bounded to one host slice even after a very expensive guest instruction.
    pub wait_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PacedSliceResult {
    pub plan: PacePlan,
    pub slice: Option<RunSliceResult>,
    pub dropped_host_ns: u64,
}

impl CoreRuntime {
    /// Shared native/WASM automatic execution path. The host must yield or
    /// service controls after returning, including when the plan says Wait.
    /// Explicit step/call budgets bypass this automatic pacing path.
    pub fn run_automatic_slice(
        &mut self,
        pacer: &mut Pacer,
        host_ns: u64,
        boundary_limit: usize,
        mut should_yield: impl FnMut(&RunProgress) -> bool,
    ) -> Result<PacedSliceResult> {
        let plan = pacer.automatic_plan(host_ns, boundary_limit)?;
        let slice = if plan.action == PaceAction::Run && plan.boundary_budget != 0 {
            let result = self.run_slice(plan.boundary_budget, |progress| {
                should_yield(progress)
                    || plan
                        .elapsed_timing_budget
                        .is_some_and(|limit| progress.elapsed_timing_units_advanced >= limit)
            })?;
            pacer.account(result.progress.elapsed_timing_units_advanced);
            Some(result)
        } else {
            None
        };
        Ok(PacedSliceResult {
            plan,
            slice,
            dropped_host_ns: pacer.dropped_host_ns(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Pacer {
    mode: ExecutionMode,
    timebase_hz: u64,
    previous_host_ns: Option<u64>,
    // Fixed point: elapsed timing units * 1e9, so repeated sub-tick host
    // intervals retain fractional credit without floating-point drift.
    credit: i128,
    dropped_credit: u128,
}

impl Pacer {
    pub fn for_model(model: DeviceModel, mode: ExecutionMode) -> Self {
        Self {
            mode,
            timebase_hz: model.timer_profile().timebase_hz,
            previous_host_ns: None,
            credit: 0,
            dropped_credit: 0,
        }
    }

    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn timebase_hz(&self) -> u64 {
        self.timebase_hz
    }

    /// Rebase on Pause/Resume, reset/load, or after explicit debugger execution.
    /// No guest state is reset and no paused wall time is silently simulated.
    pub fn rebase(&mut self) {
        self.previous_host_ns = None;
        self.credit = 0;
    }

    pub fn set_mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
        self.rebase();
    }

    /// Nominal wall time omitted by the bounded catch-up policy, cumulative.
    pub fn dropped_host_ns(&self) -> u64 {
        u64::try_from(self.dropped_credit / u128::from(self.timebase_hz)).unwrap_or(u64::MAX)
    }

    pub fn automatic_plan(&mut self, host_ns: u64, requested: usize) -> Result<PacePlan> {
        let mut plan = PacePlan {
            action: PaceAction::Run,
            boundary_budget: requested.min(MAX_HOST_SLICE_BOUNDARIES),
            elapsed_timing_budget: None,
            wait_ns: 0,
        };
        if self.mode == ExecutionMode::Deterministic {
            plan.action = PaceAction::ExplicitBudgetRequired;
            plan.boundary_budget = 0;
            return Ok(plan);
        }
        if self.mode == ExecutionMode::Turbo {
            return Ok(plan);
        }
        if let Some(previous) = self.previous_host_ns {
            let elapsed = host_ns.checked_sub(previous).ok_or_else(|| {
                CoreError::Other("host pacing clock moved backwards; rebase before resuming".into())
            })?;
            self.credit += i128::from(elapsed) * i128::from(self.timebase_hz);
            let cap = i128::from(MAX_CATCH_UP_HOST_US * 1_000) * i128::from(self.timebase_hz);
            if self.credit > cap {
                self.dropped_credit = self
                    .dropped_credit
                    .saturating_add((self.credit - cap) as u128);
                self.credit = cap;
            }
        }
        self.previous_host_ns = Some(host_ns);
        if self.credit < NS_PER_SECOND {
            plan.action = PaceAction::Wait;
            plan.boundary_budget = 0;
            let missing = NS_PER_SECOND.saturating_sub(self.credit);
            let wait = missing.saturating_add(i128::from(self.timebase_hz) - 1)
                / i128::from(self.timebase_hz);
            plan.wait_ns = u64::try_from(wait)
                .unwrap_or(u64::MAX)
                .min(HOST_SLICE_TARGET_US * 1_000);
            return Ok(plan);
        }
        let available = u64::try_from(self.credit / NS_PER_SECOND).unwrap_or(u64::MAX);
        plan.elapsed_timing_budget = Some(available);
        // One boundary usually costs at least one timing unit. The elapsed
        // target handles wider instructions and rare zero-time wake boundaries.
        plan.boundary_budget = plan
            .boundary_budget
            .min(usize::try_from(available).unwrap_or(usize::MAX));
        Ok(plan)
    }

    /// Charge successful scheduler progress only, never requested budget.
    /// Faults preserve the scheduler's poison contract; do not resume them by
    /// interpreting missing progress as a successful zero-cost instruction.
    pub fn account(&mut self, elapsed_timing_units: u64) {
        if self.mode == ExecutionMode::Interactive {
            self.credit = self
                .credit
                .saturating_sub(i128::from(elapsed_timing_units) * NS_PER_SECOND);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paced() -> Pacer {
        Pacer::for_model(DeviceModel::PcE500, ExecutionMode::Interactive)
    }

    #[test]
    fn nominal_pacing_uses_elapsed_timing_and_retains_fractional_credit() {
        let mut pacer = paced();
        assert_eq!(
            pacer.automatic_plan(0, 100_000).unwrap().action,
            PaceAction::Wait
        );
        assert_eq!(
            pacer
                .automatic_plan(1_000_000, 100_000)
                .unwrap()
                .boundary_budget,
            1024
        );
        pacer.account(1024);
        for ns in [1_000_100, 1_000_200, 1_000_300] {
            assert_eq!(
                pacer.automatic_plan(ns, 100).unwrap().action,
                PaceAction::Wait
            );
        }
        assert_eq!(
            pacer
                .automatic_plan(1_001_000, 100)
                .unwrap()
                .boundary_budget,
            1
        );
        pacer.account(1);
        assert_eq!(
            pacer.automatic_plan(1_001_000, 100).unwrap().action,
            PaceAction::Wait
        );
    }

    #[test]
    fn slow_host_backlog_is_bounded_and_reported_without_guest_time_skipping() {
        let mut pacer = paced();
        pacer.automatic_plan(0, 100).unwrap();
        let plan = pacer.automatic_plan(10_000_000_000, 1_000_000).unwrap();
        assert_eq!(plan.elapsed_timing_budget, Some(51_200));
        assert_eq!(pacer.dropped_host_ns(), 9_950_000_000);
        pacer.account(51_200);
        assert_eq!(
            pacer.automatic_plan(10_000_000_000, 100).unwrap().action,
            PaceAction::Wait
        );
    }

    #[test]
    fn instruction_overshoot_remains_debt_but_never_requests_a_long_blocking_sleep() {
        let mut pacer = paced();
        pacer.automatic_plan(0, 100).unwrap();
        pacer.account(10_240_000); // Ten nominal seconds from one atomic operation.
        let plan = pacer.automatic_plan(1_000_000, 100).unwrap();
        assert_eq!(plan.action, PaceAction::Wait);
        assert_eq!(plan.wait_ns, HOST_SLICE_TARGET_US * 1000);
        assert_eq!(
            pacer.automatic_plan(10_000_000_000, 100).unwrap().action,
            PaceAction::Wait
        );
        assert_eq!(pacer.dropped_host_ns(), 0);
        assert_eq!(
            pacer
                .automatic_plan(10_001_000_000, 100)
                .unwrap()
                .boundary_budget,
            100
        );
    }

    #[test]
    fn rebase_never_charges_paused_wall_time_or_old_turbo_work() {
        let mut pacer = paced();
        pacer.automatic_plan(0, 100).unwrap();
        pacer.account(99_999);
        pacer.rebase();
        assert_eq!(
            pacer.automatic_plan(10_000_000_000, 100).unwrap().action,
            PaceAction::Wait
        );
        assert_eq!(pacer.dropped_host_ns(), 0);
        pacer.set_mode(ExecutionMode::Turbo);
        pacer.account(u64::MAX);
        assert_eq!(
            pacer.automatic_plan(0, usize::MAX).unwrap().boundary_budget,
            MAX_HOST_SLICE_BOUNDARIES
        );
        pacer.set_mode(ExecutionMode::Interactive);
        assert_eq!(
            pacer.automatic_plan(42, 100).unwrap().action,
            PaceAction::Wait
        );
    }

    #[test]
    fn deterministic_mode_requires_explicit_budgets_and_ignores_host_clock() {
        let mut pacer = Pacer::for_model(DeviceModel::Iq7000, ExecutionMode::Deterministic);
        for now in [0, u64::MAX, 1] {
            let plan = pacer.automatic_plan(now, usize::MAX).unwrap();
            assert_eq!(plan.action, PaceAction::ExplicitBudgetRequired);
            assert_eq!(plan.boundary_budget, 0);
        }
    }

    #[test]
    fn backwards_interactive_clock_fails_without_changing_credit_or_epoch() {
        let mut pacer = paced();
        pacer.automatic_plan(100, 100).unwrap();
        let credit = pacer.credit;
        assert!(pacer
            .automatic_plan(99, 100)
            .unwrap_err()
            .to_string()
            .contains("backwards"));
        assert_eq!(pacer.previous_host_ns, Some(100));
        assert_eq!(pacer.credit, credit);
    }
}
