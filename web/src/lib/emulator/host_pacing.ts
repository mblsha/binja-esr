import { checkBoundaryBudget, limitedBudget, type SliceResult } from './bounded_step';

export type ExecutionMode = 'interactive' | 'turbo' | 'deterministic';
export type PacingStatus = {
	mode: ExecutionMode;
	nominal_timebase_hz: number;
	calibration: string;
	dropped_host_ns: number | bigint;
};
export type AutomaticResult = {
	plan: {
		action: 'run' | 'wait' | 'explicit_budget_required';
		boundary_budget: number;
		wait_ns: number | bigint;
	};
	slice?: SliceResult | null;
};

/** Rust owns pacing. JS only services input deadlines and yields to the host. */
export function automaticHostSlice(
	emulator: { automatic_slice(boundaries: number): AutomaticResult },
	boundaries: number,
	inputs: { limitBudget: (requested: number) => number; advance: (used: number) => void },
): number {
	checkBoundaryBudget(boundaries);
	if (typeof emulator.automatic_slice !== 'function')
		throw new Error('Pacing requires updated Rust/WASM; rebuild and reload');
	const budget = limitedBudget(boundaries, inputs.limitBudget);
	const result = emulator.automatic_slice(budget);
	if (result.plan.action === 'explicit_budget_required')
		throw new Error('Deterministic mode requires explicit Step or Function Runner budgets');
	const planned = result.plan.boundary_budget;
	const used = result.slice?.progress.boundary_budget_used ?? 0;
	const waitMs = Number(result.plan.wait_ns) / 1e6;
	if (
		!Number.isSafeInteger(planned) ||
		planned < 0 ||
		planned > budget ||
		!Number.isSafeInteger(used) ||
		used < 0 ||
		used > planned ||
		!Number.isFinite(waitMs) ||
		waitMs < 0 ||
		waitMs > 4 ||
		(result.plan.action === 'wait'
			? planned !== 0 || result.slice != null
			: result.plan.action !== 'run' || (planned > 0 && used === 0))
	) {
		throw new Error('Invalid automatic pacing progress from Rust/WASM');
	}
	// Input holds are defined in executed scheduler boundaries, not CPU timing,
	// host milliseconds, elapsed OFF time, or the requested budget.
	inputs.advance(used);
	return waitMs;
}
