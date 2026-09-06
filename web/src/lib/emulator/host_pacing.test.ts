import { describe, expect, it, vi } from 'vitest';
import { automaticHostSlice, type AutomaticResult } from './host_pacing';

const progress = (used: number) => ({ boundary_budget_used: used, instructions_retired: 0, timing_units_advanced: 0 });
describe('automatic Rust pacing adapter', () => {
	it('limits input deadlines and charges only actual boundaries, including OFF', () => {
		const automatic_slice = vi.fn(
			(): AutomaticResult => ({
				plan: { action: 'run', boundary_budget: 7, wait_ns: 0 },
				slice: { reason: 'host_yield', progress: progress(3) },
			}),
		);
		const inputs = { limitBudget: () => 7, advance: vi.fn() };
		expect(automaticHostSlice({ automatic_slice }, 100, inputs)).toBe(0);
		expect(automatic_slice).toHaveBeenCalledWith(7);
		expect(inputs.advance).toHaveBeenCalledWith(3);
	});
	it('waits without expiring input and refuses an automatic deterministic loop', () => {
		const inputs = { limitBudget: (n: number) => n, advance: vi.fn() };
		const result: AutomaticResult = { plan: { action: 'wait', boundary_budget: 0, wait_ns: 4_000_000n }, slice: null };
		expect(automaticHostSlice({ automatic_slice: () => result }, 10, inputs)).toBe(4);
		expect(inputs.advance).toHaveBeenCalledWith(0);
		result.plan.action = 'explicit_budget_required';
		expect(() => automaticHostSlice({ automatic_slice: () => result }, 10, inputs)).toThrow('explicit Step');
	});
	it('fails closed on bad progress or a long blocking wait', () => {
		for (const [planned, used, wait] of [
			[11, 1, 0],
			[10, 11, 0],
			[10, 0, 0],
			[10, 1, 5e6],
			[10, 1, NaN],
		]) {
			const inputs = { limitBudget: (n: number) => n, advance: vi.fn() };
			const result: AutomaticResult = {
				plan: { action: 'run', boundary_budget: planned, wait_ns: wait },
				slice: { reason: 'host_yield', progress: progress(used) },
			};
			expect(() => automaticHostSlice({ automatic_slice: () => result }, 10, inputs)).toThrow('Invalid automatic');
			expect(inputs.advance).not.toHaveBeenCalled();
		}
	});
});
