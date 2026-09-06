import { describe, expect, it, vi } from 'vitest';
import { automaticHostSlice, type AutomaticResult, type PacingStatus } from './host_pacing';

const progress = (used: number) => ({ boundary_budget_used: used, instructions_retired: 0, timing_units_advanced: 0 });
describe('automatic Rust pacing adapter', () => {
	it('speeds up buffered scan work with a bounded slice and rebases afterwards', () => {
		const emu = {
			pacing_status: (): PacingStatus => ({
				mode: 'interactive',
				calibration: 'test',
				nominal_timebase_hz: 1,
				dropped_host_ns: 0,
			}),
			run_slice: vi.fn(() => ({ reason: 'host_yield' as const, progress: progress(3) })),
			rebase_pacing: vi.fn(),
			automatic_slice: vi.fn(),
		};
		const inputs = { limitBudget: () => 7, typingBoostBudget: () => 5, advance: vi.fn() };
		expect(automaticHostSlice(emu, 100, inputs, true)).toBe(0);
		expect(emu.run_slice).toHaveBeenCalledWith(5, 4);
		expect(inputs.advance).toHaveBeenCalledWith(3);
		expect(emu.rebase_pacing).toHaveBeenCalledOnce();
		expect(emu.automatic_slice).not.toHaveBeenCalled();
	});
	it('does not boost explicit deterministic execution or idle/held keys', () => {
		for (const mode of ['interactive', 'turbo', 'deterministic'] as const) {
			const emu = {
				pacing_status: (): PacingStatus => ({ mode, calibration: 'test', nominal_timebase_hz: 1, dropped_host_ns: 0 }),
				run_slice: vi.fn(),
				rebase_pacing: vi.fn(),
				automatic_slice: vi.fn(
					(): AutomaticResult => ({
						plan: {
							action: mode === 'deterministic' ? 'explicit_budget_required' : 'wait',
							boundary_budget: 0,
							wait_ns: 4e6,
						},
					}),
				),
			};
			const inputs = {
				limitBudget: (n: number) => n,
				typingBoostBudget: () => (mode === 'interactive' ? 0 : 5),
				advance: vi.fn(),
			};
			if (mode === 'deterministic') expect(() => automaticHostSlice(emu, 10, inputs, true)).toThrow('explicit Step');
			else expect(automaticHostSlice(emu, 10, inputs, true)).toBe(4);
			expect(emu.run_slice).not.toHaveBeenCalled();
			expect(emu.rebase_pacing).not.toHaveBeenCalled();
		}
	});
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
