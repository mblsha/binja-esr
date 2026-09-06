import { describe, expect, it, vi } from 'vitest';
import { ExecutionCancelled, runHostSlice, stepBounded, type SliceResult } from './bounded_step';

function result(used: number): SliceResult {
	return {
		reason: 'host_yield',
		progress: { boundary_budget_used: used, instructions_retired: 0, timing_units_advanced: 0 },
	};
}

describe('bounded Rust/WASM stepping', () => {
	it('accounts for returned boundary progress, not requested or retired counts', async () => {
		const run_slice = vi.fn((n: number) => result(Math.min(n, 64)));
		const onProgress = vi.fn();
		const yieldHost = vi.fn(async () => {});
		expect(await stepBounded({ run_slice }, 130, { onProgress, yieldHost })).toBe(130);
		expect(run_slice.mock.calls.map(([n]) => n)).toEqual([130, 66, 2]);
		expect(onProgress.mock.calls.map(([n]) => n)).toEqual([64, 64, 2]);
		expect(yieldHost).toHaveBeenCalledTimes(3);
	});

	it('processes host tasks even after tiny and zero steps', async () => {
		for (const boundaries of [0, 1]) {
			let hostRan = false;
			setTimeout(() => {
				hostRan = true;
			}, 0);
			await stepBounded({ run_slice: (n) => result(n) }, boundaries);
			expect(hostRan).toBe(true);
		}
	});

	it('cancels at the next yield and never executes the remaining budget', async () => {
		const controller = new AbortController();
		const run_slice = vi.fn(() => result(64));
		const run = stepBounded({ run_slice }, 100_000, {
			signal: controller.signal,
			yieldHost: async () => {
				controller.abort();
			},
		});
		await expect(run).rejects.toMatchObject({ name: 'ExecutionCancelled', boundaryBudgetUsed: 64 });
		expect(run_slice).toHaveBeenCalledTimes(1);
	});

	it('does not enter WASM for an already cancelled request', async () => {
		const controller = new AbortController();
		controller.abort();
		const run_slice = vi.fn();
		await expect(stepBounded({ run_slice }, 10, { signal: controller.signal })).rejects.toBeInstanceOf(
			ExecutionCancelled,
		);
		expect(run_slice).not.toHaveBeenCalled();
	});

	it('can yield with zero progress without spinning in the worker turn', async () => {
		const run_slice = vi.fn().mockReturnValueOnce(result(0)).mockReturnValueOnce(result(2));
		const yieldHost = vi.fn(async () => {});
		expect(await stepBounded({ run_slice }, 2, { yieldHost })).toBe(2);
		expect(yieldHost).toHaveBeenCalledTimes(2);
	});

	it('rejects bad budgets and stale WASM instead of falling back to unbounded execution', async () => {
		const run_slice = vi.fn();
		for (const bad of [-1, 1.5, Infinity, NaN, 0x100000000]) {
			await expect(stepBounded({ run_slice }, bad)).rejects.toThrow('Scheduler-boundary budget');
		}
		expect(run_slice).not.toHaveBeenCalled();
		expect(() => runHostSlice({ step: vi.fn() } as any, 1)).toThrow('updated Rust/WASM');
		for (const used of [-1, 3, NaN]) {
			expect(() => runHostSlice({ run_slice: () => result(used) }, 2)).toThrow('Invalid boundary progress');
		}
	});
});
