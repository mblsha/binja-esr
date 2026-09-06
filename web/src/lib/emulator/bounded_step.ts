/** Host scheduling only. All machine boundaries still execute in Rust. */
export type SliceResult = {
	reason: 'boundary_budget' | 'host_yield';
	progress: {
		boundary_budget_used: number;
		instructions_retired: number | bigint;
		timing_units_advanced: number | bigint;
	};
};

export type SlicedEmulator = { run_slice(boundaries: number, hostMs: number): SliceResult };
export const HOST_SLICE_MS = 4;
export const yieldToHost = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export class ExecutionCancelled extends Error {
	constructor(public readonly boundaryBudgetUsed: number) {
		super(`Execution cancelled after ${boundaryBudgetUsed} scheduler boundaries of this step request`);
		this.name = 'ExecutionCancelled';
	}
}

export function checkBoundaryBudget(boundaries: number): void {
	if (!Number.isSafeInteger(boundaries) || boundaries < 0 || boundaries > 0xffffffff) {
		throw new Error('Scheduler-boundary budget must be an integer in 0..=4294967295');
	}
}

export function limitedBudget(requested: number, limit?: (requested: number) => number): number {
	const budget = limit?.(requested) ?? requested;
	if (!Number.isSafeInteger(budget) || budget < (requested > 0 ? 1 : 0) || budget > requested)
		throw new Error('Invalid input-deadline boundary budget');
	return budget;
}

export function runHostSlice(emulator: SlicedEmulator, boundaries: number): number {
	checkBoundaryBudget(boundaries);
	if (typeof emulator.run_slice !== 'function') {
		throw new Error('Bounded execution requires updated Rust/WASM exports; rebuild WASM and reload');
	}
	const result = emulator.run_slice(boundaries, HOST_SLICE_MS);
	const used = result.progress.boundary_budget_used;
	if (!Number.isSafeInteger(used) || used < 0 || used > boundaries) {
		throw new Error('Invalid boundary progress from Rust/WASM');
	}
	return used;
}

export async function stepBounded(
	emulator: SlicedEmulator,
	boundaries: number,
	options: {
		signal?: AbortSignal;
		limitBudget?: (requested: number) => number;
		onProgress?: (used: number) => void;
		yieldHost?: () => Promise<void>;
	} = {},
): Promise<number> {
	checkBoundaryBudget(boundaries);
	let completed = 0;
	do {
		if (options.signal?.aborted) throw new ExecutionCancelled(completed);
		if (completed < boundaries) {
			const used = runHostSlice(emulator, limitedBudget(boundaries - completed, options.limitBudget));
			completed += used;
			options.onProgress?.(used);
		}
		// Also yield after tiny/zero/final steps. A script repeatedly awaiting
		// already-resolved promises would otherwise starve worker messages.
		await (options.yieldHost ?? yieldToHost)();
	} while (completed < boundaries);
	return completed;
}
