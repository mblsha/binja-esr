import type { CallArtifacts } from '../debug/sc62015_eval_api';
import { checkBoundaryBudget, ExecutionCancelled, HOST_SLICE_MS, limitedBudget, yieldToHost } from './bounded_step';

export type FunctionStubRequest = {
	id: number;
	sequence: number;
	pc: number;
	regs: { name: string; value: number }[];
	flags: { name: string; value: number }[];
};

export type FunctionCallSlice = {
	state: 'running' | 'stub' | 'complete';
	steps: number;
	scheduler_boundaries: number;
	stub_request?: FunctionStubRequest | null;
};

export type ResumableCallEmulator = {
	call_function_begin(address: number, maxSteps: number, options: unknown): number;
	call_function_slice(id: number, boundaries: number, hostMs: number): FunctionCallSlice;
	call_function_apply_stub(id: number, sequence: number, patch: unknown): void;
	call_function_stub_failed(id: number, sequence: number, message: string): void;
	call_function_finish(id: number): string;
	call_function_cancel(id: number): string;
};

/** Keeps the same Rust machine alive across yields. Cancellation removes the
 * debugger sentinel, not guest writes, timers, or peripheral effects. */
export async function callBounded(
	emulator: ResumableCallEmulator,
	address: number,
	maxSteps: number,
	callOptions: unknown,
	controls: {
		signal?: AbortSignal;
		limitBudget?: (requested: number) => number;
		dispatchStub?: (request: FunctionStubRequest) => unknown | Promise<unknown>;
		onProgress?: (schedulerBoundaries: number, slice: FunctionCallSlice) => void;
		yieldHost?: () => Promise<void>;
	} = {},
): Promise<CallArtifacts> {
	checkBoundaryBudget(maxSteps);
	if (controls.signal?.aborted) throw new ExecutionCancelled(0);
	if (typeof emulator.call_function_begin !== 'function') {
		throw new Error('Resumable calls require updated Rust/WASM exports; rebuild WASM and reload');
	}
	const id = emulator.call_function_begin(address, maxSteps, callOptions);
	let owned = true;
	let previousBoundaries = 0;
	try {
		for (;;) {
			if (controls.signal?.aborted) {
				owned = false; // Rust restores scaffolding even if artifact encoding fails.
				return JSON.parse(emulator.call_function_cancel(id));
			}
			const slice = emulator.call_function_slice(id, limitedBudget(200_000, controls.limitBudget), HOST_SLICE_MS);
			controls.onProgress?.(slice.scheduler_boundaries - previousBoundaries, slice);
			previousBoundaries = slice.scheduler_boundaries;
			if (slice.state === 'complete') {
				owned = false;
				const result = JSON.parse(emulator.call_function_finish(id));
				// Repeated tiny calls must also allow the host to receive control messages.
				await (controls.yieldHost ?? yieldToHost)();
				return result;
			}
			if (slice.state === 'stub') {
				try {
					if (!slice.stub_request || !controls.dispatchStub) throw new Error('No function stub dispatcher');
					const patch = await controls.dispatchStub(slice.stub_request);
					if (!controls.signal?.aborted) emulator.call_function_apply_stub(id, slice.stub_request.sequence, patch);
				} catch (error) {
					if (!slice.stub_request) throw error;
					if (!controls.signal?.aborted)
						emulator.call_function_stub_failed(id, slice.stub_request.sequence, String(error));
				}
			}
			await (controls.yieldHost ?? yieldToHost)();
		}
	} catch (error) {
		if (owned) {
			try {
				emulator.call_function_cancel(id);
			} catch (cleanupError) {
				throw new Error(`Function call failed: ${String(error)}; cleanup also failed: ${String(cleanupError)}`);
			}
		}
		throw error;
	}
}
