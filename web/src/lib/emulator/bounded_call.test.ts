import { describe, expect, it, vi } from 'vitest';
import { callBounded, type FunctionCallSlice } from './bounded_call';
import { HostInputs } from './host_inputs';

const finished = JSON.stringify({ report: { reason: 'returned', steps: 65 } });
const cancelled = JSON.stringify({ report: { reason: 'cancelled', steps: 64 } });
const status = (state: FunctionCallSlice['state'], boundaries = 64): FunctionCallSlice => ({
	state,
	steps: boundaries,
	scheduler_boundaries: boundaries,
	stub_request: state === 'stub' ? { id: 7, sequence: 1, pc: 0x10000, regs: [], flags: [] } : null,
});
const machine = () => ({
	call_function_begin: vi.fn(() => 3),
	call_function_slice: vi.fn(() => status('complete', 65)),
	call_function_finish: vi.fn(() => finished),
	call_function_cancel: vi.fn(() => cancelled),
	call_function_apply_stub: vi.fn(),
	call_function_stub_failed: vi.fn(),
});

describe('resumable Rust function calls', () => {
	it('limits call slices at input release boundaries and does not charge debugger stub actions', async () => {
		const emulator = machine();
		let boundaries = 0;
		const releaseAt: number[] = [];
		const inputs = new HostInputs((_contact, down) => {
			if (!down) releaseAt.push(boundaries);
		});
		inputs.set({ source: 'virtual', owner: 'test', contact: 1, down: true, minimumHold: 40 });
		inputs.set({ source: 'virtual', owner: 'test', contact: 1, down: false });
		emulator.call_function_slice.mockImplementationOnce(() => ({ ...status('stub', 0), steps: 1 }));
		emulator.call_function_slice.mockImplementation((_id?: number, budget?: number) => {
			boundaries += Math.min(budget!, 100 - boundaries);
			return { ...status(boundaries === 100 ? 'complete' : 'running', boundaries), steps: boundaries + 1 };
		});
		await callBounded(
			emulator,
			0x10000,
			1000,
			{},
			{
				limitBudget: inputs.limitBudget,
				onProgress: inputs.advance,
				dispatchStub: () => ({}),
				yieldHost: async () => {},
			},
		);
		expect(releaseAt).toEqual([40]);
	});
	it('yields between slices, accounts actual scheduler progress, and finishes once', async () => {
		const emulator = machine();
		emulator.call_function_slice.mockReturnValueOnce(status('running'));
		const yieldHost = vi.fn(async () => {});
		const onProgress = vi.fn();
		expect(await callBounded(emulator, 0x10000, 1000, {}, { yieldHost, onProgress })).toEqual(JSON.parse(finished));
		expect(onProgress.mock.calls.map(([used]) => used)).toEqual([64, 1]);
		expect(yieldHost).toHaveBeenCalledTimes(2); // Includes final/tiny calls.
		expect(emulator.call_function_finish).toHaveBeenCalledExactlyOnceWith(3);
		expect(emulator.call_function_cancel).not.toHaveBeenCalled();
	});

	it('cancels the owned call at the next yield and retains its partial artifacts', async () => {
		const emulator = machine();
		emulator.call_function_slice.mockReturnValue(status('running'));
		const controller = new AbortController();
		const result = await callBounded(
			emulator,
			0x10000,
			4_000_000_000,
			{},
			{
				signal: controller.signal,
				yieldHost: async () => controller.abort(),
			},
		);
		expect(result).toEqual(JSON.parse(cancelled));
		expect(emulator.call_function_slice).toHaveBeenCalledTimes(1);
		expect(emulator.call_function_cancel).toHaveBeenCalledExactlyOnceWith(3);
		expect(emulator.call_function_finish).not.toHaveBeenCalled();
	});

	it('hands stubs to the host after the WASM borrow ends, before another slice', async () => {
		const emulator = machine();
		emulator.call_function_slice.mockReturnValueOnce(status('stub', 0));
		const patch = { ret: { kind: 'retf' } };
		await callBounded(
			emulator,
			0x10000,
			1000,
			{},
			{
				dispatchStub: async (request) => {
					expect(request.id).toBe(7);
					expect(emulator.call_function_slice).toHaveBeenCalledTimes(1);
					expect(emulator.call_function_apply_stub).not.toHaveBeenCalled();
					return patch;
				},
				yieldHost: async () => {},
			},
		);
		expect(emulator.call_function_apply_stub).toHaveBeenCalledExactlyOnceWith(3, 1, patch);
	});

	it('does not apply a late stub response after cancellation', async () => {
		const emulator = machine();
		emulator.call_function_slice.mockReturnValueOnce(status('stub', 0));
		const controller = new AbortController();
		await callBounded(
			emulator,
			0x10000,
			1000,
			{},
			{
				signal: controller.signal,
				dispatchStub: async () => {
					controller.abort();
					return { ret: { kind: 'retf' } };
				},
				yieldHost: async () => {},
			},
		);
		expect(emulator.call_function_apply_stub).not.toHaveBeenCalled();
		expect(emulator.call_function_slice).toHaveBeenCalledTimes(1);
		expect(emulator.call_function_cancel).toHaveBeenCalledExactlyOnceWith(3);
	});

	it('reports a callback fault through Rust and cleans up on unexpected host errors', async () => {
		const emulator = machine();
		emulator.call_function_slice.mockReturnValueOnce(status('stub', 0));
		await callBounded(
			emulator,
			0x10000,
			1000,
			{},
			{
				dispatchStub: () => {
					throw new Error('callback failed');
				},
				yieldHost: async () => {},
			},
		);
		expect(emulator.call_function_stub_failed).toHaveBeenCalledExactlyOnceWith(3, 1, 'Error: callback failed');
		emulator.call_function_slice.mockImplementation(() => {
			throw new Error('clock failed');
		});
		await expect(callBounded(emulator, 0x10000, 1000, {})).rejects.toThrow('clock failed');
		expect(emulator.call_function_cancel).toHaveBeenCalledExactlyOnceWith(3);
	});

	it('does not cancel an already-restored session again after artifact encoding fails', async () => {
		const emulator = machine();
		emulator.call_function_finish.mockImplementation(() => {
			throw new Error('encoding failed');
		});
		await expect(callBounded(emulator, 0x10000, 1000, {})).rejects.toThrow('encoding failed');
		expect(emulator.call_function_cancel).not.toHaveBeenCalled();
	});

	it('rejects invalid budgets or pre-cancellation without starting a call', async () => {
		const emulator = machine();
		for (const bad of [-1, 1.5, NaN, Infinity, 0x100000000]) {
			await expect(callBounded(emulator, 0x10000, bad, {})).rejects.toThrow('Scheduler-boundary budget');
		}
		const controller = new AbortController();
		controller.abort();
		await expect(callBounded(emulator, 0x10000, 1000, {}, { signal: controller.signal })).rejects.toThrow(
			'Execution cancelled',
		);
		expect(emulator.call_function_begin).not.toHaveBeenCalled();
	});
});
