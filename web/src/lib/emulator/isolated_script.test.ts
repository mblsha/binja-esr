import { describe, expect, it, vi } from 'vitest';
import { createEvalApi, type EvalApi } from '../debug/sc62015_eval_api';
import { createIsolatedScriptApi } from './isolated_script_api';
import { runIsolatedScript } from './isolated_script';
import {
	ASYNC_SCRIPT_METHODS,
	SYNC_SCRIPT_METHODS,
	encodeScriptValue,
	readScriptReply,
	writeScriptReply,
	SCRIPT_HEADER_BYTES,
} from './script_protocol';

const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
class FakeScriptWorker {
	onmessage: any = null;
	onerror: any = null;
	onmessageerror: any = null;
	postMessage = vi.fn();
	terminate = vi.fn();
	send(data: any) {
		this.onmessage?.({ data });
	}
	rpc(id: number, path: string, args: unknown[] = [], sync = false) {
		this.send({ type: 'rpc', id, path, args: encodeScriptValue(args), sync });
	}
}

describe('isolated script protocol', () => {
	it('covers the complete EvalApi surface without arbitrary property access', () => {
		const paths: string[] = [];
		const visit = (value: any, prefix = '') => {
			for (const [key, child] of Object.entries(value)) {
				const path = prefix + key;
				if (typeof child === 'function' || Array.isArray(child)) paths.push(path);
				else if (child && typeof child === 'object') visit(child, path + '.');
			}
		};
		visit(createEvalApi({} as any));
		expect(paths.sort()).toEqual(
			[...ASYNC_SCRIPT_METHODS, ...SYNC_SCRIPT_METHODS.filter((path) => path !== 'stub.read8')].sort(),
		);
	});
	it('does not silently turn invalid numeric arguments into null', () => {
		for (const value of [NaN, Infinity, -Infinity]) expect(() => encodeScriptValue([value])).toThrow('finite');
	});
	it('publishes replies with identity, rejects stale/errors and oversized UTF-8', () => {
		const buffer = new SharedArrayBuffer(SCRIPT_HEADER_BYTES + 200);
		writeScriptReply(buffer, 7, true, { value: 42 });
		expect(readScriptReply(buffer, 7)).toEqual({ value: 42 });
		expect(() => readScriptReply(buffer, 8)).toThrow('Stale');
		writeScriptReply(buffer, 8, false, 'not writable');
		expect(() => readScriptReply(buffer, 8)).toThrow('not writable');
		writeScriptReply(buffer, 9, true, '😀'.repeat(100));
		expect(() => readScriptReply(buffer, 9)).toThrow('capacity');
	});
	it('forwards live reads and runs only local stub/callback closures', async () => {
		let value = 1;
		const sync = vi.fn((path: string) => (path === 'stub' ? { id: 3, pc: 20, name: null } : value));
		const async = vi.fn(async () => 42);
		const remote = createIsolatedScriptApi({ sync, async });
		expect(remote.api.reg('A')).toBe(1);
		value = 2;
		expect(remote.api.reg('A')).toBe(2);
		remote.api.stub(20, null, (memory) => ({ regs: { A: memory.read8(10) } }));
		expect(remote.invoke('stub', 3, [[], []])).toMatchObject({ regs: [{ name: 'A', value: 2 }] });
		await remote.api.withProbe(
			20,
			() => {},
			async () => {},
		);
		expect(async).toHaveBeenCalledWith('withProbe', [20, 1, 2]);
		expect(() => remote.invoke('callback', 1, [])).toThrow('Stale');
	});
});

describe('script worker supervisor', () => {
	it('cleans up a script worker if API initialization fails', async () => {
		const worker = new FakeScriptWorker();
		await expect(
			runIsolatedScript({
				source: '',
				createApi: () => {
					throw new Error('init');
				},
				read8: () => 0,
				workerFactory: () => worker,
			}),
		).rejects.toThrow('init');
		expect(worker.terminate).toHaveBeenCalled();
	});
	it('keeps owner artifacts and rejects RPC after termination', async () => {
		const worker = new FakeScriptWorker();
		const controller = new AbortController();
		const api = createEvalApi({} as any);
		const result = runIsolatedScript({
			source: '',
			signal: controller.signal,
			createApi: () => api,
			read8: () => 0,
			workerFactory: () => worker,
		});
		worker.rpc(1, 'print', ['before loop'], true);
		await turn();
		controller.abort();
		worker.rpc(2, 'print', ['too late'], true);
		const output = await result;
		expect(output.prints).toEqual([{ index: 0, value: 'before loop' }]);
		expect(output.error).toContain('cancelled');
		expect(worker.terminate).toHaveBeenCalled();
	});
	it('waits for interrupted machine cleanup before acknowledging cancellation', async () => {
		const worker = new FakeScriptWorker();
		const controller = new AbortController();
		let cleanup!: () => void;
		let signal!: AbortSignal;
		let done = false;
		const api = createEvalApi({
			step: () =>
				new Promise<void>((resolve) => {
					cleanup = resolve;
				}),
		} as any);
		const result = runIsolatedScript({
			source: '',
			signal: controller.signal,
			createApi: (host) => {
				signal = host.signal;
				return api;
			},
			read8: () => 0,
			workerFactory: () => worker,
		}).then((value) => {
			done = true;
			return value;
		});
		worker.rpc(1, 'step', [100]);
		controller.abort();
		await turn();
		expect(signal.aborted).toBe(true);
		expect(done).toBe(false);
		cleanup();
		await result;
		expect(done).toBe(true);
	});
	it('cancels an infinite callback and awaits its owner finally block', async () => {
		const worker = new FakeScriptWorker();
		const controller = new AbortController();
		let cleaned = false;
		const api = {
			events: [],
			calls: [],
			prints: [],
			perfetto: {
				trace: async (_name: string, body: () => Promise<unknown>) => {
					try {
						await body();
					} finally {
						cleaned = true;
					}
				},
			},
		} as unknown as EvalApi;
		const result = runIsolatedScript({
			source: '',
			signal: controller.signal,
			createApi: () => api,
			read8: () => 0,
			workerFactory: () => worker,
		});
		worker.rpc(1, 'perfetto.trace', ['outer', 7]);
		expect(worker.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'callback', callback: 7 }));
		controller.abort();
		await result;
		expect(cleaned).toBe(true);
	});
	it('rejects unknown paths and does not run unawaited operations after completion', async () => {
		const worker = new FakeScriptWorker();
		const api = createEvalApi({} as any);
		const result = runIsolatedScript({ source: '', createApi: () => api, read8: () => 0, workerFactory: () => worker });
		worker.rpc(1, '__proto__.toString');
		await turn();
		expect(worker.postMessage).toHaveBeenLastCalledWith(expect.objectContaining({ ok: false }));
		worker.send({ type: 'done', resultJson: '42', error: null });
		expect((await result).resultJson).toBe('42');
	});
	it('handles worker failures and host deadlines without terminating the machine', async () => {
		for (const failure of ['error', 'messageerror', 'timeout']) {
			const worker = new FakeScriptWorker();
			const result = runIsolatedScript({
				source: '',
				createApi: () => createEvalApi({} as any),
				read8: () => 0,
				workerFactory: () => worker,
				maxHostMs: 10,
			});
			if (failure === 'error') worker.onerror({ message: 'crashed' });
			if (failure === 'messageerror') worker.onmessageerror({});
			expect((await result).error).toBeTruthy();
			expect(worker.terminate).toHaveBeenCalled();
		}
	});
});
