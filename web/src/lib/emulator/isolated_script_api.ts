import type { EvalApi } from '../debug/sc62015_eval_api';
import { createStubDispatcher } from '../debug/sc62015_stub_dispatch';
import { ASYNC_SCRIPT_METHODS, SYNC_SCRIPT_METHODS } from './script_protocol';

export interface ScriptRpc {
	sync(path: string, args: unknown[]): any;
	async(path: string, args: unknown[]): Promise<any>;
}

/** Only user code, closures, and stub normalization live here. Register and
 * memory reads always ask the owner; there is no cached/fabricated machine. */
export function createIsolatedScriptApi(rpc: ScriptRpc) {
	const callbacks = new Map<number, (...args: any[]) => unknown>();
	let nextCallback = 0;
	const keep = (callback: (...args: any[]) => unknown) => {
		const id = ++nextCallback;
		callbacks.set(id, callback);
		return id;
	};
	const dispatcher = createStubDispatcher({ read8: (addr) => rpc.sync('stub.read8', [addr]) });
	const api: any = {};
	for (const path of [...SYNC_SCRIPT_METHODS, ...ASYNC_SCRIPT_METHODS]) {
		if (['stub', 'stub.read8', 'clearStubs', 'withProbe', 'perfetto.trace'].includes(path)) continue;
		if (['calls', 'events', 'prints', 'perfettoTraces'].includes(path)) {
			Object.defineProperty(api, path, { get: () => rpc.sync(path, []) });
			continue;
		}
		const parts = path.split('.');
		let target = api;
		for (const part of parts.slice(0, -1)) target = target[part] ??= {};
		target[parts.at(-1)!] = (...args: unknown[]) =>
			(SYNC_SCRIPT_METHODS as readonly string[]).includes(path) ? rpc.sync(path, args) : rpc.async(path, args);
	}
	api.stub = (pc: number, name: string | null, handler: any) => {
		if (typeof handler !== 'function') throw new Error('Stub handler must be a function');
		const registration = { ...rpc.sync('stub', [pc, name]), handler };
		dispatcher.registerStub(registration);
		return registration;
	};
	api.clearStubs = () => {
		rpc.sync('clearStubs', []);
		dispatcher.clearStubs();
	};
	api.withProbe = async (pc: number, handler: any, body: any) => {
		const handlerId = keep(handler);
		const bodyId = keep(body);
		try {
			return await rpc.async('withProbe', [pc, handlerId, bodyId]);
		} finally {
			callbacks.delete(handlerId);
			callbacks.delete(bodyId);
		}
	};
	api.perfetto = {
		trace: async (name: string, body: any) => {
			const id = keep(body);
			try {
				return await rpc.async('perfetto.trace', [name, id]);
			} finally {
				callbacks.delete(id);
			}
		},
	};
	return {
		api: api as EvalApi,
		invoke(kind: string, id: number, args: any[]) {
			if (kind === 'stub') return dispatcher.dispatch(id, args[0], args[1]);
			const callback = callbacks.get(id);
			if (!callback) throw new Error('Stale script callback');
			return callback(...args);
		},
	};
}
