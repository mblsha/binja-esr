import type { EvalApi } from '../debug/sc62015_eval_api';
import type { FunctionStubRequest } from './bounded_call';
import {
	ASYNC_SCRIPT_METHODS,
	SYNC_SCRIPT_METHODS,
	decodeScriptValue,
	encodeScriptValue,
	SCRIPT_HEADER_BYTES,
	SCRIPT_REPLY_BYTES,
	SCRIPT_MAX_HOST_MS,
	writeScriptReply,
} from './script_protocol';

type ScriptWorker = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type IsolatedScriptHost = {
	signal: AbortSignal;
	dispatchStub(request: FunctionStubRequest): Promise<unknown>;
};

/** The emulator and EvalApi stay on this side. Killing a runaway script rejects
 * its callbacks and drains cooperative machine operations before Stop can ack.
 * This is a responsiveness boundary, not a security sandbox for hostile code. */
export async function runIsolatedScript(options: {
	source: string;
	signal?: AbortSignal;
	createApi(host: IsolatedScriptHost): EvalApi;
	read8(addr: number): number;
	workerFactory?: () => ScriptWorker;
	maxHostMs?: number;
	onRequest?: (path: string) => void;
}) {
	if (typeof SharedArrayBuffer === 'undefined' || (!options.workerFactory && !globalThis.crossOriginIsolated)) {
		throw new Error(
			'Function Runner requires cross-origin isolated workers (COOP/COEP headers). Scripts will not run on the machine or UI thread.',
		);
	}
	const controller = new AbortController();
	const mailbox = new SharedArrayBuffer(SCRIPT_HEADER_BYTES + SCRIPT_REPLY_BYTES);
	const worker =
		options.workerFactory?.() ?? new Worker(new URL('./script.worker.ts', import.meta.url), { type: 'module' });
	let accepting = true;
	let nextCallback = 0;
	const callbacks = new Map<number, { resolve(value: any): void; reject(error: Error): void }>();
	const jobs = new Set<Promise<void>>();
	const cleanupErrors = new Set<string>();
	let finish!: (result: { resultJson: string | null; error: string | null }) => void;
	const completion = new Promise<{ resultJson: string | null; error: string | null }>((resolve) => {
		finish = resolve;
	});
	const invoke = (kind: string, callback: number, args: unknown[]): Promise<any> => {
		if (!accepting) return Promise.reject(new Error('Script cancelled'));
		const id = ++nextCallback;
		return new Promise((resolve, reject) => {
			callbacks.set(id, { resolve, reject });
			try {
				worker.postMessage({ type: 'callback', id, kind, callback, args: encodeScriptValue(args) });
			} catch (error) {
				callbacks.delete(id);
				reject(error);
			}
		});
	};
	let api: EvalApi;
	try {
		api = options.createApi({
			signal: controller.signal,
			dispatchStub: (request) => invoke('stub', request.id, [request.regs, request.flags]),
		});
	} catch (error) {
		worker.terminate();
		throw error;
	}
	const end = (resultJson: string | null, error: string | null) => {
		if (!accepting) return;
		accepting = false;
		worker.terminate(); // Only user code; never the machine-owning worker.
		controller.abort();
		for (const callback of callbacks.values()) callback.reject(new Error(error ?? 'Script finished'));
		callbacks.clear();
		finish({ resultJson, error });
	};
	const abort = () => end(null, 'Script cancelled; machine retained and pending execution cancelled.');
	const dispatch = (path: string, args: any[]): any => {
		if (path === 'stub.read8') return options.read8(args[0]);
		if (path === 'stub') {
			const stub = api.stub(args[0], args[1], () => {
				throw new Error('Remote stub must use isolated dispatch');
			});
			return { id: stub.id, pc: stub.pc, name: stub.name };
		}
		if (path === 'withProbe')
			return api.withProbe(
				args[0],
				(sample) => invoke('callback', args[1], [sample]),
				() => invoke('callback', args[2], []),
			);
		if (path === 'perfetto.trace') return api.perfetto.trace(args[0], () => invoke('callback', args[1], []));
		let owner: any = api;
		const parts = path.split('.');
		for (const part of parts.slice(0, -1)) owner = owner[part];
		const value = owner[parts.at(-1)!];
		return typeof value === 'function' ? value.apply(owner, args) : value;
	};
	const handleRequest = async (message: any) => {
		let ok = true;
		let result: unknown;
		try {
			const methods: readonly string[] = message.sync ? SYNC_SCRIPT_METHODS : ASYNC_SCRIPT_METHODS;
			if (!Number.isSafeInteger(message.id) || message.id <= 0 || !methods.includes(message.path))
				throw new Error('Invalid script RPC request');
			const args = decodeScriptValue(message.args);
			if (!Array.isArray(args)) throw new Error('Script RPC arguments must be an array');
			options.onRequest?.(message.path);
			result = await dispatch(message.path, args);
		} catch (error) {
			ok = false;
			result = error instanceof Error ? error.message : String(error);
			if (!accepting && cleanupErrors.size < 64) cleanupErrors.add(String(result));
		}
		if (!accepting) return; // A terminated script may never submit a late mutation.
		if (message.sync) writeScriptReply(mailbox, message.id, ok, result);
		else {
			try {
				worker.postMessage({ type: 'rpc-result', id: message.id, ok, payload: encodeScriptValue(result) });
			} catch (error) {
				end(null, `Script reply failed: ${String(error)}`);
			}
		}
	};
	worker.onmessage = ({ data }) => {
		if (!accepting) return;
		if (data?.type === 'rpc') {
			if (jobs.size >= 64) {
				end(null, 'Too many unawaited script requests; execution cancelled.');
				return;
			}
			const job = handleRequest(data);
			jobs.add(job);
			void job.then(
				() => jobs.delete(job),
				(error) => {
					jobs.delete(job);
					end(null, `Script RPC failed: ${String(error)}`);
				},
			);
		} else if (data?.type === 'callback-result') {
			const callback = callbacks.get(data.id);
			if (!callback) return;
			callbacks.delete(data.id);
			try {
				const value = decodeScriptValue(data.payload);
				if (data.ok) callback.resolve(value);
				else callback.reject(new Error(String(value)));
			} catch (error) {
				callback.reject(error as Error);
			}
		} else if (data?.type === 'done') {
			end(
				data.resultJson,
				jobs.size ? 'Script finished with unawaited machine operations; execution cancelled.' : data.error,
			);
		} else if (data?.type === 'fatal') {
			end(null, String(data.error));
		}
	};
	worker.onerror = (event) => {
		event.preventDefault?.();
		end(null, `Script worker failed: ${event.message || 'unknown worker failure'}`);
	};
	worker.onmessageerror = () => end(null, 'Script worker message could not be decoded');
	options.signal?.addEventListener('abort', abort, { once: true });
	const timer = setTimeout(
		() => end(null, 'Script exceeded its 270-second host limit; execution cancelled.'),
		options.maxHostMs ?? SCRIPT_MAX_HOST_MS,
	);
	try {
		if (options.signal?.aborted) abort();
		else worker.postMessage({ type: 'start', source: options.source, mailbox });
		const result = await completion;
		// Includes interrupted step/call, callback scopes and their finally blocks.
		// No new RPC is accepted once completion starts.
		await Promise.allSettled([...jobs]);
		if (cleanupErrors.size) result.error = [result.error, ...cleanupErrors].filter(Boolean).join('\n');
		return { events: api.events, calls: api.calls, prints: api.prints, ...result };
	} finally {
		clearTimeout(timer);
		options.signal?.removeEventListener('abort', abort);
		worker.terminate();
	}
}
