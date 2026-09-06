import { runUserJs } from '../debug/run_user_js';
import { Reg, Flag } from '../debug/sc62015_eval_api';
import { IOCS } from '../debug/iocs';
import { createIsolatedScriptApi } from './isolated_script_api';
import { decodeScriptValue, encodeScriptValue, readScriptReply, SCRIPT_SYNC_TIMEOUT_MS } from './script_protocol';

let nextRequest = 0;
let failed = false;
let mailbox: SharedArrayBuffer;
let remote: ReturnType<typeof createIsolatedScriptApi>;
const pending = new Map<number, { resolve(value: any): void; reject(error: Error): void }>();

self.onmessage = async ({ data }) => {
	if (data.type === 'rpc-result') {
		const request = pending.get(data.id);
		if (!request) return;
		pending.delete(data.id);
		try {
			if (data.ok) request.resolve(decodeScriptValue(data.payload));
			else request.reject(new Error(decodeScriptValue(data.payload)));
		} catch (error) {
			request.reject(error as Error);
		}
		return;
	}
	if (data.type === 'callback') {
		try {
			const result = await remote.invoke(data.kind, data.callback, decodeScriptValue(data.args));
			self.postMessage({ type: 'callback-result', id: data.id, ok: true, payload: encodeScriptValue(result) });
		} catch (error) {
			self.postMessage({ type: 'callback-result', id: data.id, ok: false, payload: encodeScriptValue(String(error)) });
		}
		return;
	}
	if (data.type !== 'start' || remote) return;
	mailbox = data.mailbox;
	remote = createIsolatedScriptApi({
		sync(path, args) {
			if (failed) throw new Error('Script RPC is closed after a timeout');
			const id = ++nextRequest;
			const payload = encodeScriptValue(args); // Execute user getters only in this worker.
			const header = new Int32Array(mailbox, 0, 3);
			Atomics.store(header, 0, 0);
			self.postMessage({ type: 'rpc', id, path, args: payload, sync: true });
			if (Atomics.wait(header, 0, 0, SCRIPT_SYNC_TIMEOUT_MS) === 'timed-out') {
				failed = true; // Never reuse a mailbox that might receive a late reply.
				self.postMessage({ type: 'fatal', error: 'Machine RPC timed out; state is unconfirmed' });
				throw new Error('Machine RPC timed out; state is unconfirmed');
			}
			return readScriptReply(mailbox, id);
		},
		async(path, args) {
			const id = ++nextRequest;
			const payload = encodeScriptValue(args);
			return new Promise((resolve, reject) => {
				pending.set(id, { resolve, reject });
				self.postMessage({ type: 'rpc', id, path, args: payload, sync: false });
			});
		},
	});
	let resultJson: string | null = null;
	let error: string | null = null;
	try {
		const result = await runUserJs(data.source, remote.api, Reg, Flag, IOCS);
		resultJson = encodeScriptValue(result);
	} catch (caught) {
		error = caught instanceof Error ? caught.message : String(caught);
	}
	self.postMessage({ type: 'done', resultJson, error });
};
