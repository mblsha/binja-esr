/** The shared buffer is an RPC reply mailbox, never emulated RAM. Only the
 * disposable script worker may wait on it; machine and UI workers never wait. */
export const SCRIPT_REPLY_BYTES = 16 * 1024 * 1024;
export const SCRIPT_HEADER_BYTES = 12;
export const SCRIPT_SYNC_TIMEOUT_MS = 30_000;
export const SCRIPT_MAX_HOST_MS = 270_000;

export const SYNC_SCRIPT_METHODS = [
	'reg',
	'flag',
	'assert',
	'print',
	'last',
	'calls',
	'events',
	'prints',
	'perfettoTraces',
	'stub',
	'clearStubs',
	'stub.read8',
] as const;
export const ASYNC_SCRIPT_METHODS = [
	'reset',
	'step',
	'call',
	'withProbe',
	'perfetto.trace',
	'memory.read',
	'memory.write',
	'lcd.text',
	'lcd.textString',
	'lcd.pixels',
	'lcd.capture',
	'lcd.assertCalendarMonth',
	'keyboard.press',
	'keyboard.release',
	'keyboard.tap',
	'keyboard.injectEvent',
	'keys.tap',
	'keys.event.press',
	'keys.event.release',
	'keys.event.tap',
	'keys.phys.press',
	'keys.phys.release',
	'keys.phys.tap',
	'keys.app.tap',
	'onKey.press',
	'onKey.release',
	'onKey.tap',
	'pclinkSerial.serve',
	'wait.lcdStable',
	'wait.screenChange',
	'wait.textIncludes',
	'proof.metadata',
	'proof.metadataYaml',
	'iocs.putc',
	'iocs.text',
	'iocs.putcXY',
] as const;

export function encodeScriptValue(value: unknown): string {
	const json =
		JSON.stringify(value, (_key, item) => {
			if (typeof item === 'number' && !Number.isFinite(item)) throw new Error('Script RPC numbers must be finite');
			if (typeof item === 'function' || typeof item === 'symbol')
				throw new Error('Script RPC data cannot contain functions or symbols');
			return typeof item === 'bigint' ? item.toString() : item;
		}) ?? 'null';
	checkPayloadSize(json);
	return json;
}

export function decodeScriptValue(json: string): any {
	if (typeof json !== 'string') throw new Error('Invalid script RPC payload');
	checkPayloadSize(json);
	return JSON.parse(json);
}

function checkPayloadSize(json: string): void {
	if (json.length > SCRIPT_REPLY_BYTES || new TextEncoder().encode(json).length > SCRIPT_REPLY_BYTES)
		throw new Error('Script RPC payload exceeds the 16 MiB limit');
}

export function writeScriptReply(buffer: SharedArrayBuffer, id: number, ok: boolean, value: unknown): void {
	const header = new Int32Array(buffer, 0, 3);
	let bytes: Uint8Array;
	try {
		bytes = new TextEncoder().encode(encodeScriptValue(value));
		if (bytes.length > buffer.byteLength - SCRIPT_HEADER_BYTES)
			throw new Error('Script RPC reply exceeds mailbox capacity');
	} catch (error) {
		ok = false;
		bytes = new TextEncoder().encode(JSON.stringify(String(error)));
	}
	new Uint8Array(buffer, SCRIPT_HEADER_BYTES, bytes.length).set(bytes);
	Atomics.store(header, 1, id);
	Atomics.store(header, 2, bytes.length);
	// Publish payload/identity before waking the sole reader.
	Atomics.store(header, 0, ok ? 1 : 2);
	Atomics.notify(header, 0);
}

export function readScriptReply(buffer: SharedArrayBuffer, id: number): any {
	const header = new Int32Array(buffer, 0, 3);
	if (Atomics.load(header, 1) !== id) throw new Error('Stale script RPC reply');
	const length = Atomics.load(header, 2);
	if (length < 0 || length > buffer.byteLength - SCRIPT_HEADER_BYTES)
		throw new Error('Invalid script RPC reply length');
	// Browser TextDecoder rejects shared views. Copy the published reply before
	// decoding; only this client can request the next write to the mailbox.
	const value = decodeScriptValue(
		new TextDecoder().decode(new Uint8Array(buffer, SCRIPT_HEADER_BYTES, length).slice()),
	);
	if (Atomics.load(header, 0) !== 1) throw new Error(String(value));
	return value;
}
