import type { EvalApi } from './sc62015_eval_api';

export async function runUserJs(
	source: string,
	api: EvalApi,
	Reg: unknown,
	Flag: unknown,
	IOCS: unknown
): Promise<unknown> {
	// Best-effort sandbox: shadow common escape hatches, but this is not a hard security boundary.
	const prelude = 'const Function = undefined; const AsyncFunction = undefined;\n';
	const runner = new Function(
		'e',
		'Reg',
		'Flag',
		'IOCS',
		'globalThis',
		'self',
		'window',
		'document',
		'fetch',
		'XMLHttpRequest',
		'WebSocket',
		'Worker',
		'importScripts',
		'postMessage',
		'onmessage',
		'MessageChannel',
		'setTimeout',
		'setInterval',
		`"use strict";\n${prelude}return (async () => {\n${source}\n})();`
	);
	if (typeof Reg === 'object' && Reg !== null) {
		try {
			Object.freeze(Reg);
		} catch {
			/* ignore freeze errors */
		}
	}
	if (typeof Flag === 'object' && Flag !== null) {
		try {
			Object.freeze(Flag);
		} catch {
			/* ignore freeze errors */
		}
	}
	if (typeof IOCS === 'object' && IOCS !== null) {
		try {
			Object.freeze(IOCS);
		} catch {
			/* ignore freeze errors */
		}
	}
	return runner(
		api,
		Reg,
		Flag,
		IOCS,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined
	);
}
