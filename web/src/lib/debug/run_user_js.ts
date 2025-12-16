import type { EvalApi } from './sc62015_eval_api';

function freezeIfObject(value: unknown): void {
	if (typeof value !== 'object' || value === null) return;
	try {
		Object.freeze(value);
	} catch {
		/* ignore freeze errors */
	}
}

export async function runUserJs(
	source: string,
	api: EvalApi,
	Reg: unknown,
	Flag: unknown,
	IOCS: unknown,
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
		`"use strict";\n${prelude}return (async () => {\n${source}\n})();`,
	);
	freezeIfObject(Reg);
	freezeIfObject(Flag);
	freezeIfObject(IOCS);
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
		undefined,
	);
}
