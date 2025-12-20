import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { createEvalApi, Flag, Reg, type EmulatorAdapter } from '../src/lib/debug/sc62015_eval_api';
import { IOCS } from '../src/lib/debug/iocs';
import { runUserJs } from '../src/lib/debug/run_user_js';
import { normalizeRomModel, romBasename, type RomModel } from '../src/lib/rom_model';

import initWasm, * as wasm from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';

type RunnerArgs = {
	model: RomModel | null;
	romPath: string | null;
	scriptPath: string | null;
	evalSource: string | null;
	stdin: boolean;
};

function usage() {
	return `
Usage:
  npm run fnr:cli -- [options] <script.js>
  npm run fnr:cli -- [options] --eval "<js>"
  cat script.js | npm run fnr:cli -- [options] --stdin

Options:
  --model <iq-7000|pc-e500>   ROM preset (default: Rust runtime default)
  --rom <path>               Explicit ROM path (overrides --model)
  --eval <js>                Inline script (async JS)
  --stdin                    Read script from stdin
  --help                     Show this help

Output:
  Prints a JSON object compatible with FunctionRunnerOutput (events/calls/prints/resultJson/error).
`;
}

function die(msg: string): never {
	console.error(msg);
	process.exit(2);
}

async function readAllStdin(): Promise<string> {
	const chunks: Buffer[] = [];
	for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
	return Buffer.concat(chunks).toString('utf8');
}

function safeJson(value: unknown): string {
	return JSON.stringify(value, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
}

function parseArgs(argv: string[]): RunnerArgs {
	let model: RomModel | null = null;
	let romPath: string | null = null;
	let evalSource: string | null = null;
	let stdin = false;
	let scriptPath: string | null = null;

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === '--help' || arg === '-h') {
			console.log(usage().trimEnd());
			process.exit(0);
		}
		if (arg === '--model') {
			const next = argv[++i];
			if (!next) die('error: --model requires a value');
			const parsed = normalizeRomModel(next);
			if (!parsed) die(`error: unknown --model '${next}' (expected: iq-7000|pc-e500)`);
			model = parsed;
			continue;
		}
		if (arg === '--rom') {
			const next = argv[++i];
			if (!next) die('error: --rom requires a path');
			romPath = next;
			continue;
		}
		if (arg === '--eval') {
			const next = argv[++i];
			if (next === undefined) die('error: --eval requires JS source');
			evalSource = next;
			continue;
		}
		if (arg === '--stdin') {
			stdin = true;
			continue;
		}
		if (arg.startsWith('-')) die(`error: unknown flag '${arg}'`);
		if (scriptPath) die('error: multiple script paths provided');
		scriptPath = arg;
	}

	return { model, romPath, scriptPath, evalSource, stdin };
}

async function ensureWasmInitialized(): Promise<void> {
	const wasmPath = fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url));
	let bytes: Uint8Array;
	try {
		bytes = new Uint8Array(await readFile(wasmPath));
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		die(
			`error: missing WASM artifact at ${wasmPath}\n` +
				`Run: cd web && npm install && npm run wasm:build\n` +
				`(${msg})`,
		);
	}
	await initWasm({ module_or_path: bytes });
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));

	const source = args.stdin
		? await readAllStdin()
		: args.evalSource ?? (args.scriptPath ? await readFile(resolve(process.cwd(), args.scriptPath), 'utf8') : null);
	if (source === null) die(usage().trimEnd());

	await ensureWasmInitialized();

	const defaultModelRaw = wasm.default_device_model?.();
	const defaultModel = normalizeRomModel(defaultModelRaw);
	const model = args.model ?? defaultModel ?? ('pc-e500' as const);

	const resolvedRomPath = args.romPath
		? resolve(process.cwd(), args.romPath)
		: resolve(repoRoot, 'data', romBasename(model));
	const romBytes = new Uint8Array(await readFile(resolvedRomPath));

	const Emulator = (wasm as any).Sc62015Emulator ?? (wasm as any).Pce500Emulator;
	if (!Emulator) die('error: wasm module missing Sc62015Emulator/Pce500Emulator export');
	const emulator: any = new Emulator();
	(emulator.load_rom_with_model?.(romBytes, model) ?? emulator.load_rom(romBytes));

	function wrapError(context: string, err: unknown): Error {
		const msg = err instanceof Error ? err.message : String(err);
		return new Error(`${context}: ${msg}`);
	}
	const runWithError = <T>(context: string, fn: () => T): T => {
		try {
			return fn();
		} catch (err) {
			throw wrapError(context, err);
		}
	};
	const runWithErrorAsync = async <T>(context: string, fn: () => Promise<T> | T): Promise<T> => {
		try {
			return await fn();
		} catch (err) {
			throw wrapError(context, err);
		}
	};

	const adapter: EmulatorAdapter = {
		callFunction: async (
			address: number,
			maxInstructions: number,
			options?: { trace?: boolean; probe?: { pc: number; maxSamples?: number } } | null,
		) =>
			runWithErrorAsync(`call(0x${address.toString(16).toUpperCase()})`, async () => {
				const raw =
					emulator.call_function_ex?.(address, maxInstructions, {
						trace: Boolean(options?.trace),
						probe_pc: options?.probe ? options.probe.pc : null,
						probe_max_samples: options?.probe?.maxSamples ?? 256,
					}) ?? emulator.call_function(address, maxInstructions);
				if (typeof raw === 'string') return JSON.parse(raw);
				return raw;
			}),
		reset: async () => runWithErrorAsync('reset()', () => Promise.resolve(emulator.reset?.())),
		step: async (instructions: number) =>
			runWithErrorAsync(`step(${instructions})`, () => Promise.resolve(emulator.step?.(instructions))),
		getReg: (name: string) => runWithError(`getReg(${name})`, () => emulator.get_reg?.(name) ?? 0),
		setReg: (name: string, value: number) =>
			runWithError(`setReg(${name}=${value})`, () => emulator.set_reg?.(name, value)),
		read8: (addr: number) =>
			runWithError(`read8(0x${addr.toString(16).toUpperCase()})`, () => emulator.read_u8?.(addr) ?? 0),
		write8: (addr: number, value: number) =>
			runWithError(`write8(0x${addr.toString(16).toUpperCase()}, ${value})`, () => emulator.write_u8?.(addr, value)),
		lcdText: () => runWithError('lcd.text()', () => emulator.lcd_text?.() ?? null),
		pressMatrixCode: (code: number) =>
			runWithError(`keyboard.press(0x${code.toString(16).toUpperCase()})`, () => emulator.press_matrix_code?.(code)),
		releaseMatrixCode: (code: number) =>
			runWithError(`keyboard.release(0x${code.toString(16).toUpperCase()})`, () => emulator.release_matrix_code?.(code)),
		injectMatrixEvent: (code: number, release: boolean) =>
			runWithError(`keyboard.inject(0x${code.toString(16).toUpperCase()}, ${release})`, () =>
				emulator.inject_matrix_event?.(code, release),
			),
		pressOnKey: () => runWithError('onkey.press()', () => emulator.press_on_key?.()),
		releaseOnKey: () => runWithError('onkey.release()', () => emulator.release_on_key?.()),
	};

	const api = createEvalApi(adapter);
	let resultJson: string | null = null;
	let error: string | null = null;
	try {
		const result = await runUserJs(source, api, Reg, Flag, IOCS);
		resultJson = safeJson(result);
	} catch (err) {
		error = err instanceof Error ? err.message : String(err);
	}

	console.log(
		safeJson({
			events: api.events,
			calls: api.calls,
			prints: api.prints,
			resultJson,
			error,
			rom: { model, path: resolvedRomPath },
		}),
	);

	process.exit(error ? 1 : 0);
}

try {
	await main();
} catch (err) {
	if (err instanceof Error) console.error(err.stack ?? err.message);
	else console.error(String(err));
	process.exit(1);
}
