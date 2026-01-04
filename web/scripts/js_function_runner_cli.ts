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
	bnidaPath: string | null;
	disableBnida: boolean;
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
  --bnida <path>             BNIDA export for function trace labels
  --no-bnida                 Disable auto-loading BNIDA symbols
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

type PerfettoSymbol = { addr: number; name: string };

function stripLeadingLineComments(raw: string): string {
	const lines = raw.split(/\r?\n/);
	let start = 0;
	while (start < lines.length && lines[start]?.trimStart().startsWith('//')) start += 1;
	return lines.slice(start).join('\n');
}

function walkParents(start: string, maxDepth = 6): string[] {
	const out: string[] = [];
	let current = start;
	for (let i = 0; i < maxDepth; i++) {
		out.push(current);
		const next = resolve(current, '..');
		if (next === current) break;
		current = next;
	}
	return out;
}

function reportRelativePath(model: RomModel): string {
	switch (model) {
		case 'iq-7000':
			return 'rom-analysis/iq-7000/bnida.json';
		case 'pc-e500':
			return 'rom-analysis/pc-e500/s3-en/bnida.json';
	}
}

function parseAddress(key: string): number | null {
	const trimmed = key.trim();
	if (!trimmed) return null;
	const value = Number.parseInt(trimmed, 10);
	if (!Number.isFinite(value)) return null;
	return value >>> 0;
}

async function loadBnidaSymbols(args: RunnerArgs, model: RomModel): Promise<PerfettoSymbol[] | null> {
	if (args.disableBnida) return null;

	const candidates: string[] = [];
	if (args.bnidaPath) candidates.push(resolve(process.cwd(), args.bnidaPath));

	const reportPath = reportRelativePath(model);
	for (const root of walkParents(process.cwd())) {
		candidates.push(resolve(root, reportPath));
		candidates.push(resolve(root, 'binja-esr-tests', reportPath));
	}

	for (const candidate of candidates) {
		try {
			const raw = await readFile(candidate, 'utf8');
			const jsonText = stripLeadingLineComments(raw);
			const bnida = JSON.parse(jsonText) as { names?: Record<string, string> };
			const symbols = Object.entries(bnida.names ?? {})
				.map(([addr, name]) => ({ addr: parseAddress(addr), name: String(name ?? '').trim() }))
				.filter((entry) => typeof entry.addr === 'number' && entry.addr !== null && entry.name.length > 0)
				.map((entry) => ({ addr: (entry.addr as number) & 0x000f_ffff, name: entry.name }));
			return symbols;
		} catch {
			// Try next candidate.
		}
	}

	return null;
}

function parseArgs(argv: string[]): RunnerArgs {
	let model: RomModel | null = null;
	let romPath: string | null = null;
	let bnidaPath: string | null = null;
	let disableBnida = false;
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
		if (arg === '--bnida') {
			const next = argv[++i];
			if (!next) die('error: --bnida requires a path');
			bnidaPath = next;
			continue;
		}
		if (arg === '--no-bnida') {
			disableBnida = true;
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

	return { model, romPath, bnidaPath, disableBnida, scriptPath, evalSource, stdin };
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

	try {
		const symbols = await loadBnidaSymbols(args, model);
		if (symbols && typeof emulator.set_perfetto_function_symbols === 'function') {
			emulator.set_perfetto_function_symbols(symbols);
		}
	} catch {
		// Ignore missing symbol sources (public CI does not ship private rom-analysis).
	}

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
		startPerfettoTrace: (name: string) =>
			runWithError(`perfetto.start(${name})`, () => {
				if (typeof emulator.perfetto_start !== 'function') {
					throw new Error('perfetto_start is not available in this runtime');
				}
				emulator.perfetto_start(name);
			}),
		stopPerfettoTrace: () =>
			runWithError('perfetto.stop()', () => {
				if (typeof emulator.perfetto_stop_b64 !== 'function') {
					throw new Error('perfetto_stop_b64 is not available in this runtime');
				}
				const raw = emulator.perfetto_stop_b64();
				if (typeof raw !== 'string') {
					throw new Error('perfetto_stop_b64 returned a non-string value');
				}
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
