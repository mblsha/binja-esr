#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

function usage() {
	return `
Usage:
  node scripts/js_function_runner_cli.mjs [options] <script.js>
  node scripts/js_function_runner_cli.mjs [options] --eval "<js>"
  cat script.js | node scripts/js_function_runner_cli.mjs [options] --stdin

Options:
  --model <iq-7000|pc-e500>   ROM preset (default: Rust runtime default)
  --rom <path>               Explicit ROM path (overrides --model)
  --eval <js>                Inline script (async JS)
  --stdin                    Read script from stdin
  --help                     Show this help

Prereqs:
  - Ensure the WASM package is built:
      cd web && npm install && npm run wasm:build

Script context:
  - e: minimal async emulator API (reset/step/call/reg/read8/write8/lcd/keyboard)
  - Reg: { A,B,BA,IL,IH,I,X,Y,U,S,PC,F,IMR,FC,FZ }
  - Flag: { C,Z }
  - IOCS: empty map (use addresses directly for now)
`;
}

function die(msg) {
	console.error(msg);
	process.exit(2);
}

function parseModel(raw) {
	const v = String(raw ?? '').trim().toLowerCase();
	if (!v) return null;
	if (v === 'iq-7000' || v === 'iq7000' || v === 'iq_7000') return 'iq-7000';
	if (v === 'pc-e500' || v === 'pce500' || v === 'pc_e500') return 'pc-e500';
	return null;
}

function romBasename(model) {
	switch (model) {
		case 'pc-e500':
			return 'pc-e500.bin';
		case 'iq-7000':
			return 'iq-7000.bin';
		default:
			return null;
	}
}

function parseAddress(value) {
	if (typeof value === 'number' && Number.isFinite(value)) return value >>> 0;
	const raw = String(value ?? '').trim();
	if (!raw) throw new Error(`invalid address: ${value}`);
	if (raw.startsWith('0x') || raw.startsWith('0X')) {
		const parsed = Number.parseInt(raw.slice(2), 16);
		if (!Number.isFinite(parsed)) throw new Error(`invalid hex address: ${value}`);
		return parsed >>> 0;
	}
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed)) throw new Error(`invalid address: ${value}`);
	return parsed >>> 0;
}

async function readAllStdin() {
	const chunks = [];
	for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
	return Buffer.concat(chunks).toString('utf8');
}

function freezeIfObject(value) {
	if (typeof value !== 'object' || value === null) return;
	try {
		Object.freeze(value);
	} catch {
		// ignore
	}
}

async function runUserJs(source, e, Reg, Flag, IOCS) {
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
		e,
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

async function loadWasm(repoRoot) {
	const wasmJsPath = resolve(repoRoot, 'web/src/lib/wasm/pce500_wasm/pce500_wasm.js');
	const wasmBgPath = resolve(repoRoot, 'web/src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm');
	if (!existsSync(wasmJsPath) || !existsSync(wasmBgPath)) {
		die(
			`error: missing WASM package under web/src/lib/wasm/pce500_wasm.\n` +
				`Run: cd web && npm install && npm run wasm:build`,
		);
	}
	const wasm = await import(pathToFileURL(wasmJsPath).href);
	if (typeof wasm.default === 'function') {
		const wasmBytes = await readFile(wasmBgPath);
		await wasm.default({ module_or_path: wasmBytes });
	}
	return wasm;
}

async function main() {
	const argv = process.argv.slice(2);
	let modelArg = null;
	let romPath = null;
	let evalSource = null;
	let stdin = false;
	let scriptPath = null;

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === '--help' || arg === '-h') {
			console.log(usage().trimEnd());
			return;
		}
		if (arg === '--model') {
			const next = argv[++i];
			if (!next) die('error: --model requires a value');
			const parsed = parseModel(next);
			if (!parsed) die(`error: unknown --model '${next}' (expected: iq-7000|pc-e500)`);
			modelArg = parsed;
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

	const repoRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));

	const source = stdin ? await readAllStdin() : evalSource ?? (scriptPath ? await readFile(scriptPath, 'utf8') : null);
	if (source === null) die(usage().trimEnd());

	const wasm = await loadWasm(repoRoot);

	let model = modelArg;
	if (!model) {
		model = parseModel(wasm.default_device_model?.());
	}
	if (!model) {
		model = 'pc-e500';
	}

	const resolvedRomPath = romPath
		? resolve(process.cwd(), romPath)
		: resolve(repoRoot, 'data', romBasename(model));
	const romBytes = new Uint8Array(await readFile(resolvedRomPath));

	const Emulator = wasm.Sc62015Emulator ?? wasm.Pce500Emulator;
	if (!Emulator) die('error: wasm module missing Sc62015Emulator/Pce500Emulator export');
	const emu = new Emulator();
	(emu.load_rom_with_model?.(romBytes, model) ?? emu.load_rom(romBytes));

	const Reg = Object.freeze({
		A: 'A',
		B: 'B',
		BA: 'BA',
		IL: 'IL',
		IH: 'IH',
		I: 'I',
		X: 'X',
		Y: 'Y',
		U: 'U',
		S: 'S',
		PC: 'PC',
		F: 'F',
		IMR: 'IMR',
		FC: 'FC',
		FZ: 'FZ',
	});
	const Flag = Object.freeze({ C: 'C', Z: 'Z' });
	const IOCS = Object.freeze({});

	const e = {
		rom: { model, path: resolvedRomPath },
		async reset(options = {}) {
			await Promise.resolve(emu.reset());
			const warmup = Number.parseInt(String(options?.warmupTicks ?? 0), 10);
			if (Number.isFinite(warmup) && warmup > 0) {
				await Promise.resolve(emu.step(warmup));
			}
		},
		async step(instructions) {
			const n = Number.parseInt(String(instructions ?? 0), 10);
			if (!Number.isFinite(n) || n < 0) throw new Error(`step(): invalid instruction count '${instructions}'`);
			await Promise.resolve(emu.step(n));
		},
		async call(address, maxInstructions = 200_000, options = null) {
			const addr = parseAddress(address) & 0x00ff_ffff;
			const budget = Number.parseInt(String(maxInstructions ?? 0), 10);
			const trace = Boolean(options?.trace);
			const raw =
				emu.call_function_ex?.(addr, budget > 0 ? budget : 1, {
					trace,
					probe_pc: options?.probe ? options.probe.pc : null,
					probe_max_samples: options?.probe?.maxSamples ?? 256,
				}) ?? emu.call_function(addr, budget > 0 ? budget : 1);
			if (typeof raw === 'string') return JSON.parse(raw);
			return raw;
		},
		reg(name) {
			return emu.get_reg(String(name));
		},
		getReg(name) {
			return emu.get_reg(String(name));
		},
		setReg(name, value) {
			emu.set_reg(String(name), Number(value) >>> 0);
		},
		read8(addr) {
			return emu.read_u8(parseAddress(addr));
		},
		write8(addr, value) {
			emu.write_u8(parseAddress(addr), Number(value) & 0xff);
		},
		lcd: {
			text() {
				return emu.lcd_text?.() ?? [];
			},
			textString() {
				const lines = emu.lcd_text?.() ?? [];
				return Array.isArray(lines) ? lines.join('\n') : String(lines ?? '');
			},
		},
		keyboard: {
			async press(code) {
				emu.inject_matrix_event(parseAddress(code) & 0x7f, false);
			},
			async release(code) {
				emu.inject_matrix_event(parseAddress(code) & 0x7f, true);
			},
			async tap(code, holdInstructions = 40_000) {
				const c = parseAddress(code) & 0x7f;
				emu.inject_matrix_event(c, false);
				const n = Number.parseInt(String(holdInstructions ?? 0), 10);
				if (Number.isFinite(n) && n > 0) await Promise.resolve(emu.step(n));
				emu.inject_matrix_event(c, true);
			},
			async injectEvent(code, release) {
				emu.inject_matrix_event(parseAddress(code) & 0x7f, Boolean(release));
			},
		},
	};

	const result = await runUserJs(source, e, Reg, Flag, IOCS);
	if (result !== undefined) {
		try {
			console.log(JSON.stringify(result, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2));
		} catch {
			console.log(String(result));
		}
	}
}

try {
	await main();
} catch (err) {
	if (err instanceof Error) {
		console.error(err.stack ?? err.message);
	} else {
		console.error(String(err));
	}
	process.exit(1);
}

