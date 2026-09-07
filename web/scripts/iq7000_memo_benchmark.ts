/** Real-ROM, physical-key benchmark. No FIFO injection, stubs or RAM writes. */
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Session } from 'node:inspector/promises';
import { performance } from 'node:perf_hooks';
import { setImmediate as yieldImmediate } from 'node:timers/promises';
import { HostInputs, applyContact, TYPING_HOLD, TYPING_GAP } from '../src/lib/emulator/host_inputs';
import { stepBounded } from '../src/lib/emulator/bounded_step';
import { physicalKey } from '../src/lib/keymap';
import { planPaste } from '../src/lib/emulator/paste_plan';

const LOREM =
	'lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
	'ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ' +
	'duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. ' +
	'excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.';
const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const RTC = '199201101050';

async function main() {
	let romPath = process.env.IQ7000_ROM_PATH ?? resolve(ROOT, 'data/iq-7000.bin');
	let out: string | null = null;
	let repeats = 1;
	let text = LOREM;
	let wasmDir = resolve(ROOT, 'web/src/lib/wasm/pce500_wasm');
	let cpuProfile = false;
	let traceBoundaries = 0;
	let adaptive = false;
	let adaptiveGap = 8000;
	let inputPhase = 0;
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === '--adaptive-memo') {
			adaptive = true;
			continue;
		}
		if (arg === '--cpu-profile') {
			cpuProfile = true;
			continue;
		}
		if (arg === '--help') {
			console.log(
				'IQ-7000 real-ROM MEMO benchmark\n--rom PATH  --out DIRECTORY  --repeats 1..20  --text "lowercase ASCII text"\n--wasm-dir DIRECTORY  --cpu-profile  --trace-boundaries 1..20000\n--adaptive-memo  --adaptive-gap 1..40000 (default 8000)  --input-phase 0..10000\nAdaptive input is an experimental MEMO-specific observer, not the default UI policy.\nProfiling requires --out. CPU and guest Perfetto profiling must use separate runs.\nDefault: complete 445-character Lorem Ipsum, fixed RTC, unthrottled bounded execution.\nNo ROM is distributed; use IQ7000_ROM_PATH or --rom. Build/compile and evidence output are outside timed phases.',
			);
			return;
		}
		const value = argv[++i];
		if (!value) throw new Error(`Missing value for ${arg}`);
		if (arg === '--rom') romPath = value;
		else if (arg === '--out') out = resolve(value);
		else if (arg === '--repeats') repeats = Number(value);
		else if (arg === '--text') text = value;
		else if (arg === '--wasm-dir') wasmDir = resolve(value);
		else if (arg === '--adaptive-gap') adaptiveGap = Number(value);
		else if (arg === '--input-phase') inputPhase = Number(value);
		else if (arg === '--trace-boundaries') {
			traceBoundaries = Number(value);
			if (!Number.isInteger(traceBoundaries) || traceBoundaries < 1 || traceBoundaries > 20_000)
				throw new Error('--trace-boundaries must be 1..20000');
		} else throw new Error(`Unknown argument ${arg}`);
	}
	if (!Number.isInteger(repeats) || repeats < 1 || repeats > 20) throw new Error('--repeats must be 1..20');
	if ((cpuProfile || traceBoundaries) && !out) throw new Error('Profiling requires --out');
	if (cpuProfile && traceBoundaries) throw new Error('Use separate runs for CPU sampling and guest tracing');
	if (!Number.isInteger(adaptiveGap) || adaptiveGap < 1 || adaptiveGap > 40_000)
		throw new Error('--adaptive-gap must be 1..40000');
	if (!Number.isInteger(inputPhase) || inputPhase < 0 || inputPhase > 10_000)
		throw new Error('--input-phase must be 0..10000');
	if (adaptive && traceBoundaries) throw new Error('Adaptive input and guest trace capture need separate runs');
	// Explicit case policy: CAPS controls characters; do not pretend the keymap
	// implements host text composition or silently alter the requested string.
	if (!text.length || text.length > 1000 || !/^[a-z0-9 ,.]+$/.test(text) || text !== text.trim())
		throw new Error('Text must be 1..1000 lowercase ASCII letters/digits/spaces/comma/period, without edge spaces.');
	const plan = planPaste(text, 'iq-7000');
	if (plan.error || plan.unsupported.length) throw new Error(`Unsupported text: ${JSON.stringify(plan)}`);
	const rom = new Uint8Array(await readFile(resolve(romPath)));
	const { initSync, Sc62015Emulator } = (await import(
		/* @vite-ignore */ pathToFileURL(resolve(wasmDir, 'pce500_wasm.js')).href
	)) as typeof import('../src/lib/wasm/pce500_wasm/pce500_wasm.js');
	const wasmBytes = await readFile(resolve(wasmDir, 'pce500_wasm_bg.wasm'));
	const wasmModule = new WebAssembly.Module(wasmBytes);
	const wasmHasFunctionNames = WebAssembly.Module.customSections(wasmModule, 'name').length > 0;
	if (cpuProfile && !wasmHasFunctionNames)
		throw new Error(
			'CPU profiling requires a named WASM build: npm run wasm:build:profile, then --wasm-dir node_modules/.cache/iq-memo-wasm-profile',
		);
	const wasm = initSync({ module: wasmModule });
	const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
	const results: any[] = [];
	if (out) await mkdir(out, { recursive: true });
	for (let run = 1; run <= repeats; run++) {
		const emulator = new Sc62015Emulator();
		const inputs = new HostInputs((contact, down) => applyContact(emulator, contact, down));
		let boundaries = 0;
		let profiler: Session | null = null;
		let traceActive = false;
		let cpuData: unknown;
		let guestTrace: Buffer | null = null;
		const adaptiveHolds: number[] = [];
		const deadline = performance.now() + 120_000;
		const advance = (count: number) =>
			stepBounded(emulator, count, {
				limitBudget: inputs.limitBudget,
				onProgress: (used) => {
					boundaries += used;
					inputs.advance(used);
					if (performance.now() > deadline) throw new Error('Benchmark exceeded 120-second host deadline');
				},
				// No setTimeout(0) clamp and no real-time pacing. Still yield between
				// <=4 ms cooperative Rust slices, respecting all input deadlines.
				yieldHost: yieldImmediate,
			});
		const tap = async (name: string) => {
			const code = physicalKey('iq-7000', name);
			if (code === null) throw new Error(`Unqualified key ${name}`);
			inputs.startPaste([code]);
			await advance(TYPING_HOLD + TYPING_GAP);
		};
		const lcdText = () => (emulator.lcd_text() as string[]).join('\n');
		const check = (condition: boolean, message: string) => {
			if (!condition) throw new Error(`${message}\nLCD:\n${lcdText()}`);
		};
		const matches = (expected = text) => {
			// Read-only exact comparison in the emulated external address space,
			// not the WASM heap that contains our host-side expected string.
			const memory = Buffer.from(
				new Uint8Array(wasm.memory.buffer, emulator.memory_external_ptr(), emulator.memory_external_len()),
			);
			const needle = Buffer.from(expected, 'ascii');
			const addresses: number[] = [];
			for (let at = memory.indexOf(needle); at !== -1; at = memory.indexOf(needle, at + 1)) addresses.push(at);
			return addresses;
		};
		try {
			const started = performance.now();
			emulator.load_rom_with_model(rom, 'iq-7000');
			emulator.set_iq7000_rtc_yyyymmddhhmm(RTC);
			emulator.set_execution_mode('turbo');
			await advance(500_000);
			const bootMs = performance.now() - started;
			check(matches().length === 0, 'Expected text already present before typing; invalid benchmark fixture');
			const memoStarted = performance.now();
			await tap('MEMO');
			check(lcdText().includes('MEMO ?'), 'MEMO prompt not reached');
			if (emulator.lcd_capture().annunciators.caps) await tap('CAPS');
			check(!emulator.lcd_capture().annunciators.caps, 'CAPS must be off for lowercase text');
			const memoMs = performance.now() - memoStarted;
			if (inputPhase) await advance(inputPhase);
			if (cpuProfile) {
				profiler = new Session();
				profiler.connect();
				await profiler.post('Profiler.enable');
				await profiler.post('Profiler.setSamplingInterval', { interval: 1000 });
				await profiler.post('Profiler.start');
			}
			const typingStarted = performance.now();
			if (adaptive) {
				// Experimental MEMO-only observer: discover a unique editor copy
				// after a conservative seed, then observe ROM-written bytes. Never
				// inject text, FIFO data, or IRQs. Other apps retain normal pacing.
				const seed = text.slice(0, Math.min(16, text.length));
				const previousAnchors = new Set(matches(seed));
				const seedPlan = planPaste(seed, 'iq-7000');
				inputs.startPaste(seedPlan.contacts);
				await advance(seedPlan.contacts.length * (TYPING_HOLD + TYPING_GAP));
				const anchors = matches(seed).filter((address) => !previousAnchors.has(address));
				check(anchors.length === 1, 'Adaptive seed must identify exactly one editor buffer');
				const anchor = anchors[0];
				const editor = () =>
					new Uint8Array(wasm.memory.buffer, emulator.memory_external_ptr(), emulator.memory_external_len());
				check(editor()[anchor + seed.length] === 0, 'Adaptive editor must have a NUL terminator');
				for (let index = seed.length; index < text.length; index++) {
					const charPlan = planPaste(text[index], 'iq-7000');
					// Modifier contacts remain conservative; acceptance observes the
					// resulting character, not an assumed scanner/FIFO acknowledgement.
					for (const code of charPlan.contacts.slice(0, -1)) {
						inputs.startPaste([code]);
						await advance(TYPING_HOLD + TYPING_GAP);
					}
					check(editor()[anchor + index] === 0, 'Unexpected extra text before adaptive contact');
					const expected = Buffer.from(text.slice(0, index + 1), 'ascii');
					const accepted = () => {
						const bytes = editor();
						return (
							bytes[anchor + expected.length] === 0 &&
							expected.every((value, offset) => bytes[anchor + offset] === value)
						);
					};
					const contact = charPlan.contacts.at(-1)!;
					inputs.set({ source: 'script', owner: 'adaptive', contact, down: true });
					let held = 0;
					while (!accepted() && held < TYPING_HOLD) {
						const quantum = Math.min(256, TYPING_HOLD - held);
						await advance(quantum);
						held += quantum;
					}
					inputs.set({ source: 'script', owner: 'adaptive', contact, down: false });
					check(accepted(), `Adaptive contact ${index} was not accepted within its budget`);
					adaptiveHolds.push(held);
					await advance(adaptiveGap);
				}
				await advance(TYPING_GAP);
			} else {
				inputs.startPaste(plan.contacts);
				const typingBoundaries = plan.contacts.length * (TYPING_HOLD + TYPING_GAP);
				const tracedBoundaries = Math.min(traceBoundaries, typingBoundaries);
				if (tracedBoundaries) {
					emulator.perfetto_start(`memo-typing-${run}`);
					traceActive = true;
					await advance(tracedBoundaries);
					traceActive = false;
					const b64 = emulator.perfetto_stop_b64();
					guestTrace = Buffer.from(b64, 'base64');
				}
				await advance(typingBoundaries - tracedBoundaries);
			}
			const typingMs = performance.now() - typingStarted;
			if (profiler) {
				cpuData = (await profiler.post('Profiler.stop')).profile;
				profiler.disconnect();
				profiler = null;
			}
			const bootToTypedMs = performance.now() - started;
			check(inputs.pasteStatus().pending === 0, 'Paste did not drain');
			const typedAddresses = matches();
			check(
				typedAddresses.length > 0,
				'Complete text missing from machine memory after typing (dropped or altered keys)',
			);
			const typedScreen = lcdText();
			const saveStarted = performance.now();
			await tap('ENTER');
			await tap('MEMO');
			check(lcdText().includes('MEMO ?'), 'MEMO prompt not reached after store');
			await tap('SEARCH_DOWN');
			const saveReopenMs = performance.now() - saveStarted;
			const reopenedScreen = lcdText();
			check(
				reopenedScreen.replace(/\s+/g, '').startsWith(text.slice(0, Math.min(12, text.length)).replace(/\s+/g, '')),
				'Stored memo did not reopen at its beginning',
			);
			const reopenedAddresses = matches();
			check(reopenedAddresses.length > 0, 'Complete exact text missing after store/reopen');
			const storedAddresses = reopenedAddresses.filter((address) => !typedAddresses.includes(address));
			check(
				storedAddresses.length > 0,
				'Store/reopen did not produce a new full-text copy outside the original editor buffer',
			);
			const contacts = emulator.input_contacts();
			check(contacts.matrix.length === 0 && !contacts.on, 'Input contacts remain held');
			const result = {
				run,
				verified: true,
				characters: text.length,
				contacts: plan.contacts.length,
				adaptiveHolds,
				bootMs,
				memoMs,
				typingMs,
				saveReopenMs,
				bootToTypedMs,
				totalWallMs: performance.now() - started,
				charactersPerSecond: (text.length * 1000) / typingMs,
				schedulerBoundaries: boundaries,
				retired: emulator.instruction_count().toString(),
				timingUnits: emulator.cycle_count().toString(),
				typedAddresses,
				reopenedAddresses,
				storedAddresses,
				typedScreen,
				reopenedScreen,
			};
			results.push(result);
			console.log(
				JSON.stringify({
					...result,
					adaptiveHolds: adaptiveHolds.length
						? {
								count: adaptiveHolds.length,
								min: Math.min(...adaptiveHolds),
								max: Math.max(...adaptiveHolds),
							}
						: undefined,
				}),
			);
			if (out) {
				if (cpuData) await writeFile(resolve(out, `run-${run}.cpuprofile`), JSON.stringify(cpuData));
				if (guestTrace) await writeFile(resolve(out, `run-${run}.perfetto-trace`), guestTrace);
				const capture = emulator.lcd_capture();
				await writeFile(
					resolve(out, `run-${run}-lcd.json`),
					JSON.stringify({ ...capture, pixels: Array.from(capture.pixels) }),
				);
			}
		} catch (error) {
			if (out)
				await writeFile(
					resolve(out, `run-${run}-failure.json`),
					JSON.stringify({
						error: String(error),
						screen: lcdText(),
						boundaries,
						adaptive,
						adaptiveGap,
						adaptiveHolds,
					}),
				);
			throw error;
		} finally {
			profiler?.disconnect();
			try {
				if (traceActive) emulator.perfetto_stop_b64();
			} finally {
				inputs.clear();
				emulator.free();
			}
		}
	}
	const median = (values: number[]) => {
		const sorted = values.slice().sort((a, b) => a - b);
		const mid = Math.floor(sorted.length / 2);
		return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
	};
	const summary = {
		runs: results.length,
		characters: text.length,
		verified: true,
		medianBootToTypedMs: median(results.map((result) => result.bootToTypedMs)),
		medianTypingMs: median(results.map((result) => result.typingMs)),
	};
	console.log(JSON.stringify({ summary }));
	const report = {
		format: 'iq7000-memo-benchmark-v1',
		text,
		rtcSeed: RTC,
		romSha256: hash(rom),
		wasmSha256: hash(wasmBytes),
		wasmDir,
		wasmHasFunctionNames,
		profiling: { cpuProfile, guestTraceBoundaries: traceBoundaries },
		inputStrategy: adaptive
			? { kind: 'experimental-memo-acceptance', gapBoundaries: adaptiveGap, quantum: 256, seedCharacters: 16 }
			: { kind: 'fixed' },
		inputPhaseBoundaries: inputPhase,
		node: process.version,
		host: `${process.platform}/${process.arch}`,
		mode: 'headless turbo',
		keyHoldBoundaries: TYPING_HOLD,
		keyGapBoundaries: TYPING_GAP,
		verification:
			'Real MEMO prompt, exact ASCII in emulated external memory absent before typing and present after typing/store/reopen, new full-text copy outside the original editor buffer after store, reopened screen prefix, released contacts. Read-only RAM observation, not a parsed record-format proof.',
		timingScope:
			'Build/WASM compile/ROM file read and initial emulator object allocation excluded; boot includes ROM load/reset. No rendering during typing. Wall times include read-only verification between phases; evidence file output excluded. First run includes V8 warm-up.',
		results,
		summary,
	};
	if (out) await writeFile(resolve(out, 'benchmark.json'), JSON.stringify(report, null, 2));
}

main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
