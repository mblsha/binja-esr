import { normalizeRomModel, type RomModel } from '../rom_model';
import { normalizeLcdKind, type LcdKind } from '../lcd_kind';
import { PCE500_KEY_FIFO_CAPACITY, resolvePce500KeyboardFifo } from './pce500_iocs_workspace';
import { ExecutionCancelled, runHostSlice, stepBounded } from './bounded_step';
import { WorkerOperations } from './worker_operations';
import { callBounded } from './bounded_call';
import { runIsolatedScript } from './isolated_script';

type DebugOptions = {
	regsOpen: boolean;
	callStackOpen: boolean;
	lcdTextOpen: boolean;
	debugStateOpen: boolean;
	keyboardDebugOpen: boolean;
};

type WorkerRequest =
	| {
			id: number;
			type: 'load_rom';
			bytes: Uint8Array;
			romSource?: string | null;
			model?: RomModel;
			generation?: number;
	  }
	| { id: number; type: 'get_model' }
	| { id: number; type: 'step'; instructions: number }
	| { id: number; type: 'start' }
	| { id: number; type: 'stop' }
	| { id: number; type: 'snapshot' }
	| { id: number; type: 'lcd_trace' }
	| { id: number; type: 'eval_js'; source: string }
	| { id: number; type: 'set_options'; targetFps?: number; debug?: Partial<DebugOptions> }
	| { id: number; type: 'virtual_key'; code: number; down: boolean }
	| { id: number; type: 'physical_key'; code: number; down: boolean };

type WorkerReply =
	| { type: 'reply'; id: number; ok: true; result?: any }
	| { type: 'reply'; id: number; ok: false; error: string };

type KeyboardDebug = {
	pc: number | null;
	instr: string | null;
	imr: number | null;
	isr: number | null;
	kol: number | null;
	koh: number | null;
	kil: number | null;
	fifoHead: number | null;
	fifoTail: number | null;
	fifo: number[];
	pressedCodes: number[];
	pendingVirtualRelease: [number, number][];
};

type Frame = {
	model: RomModel;
	generation: number;
	lcdPixels: ArrayBuffer;
	lcdAnnunciatorBytes: ArrayBuffer;
	lcdChipPixels: ArrayBuffer;
	lcdCols: number;
	lcdPixelScale: number;
	lcdRows: number;
	lcdKind: LcdKind;
	pc: number | null;
	instructionCount: string | null;
	halted: boolean;
	buildInfo: { version: string; git_commit: string; build_timestamp: string } | null;
	lcdText: string[] | null;
	regs: any | null;
	callStack: number[] | null;
	debugState: any | null;
	keyboardDebug: KeyboardDebug | null;
	keyboardDebugJson: string | null;
};

const MIN_VIRTUAL_HOLD_INSTRUCTIONS = 40_000;
const IMEM_BASE = 0x100000;

const RUN_SLICE_MAX_INSTRUCTIONS = 200_000;
const RUN_YIELD_MS = 0;

const LCD_TEXT_UPDATE_INTERVAL_MS = 250;

function formatHostUtcRtcSeed(now = new Date()): string {
	const pad2 = (value: number) => String(value).padStart(2, '0');
	return `${now.getUTCFullYear()}${pad2(now.getUTCMonth() + 1)}${pad2(now.getUTCDate())}${pad2(now.getUTCHours())}${pad2(now.getUTCMinutes())}`;
}

let wasm: any = null;
let emulator: any = null;
let emulatorReady: Promise<any> | null = null;
const operations = new WorkerOperations();
let buildInfo: { version: string; git_commit: string; build_timestamp: string } | null = null;
let romModel: RomModel = 'pc-e500';
let machineGeneration = 0;

let running = false;
let targetFps = 30;
let debugOptions: DebugOptions = {
	regsOpen: false,
	callStackOpen: true,
	lcdTextOpen: true,
	debugStateOpen: false,
	keyboardDebugOpen: false,
};

let runLoopId = 0;
let lastLcdTextUpdateMs = 0;
let lastLcdText: string[] | null = null;

const pressedCodes = new Set<number>();
const pendingVirtualRelease = new Map<number, number>();
let perfettoSymbolsPromise: Promise<void> | null = null;

function safeJson(value: any): string {
	return JSON.stringify(value, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
}

function isLikelyWasmTrap(message: string): boolean {
	const lower = message.toLowerCase();
	return /\bunreachable\b/.test(lower) || lower.includes('out of memory') || lower.includes('memory allocation');
}

function isWasmBindgenBorrowError(message: string): boolean {
	const lower = message.toLowerCase();
	return lower.includes('recursive use of an object detected') || lower.includes('unsafe aliasing');
}

async function ensurePerfettoSymbols(): Promise<void> {
	if (perfettoSymbolsPromise) return perfettoSymbolsPromise;
	perfettoSymbolsPromise = (async () => {
		try {
			await ensureEmulator();
			const res = await fetch(`/api/symbols?model=${encodeURIComponent(romModel)}`);
			if (!res.ok) return;
			const payload = (await res.json()) as any;
			emulator.set_perfetto_function_symbols(payload.symbols);
		} catch {
			// Ignore missing symbol sources (public CI does not ship private rom-analysis).
		}
	})();
	return perfettoSymbolsPromise;
}

async function evalScript(source: string, signal?: AbortSignal): Promise<any> {
	const { createEvalApi } = await import('../debug/sc62015_eval_api');
	let lastScriptProgress = -Infinity;
	let reportedPrint = false;
	return runIsolatedScript({
		source,
		signal,
		read8: (addr) => emulator.read_u8(addr),
		onRequest: (path) => {
			const now = performance.now();
			if (now - lastScriptProgress >= 100 || (path === 'print' && !reportedPrint)) {
				if (path === 'print') reportedPrint = true;
				lastScriptProgress = now;
				self.postMessage({ type: 'script_progress', generation: machineGeneration, operation: path });
			}
		},
		createApi: ({ signal, dispatchStub }) => {
			let mutating = false;
			const requireIdle = () => {
				if (mutating) throw new Error('Await the active machine operation before starting another mutation');
			};
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
				requireIdle();
				mutating = true;
				try {
					if (signal?.aborted) throw new ExecutionCancelled(0);
					return await fn();
				} catch (err) {
					throw wrapError(context, err);
				} finally {
					mutating = false;
				}
			};
			return createEvalApi({
				callFunction: async (
					address: number,
					maxInstructions: number,
					options?: {
						trace?: boolean;
						probe?: { pc: number; maxSamples?: number };
						stubs?: Array<{ id: number; pc: number }>;
					} | null,
				) =>
					runWithErrorAsync(`call(0x${address.toString(16).toUpperCase()})`, async () => {
						if (options?.trace) await ensurePerfettoSymbols();
						let lastProgress = -Infinity;
						return callBounded(
							emulator,
							address,
							maxInstructions,
							{
								trace: Boolean(options?.trace),
								probe_pc: options?.probe ? options.probe.pc : null,
								probe_max_samples: options?.probe?.maxSamples ?? 256,
								stubs: options?.stubs ?? [],
							},
							{
								signal,
								onProgress: (used, slice) => {
									applyVirtualReleaseBudget(used);
									const now = performance.now();
									if (now - lastProgress >= 100 || slice.state === 'complete') {
										lastProgress = now;
										(self as any).postMessage({
											type: 'execution_progress',
											generation: machineGeneration,
											address,
											steps: slice.steps,
											schedulerBoundaries: slice.scheduler_boundaries,
										});
									}
								},
								dispatchStub,
							},
						);
					}),
				startPerfettoTrace: async (name: string) =>
					runWithErrorAsync(`perfetto.start(${name})`, async () => {
						await ensurePerfettoSymbols();
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
					runWithErrorAsync(`step(${instructions})`, async () => {
						await stepBounded(emulator, instructions, { signal, onProgress: applyVirtualReleaseBudget });
					}),
				getReg: (name: string) => runWithError(`getReg(${name})`, () => emulator.get_reg?.(name) ?? 0),
				setReg: (name: string, value: number) =>
					runWithError(`setReg(${name}=${value})`, () => {
						requireIdle();
						emulator.set_reg?.(name, value);
					}),
				read8: (addr: number) =>
					runWithError(`read8(0x${addr.toString(16).toUpperCase()})`, () => emulator.read_u8?.(addr) ?? 0),
				write8: (addr: number, value: number) =>
					runWithError(`write8(0x${addr.toString(16).toUpperCase()}, ${value})`, () => {
						requireIdle();
						emulator.write_u8?.(addr, value);
					}),
				lcdText: () => runWithError('lcd.text()', () => emulator.lcd_text?.() ?? null),
				lcdPixels: () => runWithError('lcd.pixels()', () => emulator.lcd_pixels()),
				lcdCapture: (scale) => runWithError('lcd.capture()', () => emulator.lcd_capture(scale)),
				pressMatrixCode: (code: number) =>
					runWithError(`keyboard.press(0x${code.toString(16).toUpperCase()})`, () =>
						emulator.press_matrix_code?.(code),
					),
				releaseMatrixCode: (code: number) =>
					runWithError(`keyboard.release(0x${code.toString(16).toUpperCase()})`, () =>
						emulator.release_matrix_code?.(code),
					),
				injectMatrixEvent: (code: number, release: boolean) =>
					runWithError(`keyboard.inject(0x${code.toString(16).toUpperCase()}, ${release})`, () =>
						emulator.inject_matrix_event?.(code, release),
					),
				// Handler closures and their registry live only in the disposable worker.
				registerStub: () => {},
				clearStubs: () => {},
			});
		},
	});
}

async function ensureEmulator(): Promise<any> {
	if (emulatorReady) return emulatorReady;
	emulatorReady = initializeEmulator().catch((error) => {
		emulatorReady = null;
		emulator = null;
		wasm = null;
		throw error;
	});
	return emulatorReady;
}

async function initializeEmulator(): Promise<any> {
	if (!wasm) {
		wasm = await import('../wasm/sc62015_wasm');
		if (typeof wasm.default === 'function') {
			const url = new URL('../wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url);
			try {
				if (Boolean((import.meta as any)?.env?.DEV)) {
					url.searchParams.set('v', String(Date.now()));
				}
			} catch {
				// ignore
			}
			await wasm.default(url);
		}
	}
	if (!emulator) {
		emulator = new wasm.Sc62015Emulator();
		try {
			buildInfo = emulator.build_info?.() ?? null;
		} catch {
			buildInfo = null;
		}
	}
	try {
		const raw = emulator?.device_model?.() ?? wasm?.default_device_model?.();
		const model = normalizeRomModel(typeof raw === 'string' ? raw : null);
		if (model) romModel = model;
	} catch {
		// ignore
	}
	return emulator;
}

function replyOk(id: number, result?: any) {
	(self as any).postMessage({
		type: 'reply',
		id,
		ok: true,
		...(result !== undefined ? { result } : {}),
	} satisfies WorkerReply);
}

function replyErr(id: number, error: unknown) {
	(self as any).postMessage({
		type: 'reply',
		id,
		ok: false,
		error: error instanceof Error ? error.message : String(error),
	} satisfies WorkerReply);
}

function applyVirtualReleaseBudget(stepped: number) {
	if (pendingVirtualRelease.size === 0) return;
	for (const [code, remaining] of pendingVirtualRelease.entries()) {
		const next = remaining - stepped;
		if (next <= 0) {
			pendingVirtualRelease.delete(code);
			pressedCodes.delete(code);
			emulator.release_matrix_code?.(code);
		} else {
			pendingVirtualRelease.set(code, next);
		}
	}
}

function stepCore(boundaries: number) {
	const used = runHostSlice(emulator, boundaries);
	applyVirtualReleaseBudget(used);
	return used;
}

function snapshotKeyboard(): { keyboardDebug: KeyboardDebug; keyboardDebugJson: string } | null {
	if (!debugOptions.keyboardDebugOpen) return null;
	try {
		const pc = emulator.get_reg?.('PC') ?? null;
		const instrRaw = emulator.instruction_count?.() ?? null;
		const instr = instrRaw?.toString?.() ?? null;
		const imr = emulator.imr?.() ?? null;
		const isr = emulator.isr?.() ?? null;
		const kol = emulator.read_u8?.(IMEM_BASE + 0xf0) ?? null;
		const koh = emulator.read_u8?.(IMEM_BASE + 0xf1) ?? null;
		const kil = emulator.read_u8?.(IMEM_BASE + 0xf2) ?? null;
		const fifoAddresses = resolvePce500KeyboardFifo((address) => emulator.read_u8?.(address));
		const fifoHead = fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoHead) ?? null) : null;
		const fifoTail = fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoTail) ?? null) : null;
		const fifo = Array.from({ length: PCE500_KEY_FIFO_CAPACITY }, (_, i) =>
			fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoBase + i) ?? 0) : 0,
		);
		const keyboardDebug: KeyboardDebug = {
			pc,
			instr,
			imr,
			isr,
			kol,
			koh,
			kil,
			fifoHead,
			fifoTail,
			fifo,
			pressedCodes: Array.from(pressedCodes.values()),
			pendingVirtualRelease: Array.from(pendingVirtualRelease.entries()),
		};
		return { keyboardDebug, keyboardDebugJson: safeJson(keyboardDebug) };
	} catch {
		return null;
	}
}

function captureFrame(forceText: boolean): Frame {
	const geometry = emulator.lcd_capture();
	const lcdCols = typeof geometry?.cols === 'number' ? geometry.cols : 240;
	const lcdRows = typeof geometry?.rows === 'number' ? geometry.rows : 32;
	const lcdKind = normalizeLcdKind(geometry?.kind) ?? 'unknown';

	const pixels = geometry.pixels;
	const pixelsCopy = new Uint8Array(pixels);
	const annunciatorBytes = emulator.lcd_annunciator_bytes?.() ?? new Uint8Array(4);
	const annunciatorBytesCopy = new Uint8Array(annunciatorBytes);
	const chipPixels = emulator.lcd_chip_pixels();
	const chipPixelsCopy = new Uint8Array(chipPixels);
	const nowMs = performance.now();

	const pc = (() => {
		try {
			return emulator.get_reg?.('PC') ?? null;
		} catch {
			return null;
		}
	})();

	const halted = (() => {
		try {
			return Boolean(emulator.halted?.());
		} catch {
			return false;
		}
	})();

	const instructionCount = (() => {
		try {
			const count = emulator.instruction_count?.();
			return count?.toString?.() ?? null;
		} catch {
			return null;
		}
	})();

	let lcdText: string[] | null = null;
	if (debugOptions.lcdTextOpen) {
		const shouldUpdate = forceText || !running || nowMs - lastLcdTextUpdateMs >= LCD_TEXT_UPDATE_INTERVAL_MS;
		if (shouldUpdate) {
			lastLcdTextUpdateMs = nowMs;
			lastLcdText = emulator.lcd_text?.() ?? lastLcdText;
		}
		lcdText = lastLcdText ?? null;
	}

	const regs = debugOptions.regsOpen ? (emulator.regs?.() ?? null) : null;
	const callStack = debugOptions.callStackOpen ? (emulator.call_stack?.() ?? null) : null;
	const debugState = debugOptions.debugStateOpen ? (emulator.debug_state?.() ?? null) : null;

	const kb = snapshotKeyboard();
	return {
		lcdPixels: pixelsCopy.buffer,
		model: romModel,
		generation: machineGeneration,
		lcdAnnunciatorBytes: annunciatorBytesCopy.buffer,
		lcdChipPixels: chipPixelsCopy.buffer,
		lcdCols,
		lcdPixelScale: geometry.pixel_scale,
		lcdRows,
		lcdKind,
		pc,
		instructionCount,
		halted,
		buildInfo,
		lcdText,
		regs,
		callStack,
		debugState,
		keyboardDebug: kb?.keyboardDebug ?? null,
		keyboardDebugJson: kb?.keyboardDebugJson ?? null,
	};
}

function postFrame(frame: Frame) {
	(self as any).postMessage({ type: 'frame', frame }, [
		frame.lcdPixels,
		frame.lcdAnnunciatorBytes,
		frame.lcdChipPixels,
	]);
}

function pumpEmulator(id: number) {
	if (!running || !emulator || id !== runLoopId) return;
	try {
		stepCore(RUN_SLICE_MAX_INSTRUCTIONS);
	} catch (err) {
		// Crash stops the run loop; render loop will stop too.
		running = false;
		(self as any).postMessage({ type: 'fatal', error: String(err) });
		return;
	}
	setTimeout(() => pumpEmulator(id), RUN_YIELD_MS);
}

function pumpRender(id: number) {
	if (!running || !emulator || id !== runLoopId) return;
	const startMs = performance.now();
	postFrame(captureFrame(false));
	const elapsedMs = performance.now() - startMs;
	const intervalMs = 1000 / Math.max(1, targetFps);
	const delayMs = Math.max(0, intervalMs - elapsedMs);
	setTimeout(() => pumpRender(id), delayMs);
}

async function handleRequest(msg: WorkerRequest, signal?: AbortSignal) {
	try {
		switch (msg.type) {
			case 'set_options': {
				if (typeof msg.targetFps === 'number') targetFps = msg.targetFps;
				if (msg.debug) debugOptions = { ...debugOptions, ...msg.debug };
				replyOk(msg.id);
				return;
			}
			case 'load_rom': {
				const emu = await ensureEmulator();
				if (signal?.aborted) throw new ExecutionCancelled(0);
				romModel = msg.model ?? romModel;
				perfettoSymbolsPromise = null;
				if (typeof emu.load_rom_with_model === 'function') emu.load_rom_with_model(msg.bytes, romModel);
				else emu.load_rom(msg.bytes);
				machineGeneration = msg.generation ?? machineGeneration + 1;
				if (romModel === 'iq-7000' && typeof emu.set_iq7000_rtc_yyyymmddhhmm === 'function') {
					emu.set_iq7000_rtc_yyyymmddhhmm(formatHostUtcRtcSeed());
				}
				lastLcdTextUpdateMs = 0;
				lastLcdText = null;
				pressedCodes.clear();
				pendingVirtualRelease.clear();
				postFrame(captureFrame(true));
				replyOk(msg.id);
				return;
			}
			case 'get_model': {
				await ensureEmulator();
				replyOk(msg.id, romModel);
				return;
			}
			case 'step': {
				await ensureEmulator();
				await stepBounded(emulator, msg.instructions, { signal, onProgress: applyVirtualReleaseBudget });
				postFrame(captureFrame(true));
				replyOk(msg.id);
				return;
			}
			case 'snapshot': {
				await ensureEmulator();
				postFrame(captureFrame(true));
				replyOk(msg.id);
				return;
			}
			case 'lcd_trace': {
				await ensureEmulator();
				const trace = emulator.lcd_trace?.() ?? null;
				replyOk(msg.id, trace);
				return;
			}
			case 'eval_js': {
				// Ensure we don't race the run loop while mutating state.
				if (running) {
					running = false;
					runLoopId += 1;
				}
				await ensureEmulator();
				if (signal?.aborted) throw new ExecutionCancelled(0);
				const res = await evalScript(msg.source, signal);
				const scriptError = typeof res?.error === 'string' ? res.error : null;
				const fatalWasmError =
					typeof scriptError === 'string' && (isLikelyWasmTrap(scriptError) || isWasmBindgenBorrowError(scriptError));

				if (fatalWasmError) {
					if (typeof res?.error === 'string' && !res.error.toLowerCase().includes('reload')) {
						res.error = `${res.error}\n(postFrame skipped: emulator may need reload after a WASM trap)`;
					}
				} else if (!signal?.aborted) {
					try {
						postFrame(captureFrame(true));
					} catch (err) {
						const msg = err instanceof Error ? err.message : String(err);
						if (res && typeof res === 'object') {
							res.error = res.error ? `${res.error}\n(postFrame) ${msg}` : `(postFrame) ${msg}`;
						}
					}
				}
				replyOk(msg.id, res);
				return;
			}
			case 'start': {
				await ensureEmulator();
				if (signal?.aborted) throw new ExecutionCancelled(0);
				if (!running) {
					running = true;
					runLoopId += 1;
					const id = runLoopId;
					setTimeout(() => {
						pumpEmulator(id);
						pumpRender(id);
					}, 0);
				}
				replyOk(msg.id);
				return;
			}
			case 'stop': {
				running = false;
				runLoopId += 1;
				await operations.stop();
				replyOk(msg.id);
				// Ownership has been released. A slow final LCD/text capture
				// must not delay acknowledgement of the architectural pause.
				try {
					if (emulator) postFrame(captureFrame(true));
				} catch (error) {
					(self as any).postMessage({ type: 'render_error', error: String(error) });
				}
				return;
			}
			case 'virtual_key': {
				await ensureEmulator();
				if (msg.down) {
					if (!pressedCodes.has(msg.code)) {
						pressedCodes.add(msg.code);
						pendingVirtualRelease.delete(msg.code);
						emulator.press_matrix_code?.(msg.code);
					}
				} else {
					if (pressedCodes.has(msg.code)) {
						pendingVirtualRelease.set(msg.code, MIN_VIRTUAL_HOLD_INSTRUCTIONS);
					}
				}
				replyOk(msg.id);
				return;
			}
			case 'physical_key': {
				await ensureEmulator();
				if (msg.down) {
					emulator.press_matrix_code?.(msg.code);
				} else {
					emulator.release_matrix_code?.(msg.code);
				}
				replyOk(msg.id);
				return;
			}
		}
	} catch (err) {
		replyErr(msg.id, err);
	}
}

async function dispatchRequest(msg: WorkerRequest) {
	if (['load_rom', 'step', 'eval_js', 'start'].includes(msg.type)) {
		try {
			await operations.run(async (signal) => {
				if (msg.type !== 'start') {
					running = false;
					runLoopId++;
				}
				await handleRequest(msg, signal);
			});
		} catch (error) {
			replyErr(msg.id, error);
		}
	} else {
		await handleRequest(msg);
	}
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
	void dispatchRequest(event.data);
};
