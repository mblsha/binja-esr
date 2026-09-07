import { normalizeRomModel, type RomModel } from '../rom_model';
import { normalizeLcdKind, type LcdKind } from '../lcd_kind';
import { PCE500_KEY_FIFO_CAPACITY, resolvePce500KeyboardFifo } from './pce500_iocs_workspace';
import { ExecutionCancelled, stepBounded } from './bounded_step';
import { automaticHostSlice, type ExecutionMode, type PacingStatus } from './host_pacing';
import { WorkerOperations } from './worker_operations';
import { callBounded } from './bounded_call';
import { runIsolatedScript } from './isolated_script';
import { applyContact, HostInputs, InputBufferOverflow, type InputContact } from './host_inputs';
import { LatestFrame } from './latest_frame';

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
	| { id: number; type: 'set_execution_mode'; mode: ExecutionMode }
	| { id: number; type: 'pacing_status' }
	| { id: number; type: 'snapshot' }
	| { id: number; type: 'lcd_trace' }
	| { id: number; type: 'eval_js'; source: string }
	| { id: number; type: 'set_options'; targetFps?: number; typingCatchUp?: boolean; debug?: Partial<DebugOptions> }
	| {
			id: number;
			type: 'virtual_key' | 'physical_key';
			code: InputContact;
			down: boolean;
			owner?: string;
			cancel?: boolean;
			minimumHold?: number;
			buffered?: boolean;
			generation?: number;
	  }
	| { id: number; type: 'release_inputs'; source: 'physical' | 'virtual'; generation?: number }
	| { id: number; type: 'input_state' }
	| { id: number; type: 'frame_consumed'; sequence: number }
	| { id: number; type: 'frame_delivery_state' };

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
	typing: ReturnType<HostInputs['typingStatus']>;
	pacing: PacingStatus;
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

const IMEM_BASE = 0x100000;

const RUN_SLICE_MAX_INSTRUCTIONS = 200_000;

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
let typingCatchUp = false;
let debugOptions: DebugOptions = {
	regsOpen: false,
	callStackOpen: false,
	lcdTextOpen: false,
	debugStateOpen: false,
	keyboardDebugOpen: false,
};

let runLoopId = 0;
let lastLcdTextUpdateMs = 0;
let lastLcdText: string[] | null = null;

const inputs = new HostInputs((contact, down) => applyContact(emulator, contact, down));
const diagnosticContacts = new Set<number>();

function injectDiagnostic(code: number, release: boolean) {
	inputs.set({ source: 'diagnostic', owner: String(code), contact: code, down: !release });
	if (release) diagnosticContacts.delete(code);
	else diagnosticContacts.add(code);
	try {
		emulator.inject_matrix_event(code, release);
	} finally {
		inputs.reapply(code);
	}
}

function releaseScriptInputs() {
	for (const code of diagnosticContacts) injectDiagnostic(code, true);
	inputs.releaseSource('diagnostic');
	inputs.releaseSource('script');
}
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
	try {
		return await runIsolatedScript({
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
									limitBudget: inputs.limitBudget,
									onProgress: (used, slice) => {
										inputs.advance(used);
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
					reset: async () =>
						runWithErrorAsync('reset()', () => {
							releaseScriptInputs();
							inputs.clear();
							emulator.reset();
						}),
					step: async (instructions: number) =>
						runWithErrorAsync(`step(${instructions})`, async () => {
							await stepBounded(emulator, instructions, {
								signal,
								limitBudget: inputs.limitBudget,
								onProgress: inputs.advance,
							});
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
							inputs.set({ source: 'script', owner: String(code), contact: code, down: true }),
						),
					releaseMatrixCode: (code: number) =>
						runWithError(`keyboard.release(0x${code.toString(16).toUpperCase()})`, () =>
							inputs.set({ source: 'script', owner: String(code), contact: code, down: false }),
						),
					injectMatrixEvent: (code: number, release: boolean) =>
						runWithError(`keyboard.inject(0x${code.toString(16).toUpperCase()}, ${release})`, () =>
							injectDiagnostic(code, release),
						),
					pressOnKey: () => inputs.set({ source: 'script', owner: 'on', contact: 'on', down: true }),
					releaseOnKey: () => inputs.set({ source: 'script', owner: 'on', contact: 'on', down: false }),
					// Handler closures and their registry live only in the disposable worker.
					registerStub: () => {},
					clearStubs: () => {},
				});
			},
		});
	} finally {
		// A script owns its raw presses only for this invocation, on success too.
		// The isolated host has already drained all pending RPC cleanup here.
		releaseScriptInputs();
	}
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

function stepCore(boundaries: number) {
	return automaticHostSlice(emulator, boundaries, inputs, typingCatchUp);
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
			...inputs.snapshot(),
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
		typing: inputs.typingStatus(),
		pacing: emulator.pacing_status(),
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

const frames = new LatestFrame<Frame>(
	(frame, sequence) => {
		(self as any).postMessage({ type: 'frame', frame, sequence }, [
			frame.lcdPixels,
			frame.lcdAnnunciatorBytes,
			frame.lcdChipPixels,
		]);
	},
	(error) => self.postMessage({ type: 'render_error', error: String(error) }),
);

function requestFrame(forceText: boolean) {
	frames.request(() => captureFrame(forceText));
}

function pumpEmulator(id: number) {
	if (!running || !emulator || id !== runLoopId) return;
	let waitMs: number;
	try {
		waitMs = stepCore(RUN_SLICE_MAX_INSTRUCTIONS);
	} catch (err) {
		// Crash stops the run loop; render loop will stop too.
		running = false;
		frames.discardPending();
		(self as any).postMessage({ type: 'fatal', error: String(err) });
		return;
	}
	setTimeout(() => pumpEmulator(id), waitMs);
}

function pumpRender(id: number) {
	if (!running || !emulator || id !== runLoopId) return;
	const startMs = performance.now();
	requestFrame(false);
	const elapsedMs = performance.now() - startMs;
	const intervalMs = 1000 / Math.max(1, targetFps);
	const delayMs = Math.max(0, intervalMs - elapsedMs);
	setTimeout(() => pumpRender(id), delayMs);
}

async function handleRequest(msg: WorkerRequest, signal?: AbortSignal) {
	try {
		switch (msg.type) {
			case 'frame_consumed': {
				frames.consumed(msg.sequence);
				return; // This is the one-way presentation credit, not another RPC.
			}
			case 'frame_delivery_state': {
				replyOk(msg.id, frames.snapshot());
				return;
			}
			case 'set_options': {
				if (typeof msg.typingCatchUp === 'boolean') typingCatchUp = msg.typingCatchUp;
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
				releaseScriptInputs();
				inputs.clear();
				if (typeof emu.load_rom_with_model === 'function') emu.load_rom_with_model(msg.bytes, romModel);
				else emu.load_rom(msg.bytes);
				machineGeneration = msg.generation ?? machineGeneration + 1;
				if (romModel === 'iq-7000' && typeof emu.set_iq7000_rtc_yyyymmddhhmm === 'function') {
					emu.set_iq7000_rtc_yyyymmddhhmm(formatHostUtcRtcSeed());
				}
				lastLcdTextUpdateMs = 0;
				lastLcdText = null;
				requestFrame(true);
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
				await stepBounded(emulator, msg.instructions, {
					signal,
					limitBudget: inputs.limitBudget,
					onProgress: inputs.advance,
				});
				requestFrame(true);
				replyOk(msg.id);
				return;
			}
			case 'snapshot': {
				await ensureEmulator();
				requestFrame(true);
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
					frames.discardPending();
					if (typeof res?.error === 'string' && !res.error.toLowerCase().includes('reload')) {
						res.error = `${res.error}\n(postFrame skipped: emulator may need reload after a WASM trap)`;
					}
				} else if (!signal?.aborted) {
					try {
						requestFrame(true);
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
			case 'pacing_status': {
				await ensureEmulator();
				replyOk(msg.id, emulator.pacing_status());
				return;
			}
			case 'set_execution_mode': {
				await ensureEmulator();
				if (signal?.aborted) throw new ExecutionCancelled(0);
				if (running) throw new Error('Pause before changing execution mode');
				emulator.set_execution_mode(msg.mode);
				replyOk(msg.id, emulator.pacing_status());
				requestFrame(false);
				return;
			}
			case 'start': {
				await ensureEmulator();
				if (signal?.aborted) throw new ExecutionCancelled(0);
				if (emulator.execution_mode() === 'deterministic')
					throw new Error('Deterministic mode requires explicit Step or Function Runner budgets');
				if (!running) {
					emulator.rebase_pacing();
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
				emulator?.rebase_pacing();
				replyOk(msg.id);
				// Ownership has been released. A slow final LCD/text capture
				// must not delay acknowledgement of the architectural pause.
				try {
					if (emulator) requestFrame(true);
				} catch (error) {
					(self as any).postMessage({ type: 'render_error', error: String(error) });
				}
				return;
			}
			case 'virtual_key':
			case 'physical_key': {
				await ensureEmulator();
				if (msg.generation !== undefined && msg.generation !== machineGeneration)
					throw new Error('Stale input generation');
				// ON is a priority power contact, never stuck behind a typing backlog.
				if (msg.code === 'on' && msg.down) inputs.clearTyping();
				inputs.set({
					source: msg.type === 'virtual_key' ? 'virtual' : 'physical',
					owner: msg.owner ?? String(msg.code),
					contact: msg.code,
					down: msg.down,
					cancel: msg.cancel,
					minimumHold: msg.type === 'virtual_key' ? (msg.minimumHold ?? 40_000) : 0,
					buffered: msg.type === 'physical_key' && (msg.buffered ?? false),
				});
				(self as any).postMessage({
					type: 'input_status',
					generation: machineGeneration,
					typing: inputs.typingStatus(),
				});
				replyOk(msg.id, {
					generation: machineGeneration,
					applied: !msg.buffered,
					queued: Boolean(msg.buffered),
					contact: msg.code,
					down: msg.down,
				});
				return;
			}
			case 'release_inputs': {
				if (msg.generation !== undefined && msg.generation !== machineGeneration)
					throw new Error('Stale input generation');
				if (msg.source !== 'physical' && msg.source !== 'virtual') throw new Error('Invalid host input source');
				inputs.releaseSource(msg.source);
				(self as any).postMessage({
					type: 'input_status',
					generation: machineGeneration,
					typing: inputs.typingStatus(),
				});
				replyOk(msg.id, { generation: machineGeneration, applied: true });
				return;
			}
			case 'input_state': {
				replyOk(msg.id, { generation: machineGeneration, ...inputs.snapshot(), rust: emulator.input_contacts() });
				return;
			}
		}
	} catch (err) {
		if (err instanceof InputBufferOverflow) {
			running = false;
			runLoopId++;
			await operations.stop();
			emulator?.rebase_pacing();
			(self as any).postMessage({ type: 'input_paused', generation: machineGeneration, error: err.message });
			requestFrame(true);
		}
		replyErr(msg.id, err);
	}
}

async function dispatchRequest(msg: WorkerRequest) {
	if (['load_rom', 'step', 'eval_js', 'start', 'set_execution_mode'].includes(msg.type)) {
		try {
			await operations.run(async (signal) => {
				if (msg.type !== 'start' && msg.type !== 'set_execution_mode') {
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
