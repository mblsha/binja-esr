type DebugOptions = {
	regsOpen: boolean;
	callStackOpen: boolean;
	lcdTextOpen: boolean;
	debugStateOpen: boolean;
	keyboardDebugOpen: boolean;
};

type WorkerRequest =
	| { id: number; type: 'load_rom'; bytes: Uint8Array; romSource?: string | null }
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
	lcdPixels: ArrayBuffer;
	lcdChipPixels: ArrayBuffer;
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
const FIFO_BASE_ADDR = 0x00bfc96;
const FIFO_HEAD_ADDR = 0x00bfc9d;
const FIFO_TAIL_ADDR = 0x00bfc9e;

const RUN_SLICE_MIN_INSTRUCTIONS = 1;
const RUN_SLICE_MAX_INSTRUCTIONS = 200_000;
const RUN_YIELD_MS = 0;

const LCD_TEXT_UPDATE_INTERVAL_MS = 250;

let wasm: any = null;
let emulator: any = null;
let buildInfo: { version: string; git_commit: string; build_timestamp: string } | null = null;

let running = false;
let targetFps = 30;
let debugOptions: DebugOptions = {
	regsOpen: false,
	callStackOpen: true,
	lcdTextOpen: true,
	debugStateOpen: false,
	keyboardDebugOpen: false
};

let runLoopId = 0;
let runSliceInstructions = 2000;
let lastLcdTextUpdateMs = 0;
let lastLcdText: string[] | null = null;

const pressedCodes = new Set<number>();
const pendingVirtualRelease = new Map<number, number>();

function safeJson(value: any): string {
	return JSON.stringify(value, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
}

async function evalScript(source: string): Promise<any> {
	const { createEvalApi, Reg, Flag } = await import('../debug/sc62015_eval_api');
	const { runUserJs } = await import('../debug/run_user_js');
	const { IOCS } = await import('../debug/iocs');
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
	const api = createEvalApi({
		callFunction: async (
			address: number,
			maxInstructions: number,
			options?: { trace?: boolean; probe?: { pc: number; maxSamples?: number } } | null
		) =>
			runWithErrorAsync(`call(0x${address.toString(16).toUpperCase()})`, async () => {
				const raw =
					emulator.call_function_ex?.(address, maxInstructions, {
						trace: Boolean(options?.trace),
						probe_pc: options?.probe ? options.probe.pc : null,
						probe_max_samples: options?.probe?.maxSamples ?? 256
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
			runWithError(`write8(0x${addr.toString(16).toUpperCase()}, ${value})`, () =>
				emulator.write_u8?.(addr, value)
			),
		pressMatrixCode: (code: number) =>
			runWithError(`keyboard.press(0x${code.toString(16).toUpperCase()})`, () =>
				emulator.press_matrix_code?.(code)
			),
		releaseMatrixCode: (code: number) =>
			runWithError(`keyboard.release(0x${code.toString(16).toUpperCase()})`, () =>
				emulator.release_matrix_code?.(code)
			),
		injectMatrixEvent: (code: number, release: boolean) =>
			runWithError(`keyboard.inject(0x${code.toString(16).toUpperCase()}, ${release})`, () =>
				emulator.inject_matrix_event?.(code, release)
			)
	});
	let resultJson: string | null = null;
	let error: string | null = null;
	try {
		const result = await runUserJs(source, api, Reg, Flag, IOCS);
		resultJson = safeJson(result);
	} catch (err) {
		error = err instanceof Error ? err.message : String(err);
	}
	return {
		events: api.events,
		calls: api.calls,
		prints: api.prints,
		resultJson,
		error
	};
}

async function ensureEmulator(): Promise<any> {
	if (!wasm) {
		wasm = await import('../wasm/pce500_wasm/pce500_wasm.js');
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
		emulator = new wasm.Pce500Emulator();
		try {
			buildInfo = emulator.build_info?.() ?? null;
		} catch {
			buildInfo = null;
		}
	}
	return emulator;
}

function replyOk(id: number, result?: any) {
	(self as any).postMessage({ type: 'reply', id, ok: true, ...(result !== undefined ? { result } : {}) } satisfies WorkerReply);
}

function replyErr(id: number, error: unknown) {
	(self as any).postMessage({
		type: 'reply',
		id,
		ok: false,
		error: error instanceof Error ? error.message : String(error)
	} satisfies WorkerReply);
}

function applyVirtualReleaseBudget(stepped: number) {
	if (pendingVirtualRelease.size === 0) return;
	for (const [code, remaining] of pendingVirtualRelease.entries()) {
		const next = remaining - stepped;
		if (next <= 0) {
			pendingVirtualRelease.delete(code);
			pressedCodes.delete(code);
			emulator.inject_matrix_event?.(code, true);
		} else {
			pendingVirtualRelease.set(code, next);
		}
	}
}

function stepCore(instructions: number) {
	emulator.step(instructions);
	applyVirtualReleaseBudget(instructions);
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
		const fifoHead = emulator.read_u8?.(FIFO_HEAD_ADDR) ?? null;
		const fifoTail = emulator.read_u8?.(FIFO_TAIL_ADDR) ?? null;
		const fifo = Array.from({ length: 16 }, (_, i) => emulator.read_u8?.(FIFO_BASE_ADDR + i) ?? 0);
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
			pendingVirtualRelease: Array.from(pendingVirtualRelease.entries())
		};
		return { keyboardDebug, keyboardDebugJson: safeJson(keyboardDebug) };
	} catch {
		return null;
	}
}

function captureFrame(forceText: boolean): Frame {
	const pixels = emulator.lcd_pixels();
	const pixelsCopy = new Uint8Array(pixels);
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
		const shouldUpdate =
			forceText || !running || nowMs - lastLcdTextUpdateMs >= LCD_TEXT_UPDATE_INTERVAL_MS;
		if (shouldUpdate) {
			lastLcdTextUpdateMs = nowMs;
			lastLcdText = emulator.lcd_text?.() ?? lastLcdText;
		}
		lcdText = lastLcdText ?? null;
	}

	const regs = debugOptions.regsOpen ? emulator.regs?.() ?? null : null;
	const callStack = debugOptions.callStackOpen ? emulator.call_stack?.() ?? null : null;
	const debugState = debugOptions.debugStateOpen ? emulator.debug_state?.() ?? null : null;

	const kb = snapshotKeyboard();
	return {
		lcdPixels: pixelsCopy.buffer,
		lcdChipPixels: chipPixelsCopy.buffer,
		pc,
		instructionCount,
		halted,
		buildInfo,
		lcdText,
		regs,
		callStack,
		debugState,
		keyboardDebug: kb?.keyboardDebug ?? null,
		keyboardDebugJson: kb?.keyboardDebugJson ?? null
	};
}

function postFrame(frame: Frame) {
	(self as any).postMessage({ type: 'frame', frame }, [frame.lcdPixels, frame.lcdChipPixels]);
}

function pumpEmulator(id: number) {
	if (!running || !emulator || id !== runLoopId) return;
	const runMaxWorkMs = 4;
	const runSliceTargetMs = 0.4;
	const startMs = performance.now();
	try {
		while (performance.now() - startMs < runMaxWorkMs) {
			const sliceStart = performance.now();
			stepCore(runSliceInstructions);
			const sliceMs = performance.now() - sliceStart;
			if (sliceMs > 0) {
				const scaled = Math.floor(runSliceInstructions * (runSliceTargetMs / sliceMs));
				runSliceInstructions = Math.max(
					RUN_SLICE_MIN_INSTRUCTIONS,
					Math.min(RUN_SLICE_MAX_INSTRUCTIONS, scaled)
				);
			}
			if (!running || id !== runLoopId) return;
		}
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

async function handleRequest(msg: WorkerRequest) {
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
				emu.load_rom(msg.bytes);
				lastLcdTextUpdateMs = 0;
				lastLcdText = null;
				pressedCodes.clear();
				pendingVirtualRelease.clear();
				postFrame(captureFrame(true));
				replyOk(msg.id);
				return;
			}
			case 'step': {
				await ensureEmulator();
				stepCore(msg.instructions);
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
				const res = await evalScript(msg.source);
				try {
					postFrame(captureFrame(true));
				} catch (err) {
					const msg = err instanceof Error ? err.message : String(err);
					if (res && typeof res === 'object') {
						res.error = res.error ? `${res.error}\n(postFrame) ${msg}` : `(postFrame) ${msg}`;
					}
				}
				replyOk(msg.id, res);
				return;
			}
			case 'start': {
				await ensureEmulator();
				if (!running) {
					running = true;
					runSliceInstructions = 2000;
					runLoopId += 1;
					pumpEmulator(runLoopId);
					pumpRender(runLoopId);
				}
				replyOk(msg.id);
				return;
			}
			case 'stop': {
				running = false;
				runLoopId += 1;
				replyOk(msg.id);
				return;
			}
			case 'virtual_key': {
				await ensureEmulator();
				if (msg.down) {
					if (!pressedCodes.has(msg.code)) {
						pressedCodes.add(msg.code);
						pendingVirtualRelease.delete(msg.code);
						emulator.inject_matrix_event?.(msg.code, false);
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

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
	void handleRequest(event.data);
};
