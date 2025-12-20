<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import { LCD_CHIP_COLS, LCD_CHIP_ROWS, LCD_COLS, LCD_ROWS } from '$lib/lcd';
	import VirtualKeyboard from '$lib/components/VirtualKeyboard.svelte';
	import { matrixCodeForKeyEvent } from '$lib/keymap';
	import { createEvalApi, Flag, Reg } from '$lib/debug/sc62015_eval_api';
	import { IOCS } from '$lib/debug/iocs';
	import { runUserJs } from '$lib/debug/run_user_js';
	import FunctionRunnerPanel from '$lib/components/FunctionRunnerPanel.svelte';
	import type { FunctionRunnerOutput } from '$lib/debug/function_runner_types';
	import { createPersistedStore } from '$lib/stores/persisted';
	import { normalizeRomModel, type RomModel } from '$lib/rom_model';

	const ROM_MODEL_STORAGE_KEY = 'sc62015:rom-model';
	const romModelStore = createPersistedStore<RomModel>(ROM_MODEL_STORAGE_KEY, 'pc-e500', {
		serialize: (value) => value,
		deserialize: (raw) => normalizeRomModel(raw) ?? 'pc-e500',
	});

	let wasm: any = null;
	let emulator: any = null;
	let worker: Worker | null = null;
	let workerNextId = 1;
	const workerPending = new Map<number, { resolve: (value: any) => void; reject: (err: unknown) => void }>();
	const canUseWorker =
		typeof window !== 'undefined' && typeof Worker !== 'undefined' && !Boolean((import.meta as any)?.env?.VITEST);

	let lcdPixels: Uint8Array | null = null;
	let lcdChipPixels: Uint8Array | null = null;
	let lcdCols = LCD_COLS;
	let lcdRows = LCD_ROWS;
	let lcdKind: string | null = null;
	const CHIP_PIXELS_LEN = LCD_CHIP_COLS * LCD_CHIP_ROWS;
	$: lcdLeftChipPixels =
		lcdChipPixels && lcdChipPixels.length >= CHIP_PIXELS_LEN ? lcdChipPixels.subarray(0, CHIP_PIXELS_LEN) : null;
	$: lcdRightChipPixels =
		lcdChipPixels && lcdChipPixels.length >= CHIP_PIXELS_LEN * 2
			? lcdChipPixels.subarray(CHIP_PIXELS_LEN, CHIP_PIXELS_LEN * 2)
			: null;
	let lcdText: string[] | null = null;
	let regs: any = null;
	let callStack: number[] | null = null;
	let debugState: any = null;
	let lastError: string | null = null;
	let romSource: string | null = null;
	let pcReg: number | null = null;
	let halted = false;
	let instructionCount: string | null = null;
	let buildInfo: { version: string; git_commit: string; build_timestamp: string } | null = null;
	let romLoaded = false;
	let romModel: RomModel = 'pc-e500';
	let romModelWasPersisted = false;
	$: romModel = $romModelStore;

	let running = false;
	let targetFps = 30;

	let functionRunnerBusy = false;
	const pressedCodes = new Set<number>();
	const physicalHeldCodes = new Set<number>();
	const pendingVirtualRelease = new Map<number, number>();
	const MIN_VIRTUAL_HOLD_INSTRUCTIONS = 40_000;
	const IMEM_BASE = 0x100000;
	const FIFO_BASE_ADDR = 0x00bfc96;
	const FIFO_HEAD_ADDR = 0x00bfc9d;
	const FIFO_TAIL_ADDR = 0x00bfc9e;
	const debugLog: string[] = [];
	let physicalKeyboardEnabled = false;
	let keyboardDebugOpen = false;
	let regsOpen = false;
	let callStackOpen = true;
	let lcdTextOpen = true;
	let debugStateOpen = false;
	let physicalKeyboardHookInstalled = false;
	let mounted = false;

	const LCD_TEXT_UPDATE_INTERVAL_MS = 250;
	let lastLcdTextUpdateMs = 0;
	$: targetFrameIntervalMs = 1000 / Math.max(1, targetFps);

	const RUN_SLICE_MIN_INSTRUCTIONS = 1;
	const RUN_SLICE_MAX_INSTRUCTIONS = 200_000;
	const runSliceTargetMs = 0.4;
	const runMaxWorkMs = 4;
	const RUN_YIELD_MS = 0;
	let runLoopId = 0;
	let runSliceInstructions = 2000;

	let debugKio: {
		pc: number | null;
		instr: any;
		imr: number | null;
		isr: number | null;
		kol: number | null;
		koh: number | null;
		kil: number | null;
		fifoHead: number | null;
		fifoTail: number | null;
		fifo: number[];
	} | null = null;
	let debugKioJson: string | null = null;

	function applyWorkerFrame(frame: any) {
		try {
			if (frame?.lcdPixels instanceof ArrayBuffer) {
				lcdPixels = new Uint8Array(frame.lcdPixels);
			}
			if (frame?.lcdChipPixels instanceof ArrayBuffer) {
				lcdChipPixels = new Uint8Array(frame.lcdChipPixels);
			}
			if (typeof frame?.lcdCols === 'number') lcdCols = frame.lcdCols;
			if (typeof frame?.lcdRows === 'number') lcdRows = frame.lcdRows;
			if (typeof frame?.lcdKind === 'string') lcdKind = frame.lcdKind;
			lcdText = frame?.lcdText ?? null;
			buildInfo = frame?.buildInfo ?? buildInfo;
			regs = frame?.regs ?? regs;
			callStack = frame?.callStack ?? callStack;
			debugState = frame?.debugState ?? null;
			pcReg = frame?.pc ?? pcReg;
			halted = Boolean(frame?.halted);
			instructionCount = frame?.instructionCount ?? instructionCount;

			if (frame?.keyboardDebug) {
				debugKio = frame.keyboardDebug;
				debugKioJson = frame.keyboardDebugJson ?? null;
				pressedCodes.clear();
				for (const code of frame.keyboardDebug.pressedCodes ?? []) pressedCodes.add(code);
				pendingVirtualRelease.clear();
				for (const [code, remaining] of frame.keyboardDebug.pendingVirtualRelease ?? []) {
					pendingVirtualRelease.set(code, remaining);
				}
			}
		} catch (err) {
			lastError = String(err);
		}
	}

	function workerPost(message: any, transfer?: Transferable[]) {
		if (!worker) return;
		if (transfer) worker.postMessage(message, transfer);
		else worker.postMessage(message);
	}

	function workerCall<T = any>(type: string, payload: any = {}, transfer?: Transferable[]): Promise<T> {
		if (!worker) return Promise.reject(new Error('worker not ready'));
		const id = workerNextId++;
		const message = { id, type, ...payload };
		return new Promise((resolve, reject) => {
			workerPending.set(id, { resolve, reject });
			workerPost(message, transfer);
		});
	}

	function pushWorkerOptions() {
		if (!worker) return;
		const id = workerNextId++;
		workerPost({
			id,
			type: 'set_options',
			targetFps,
			debug: { regsOpen, callStackOpen, lcdTextOpen, debugStateOpen, keyboardDebugOpen },
		});
	}

	async function ensureWorker(): Promise<void> {
		if (!canUseWorker || worker) return;
		worker = new Worker(new URL('../lib/emulator/pce500.worker.ts', import.meta.url), { type: 'module' });
		worker.onmessage = (event: MessageEvent<any>) => {
			const data = event.data;
			if (!data) return;
			if (data.type === 'reply') {
				const pending = workerPending.get(data.id);
				if (!pending) return;
				workerPending.delete(data.id);
				if (data.ok) pending.resolve(data.result);
				else pending.reject(new Error(data.error ?? 'worker error'));
				return;
			}
			if (data.type === 'frame') {
				applyWorkerFrame(data.frame);
				return;
			}
			if (data.type === 'fatal') {
				lastError = `Worker error: ${data.error ?? 'unknown error'}`;
				running = false;
			}
		};
		worker.onerror = (event) => {
			lastError = `Worker crashed: ${String(event)}`;
			running = false;
		};
		pushWorkerOptions();
	}

	function isDevBuild(): boolean {
		try {
			return Boolean((import.meta as any)?.env?.DEV) && !Boolean((import.meta as any)?.env?.VITEST);
		} catch {
			return false;
		}
	}

	function safeJson(value: any): string {
		return JSON.stringify(value, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
	}

	let perfettoSymbolsPromise: Promise<void> | null = null;

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

	async function runFunctionRunner(source: string): Promise<FunctionRunnerOutput> {
		functionRunnerBusy = true;
		try {
			if (running) stop();
			await ensureWorker();
			if (worker) {
				return await workerCall('eval_js', { source });
			}

			const emu = await ensureEmulator();
			const api = createEvalApi({
				callFunction: async (
					address: number,
					maxInstructions: number,
					options?: { trace?: boolean; probe?: { pc: number; maxSamples?: number } } | null,
				) => {
					if (options?.trace) await ensurePerfettoSymbols();
					const raw =
						emu.call_function_ex?.(address, maxInstructions, {
							trace: Boolean(options?.trace),
							probe_pc: options?.probe ? options.probe.pc : null,
							probe_max_samples: options?.probe?.maxSamples ?? 256,
						}) ?? emu.call_function(address, maxInstructions);
					if (typeof raw === 'string') return JSON.parse(raw);
					return raw;
				},
				reset: async () => {
					await Promise.resolve(emu.reset?.());
				},
				step: async (instructions: number) => {
					await Promise.resolve(emu.step?.(instructions));
				},
				getReg: (name: string) => emu.get_reg?.(name) ?? 0,
				setReg: (name: string, value: number) => emu.set_reg?.(name, value),
				read8: (addr: number) => emu.read_u8?.(addr) ?? 0,
				write8: (addr: number, value: number) => emu.write_u8?.(addr, value),
				pressMatrixCode: (code: number) => emu.press_matrix_code?.(code),
				releaseMatrixCode: (code: number) => emu.release_matrix_code?.(code),
				injectMatrixEvent: (code: number, release: boolean) => emu.inject_matrix_event?.(code, release),
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
				error,
			};
		} catch (err) {
			return {
				events: [],
				calls: [],
				prints: [],
				resultJson: null,
				error: err instanceof Error ? err.message : String(err),
			};
		} finally {
			functionRunnerBusy = false;
		}
	}

	function hex(value: number | null | undefined, width = 5): string {
		if (value === null || value === undefined) return '—';
		return `0x${value.toString(16).toUpperCase().padStart(width, '0')}`;
	}

	function formatBuildInfo(info: typeof buildInfo): string {
		if (!info) return '—';
		const ts = (() => {
			if (!info.build_timestamp) return 'ts=?';
			const raw = Number.parseInt(info.build_timestamp, 10);
			if (!Number.isFinite(raw)) return `ts=${info.build_timestamp}`;
			const ms = raw > 1_000_000_000_000 ? raw : raw * 1000;
			const date = new Date(ms);
			if (Number.isNaN(date.getTime())) return `ts=${info.build_timestamp}`;
			const offsetMinutes = -date.getTimezoneOffset();
			const sign = offsetMinutes >= 0 ? '+' : '-';
			const abs = Math.abs(offsetMinutes);
			const hh = String(Math.floor(abs / 60)).padStart(2, '0');
			const mm = String(abs % 60).padStart(2, '0');
			const localIso = new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().replace('Z', '');
			return `ts=${localIso}${sign}${hh}:${mm}`;
		})();
		return `v${info.version} ${info.git_commit} ${ts}`;
	}

	function formatFunction(pc: number): string {
		return `sub_${pc.toString(16).toUpperCase().padStart(5, '0')}`;
	}

	function getReg(name: string): number | null {
		for (const [key, value] of regsEntries(regs)) {
			if (key === name) return value;
		}
		return null;
	}

	function regsEntries(input: any): [string, number][] {
		if (!input) return [];
		try {
			if (typeof input.entries === 'function') {
				return (Array.from(input.entries()) as [unknown, unknown][]).filter(
					(entry): entry is [string, number] =>
						Array.isArray(entry) && entry.length === 2 && typeof entry[0] === 'string' && typeof entry[1] === 'number',
				);
			}
			return Object.entries(input).filter(([k, v]) => typeof k === 'string' && typeof v === 'number') as [
				string,
				number,
			][];
		} catch {
			return [];
		}
	}

	function logDebug(line: string) {
		debugLog.unshift(line);
		if (debugLog.length > 50) debugLog.pop();
	}

	async function copyDebugJson() {
		if (!debugKioJson) return;
		try {
			await navigator.clipboard.writeText(debugKioJson);
			logDebug('copied debug JSON to clipboard');
		} catch {
			logDebug('failed to copy debug JSON (clipboard unavailable)');
		}
	}

	function setMatrixCode(code: number, down: boolean) {
		if (worker) {
			const id = workerNextId++;
			workerPost({ id, type: 'virtual_key', code, down });
			logDebug(`${down ? 'press' : 'release'} ${hex(code, 2)}`);
			return;
		}
		if (!emulator) return;
		if (down) {
			if (pressedCodes.has(code)) return;
			pressedCodes.add(code);
			// Inject directly so short taps are observable even if firmware polling is sparse.
			emulator.inject_matrix_event?.(code, false);
			logDebug(`press ${hex(code, 2)}`);
		} else {
			if (!pressedCodes.has(code)) return;
			pressedCodes.delete(code);
			pendingVirtualRelease.delete(code);
			emulator.inject_matrix_event?.(code, true);
			logDebug(`release ${hex(code, 2)}`);
		}
	}

	function setPhysicalMatrixCode(code: number, down: boolean) {
		if (worker) {
			const id = workerNextId++;
			workerPost({ id, type: 'physical_key', code, down });
			return;
		}
		if (!emulator) return;
		if (down) {
			emulator.press_matrix_code?.(code);
		} else {
			emulator.release_matrix_code?.(code);
		}
	}

	function releaseAllPhysicalHeldCodes() {
		if (physicalHeldCodes.size === 0) return;
		for (const code of physicalHeldCodes.values()) {
			setPhysicalMatrixCode(code, false);
		}
		physicalHeldCodes.clear();
	}

	function installPhysicalKeyboardHook() {
		if (physicalKeyboardHookInstalled) return;
		window.addEventListener('keydown', onKeyDown, { passive: false });
		window.addEventListener('keyup', onKeyUp, { passive: false });
		physicalKeyboardHookInstalled = true;
	}

	function uninstallPhysicalKeyboardHook() {
		if (!physicalKeyboardHookInstalled) return;
		window.removeEventListener('keydown', onKeyDown);
		window.removeEventListener('keyup', onKeyUp);
		physicalKeyboardHookInstalled = false;
	}

	function virtualPress(code: number) {
		if (worker) {
			setMatrixCode(code, true);
			return;
		}
		setMatrixCode(code, true);
		pendingVirtualRelease.delete(code);
	}

	function virtualRelease(code: number) {
		if (worker) {
			setMatrixCode(code, false);
			return;
		}
		if (!pressedCodes.has(code)) return;
		pendingVirtualRelease.set(code, MIN_VIRTUAL_HOLD_INSTRUCTIONS);
	}

	function applyVirtualReleaseBudget(stepped: number) {
		if (pendingVirtualRelease.size === 0) return;
		for (const [code, remaining] of pendingVirtualRelease.entries()) {
			const next = remaining - stepped;
			if (next <= 0) {
				pendingVirtualRelease.delete(code);
				setMatrixCode(code, false);
			} else {
				pendingVirtualRelease.set(code, next);
			}
		}
	}

	async function ensureEmulator(): Promise<any> {
		if (!wasm) {
			wasm = await import('$lib/wasm/sc62015_wasm');
			if (typeof wasm.default === 'function') {
				const url = new URL('$lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url);
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
			if (model && !romModelWasPersisted) $romModelStore = model;
		} catch {
			// ignore
		}
		return emulator;
	}

	async function syncRomModelFromRuntime(): Promise<void> {
		if (romModelWasPersisted) return;
		if (worker) {
			try {
				const model = (await workerCall('get_model')) as any;
				if (typeof model === 'string') $romModelStore = model as RomModel;
			} catch {
				// ignore
			}
			return;
		}
		await ensureEmulator();
	}

	async function tryAutoLoadRom(force = false) {
		await ensureWorker();
		if (!force) await syncRomModelFromRuntime();
		if (!force && !worker && emulator?.has_rom?.()) return;
		try {
			const res = await fetch(`/api/rom?model=${encodeURIComponent(romModel)}`);
			if (!res.ok) return;
			perfettoSymbolsPromise = null;
			romSource = res.headers.get('x-rom-source');
			const bytes = new Uint8Array(await res.arrayBuffer());
			if (worker) {
				await workerCall('load_rom', { bytes, romSource, model: romModel }, [bytes.buffer]);
				romLoaded = true;
				return;
			}
			const emu = await ensureEmulator();
			emu.load_rom_with_model?.(bytes, romModel) ?? emu.load_rom(bytes);
			romLoaded = true;
			refreshAllNow();
		} catch (err) {
			lastError = `Auto-load failed: ${String(err)}`;
			romLoaded = false;
		}
	}

	function refreshFast() {
		if (worker) return;
		if (!emulator) return;
		try {
			const geometry = emulator.lcd_geometry?.() ?? null;
			if (geometry && typeof geometry === 'object') {
				const kind = (geometry as any).kind;
				const cols = (geometry as any).cols;
				const rows = (geometry as any).rows;
				if (typeof kind === 'string') lcdKind = kind;
				if (typeof cols === 'number') lcdCols = cols;
				if (typeof rows === 'number') lcdRows = rows;
			}
		} catch {
			// ignore
		}
		lcdPixels = emulator.lcd_pixels();
		lcdChipPixels = emulator.lcd_chip_pixels();
		try {
			pcReg = emulator.get_reg?.('PC') ?? null;
		} catch {
			pcReg = null;
		}
		try {
			halted = Boolean(emulator.halted?.());
		} catch {
			halted = false;
		}
		try {
			const count = emulator.instruction_count?.();
			instructionCount = count?.toString?.() ?? null;
		} catch {
			instructionCount = null;
		}
	}

	function refreshUi(nowMs: number) {
		if (worker) return;
		if (!emulator) return;
		if (regsOpen) regs = emulator.regs?.() ?? regs;
		if (callStackOpen) callStack = emulator.call_stack?.() ?? callStack;
		debugState = debugStateOpen ? (emulator.debug_state?.() ?? debugState) : null;
		if (lcdTextOpen && (!running || nowMs - lastLcdTextUpdateMs >= LCD_TEXT_UPDATE_INTERVAL_MS)) {
			lastLcdTextUpdateMs = nowMs;
			lcdText = emulator.lcd_text?.() ?? lcdText;
		}
	}

	function refreshAllNow() {
		if (worker) {
			void workerCall('snapshot');
			return;
		}
		if (!emulator) return;
		lastLcdTextUpdateMs = 0;
		refreshFast();
		regs = emulator.regs?.() ?? regs;
		callStack = emulator.call_stack?.() ?? callStack;
		refreshUi(performance.now());
	}

	function snapshotKeyboardState() {
		if (worker) {
			void workerCall('snapshot');
			return;
		}
		if (!emulator) {
			debugKio = null;
			debugKioJson = null;
			return;
		}
		try {
			const pc = emulator.get_reg?.('PC') ?? null;
			const instr = emulator.instruction_count?.() ?? null;
			const imr = emulator.imr?.() ?? null;
			const isr = emulator.isr?.() ?? null;
			const kol = emulator.read_u8?.(IMEM_BASE + 0xf0) ?? null;
			const koh = emulator.read_u8?.(IMEM_BASE + 0xf1) ?? null;
			const kil = emulator.read_u8?.(IMEM_BASE + 0xf2) ?? null;
			const fifoHead = emulator.read_u8?.(FIFO_HEAD_ADDR) ?? null;
			const fifoTail = emulator.read_u8?.(FIFO_TAIL_ADDR) ?? null;
			const fifo = Array.from({ length: 16 }, (_, i) => emulator.read_u8?.(FIFO_BASE_ADDR + i) ?? 0);
			debugKio = { pc, instr, imr, isr, kol, koh, kil, fifoHead, fifoTail, fifo };
			debugKioJson = safeJson({
				...debugKio,
				pressedCodes: Array.from(pressedCodes.values()),
				pendingVirtualRelease: Array.from(pendingVirtualRelease.entries()),
			});
		} catch {
			debugKio = null;
			debugKioJson = null;
		}
	}

	function dumpKeyboardState(tag = 'dump') {
		if (worker) {
			if (debugKioJson) {
				console.log(`[pce500] ${tag}`, debugKioJson);
			} else {
				console.log(`[pce500] ${tag}: no snapshot yet (use Refresh)`);
			}
			return;
		}
		if (!emulator) {
			console.log(`[pce500] ${tag}: emulator not ready`);
			return;
		}
		try {
			const pc = emulator.get_reg?.('PC');
			const instr = emulator.instruction_count?.();
			const imr = emulator.imr?.();
			const isr = emulator.isr?.();
			const kol = emulator.read_u8?.(IMEM_BASE + 0xf0);
			const koh = emulator.read_u8?.(IMEM_BASE + 0xf1);
			const kil = emulator.read_u8?.(IMEM_BASE + 0xf2);
			const fifoHead = emulator.read_u8?.(FIFO_HEAD_ADDR);
			const fifoTail = emulator.read_u8?.(FIFO_TAIL_ADDR);
			const fifo = Array.from({ length: 8 }, (_, i) => emulator.read_u8?.(FIFO_BASE_ADDR + i) ?? 0);
			console.log(`[pce500] ${tag}`, {
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
			});
		} catch (err) {
			console.log(`[pce500] ${tag}: dump failed`, err);
		}
	}

	function installDevtoolsDebugHelpers() {
		if (!isDevBuild()) return;
		(globalThis as any).__pce500 = {
			get emulator() {
				return worker ? null : emulator;
			},
			dump: dumpKeyboardState,
			step: (n: number) => stepOnce(n),
			read: (addr: number) => emulator?.read_u8?.(addr),
			readInternal: (offset: number) => emulator?.read_u8?.(IMEM_BASE + offset),
			lcdTrace: async () => {
				if (worker) return await workerCall('lcd_trace');
				return emulator?.lcd_trace?.();
			},
			press: (code: number) => virtualPress(code),
			release: (code: number) => virtualRelease(code),
			tap: (code: number, stepCount = MIN_VIRTUAL_HOLD_INSTRUCTIONS) => {
				virtualPress(code);
				stepOnce(stepCount);
				virtualRelease(code);
			},
			pressPF1: () => virtualPress(0x56),
			releasePF1: () => virtualRelease(0x56),
			tapPF1: (stepCount = MIN_VIRTUAL_HOLD_INSTRUCTIONS) => {
				virtualPress(0x56);
				stepOnce(stepCount);
				virtualRelease(0x56);
			},
		};
		console.log('[pce500] devtools helpers installed: __pce500.dump(), __pce500.tapPF1(), __pce500.readInternal(0xF2)');
	}

	$: statusLabel = running ? 'RUNNING' : halted ? 'HALTED' : 'STOPPED';
	$: pc = pcReg;
	$: if (worker) {
		targetFps;
		regsOpen;
		callStackOpen;
		lcdTextOpen;
		debugStateOpen;
		keyboardDebugOpen;
		pushWorkerOptions();
	}

	function sortedRegs(input: any): [string, number][] {
		return regsEntries(input).sort(([a], [b]) => a.localeCompare(b));
	}

	async function stepOnce(count: number) {
		if (worker) {
			try {
				await workerCall('step', { instructions: count });
			} catch (err) {
				lastError = String(err);
				running = false;
			}
			return;
		}
		if (!emulator) return;
		try {
			stepCore(count);
			refreshFast();
			const nowMs = performance.now();
			refreshUi(nowMs);
			if (keyboardDebugOpen) snapshotKeyboardState();
		} catch (err) {
			lastError = String(err);
			running = false;
		}
	}

	function stepCore(count: number) {
		if (!emulator) return;
		emulator.step(count);
		applyVirtualReleaseBudget(count);
	}

	function pumpEmulator(id: number) {
		if (!running || !emulator || id !== runLoopId) return;
		const startMs = performance.now();
		try {
			while (performance.now() - startMs < runMaxWorkMs) {
				const sliceStart = performance.now();
				stepCore(runSliceInstructions);
				const sliceMs = performance.now() - sliceStart;
				if (sliceMs > 0) {
					const scaled = Math.floor(runSliceInstructions * (runSliceTargetMs / sliceMs));
					runSliceInstructions = Math.max(RUN_SLICE_MIN_INSTRUCTIONS, Math.min(RUN_SLICE_MAX_INSTRUCTIONS, scaled));
				}
				if (!running || id !== runLoopId) return;
			}
		} catch (err) {
			lastError = String(err);
			running = false;
			return;
		}
		setTimeout(() => pumpEmulator(id), RUN_YIELD_MS);
	}

	function pumpRender(id: number) {
		if (!running || !emulator || id !== runLoopId) return;
		const startMs = performance.now();
		refreshFast();
		refreshUi(startMs);
		if (keyboardDebugOpen) snapshotKeyboardState();
		const elapsedMs = performance.now() - startMs;
		const delayMs = Math.max(0, targetFrameIntervalMs - elapsedMs);
		setTimeout(() => pumpRender(id), delayMs);
	}

	async function onSelectRom(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		lastError = null;
		try {
			const bytes = new Uint8Array(await file.arrayBuffer());
			perfettoSymbolsPromise = null;
			romSource = file.name;
			await ensureWorker();
			if (worker) {
				await workerCall('load_rom', { bytes, romSource, model: romModel }, [bytes.buffer]);
				romLoaded = true;
				return;
			}
			const emu = await ensureEmulator();
			emu.load_rom_with_model?.(bytes, romModel) ?? emu.load_rom(bytes);
			romLoaded = true;
			refreshAllNow();
		} catch (err) {
			lastError = String(err);
			romLoaded = false;
		}
	}

	function start() {
		if (!romLoaded) return;
		if (running) return;
		lastLcdTextUpdateMs = 0;
		running = true;
		if (worker) {
			void workerCall('start');
			return;
		}
		runSliceInstructions = 2000;
		runLoopId += 1;
		pumpEmulator(runLoopId);
		pumpRender(runLoopId);
	}

	function stop() {
		running = false;
		if (worker) {
			void workerCall('stop');
			return;
		}
		runLoopId += 1;
	}

	function onKeyDown(event: KeyboardEvent) {
		if (event.repeat) return;
		const code = matrixCodeForKeyEvent(event);
		if (code === null) return;
		if (physicalHeldCodes.has(code)) return;
		physicalHeldCodes.add(code);
		setPhysicalMatrixCode(code, true);
		event.preventDefault();
	}

	function onKeyUp(event: KeyboardEvent) {
		const code = matrixCodeForKeyEvent(event);
		if (code === null) return;
		if (!physicalHeldCodes.has(code)) return;
		physicalHeldCodes.delete(code);
		setPhysicalMatrixCode(code, false);
		event.preventDefault();
	}

	onMount(() => {
		mounted = true;
		try {
			romModelWasPersisted = normalizeRomModel(window.localStorage.getItem(ROM_MODEL_STORAGE_KEY)) !== null;
		} catch {
			romModelWasPersisted = false;
		}
		void tryAutoLoadRom();
		void ensureWorker();
		installDevtoolsDebugHelpers();
	});

	$: if (mounted) {
		if (physicalKeyboardEnabled) {
			installPhysicalKeyboardHook();
		} else {
			uninstallPhysicalKeyboardHook();
			releaseAllPhysicalHeldCodes();
		}
	}

	onDestroy(() => {
		if (worker) {
			worker.terminate();
			worker = null;
		}
		uninstallPhysicalKeyboardHook();
		releaseAllPhysicalHeldCodes();
	});
</script>

<main>
	<h1>SC62015 Web Emulator (LLAMA/WASM)</h1>

	<label>
		ROM preset:
		<select
			bind:value={$romModelStore}
			on:change={() => {
				romModelWasPersisted = true;
				void tryAutoLoadRom(true);
			}}
			data-testid="rom-model"
		>
			<option value="iq-7000">IQ-7000</option>
			<option value="pc-e500">PC-E500</option>
		</select>
	</label>

	<label>
		Load ROM file:
		<input type="file" accept=".bin,.rom,.img" on:change={onSelectRom} />
	</label>

	{#if romSource}
		<p class="hint">Loaded ROM ({romModel}) via {romSource}</p>
	{/if}

	<div class="controls">
		<button on:click={() => stepOnce(1_000)} disabled={!romLoaded}>Step 1k</button>
		<button on:click={() => stepOnce(20_000)} disabled={!romLoaded}>Step 20k</button>
		<button on:click={start} disabled={!romLoaded || running}>Run</button>
		<button on:click={stop} disabled={!running}>Stop</button>
		<label>
			Target FPS:
			<input type="number" min="1" max="60" step="1" bind:value={targetFps} />
		</label>
	</div>

	<p class="hint" data-testid="emu-status">Status: {statusLabel} • PC: {hex(pc)} • Instr: {instructionCount ?? '—'}</p>
	<p class="hint" data-testid="build-info">WASM: {formatBuildInfo(buildInfo)}</p>

	{#if romLoaded}
		<p class="hint">LCD: {lcdKind ?? '—'} ({lcdCols}×{lcdRows})</p>
	{/if}

	<LcdCanvas pixels={lcdPixels} cols={lcdCols} rows={lcdRows} />

	{#if lcdKind === 'hd61202'}
		<details>
			<summary>LCD controller (64×64 chips)</summary>
			<div class="lcd-chips">
				<div class="lcd-chip">
					<div class="hint">Left chip</div>
					<LcdCanvas pixels={lcdLeftChipPixels} cols={LCD_CHIP_COLS} rows={LCD_CHIP_ROWS} scale={2} />
				</div>
				<div class="lcd-chip">
					<div class="hint">Right chip</div>
					<LcdCanvas pixels={lcdRightChipPixels} cols={LCD_CHIP_COLS} rows={LCD_CHIP_ROWS} scale={2} />
				</div>
			</div>
		</details>
	{/if}

	<VirtualKeyboard
		disabled={!romLoaded}
		onPress={(code) => virtualPress(code)}
		onRelease={(code) => virtualRelease(code)}
	/>

	<label>
		<input type="checkbox" data-testid="physical-keyboard-toggle" bind:checked={physicalKeyboardEnabled} />
		Enable physical keyboard input (F1/F2, arrows)
	</label>

	{#if romLoaded}
		<details bind:open={keyboardDebugOpen}>
			<summary>Debug (keyboard)</summary>
			<div class="debug-row">
				<button type="button" on:click={() => snapshotKeyboardState()}>Refresh</button>
				<button type="button" on:click={() => dumpKeyboardState('ui')}>Dump to console</button>
				<button type="button" on:click={() => (debugLog.length = 0)}>Clear log</button>
				<button type="button" on:click={() => copyDebugJson()} disabled={!debugKioJson}>Copy JSON</button>
			</div>
			{#if debugKio}
				<table class="regs" data-testid="keyboard-debug-table">
					<tbody>
						<tr>
							<td class="name">PC</td>
							<td class="val">{hex(debugKio.pc)}</td>
						</tr>
						<tr>
							<td class="name">Instr</td>
							<td class="val">{debugKio.instr?.toString?.() ?? '—'}</td>
						</tr>
						<tr>
							<td class="name">IMR</td>
							<td class="val">{hex(debugKio.imr, 2)}</td>
						</tr>
						<tr>
							<td class="name">ISR</td>
							<td class="val">{hex(debugKio.isr, 2)}</td>
						</tr>
						<tr>
							<td class="name">KOL/KOH/KIL</td>
							<td class="val">
								{hex(debugKio.kol, 2)} / {hex(debugKio.koh, 2)} / {hex(debugKio.kil, 2)}
							</td>
						</tr>
						<tr>
							<td class="name">FIFO head/tail</td>
							<td class="val">{hex(debugKio.fifoHead, 2)} / {hex(debugKio.fifoTail, 2)}</td>
						</tr>
						<tr>
							<td class="name">FIFO[0..15]</td>
							<td class="val">{debugKio.fifo.map((b) => hex(b, 2)).join(' ')}</td>
						</tr>
						<tr>
							<td class="name">Pressed</td>
							<td class="val"
								>{Array.from(pressedCodes)
									.map((c) => hex(c, 2))
									.join(' ') || '—'}</td
							>
						</tr>
						<tr>
							<td class="name">Pending release</td>
							<td class="val">
								{Array.from(pendingVirtualRelease.entries())
									.map(([c, n]) => `${hex(c, 2)}:${n}`)
									.join(' ') || '—'}
							</td>
						</tr>
					</tbody>
				</table>
				<details>
					<summary>Debug JSON</summary>
					<pre class="log" data-testid="keyboard-debug-json">{debugKioJson ?? ''}</pre>
				</details>
			{:else}
				<p class="hint">No keyboard snapshot available yet.</p>
			{/if}
			{#if debugLog.length > 0}
				<pre class="log" data-testid="keyboard-debug-log">{debugLog.join('\n')}</pre>
			{:else}
				<p class="hint">No events yet.</p>
			{/if}
		</details>
	{/if}

	{#if romLoaded}
		<details
			bind:open={callStackOpen}
			on:toggle={() => {
				if (callStackOpen) refreshAllNow();
			}}
		>
			<summary>Call stack</summary>
			{#if callStack && callStack.length > 0}
				<ol class="stack" data-testid="call-stack">
					{#each callStack as frame}
						<li>{formatFunction(frame)} ({hex(frame)})</li>
					{/each}
				</ol>
			{:else}
				<p class="hint" data-testid="call-stack-empty">No frames</p>
			{/if}
		</details>

		<details
			bind:open={regsOpen}
			on:toggle={() => {
				if (regsOpen) refreshAllNow();
			}}
		>
			<summary>Registers</summary>
			{#if regs}
				<table class="regs" data-testid="regs-table">
					<tbody>
						{#each sortedRegs(regs) as [name, value]}
							<tr>
								<td class="name">{name}</td>
								<td class="val">{hex(value, 6)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{:else}
				<p class="hint">Open to fetch registers.</p>
			{/if}
		</details>

		<details
			bind:open={lcdTextOpen}
			on:toggle={() => {
				if (lcdTextOpen) refreshAllNow();
			}}
		>
			<summary>LCD (decoded text)</summary>
			{#if lcdText && lcdText.length > 0}
				<pre data-testid="lcd-text">{lcdText.join('\n')}</pre>
			{:else}
				<p class="hint">Open to decode LCD text.</p>
			{/if}
		</details>
	{/if}

	{#if lastError}
		<p class="error">{lastError}</p>
	{/if}

	{#if romLoaded}
		<details
			bind:open={debugStateOpen}
			on:toggle={() => {
				if (debugStateOpen) refreshAllNow();
			}}
		>
			<summary>Debug state</summary>
			{#if debugState}
				<pre>{safeJson(debugState)}</pre>
			{:else}
				<p class="hint">Open to fetch debug state.</p>
			{/if}
		</details>
	{/if}

	{#if romLoaded}
		<FunctionRunnerPanel
			disabled={!romLoaded}
			busy={functionRunnerBusy}
			onRun={runFunctionRunner}
			onBeforeRun={() => {
				if (running) stop();
			}}
		/>
	{/if}

	<p class="hint">Keyboard: F1/F2 (PF1/PF2), arrows (cursor keys). Virtual keyboard supports PF1/PF2 + arrows.</p>
</main>

<style>
	main {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding: 16px;
		font-family: system-ui, sans-serif;
	}

	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}

	.debug-row {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 8px 0;
	}

	.error {
		color: #ff5c5c;
	}

	.hint {
		color: #9aa4b2;
	}

	pre {
		overflow: auto;
		max-height: 50vh;
		background: #0c0f12;
		color: #dbe7ff;
		padding: 12px;
		border-radius: 8px;
	}
	button {
		padding: 6px 10px;
	}
	input[type='number'] {
		width: 140px;
	}
	label {
		display: inline-flex;
		gap: 8px;
		align-items: center;
	}

	.lcd-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 16px;
		margin-top: 8px;
	}

	.lcd-chip {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.stack {
		margin: 0;
		padding-left: 18px;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
	}

	.regs {
		border-collapse: collapse;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
	}

	.regs td {
		padding: 2px 8px;
		border-bottom: 1px solid #243041;
	}

	.regs td.name {
		color: #9aa4b2;
	}

	.log {
		margin: 8px 0 0;
		max-height: 200px;
	}

	/* Function runner styles live in FunctionRunnerPanel/Results components. */
</style>
