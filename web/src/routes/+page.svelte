<script lang="ts">
	import { onDestroy, onMount, tick } from 'svelte';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import { LCD_CHIP_COLS, LCD_CHIP_ROWS, LCD_COLS, LCD_ROWS } from '$lib/lcd';
	import VirtualKeyboard from '$lib/components/VirtualKeyboard.svelte';
	import { matrixCodeForKeyEvent } from '$lib/keymap';
	import { normalizeLcdKind, type LcdKind } from '$lib/lcd_kind';
	import FunctionRunnerPanel from '$lib/components/FunctionRunnerPanel.svelte';
	import FunctionRunnerExamplesPanel from '$lib/components/FunctionRunnerExamplesPanel.svelte';
	import type { FunctionRunnerOutput } from '$lib/debug/function_runner_types';
	import { createPersistedStore } from '$lib/stores/persisted';
	import { normalizeRomModel, type RomModel } from '$lib/rom_model';
	import { PCE500_KEY_FIFO_CAPACITY, resolvePce500KeyboardFifo } from '$lib/emulator/pce500_iocs_workspace';
	import { stepBounded } from '$lib/emulator/bounded_step';
	import { automaticHostSlice, type ExecutionMode, type PacingStatus } from '$lib/emulator/host_pacing';
	import { WorkerRequests } from '$lib/emulator/worker_requests';
	import { HostInputs, applyContact, type InputContact } from '$lib/emulator/host_inputs';

	const ROM_MODEL_STORAGE_KEY = 'sc62015:rom-model';
	const romModelStore = createPersistedStore<RomModel>(ROM_MODEL_STORAGE_KEY, 'pc-e500', {
		serialize: (value) => value,
		deserialize: (raw) => normalizeRomModel(raw) ?? 'pc-e500',
	});

	let wasm: any = null;
	let emulator: any = null;
	let emulatorReady: Promise<any> | null = null;
	let worker: Worker | null = null;
	let workerNextId = 1;
	let workerRequests: WorkerRequests | null = null;
	let workerHealth: 'ready' | 'unresponsive' | 'faulted' = 'ready';
	let controlPending: 'start' | 'stop' | 'mode' | null = null;
	let executionMode: ExecutionMode = 'interactive';
	let pacingStatus: PacingStatus | null = null;
	let stepBusy = false;
	let fallbackStepAbort: AbortController | null = null;
	const canUseWorker =
		typeof window !== 'undefined' && typeof Worker !== 'undefined' && !Boolean((import.meta as any)?.env?.VITEST);

	let lcdPixels: Uint8Array | null = null;
	let lcdAnnunciatorBytes: Uint8Array | null = null;
	let lcdChipPixels: Uint8Array | null = null;
	let lcdCols = LCD_COLS;
	let lcdPixelScale = 1;
	let lcdRows = LCD_ROWS;
	let lcdKind: LcdKind | null = null;
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
	let romLoadGeneration = 0;
	let loadingRom = false;
	let romModel: RomModel = 'pc-e500';
	let romModelWasPersisted = false;
	$: romModel = $romModelStore;

	let running = false;
	let targetFps = 30;

	let functionRunnerBusy = false;
	let functionProgress: string | null = null;
	const pressedCodes = new Set<number>();
	const physicalHeldCodes = new Map<string, InputContact>();
	const pendingVirtualRelease = new Map<number, number>();
	const fallbackInputs = new HostInputs((contact, down) => applyContact(emulator, contact, down));
	let assistedTaps = true;
	let lastInputAck: string | null = null;
	const IMEM_BASE = 0x100000;
	const debugLog: string[] = [];
	let physicalKeyboardEnabled = false;
	let keyboardDebugOpen = false;
	let regsOpen = false;
	let callStackOpen = false;
	let lcdTextOpen = false;
	let debugStateOpen = false;
	let physicalKeyboardHookInstalled = false;
	let mounted = false;

	const LCD_TEXT_UPDATE_INTERVAL_MS = 250;
	let lastLcdTextUpdateMs = 0;
	$: targetFrameIntervalMs = 1000 / Math.max(1, targetFps);
	type SymbolEntry = { addr: number; name: string };
	type RawSymbolEntry = { addr: number | null; name: string };
	let symbolMap = new Map<number, string>();
	let symbolsPromise: Promise<SymbolEntry[] | null> | null = null;
	let symbolsPromiseModel: RomModel | null = null;

	const RUN_SLICE_MAX_INSTRUCTIONS = 200_000;
	let runLoopId = 0;

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
		if (typeof frame?.generation === 'number' && frame.generation !== romLoadGeneration) return;
		if (frame?.model && frame.model !== romModel) return;
		try {
			if (frame?.lcdPixels instanceof ArrayBuffer) {
				lcdPixels = new Uint8Array(frame.lcdPixels);
			}
			if (frame?.lcdChipPixels instanceof ArrayBuffer) {
				lcdChipPixels = new Uint8Array(frame.lcdChipPixels);
			}
			if (frame?.lcdAnnunciatorBytes instanceof ArrayBuffer) {
				lcdAnnunciatorBytes = new Uint8Array(frame.lcdAnnunciatorBytes);
			}
			if (typeof frame?.lcdCols === 'number') lcdCols = frame.lcdCols;
			if (typeof frame?.lcdPixelScale === 'number') lcdPixelScale = frame.lcdPixelScale;
			if (typeof frame?.lcdRows === 'number') lcdRows = frame.lcdRows;
			const nextKind = normalizeLcdKind(frame?.lcdKind);
			if (nextKind) lcdKind = nextKind;
			lcdText = frame?.lcdText ?? null;
			buildInfo = frame?.buildInfo ?? buildInfo;
			regs = frame?.regs ?? regs;
			callStack = frame?.callStack ?? callStack;
			debugState = frame?.debugState ?? null;
			pcReg = frame?.pc ?? pcReg;
			halted = Boolean(frame?.halted);
			instructionCount = frame?.instructionCount ?? instructionCount;
			// Frames are observations, not control acknowledgements. An older
			// in-flight frame must never revert an acknowledged mode selection.
			pacingStatus = frame?.pacing ?? pacingStatus;

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
		if (!workerRequests) return Promise.reject(new Error('worker not ready'));
		const id = workerNextId++;
		const message = { id, type, ...payload };
		const timeoutMs = type === 'stop' ? 1_000 : type === 'eval_js' ? 300_000 : 30_000;
		return workerRequests.request<T>(message, transfer, timeoutMs);
	}

	function failWorker(message: string) {
		lastError = `${message}. Reload the page to replace the worker; unsaved emulator state will be lost.`;
		workerHealth = 'faulted';
		running = false;
		romLoaded = false;
		workerRequests?.fail(new Error(message));
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
		workerRequests = new WorkerRequests(workerPost, (error) => {
			lastError = error.message;
			workerHealth = 'unresponsive';
		});
		worker.onmessage = async (event: MessageEvent<any>) => {
			const data = event.data;
			if (!data) return;
			if (data.type === 'reply') {
				workerRequests?.reply(data);
				return;
			}
			if (data.type === 'frame') {
				try {
					applyWorkerFrame(data.frame);
					await tick(); // Return credit after the canvas/component update, not just message arrival.
				} catch (error) {
					lastError = `Display update failed: ${String(error)}`;
				} finally {
					if (typeof data.sequence === 'number')
						workerPost({ id: workerNextId++, type: 'frame_consumed', sequence: data.sequence });
				}
				return;
			}
			if (data.type === 'execution_progress' && data.generation === romLoadGeneration && functionRunnerBusy) {
				functionProgress = `Call ${hex(data.address)}: ${data.steps} call-budget steps, ${data.schedulerBoundaries} scheduler boundaries`;
				return;
			}
			if (data.type === 'script_progress' && data.generation === romLoadGeneration && functionRunnerBusy) {
				functionProgress = `Isolated script: ${data.operation}`;
				return;
			}
			if (data.type === 'fatal') {
				failWorker(`Worker error: ${data.error ?? 'unknown error'}`);
			}
			if (data.type === 'render_error') lastError = `Display refresh failed (not a CPU pause or reset): ${data.error}`;
		};
		worker.onerror = (event) => {
			failWorker(
				`Worker crashed: ${event.message || 'failed to load or execute worker'} (${event.filename || 'unknown source'})`,
			);
		};
		worker.onmessageerror = () => failWorker('Worker message could not be decoded');
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

	function resetSymbols() {
		symbolsPromise = null;
		symbolsPromiseModel = null;
		symbolMap = new Map();
	}

	async function ensureSymbols(): Promise<SymbolEntry[] | null> {
		const model = romModel;
		if (symbolsPromise && symbolsPromiseModel === model) return symbolsPromise;
		symbolsPromiseModel = model;
		symbolsPromise = (async () => {
			try {
				const res = await fetch(`/api/symbols?model=${encodeURIComponent(model)}`);
				if (!res.ok) return null;
				const payload = (await res.json()) as any;
				const raw = Array.isArray(payload?.symbols) ? payload.symbols : [];
				const entries: SymbolEntry[] = raw
					.map((entry: any) => ({
						addr: typeof entry?.addr === 'number' ? entry.addr >>> 0 : null,
						name: String(entry?.name ?? '').trim(),
					}))
					.filter((entry: RawSymbolEntry): entry is SymbolEntry => Number.isFinite(entry.addr) && entry.name.length > 0)
					.map((entry: SymbolEntry) => ({ addr: entry.addr & 0x000f_ffff, name: entry.name }));
				if (romModel !== model) return null;
				symbolMap = new Map(entries.map((entry) => [entry.addr, entry.name]));
				return entries;
			} catch {
				return null;
			}
		})();
		return symbolsPromise;
	}

	async function runFunctionRunner(source: string): Promise<FunctionRunnerOutput> {
		functionRunnerBusy = true;
		functionProgress = null;
		try {
			if (running && !(await stop())) throw new Error('Pause was not acknowledged');
			await ensureWorker();
			if (worker) {
				return await workerCall('eval_js', { source });
			}

			throw new Error('Function Runner requires isolated workers; scripts are disabled in the no-Worker fallback.');
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
		const addr = pc & 0x000f_ffff;
		const name = symbolMap.get(addr);
		return name ?? `sub_${addr.toString(16).toUpperCase().padStart(5, '0')}`;
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

	function setContact(
		source: 'physical' | 'virtual',
		code: InputContact,
		down: boolean,
		owner = String(code),
		cancel = false,
	) {
		const generation = romLoadGeneration;
		const minimumHold = source === 'virtual' && assistedTaps ? 40_000 : 0;
		const label = code === 'on' ? 'ON' : hex(code, 2);
		if (down && (!romLoaded || workerHealth !== 'ready')) return;
		const recordAck = () => {
			if (generation !== romLoadGeneration) return;
			lastInputAck = `${label} ${down ? 'down' : cancel ? 'cancelled' : 'up'} applied to input controller; ROM consumption not confirmed`;
			logDebug(lastInputAck);
		};
		if (worker) {
			void workerCall(`${source}_key`, { code, down, owner, cancel, minimumHold, generation })
				.then(recordAck)
				.catch((error) => {
					if (generation === romLoadGeneration) lastError = `Input not acknowledged: ${String(error)}`;
				});
		} else if (emulator) {
			try {
				fallbackInputs.set({ source, owner, contact: code, down, cancel, minimumHold });
				recordAck();
			} catch (error) {
				lastError = `Input failed: ${String(error)}`;
			}
		}
	}

	function setMatrixCode(code: InputContact, down: boolean) {
		setContact('virtual', code, down);
	}
	function setPhysicalMatrixCode(code: InputContact, down: boolean, owner = String(code)) {
		setContact('physical', code, down, owner);
	}

	function releaseInputSource(source: 'physical' | 'virtual') {
		const generation = romLoadGeneration;
		if (worker)
			void workerCall('release_inputs', { source, generation }).catch((error) => {
				if (generation === romLoadGeneration) lastError = `Input cleanup not acknowledged: ${String(error)}`;
			});
		else if (emulator) fallbackInputs.releaseSource(source);
	}

	function releaseAllPhysicalHeldCodes() {
		if (physicalHeldCodes.size === 0) return;
		releaseInputSource('physical');
		physicalHeldCodes.clear();
	}

	function installPhysicalKeyboardHook() {
		if (physicalKeyboardHookInstalled) return;
		window.addEventListener('keydown', onKeyDown, { passive: false });
		window.addEventListener('keyup', onKeyUp, { passive: false });
		window.addEventListener('blur', releaseAllPhysicalHeldCodes);
		document.addEventListener('visibilitychange', onVisibilityChange);
		document.addEventListener('focusin', onFocusIn);
		physicalKeyboardHookInstalled = true;
	}

	function uninstallPhysicalKeyboardHook() {
		if (!physicalKeyboardHookInstalled) return;
		window.removeEventListener('keydown', onKeyDown);
		window.removeEventListener('keyup', onKeyUp);
		window.removeEventListener('blur', releaseAllPhysicalHeldCodes);
		document.removeEventListener('visibilitychange', onVisibilityChange);
		document.removeEventListener('focusin', onFocusIn);
		physicalKeyboardHookInstalled = false;
	}

	function virtualPress(code: InputContact, owner = String(code)) {
		setContact('virtual', code, true, owner);
	}

	function virtualRelease(code: InputContact, owner = String(code), cancel = false) {
		setContact('virtual', code, false, owner, cancel);
	}

	function applyVirtualReleaseBudget(stepped: number) {
		fallbackInputs.advance(stepped);
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
				if (!romModelWasPersisted && typeof model === 'string') $romModelStore = model as RomModel;
			} catch {
				// ignore
			}
			return;
		}
		await ensureEmulator();
	}

	async function tryAutoLoadRom(force = false) {
		releaseAllPhysicalHeldCodes();
		releaseInputSource('virtual');
		const generation = ++romLoadGeneration;
		loadingRom = true;
		romLoaded = false;
		try {
			await ensureWorker();
			if (!force) await syncRomModelFromRuntime();
			if (generation !== romLoadGeneration) return;
			const model = romModel;
			const res = await fetch(`/api/rom?model=${encodeURIComponent(model)}`);
			if (!res.ok) throw new Error(`ROM fetch failed (${res.status}) for ${model}`);
			const bytes = new Uint8Array(await res.arrayBuffer());
			await installRom(bytes, model, res.headers.get('x-rom-source'), generation);
		} catch (err) {
			if (generation === romLoadGeneration) lastError = `Auto-load failed: ${String(err)}`;
		} finally {
			if (generation === romLoadGeneration) loadingRom = false;
		}
	}

	async function installRom(bytes: Uint8Array, model: RomModel, source: string | null, generation: number) {
		if (generation !== romLoadGeneration) return;
		if (!(await stop())) throw new Error('Pause was not acknowledged; ROM was not replaced');
		if (generation !== romLoadGeneration) return;
		releaseAllPhysicalHeldCodes();
		resetSymbols();
		if (worker) {
			await workerCall('load_rom', { bytes, romSource: source, model, generation }, [bytes.buffer]);
		} else {
			const emu = await ensureEmulator();
			if (generation !== romLoadGeneration) return;
			fallbackInputs.clear();
			if (typeof emu.load_rom_with_model === 'function') emu.load_rom_with_model(bytes, model);
			else emu.load_rom(bytes);
			refreshAllNow();
		}
		if (generation !== romLoadGeneration) return;
		romSource = source;
		romLoaded = true;
		lastError = null;
		if (callStackOpen) void ensureSymbols();
	}

	function refreshFast() {
		if (worker) return;
		if (!emulator) return;
		pacingStatus = emulator.pacing_status?.() ?? null;
		try {
			const geometry = emulator.lcd_capture();
			if (geometry && typeof geometry === 'object') {
				const kind = normalizeLcdKind((geometry as any).kind);
				const cols = (geometry as any).cols;
				const rows = (geometry as any).rows;
				if (kind) lcdKind = kind;
				if (typeof cols === 'number') lcdCols = cols;
				lcdPixelScale = geometry.pixel_scale;
				if (typeof rows === 'number') lcdRows = rows;
				lcdPixels = new Uint8Array(geometry.pixels);
			}
		} catch {
			// ignore
		}
		lcdAnnunciatorBytes = emulator.lcd_annunciator_bytes?.() ?? null;
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
		if (callStackOpen) void ensureSymbols();
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
			const fifoAddresses = resolvePce500KeyboardFifo((address) => emulator.read_u8?.(address));
			const fifoHead = fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoHead) ?? null) : null;
			const fifoTail = fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoTail) ?? null) : null;
			const fifo = Array.from({ length: PCE500_KEY_FIFO_CAPACITY }, (_, i) =>
				fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoBase + i) ?? 0) : 0,
			);
			debugKio = { pc, instr, imr, isr, kol, koh, kil, fifoHead, fifoTail, fifo };
			const inputState = fallbackInputs.snapshot();
			pressedCodes.clear();
			for (const code of inputState.pressedCodes) pressedCodes.add(code);
			pendingVirtualRelease.clear();
			for (const [code, remaining] of inputState.pendingVirtualRelease) pendingVirtualRelease.set(code, remaining);
			debugKioJson = safeJson({
				...debugKio,
				...fallbackInputs.snapshot(),
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
			const fifoAddresses = resolvePce500KeyboardFifo((address) => emulator.read_u8?.(address));
			const fifoHead = fifoAddresses ? emulator.read_u8?.(fifoAddresses.fifoHead) : undefined;
			const fifoTail = fifoAddresses ? emulator.read_u8?.(fifoAddresses.fifoTail) : undefined;
			const fifo = Array.from({ length: PCE500_KEY_FIFO_CAPACITY }, (_, i) =>
				fifoAddresses ? (emulator.read_u8?.(fifoAddresses.fifoBase + i) ?? 0) : 0,
			);
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
				...fallbackInputs.snapshot(),
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
			tap: async (code: number, stepCount = 40_000) => {
				virtualPress(code);
				try {
					await stepOnce(stepCount);
				} finally {
					virtualRelease(code, String(code), true);
				}
			},
			pressPF1: () => virtualPress(0x56),
			releasePF1: () => virtualRelease(0x56),
			tapPF1: async (stepCount = 40_000) => {
				virtualPress(0x56);
				try {
					await stepOnce(stepCount);
				} finally {
					virtualRelease(0x56, String(0x56), true);
				}
			},
		};
		console.log('[pce500] devtools helpers installed: __pce500.dump(), __pce500.tapPF1(), __pce500.readInternal(0xF2)');
	}

	$: statusLabel =
		workerHealth === 'faulted'
			? 'FAULTED'
			: workerHealth === 'unresponsive'
				? 'UNRESPONSIVE — STATE UNCONFIRMED'
				: controlPending === 'stop'
					? 'PAUSE REQUESTED'
					: controlPending === 'start'
						? 'START REQUESTED'
						: loadingRom
							? 'LOADING ROM'
							: functionRunnerBusy
								? 'EXECUTING SCRIPT'
								: stepBusy
									? 'STEPPING'
									: running
										? 'RUNNING'
										: halted
											? 'HALTED'
											: 'STOPPED';
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
		if (stepBusy || functionRunnerBusy || controlPending || workerHealth !== 'ready') return;
		if (running && !(await stop())) return;
		stepBusy = true;
		if (worker) {
			try {
				await workerCall('step', { instructions: count });
			} catch (err) {
				lastError = String(err);
				running = false;
			} finally {
				stepBusy = false;
			}
			return;
		}
		if (!emulator) {
			stepBusy = false;
			return;
		}
		fallbackStepAbort = new AbortController();
		try {
			await stepBounded(emulator, count, {
				signal: fallbackStepAbort.signal,
				limitBudget: fallbackInputs.limitBudget,
				onProgress: applyVirtualReleaseBudget,
			});
			refreshFast();
			const nowMs = performance.now();
			refreshUi(nowMs);
			if (keyboardDebugOpen) snapshotKeyboardState();
		} catch (err) {
			lastError = String(err);
			running = false;
		} finally {
			fallbackStepAbort = null;
			stepBusy = false;
		}
	}

	function stepCore(count: number) {
		return automaticHostSlice(emulator, count, fallbackInputs);
	}

	function pumpEmulator(id: number) {
		if (!running || !emulator || id !== runLoopId) return;
		let waitMs: number;
		try {
			waitMs = stepCore(RUN_SLICE_MAX_INSTRUCTIONS);
		} catch (err) {
			lastError = String(err);
			running = false;
			return;
		}
		setTimeout(() => pumpEmulator(id), waitMs);
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
		const generation = ++romLoadGeneration;
		const model = romModel;
		loadingRom = true;
		romLoaded = false;
		lastError = null;
		try {
			const bytes = new Uint8Array(await file.arrayBuffer());
			await ensureWorker();
			await installRom(bytes, model, file.name, generation);
		} catch (err) {
			if (generation === romLoadGeneration) lastError = String(err);
		} finally {
			if (generation === romLoadGeneration) loadingRom = false;
		}
	}

	async function setExecutionMode(event: Event) {
		const select = event.currentTarget as HTMLSelectElement;
		const mode = select.value as ExecutionMode;
		if (!romLoaded || running || controlPending || stepBusy || functionRunnerBusy || workerHealth !== 'ready') {
			select.value = executionMode;
			return;
		}
		controlPending = 'mode';
		try {
			if (worker) pacingStatus = await workerCall('set_execution_mode', { mode });
			else {
				emulator.set_execution_mode(mode);
				pacingStatus = emulator.pacing_status();
			}
			executionMode = mode;
		} catch (error) {
			lastError = String(error);
			select.value = executionMode;
		} finally {
			if (controlPending === 'mode') controlPending = null;
		}
	}

	async function start() {
		if (!romLoaded || running || controlPending || stepBusy || functionRunnerBusy || workerHealth !== 'ready') return;
		if (executionMode === 'deterministic') return;
		lastLcdTextUpdateMs = 0;
		if (worker) {
			controlPending = 'start';
			try {
				await workerCall('start');
				running = true;
			} catch (error) {
				lastError = String(error);
			} finally {
				if (controlPending === 'start') controlPending = null;
			}
			return;
		}
		emulator.rebase_pacing();
		running = true;
		runLoopId += 1;
		pumpEmulator(runLoopId);
		pumpRender(runLoopId);
	}

	async function stop(): Promise<boolean> {
		if (worker) {
			controlPending = 'stop';
			try {
				await workerCall('stop');
				running = false;
				workerHealth = 'ready';
				return true;
			} catch (error) {
				lastError = String(error);
				return false;
			} finally {
				controlPending = null;
			}
		}
		fallbackStepAbort?.abort();
		emulator?.rebase_pacing();
		running = false;
		runLoopId += 1;
		return true;
	}

	function onKeyDown(event: KeyboardEvent) {
		if (
			event.repeat ||
			!romLoaded ||
			event.isComposing ||
			event.metaKey ||
			event.ctrlKey ||
			event.altKey ||
			isHostControl(event.target)
		)
			return;
		if (
			event.target instanceof Element &&
			event.target.closest('button') &&
			(event.key === 'Enter' || event.key === ' ')
		)
			return;
		const code = matrixCodeForKeyEvent(event, romModel);
		if (code === null) return;
		if (physicalHeldCodes.has(event.code)) return;
		physicalHeldCodes.set(event.code, code);
		setPhysicalMatrixCode(code, true, event.code);
		event.preventDefault();
	}

	function onKeyUp(event: KeyboardEvent) {
		const code = physicalHeldCodes.get(event.code);
		if (code === undefined) return;
		physicalHeldCodes.delete(event.code);
		setPhysicalMatrixCode(code, false, event.code);
		event.preventDefault();
	}

	function isHostControl(target: EventTarget | null): boolean {
		return (
			target instanceof Element &&
			Boolean(
				target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"]'),
			)
		);
	}
	function onVisibilityChange() {
		if (document.hidden) releaseAllPhysicalHeldCodes();
	}
	function onFocusIn(event: FocusEvent) {
		if (isHostControl(event.target)) releaseAllPhysicalHeldCodes();
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
		uninstallPhysicalKeyboardHook();
		physicalHeldCodes.clear();
		if (!worker && emulator) fallbackInputs.clear();
		workerRequests?.fail(new Error('Emulator page closed'));
		fallbackStepAbort?.abort();
		running = false;
		runLoopId++;
		if (worker) {
			worker.terminate();
			worker = null;
		}
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
		<label>
			Execution mode:
			<select
				data-testid="execution-mode"
				value={executionMode}
				on:change={setExecutionMode}
				disabled={!romLoaded ||
					running ||
					stepBusy ||
					functionRunnerBusy ||
					!!controlPending ||
					workerHealth !== 'ready'}
			>
				<option value="interactive">Interactive (nominal)</option>
				<option value="turbo">Turbo (unthrottled)</option>
				<option value="deterministic">Deterministic (explicit budgets)</option>
			</select>
		</label>
		<button
			on:click={() => stepOnce(1_000)}
			disabled={!romLoaded || stepBusy || functionRunnerBusy || !!controlPending || workerHealth !== 'ready'}
			>Step 1k</button
		>
		<button
			on:click={() => stepOnce(20_000)}
			disabled={!romLoaded || stepBusy || functionRunnerBusy || !!controlPending || workerHealth !== 'ready'}
			>Step 20k</button
		>
		<button
			on:click={start}
			disabled={!romLoaded ||
				executionMode === 'deterministic' ||
				running ||
				stepBusy ||
				functionRunnerBusy ||
				!!controlPending ||
				workerHealth !== 'ready'}>Run</button
		>
		<button
			on:click={stop}
			disabled={workerHealth === 'faulted' ||
				controlPending === 'stop' ||
				(!running && !stepBusy && !functionRunnerBusy && controlPending !== 'start' && workerHealth !== 'unresponsive')}
			>Stop</button
		>
		<label>
			Target FPS:
			<input type="number" min="1" max="60" step="1" bind:value={targetFps} />
		</label>
	</div>

	<p class="hint" data-testid="emu-status">Status: {statusLabel} • PC: {hex(pc)} • Instr: {instructionCount ?? '—'}</p>
	<p class="hint" data-testid="pacing-status">
		{executionMode}: Interactive pacing uses {pacingStatus?.nominal_timebase_hz ?? '—'} compatibility timing units/s, not
		hardware-calibrated MHz. IQ-7000 currently uses the PC-E500 fallback timebase. The RTC follows emulated elapsed time;
		paused wall time is not simulated. Catch-up is capped at 50 ms; dropped host backlog: {(
			Number(pacingStatus?.dropped_host_ns ?? 0) / 1e6
		).toFixed(1)} ms. Step and Function Runner use explicit, unthrottled budgets in every mode.
	</p>
	{#if executionMode === 'deterministic'}
		<p class="hint">
			Automatic Run is disabled. Repeatable results require the same ROM/state, a fixed RTC seed (the default seed comes
			from host time), and identical inputs at identical scheduler boundaries. This selection does not reset your
			machine or make live human input deterministic.
		</p>
	{/if}
	{#if functionRunnerBusy && functionProgress}
		<p class="hint" data-testid="execution-progress">{functionProgress}</p>
	{/if}
	<p class="hint" data-testid="build-info">WASM: {formatBuildInfo(buildInfo)}</p>

	{#if romLoaded}
		<p class="hint">LCD: {lcdKind ?? '—'} ({lcdCols}×{lcdRows})</p>
	{/if}

	<div class="lcd-display" aria-label="Emulated LCD including fixed segments">
		<LcdCanvas pixels={lcdPixels} cols={lcdCols} rows={lcdRows} scale={4 / lcdPixelScale} pixelFormat="gray8" />
	</div>
	{#if lcdKind === 'iq7000-vram'}
		<p class="hint">
			Fixed segments: ROM-derived mapping; BATT/CARD/beep/alarm/arrows remain provisional. LCD bytes: {Array.from(
				lcdAnnunciatorBytes ?? [],
			)
				.map((b) => hex(b, 2))
				.join(' ')}
		</p>
	{/if}

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
		disabled={!romLoaded || workerHealth !== 'ready'}
		model={romModel}
		onPress={virtualPress}
		onRelease={virtualRelease}
		onCancelAll={() => releaseInputSource('virtual')}
	/>
	<label>
		<input
			type="checkbox"
			data-testid="assisted-taps-toggle"
			bind:checked={assistedTaps}
			on:change={() => releaseInputSource('virtual')}
		/>
		Assist virtual taps (minimum 40,000 scheduler boundaries from press; no execution while paused)
	</label>
	<p class="hint">
		Disable assistance for immediate raw contact releases. ON uses the power-key input, not a forced interrupt.
	</p>
	<p class="hint">
		{#if romModel === 'iq-7000'}
			IQ controls: F1–F5 = Calendar/Schedule/TEL/MEMO/Calc; F6–F8 = Card/World/Home; Page Up/Down = Search; Enter =
			Store.
		{:else}
			PC-E500 controls: F1–F5 = PF1–PF5, arrows, Enter, Backspace, Delete, Insert and Space.
		{/if}
		Both: Shift, Caps Lock and F12 = ON. This is an initial control subset; full text-key mapping is still being qualified.
	</p>
	{#if lastInputAck}<p class="hint" data-testid="input-ack">{lastInputAck}</p>{/if}

	<label>
		<input type="checkbox" data-testid="physical-keyboard-toggle" bind:checked={physicalKeyboardEnabled} />
		Enable physical keyboard input (model-specific keys; F12 = ON; ignored in text fields and controls)
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
		<FunctionRunnerExamplesPanel />
	{/if}

	{#if romLoaded}
		<FunctionRunnerPanel
			disabled={!romLoaded || stepBusy || !!controlPending || workerHealth !== 'ready'}
			busy={functionRunnerBusy}
			onRun={runFunctionRunner}
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

	.lcd-display {
		display: flex;
		align-items: flex-start;
		gap: 8px;
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
