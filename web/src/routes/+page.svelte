<script lang="ts">
	import { onDestroy, onMount, tick } from 'svelte';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import { LCD_CHIP_COLS, LCD_CHIP_ROWS, LCD_COLS, LCD_ROWS } from '$lib/lcd';
	import DeviceShell from '$lib/components/DeviceShell.svelte';
	import { lcdPng, downloadBlob } from '$lib/lcd_export';
	import { contactsForKeyEvent, type HostKeyboardMode } from '$lib/keymap';
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
	import { HostInputs, InputBufferOverflow, applyContact, type InputContact } from '$lib/emulator/host_inputs';
	import { planPaste, MAX_PASTE_CHARACTERS } from '$lib/emulator/paste_plan';

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
	let lcdFrameGeneration = -1;
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
	let installedRom: { bytes: Uint8Array; model: RomModel; source: string | null } | null = null;
	let pcReg: number | null = null;
	let halted = false;
	let powerState = 'unknown';
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
	const physicalHeldCodes = new Map<string, InputContact[]>();
	let physicalHighlights = new Set<InputContact>();
	let deliveredContacts = new Set<InputContact>();
	let shortcutsOpen = false;
	let accuracyOpen = false;
	let lastCapture: { filename: string; metadata: object } | null = null;
	let exportingLcd = false;
	let typingStatus = { pending: 0, blocked: false, capacity: 128 };
	let pasteStatus = { pending: 0, total: 0 };
	let pasteOpen = false;
	let pasteText = '';
	let pasteSubmitting = false;
	let pasteEditor: HTMLTextAreaElement;
	$: pastePlan = planPaste(pasteText, romModel);
	let typingCatchUp = false;
	let advancedOpen = false;
	let menuOpen = false;
	let lcdOnly = false;
	const pendingVirtualRelease = new Map<number, number>();
	const fallbackInputs = new HostInputs((contact, down) => applyContact(emulator, contact, down));
	let assistedTaps = true;
	let lastInputAck: string | null = null;
	const IMEM_BASE = 0x100000;
	const debugLog: string[] = [];
	let physicalKeyboardEnabled = false;
	let hostKeyboardMode: HostKeyboardMode = 'symbols';
	let activeHostKeyboardMode: HostKeyboardMode = hostKeyboardMode;
	let keyboardTarget: HTMLDivElement;
	let keyboardNotice = '';
	let keyboardDebugOpen = false;
	let lcdChipsOpen = false;
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
		if (frame.inputContacts) updateDeliveredContacts(frame.inputContacts);
		if (frame.typing) typingStatus = frame.typing;
		if (frame.paste) pasteStatus = frame.paste;
		if (frame?.model && frame.model !== romModel) return;
		try {
			if (frame?.lcdPixels instanceof ArrayBuffer) {
				lcdPixels = new Uint8Array(frame.lcdPixels);
				lcdFrameGeneration = frame.generation;
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
			powerState = frame?.powerState ?? 'unknown';
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
		lastError = `${message}. Use Reset machine to replace the worker; unsaved emulator state will be lost.`;
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
			typingCatchUp,
			debug: { regsOpen, callStackOpen, lcdTextOpen, debugStateOpen, keyboardDebugOpen, lcdChipsOpen },
		});
	}

	async function ensureWorker(): Promise<void> {
		if (!canUseWorker || worker) return;
		worker = new Worker(new URL('../lib/emulator/pce500.worker.ts', import.meta.url), { type: 'module' });
		const thisWorker = worker;
		workerRequests = new WorkerRequests(workerPost, (error) => {
			lastError = error.message;
			workerHealth = 'unresponsive';
		});
		worker.onmessage = async (event: MessageEvent<any>) => {
			if (worker !== thisWorker) return;
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
					if (worker === thisWorker && typeof data.sequence === 'number')
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
			if (data.type === 'input_paused' && data.generation === romLoadGeneration) {
				running = false;
				lastError = data.error;
			}
			if (data.type === 'input_status' && data.generation === romLoadGeneration) {
				typingStatus = data.typing;
				if (data.paste) pasteStatus = data.paste;
				if (data.inputContacts) updateDeliveredContacts(data.inputContacts);
			}
			if (data.type === 'render_error') lastError = `Display refresh failed (not a CPU pause or reset): ${data.error}`;
		};
		worker.onerror = (event) => {
			if (worker !== thisWorker) return;
			failWorker(
				`Worker crashed: ${event.message || 'failed to load or execute worker'} (${event.filename || 'unknown source'})`,
			);
		};
		worker.onmessageerror = () => {
			if (worker === thisWorker) failWorker('Worker message could not be decoded');
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
		holdOverride?: number,
	) {
		const generation = romLoadGeneration;
		const minimumHold = holdOverride ?? (source === 'virtual' && assistedTaps ? 40_000 : 0);
		const buffered = source === 'physical' && hostKeyboardMode === 'symbols' && code !== 'on';
		const label = code === 'on' ? 'ON' : hex(code, 2);
		if (down && (!romLoaded || workerHealth !== 'ready')) return;
		if (down && pasteStatus.pending && code !== 'on') {
			keyboardNotice = 'Paste in progress; cancel queued keys before entering other input.';
			return;
		}
		const recordAck = () => {
			if (generation !== romLoadGeneration) return;
			lastInputAck = `${label} ${down ? 'down' : cancel ? 'cancelled' : 'up'} ${buffered ? 'accepted by typing buffer' : 'applied to input controller'}; ROM consumption not confirmed`;
			logDebug(lastInputAck);
		};
		if (worker) {
			void workerCall(`${source}_key`, { code, down, owner, cancel, minimumHold, buffered, generation })
				.then(recordAck)
				.catch((error) => {
					if (generation === romLoadGeneration) lastError = `Input not acknowledged: ${String(error)}`;
				});
		} else if (emulator) {
			try {
				if (code === 'on' && down) fallbackInputs.clearTyping();
				fallbackInputs.set({ source, owner, contact: code, down, cancel, minimumHold, buffered });
				recordAck();
			} catch (error) {
				lastError = `Input failed: ${String(error)}`;
				if (error instanceof InputBufferOverflow) void stop();
			} finally {
				typingStatus = fallbackInputs.typingStatus();
				pasteStatus = fallbackInputs.pasteStatus();
				updateDeliveredContacts(emulator.input_contacts());
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
		else if (emulator) {
			fallbackInputs.releaseSource(source);
			typingStatus = fallbackInputs.typingStatus();
			pasteStatus = fallbackInputs.pasteStatus();
			updateDeliveredContacts(emulator.input_contacts());
		}
	}

	function releaseAllPhysicalHeldCodes() {
		// Released host keys can still be waiting for their turn in the buffer.
		releaseInputSource('physical');
		physicalHeldCodes.clear();
		physicalHighlights = new Set();
		if (lastError?.includes('Typing buffer overflow')) lastError = null;
	}

	async function focusDeviceKeyboard() {
		physicalKeyboardEnabled = true;
		keyboardNotice = '';
		await tick();
		keyboardTarget?.focus({ preventScroll: true });
	}

	async function openPaste(text = '') {
		pasteText = text;
		pasteOpen = true;
		menuOpen = false;
		await tick();
		pasteEditor?.focus();
	}

	function onPaste(event: ClipboardEvent) {
		if (!physicalKeyboardEnabled || !romLoaded || isHostControl(event.target)) return;
		event.preventDefault();
		void openPaste(event.clipboardData?.getData('text/plain') ?? '');
	}

	async function submitPaste() {
		if (
			!romLoaded ||
			loadingRom ||
			pasteSubmitting ||
			functionRunnerBusy ||
			stepBusy ||
			controlPending ||
			workerHealth !== 'ready' ||
			!pastePlan.contacts.length ||
			pastePlan.error ||
			pastePlan.unsupported.length
		)
			return;
		pasteSubmitting = true;
		const generation = romLoadGeneration;
		try {
			if (worker) {
				const result = await workerCall('paste_text', { text: pasteText, generation });
				if (generation !== romLoadGeneration) return;
				pasteStatus = result;
			} else {
				fallbackInputs.startPaste(pastePlan.contacts);
				pasteStatus = fallbackInputs.pasteStatus();
				refreshFast();
			}
			if (generation !== romLoadGeneration) return;
			pasteOpen = false;
			pasteText = '';
			await focusDeviceKeyboard();
		} catch (error) {
			if (generation === romLoadGeneration) lastError = `Paste not started: ${String(error)}`;
		} finally {
			pasteSubmitting = false;
		}
	}

	function updateDeliveredContacts(contacts: { matrix: number[]; on: boolean }) {
		deliveredContacts = new Set<InputContact>(contacts.matrix);
		if (contacts.on) deliveredContacts.add('on');
	}

	async function exportLcd() {
		if (!lcdPixels || exportingLcd || !romLoaded || lcdFrameGeneration !== romLoadGeneration) return;
		exportingLcd = true;
		const filename = `${romModel}-lcd-${new Date().toISOString().replace(/[:.]/g, '-')}`;
		// Copy a single observed frame before awaiting PNG encoding. The machine
		// can keep running; a later frame must not relabel this capture's metadata.
		const pixels = lcdPixels.slice();
		const cols = lcdCols;
		const rows = lcdRows;
		const metadata = {
			format: 'sc62015-lcd-observation-v1',
			model: romModel,
			romSource,
			build: buildInfo,
			cols,
			rows,
			pixelFormat: 'gray8',
			pixelScale: lcdPixelScale,
			pc: pcReg,
			instructionCount,
			annunciatorBytes: Array.from(lcdAnnunciatorBytes ?? []),
			accuracy:
				'Actual observed emulator LCD including implemented annunciators; physical mappings and timing retain documented provisional limits. Not a machine snapshot.',
		};
		try {
			downloadBlob(await lcdPng(pixels, cols, rows), `${filename}.png`);
			lastCapture = { filename, metadata };
		} catch (error) {
			lastError = `LCD export failed: ${String(error)}`;
		} finally {
			exportingLcd = false;
		}
	}

	function exportCaptureMetadata() {
		if (lastCapture)
			downloadBlob(
				new Blob([JSON.stringify(lastCapture.metadata, null, 2)], { type: 'application/json' }),
				`${lastCapture.filename}.json`,
			);
	}

	function installPhysicalKeyboardHook() {
		if (physicalKeyboardHookInstalled) return;
		window.addEventListener('keydown', onKeyDown, { passive: false });
		window.addEventListener('keyup', onKeyUp, { passive: false });
		window.addEventListener('blur', releaseAllPhysicalHeldCodes);
		document.addEventListener('visibilitychange', onVisibilityChange);
		document.addEventListener('focusin', onFocusIn);
		document.addEventListener('compositionstart', releaseAllPhysicalHeldCodes);
		physicalKeyboardHookInstalled = true;
	}

	function uninstallPhysicalKeyboardHook() {
		if (!physicalKeyboardHookInstalled) return;
		window.removeEventListener('keydown', onKeyDown);
		window.removeEventListener('keyup', onKeyUp);
		window.removeEventListener('blur', releaseAllPhysicalHeldCodes);
		document.removeEventListener('visibilitychange', onVisibilityChange);
		document.removeEventListener('focusin', onFocusIn);
		document.removeEventListener('compositionstart', releaseAllPhysicalHeldCodes);
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
		const originalBytes = bytes.slice();
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
		installedRom = { bytes: originalBytes, model, source };
		romLoaded = true;
		lastError = null;
		if (callStackOpen) void ensureSymbols();
	}

	function refreshFast() {
		if (worker) return;
		if (!emulator) return;
		updateDeliveredContacts(emulator.input_contacts());
		powerState = emulator.power_state();
		typingStatus = fallbackInputs.typingStatus();
		pasteStatus = fallbackInputs.pasteStatus();
		pacingStatus = emulator.pacing_status?.() ?? null;
		try {
			const geometry = emulator.lcd_capture_if_changed(false);
			if (geometry && typeof geometry === 'object') {
				const kind = normalizeLcdKind((geometry as any).kind);
				const cols = (geometry as any).cols;
				const rows = (geometry as any).rows;
				if (kind) lcdKind = kind;
				if (typeof cols === 'number') lcdCols = cols;
				lcdPixelScale = geometry.pixel_scale;
				if (typeof rows === 'number') lcdRows = rows;
				lcdPixels = geometry.pixels;
				lcdFrameGeneration = romLoadGeneration;
			}
		} catch {
			// ignore
		}
		lcdAnnunciatorBytes = emulator.lcd_annunciator_bytes?.() ?? null;
		if (lcdChipsOpen) lcdChipPixels = emulator.lcd_chip_pixels();
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
		lcdChipsOpen;
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
		return automaticHostSlice(emulator, count, fallbackInputs, typingCatchUp);
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
		if (
			installedRom &&
			!window.confirm(
				'Replace the current ROM and discard this session? Browser sessions cannot currently be restored.',
			)
		) {
			input.value = '';
			return;
		}
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

	function selectModel(event: Event) {
		const select = event.currentTarget as HTMLSelectElement;
		const model = normalizeRomModel(select.value);
		if (!model || model === romModel) return;
		if (
			installedRom &&
			!window.confirm('Change device model and discard this session? Browser sessions cannot currently be restored.')
		) {
			select.value = romModel;
			return;
		}
		romModelWasPersisted = true;
		$romModelStore = model;
		pasteOpen = false;
		pasteText = '';
		void tryAutoLoadRom(true);
	}

	function diagnosticReport() {
		return safeJson({
			format: 'sc62015-ui-diagnostics-v1',
			model: installedRom?.model ?? romModel,
			romSource,
			build: buildInfo,
			error: lastError,
			workerHealth,
			running,
			powerState,
			lastObserved: { pc: pcReg, instructionCount, regs, debugState, typingStatus, pasteStatus, pacingStatus },
			limitations:
				'Diagnostic observations only, possibly preceding the fault. Not a restorable session. No complete RAM/peripheral/RTC snapshot is exported.',
		});
	}

	async function copyDiagnostics() {
		try {
			await navigator.clipboard.writeText(diagnosticReport());
		} catch {
			keyboardNotice = 'Clipboard unavailable. Use Download diagnostics instead.';
		}
	}

	function exportDiagnostics() {
		downloadBlob(
			new Blob([diagnosticReport()], { type: 'application/json' }),
			`${installedRom?.model ?? romModel}-diagnostics.json`,
		);
	}

	async function resetSession() {
		if (!installedRom || loadingRom) return;
		if (
			!window.confirm(
				'Reset the machine and discard all session changes? This reloads the last successfully loaded ROM, not a saved session.',
			)
		)
			return;
		const savedRom = installedRom;
		if (workerHealth === 'ready' && !(await stop())) return;
		if (!worker && (stepBusy || functionRunnerBusy)) {
			lastError = 'Wait for the cancelled operation to finish before resetting.';
			return;
		}
		const generation = ++romLoadGeneration;
		workerRequests?.fail(new Error('Machine reset by user'));
		worker?.terminate();
		worker = null;
		workerRequests = null;
		if (emulator) {
			fallbackInputs.clear();
			emulator.free?.();
		}
		emulator = null;
		emulatorReady = null;
		workerHealth = 'ready';
		running = false;
		stepBusy = false;
		functionRunnerBusy = false;
		controlPending = null;
		executionMode = 'interactive';
		typingCatchUp = false;
		romLoaded = false;
		loadingRom = true;
		pasteOpen = false;
		pasteText = '';
		typingStatus = { pending: 0, blocked: false, capacity: 128 };
		pasteStatus = { pending: 0, total: 0 };
		deliveredContacts = new Set();
		physicalHeldCodes.clear();
		physicalHighlights = new Set();
		$romModelStore = savedRom.model;
		try {
			await ensureWorker();
			await installRom(savedRom.bytes.slice(), savedRom.model, savedRom.source, generation);
		} catch (error) {
			if (generation === romLoadGeneration) lastError = `Reset failed: ${String(error)}`;
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
		// Host shortcuts must not leave a guest modifier held when the browser
		// consumes the matching key-up (e.g. opening a new tab with Cmd/Ctrl).
		if (event.metaKey || event.ctrlKey || event.altKey) {
			releaseAllPhysicalHeldCodes();
			return;
		}
		if (
			event.defaultPrevented ||
			!romLoaded ||
			workerHealth !== 'ready' ||
			event.isComposing ||
			isHostControl(event.target)
		)
			return;
		if (
			event.target instanceof Element &&
			event.target.closest('button') &&
			(event.key === 'Enter' || event.key === ' ')
		)
			return;
		if (physicalHeldCodes.has(event.code)) {
			event.preventDefault(); // ROM owns repeat; don't scroll the host page.
			return;
		}
		if (event.repeat) return;
		const contacts = contactsForKeyEvent(event, romModel, hostKeyboardMode);
		if (contacts.length === 0) {
			if (event.key.length === 1)
				keyboardNotice = `No qualified ${romModel} key mapping for “${event.key}”.${romModel === 'iq-7000' && event.key === ',' ? ' For comma, press F9, release it, then press K.' : ''}`;
			return;
		}
		keyboardNotice = '';
		physicalHeldCodes.set(event.code, contacts);
		contacts.forEach((code, index) => setPhysicalMatrixCode(code, true, `${event.code}:${index}`));
		physicalHighlights = new Set([...physicalHeldCodes.values()].flat());
		event.preventDefault();
	}

	function onKeyUp(event: KeyboardEvent) {
		const contacts = physicalHeldCodes.get(event.code);
		if (contacts === undefined) return;
		physicalHeldCodes.delete(event.code);
		// Release the original chord, not a fresh mapping of event.key: Shift or
		// the host layout may have changed since key-down. Release modifier last.
		for (let index = contacts.length - 1; index >= 0; index--)
			setPhysicalMatrixCode(contacts[index], false, `${event.code}:${index}`);
		physicalHighlights = new Set([...physicalHeldCodes.values()].flat());
		event.preventDefault();
	}

	function isHostControl(target: EventTarget | null): boolean {
		return (
			target instanceof Element &&
			(target.matches('[data-host-scroll]') ||
				Boolean(
					target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"]'),
				))
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
	$: if (hostKeyboardMode !== activeHostKeyboardMode) {
		releaseAllPhysicalHeldCodes();
		keyboardNotice = '';
		activeHostKeyboardMode = hostKeyboardMode;
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

<svelte:window on:paste={onPaste} />

<main>
	<header class="device-toolbar">
		<h1>{romModel.toUpperCase()}</h1>
		<div class="toolbar-actions">
			<button
				disabled={!romLoaded || workerHealth !== 'ready'}
				on:click={() => {
					setContact('virtual', 'on', true, 'toolbar-on', false, 40_000);
					setContact('virtual', 'on', false, 'toolbar-on', false, 40_000);
				}}
				title="Press the device ON key; does not reset or toggle emulator pause"
				data-testid="power-on">Power / ON</button
			>
			<button
				data-testid="pause-resume"
				disabled={!romLoaded ||
					!!controlPending ||
					workerHealth === 'faulted' ||
					(!running && executionMode === 'deterministic')}
				on:click={() =>
					running || stepBusy || functionRunnerBusy || workerHealth === 'unresponsive' ? stop() : start()}
				>{running || stepBusy || functionRunnerBusy ? 'Pause' : 'Resume'}</button
			>
			<button aria-label="More options" aria-expanded={menuOpen} on:click={() => (menuOpen = !menuOpen)}>⋯</button>
		</div>
	</header>
	{#if menuOpen}
		<section class="quick-options" aria-label="Device options">
			<label
				>View <select bind:value={lcdOnly} data-testid="display-view"
					><option value={false}>Device</option><option value={true}>LCD only</option></select
				></label
			>
			<label
				>Pace
				<select
					data-testid="pace-preset"
					value={typingCatchUp ? 'responsive' : 'device'}
					disabled={executionMode !== 'interactive'}
					on:change={(event) => {
						typingCatchUp = event.currentTarget.value === 'responsive';
						pushWorkerOptions();
					}}
				>
					<option value="device">Device pace</option><option value="responsive">Responsive</option>
				</select>
			</label>
			<button on:click={focusDeviceKeyboard} disabled={!romLoaded}>Keyboard focus</button>
			<button data-testid="open-paste" on:click={() => openPaste()} disabled={!romLoaded}>Paste text…</button>
			<button
				data-testid="accuracy-toggle"
				on:click={() => {
					accuracyOpen = !accuracyOpen;
					menuOpen = false;
				}}>Accuracy & controls</button
			>
			<button data-testid="reset-session" on:click={resetSession} disabled={!installedRom || loadingRom}
				>Reset machine…</button
			>
			<button
				data-testid="export-lcd"
				on:click={exportLcd}
				disabled={!lcdPixels || exportingLcd || !romLoaded || lcdFrameGeneration !== romLoadGeneration}
				>Save LCD PNG</button
			>
			{#if lastCapture}<button data-testid="export-lcd-metadata" on:click={exportCaptureMetadata}
					>Last capture metadata</button
				>{/if}
			<button
				on:click={() => {
					shortcutsOpen = !shortcutsOpen;
					menuOpen = false;
				}}>Shortcuts</button
			>
			<button
				on:click={() => {
					advancedOpen = !advancedOpen;
					menuOpen = false;
				}}>Advanced</button
			>
			<p class="hint">
				Device pace is nominal, not hardware-calibrated. Responsive accelerates buffered typing and the RTC together.
			</p>
		</section>
	{/if}
	{#if shortcutsOpen}
		<section class="quick-options" aria-label="Keyboard shortcuts" data-testid="keyboard-shortcuts">
			<strong>Keyboard shortcuts</strong>
			<button on:click={() => (shortcutsOpen = false)}>Close shortcuts</button>
			<p class="hint">
				Click the device to type. F9 = device SHIFT · F10 = CAPS · F12 = ON. Device CAPS controls letter case; host
				Shift selects mapped punctuation.
			</p>
			<p class="hint">
				{romModel === 'iq-7000'
					? 'F1–F8: Calendar, Schedule, TEL, MEMO, Calc, Card, World, Home. Enter stores; F11 or Shift+Enter inserts a newline. Page Up/Down searches.'
					: 'F1–F5: PF1–PF5. F6: BASIC. F7: MENU. F8: Clear. F11: device CTRL. Enter: ENTER.'}
			</p>
			<p class="hint">
				Letters, digits, Space, arrows, Backspace, Delete and Insert use device keys. Hover a key for its host binding.
				Depressed keys show delivered contacts; cyan outlines show host-held keys, not ROM acknowledgement. Browser
				shortcuts and text fields stay with the host.
			</p>
		</section>
	{/if}
	{#if !romLoaded || !running || controlPending || functionRunnerBusy || stepBusy || workerHealth !== 'ready' || powerState === 'off'}
		<p class="hint" role="status" data-testid="device-status">
			{workerHealth !== 'ready'
				? statusLabel
				: controlPending === 'mode'
					? 'Changing execution mode…'
					: controlPending || functionRunnerBusy || stepBusy
						? statusLabel
						: loadingRom
							? 'Loading ROM…'
							: !romLoaded
								? 'No ROM loaded — open Advanced to select a ROM.'
								: !running && workerHealth === 'ready'
									? `Paused${powerState === 'off' ? ' (device OFF)' : ''} — device time is frozen. Resume to use the device.`
									: powerState === 'off' && running
										? `Device OFF — emulator running. Power / ON wakes the device.${romModel === 'iq-7000' ? ' Emulated RTC time still advances.' : ''}`
										: statusLabel}
		</p>
	{/if}
	{#if lastError}
		<section class="fault-actions" aria-label="Emulator problem">
			<p class="error" role="alert">{lastError}</p>
			<button on:click={copyDiagnostics}>Copy diagnostics</button>
			<button data-testid="download-diagnostics" on:click={exportDiagnostics}>Download diagnostics</button>
			<button on:click={resetSession} disabled={!installedRom || loadingRom}>Reset machine…</button>
			<p class="hint">Diagnostics are observations, not a restorable session. Reset discards unsaved device data.</p>
		</section>
	{/if}
	{#if accuracyOpen}
		<section class="quick-options" aria-label="Accuracy and controls" data-testid="accuracy-panel">
			<strong>Accuracy & controls</strong><button on:click={() => (accuracyOpen = false)}>Close accuracy notes</button>
			<p class="hint">
				Reference-based 2D case and key proportions, not measured or scan-derived. Card artwork is a replaceable
				placeholder, not a second display. Dotted, disabled keys have no qualified input mapping. On narrow screens, pan
				the case or choose LCD only.
			</p>
			<p class="hint">
				LCD pixels come directly from emulator display state. IQ fixed segments use LCD state; BATT, CARD, beep, alarm
				and arrows remain provisional physical assignments. SHIFT/CAPS mapping has stronger ROM evidence. No screenshot
				text is reconstructed or patched.
			</p>
			<p class="hint">
				Device pace is not hardware-calibrated. IQ timing uses a provisional PC-compatible timebase. Responsive
				accelerates RTC time too; Pause freezes it. OFF is a device state, not a pause. Full session restoration is
				unavailable; see Advanced → Session safety & recovery.
			</p>
		</section>
	{/if}
	{#if pasteOpen}
		<section class="paste-preview" aria-label="Paste preview" data-testid="paste-preview">
			<label for="paste-editor">Text to type through device keys (up to {MAX_PASTE_CHARACTERS} characters)</label>
			<textarea
				id="paste-editor"
				data-testid="paste-editor"
				bind:this={pasteEditor}
				bind:value={pasteText}
				rows="5"
				disabled={pasteSubmitting}
			></textarea>
			<p class="hint">
				Device CAPS controls case. This is a key sequence, not direct text insertion. Check the current app and CAPS
				state first. {romModel === 'iq-7000'
					? 'Newlines use the newline key; comma uses SHIFT, release, K.'
					: 'Newlines press ENTER and may execute a BASIC command or calculator expression.'} Paused paste waits for Resume.
			</p>
			{#if pastePlan.error}<p class="error" role="alert">{pastePlan.error}</p>{/if}
			{#if pastePlan.unsupported.length}<p class="error" role="alert" data-testid="paste-unsupported">
					Unsupported characters — nothing will be typed: {pastePlan.unsupported
						.slice(0, 16)
						.map((entry) => `${JSON.stringify(entry.character)} at ${entry.position}`)
						.join(', ')}{pastePlan.unsupported.length > 16 ? ` (${pastePlan.unsupported.length} total)` : ''}
				</p>{/if}
			<button
				data-testid="submit-paste"
				on:click={submitPaste}
				disabled={!romLoaded ||
					pasteSubmitting ||
					stepBusy ||
					functionRunnerBusy ||
					!!controlPending ||
					workerHealth !== 'ready' ||
					!!pastePlan.error ||
					pastePlan.unsupported.length > 0 ||
					!pastePlan.contacts.length}>Type {pastePlan.contacts.length} device keys</button
			>
			<button
				on:click={() => {
					pasteOpen = false;
					pasteText = '';
				}}
				disabled={pasteSubmitting}>Close paste preview</button
			>
		</section>
	{/if}
	{#if typingStatus.pending || typingStatus.blocked || pasteStatus.pending}
		<div class="queue-status" role="status">
			{pasteStatus.pending || typingStatus.pending} keys pending{pasteStatus.pending
				? ' — paste; delivered contacts, not ROM acknowledgement'
				: ''}{typingStatus.blocked ? ' — input buffer blocked' : ''}<button on:click={releaseAllPhysicalHeldCodes}
				>Cancel queued keys</button
			>
		</div>
	{/if}
	{#if keyboardNotice}<p class="hint" role="status" data-testid="keyboard-notice">{keyboardNotice}</p>{/if}
	<div
		class="keyboard-target"
		role="group"
		tabindex="-1"
		aria-label="Device keyboard input"
		aria-describedby="keyboard-help"
		data-testid="keyboard-target"
		bind:this={keyboardTarget}
		on:pointerup={(event) => {
			if (romLoaded && !(event.target instanceof Element && event.target.matches('[data-host-scroll]')))
				void focusDeviceKeyboard();
		}}
	>
		{#if lcdOnly}
			<div class="lcd-only" aria-label="Emulated LCD including fixed segments">
				<LcdCanvas pixels={lcdPixels} cols={lcdCols} rows={lcdRows} scale={4 / lcdPixelScale} pixelFormat="gray8" fit />
			</div>
		{:else}
			<DeviceShell
				model={romModel}
				disabled={!romLoaded || workerHealth !== 'ready'}
				{hostKeyboardMode}
				{physicalHighlights}
				{deliveredContacts}
				onPress={virtualPress}
				onRelease={virtualRelease}
				onCancelAll={() => releaseInputSource('virtual')}
			>
				<div class="lcd-display" aria-label="Emulated LCD including fixed segments">
					<LcdCanvas
						pixels={lcdPixels}
						cols={lcdCols}
						rows={lcdRows}
						scale={4 / lcdPixelScale}
						pixelFormat="gray8"
						fit
					/>
				</div>
			</DeviceShell>
		{/if}
	</div>
	<details class="advanced" bind:open={advancedOpen} data-testid="advanced-panel">
		<summary>Advanced</summary>
		<details data-testid="session-safety">
			<summary>Session safety & recovery</summary>
			<p class="hint">
				Browser sessions are held in memory only. Reload, reset or ROM/model replacement loses device RAM changes.
				Complete WASM snapshot restoration is not available: native snapshot routines are not exposed here, and they
				reject active RTC/peripheral/serial state they cannot represent. We do not silently save a partial snapshot.
			</p>
			<p class="hint">
				LCD PNGs and diagnostics can preserve evidence, not a resumable machine. Device OFF is not emulator Pause: the
				IQ RTC continues when the emulator runs, even with the device OFF. Pausing freezes all emulated time.
			</p>
			<button on:click={exportDiagnostics}>Download diagnostic observations</button>
		</details>
		<header class="page-header">
			<div>
				<p class="eyebrow">SC62015 / RUST + WASM</p>
				<h1>Pocket device bench</h1>
			</div>
			<div class="source-controls">
				<label>
					ROM preset:
					<select value={$romModelStore} on:change={selectModel} data-testid="rom-model">
						<option value="iq-7000">IQ-7000</option>
						<option value="pc-e500">PC-E500</option>
					</select>
				</label>

				<label>
					Load ROM file:
					<input type="file" accept=".bin,.rom,.img" on:change={onSelectRom} />
				</label>
			</div>
		</header>

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
				class="run-button"
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
					(!running &&
						!stepBusy &&
						!functionRunnerBusy &&
						controlPending !== 'start' &&
						workerHealth !== 'unresponsive')}>Stop</button
			>
			<label>
				Target FPS:
				<input type="number" min="1" max="60" step="1" bind:value={targetFps} />
			</label>
		</div>

		<p class="hint" data-testid="emu-status">
			Status: {statusLabel} • PC: {hex(pc)} • Instr: {instructionCount ?? '—'}
		</p>
		{#if executionMode === 'deterministic'}
			<p class="hint">
				Automatic Run is disabled. Repeatable results require the same ROM/state, a fixed RTC seed (the default seed
				comes from host time), and identical inputs at identical scheduler boundaries. This selection does not reset
				your machine or make live human input deterministic.
			</p>
		{/if}
		{#if functionRunnerBusy && functionProgress}
			<p class="hint" data-testid="execution-progress">{functionProgress}</p>
		{/if}
		<div class="keyboard-controls">
			<button data-testid="keyboard-focus" on:click={focusDeviceKeyboard} disabled={!romLoaded}>Type on device</button>
			<label
				><input type="checkbox" data-testid="physical-keyboard-toggle" bind:checked={physicalKeyboardEnabled} />
				Physical keyboard {physicalKeyboardEnabled ? 'enabled' : 'off'}</label
			>
			<label
				>Mapping:
				<select data-testid="physical-keyboard-mode" bind:value={hostKeyboardMode}>
					<option value="symbols">Letters & symbols (buffered)</option>
					<option value="keycaps">Device keycaps (raw Shift)</option>
				</select></label
			>
			<button data-testid="clear-typing" on:click={releaseAllPhysicalHeldCodes}>Clear queued keys</button>
			<label
				><input
					type="checkbox"
					data-testid="typing-catch-up"
					bind:checked={typingCatchUp}
					on:change={pushWorkerOptions}
				/>
				Speed up while typing</label
			>
		</div>
		<p class="hint" data-testid="typing-status">
			Typing buffer: {typingStatus.pending}/{typingStatus.capacity}{typingStatus.blocked
				? ' — blocked; clear queued keys to recover'
				: ''}.
			{hostKeyboardMode === 'symbols'
				? 'Fast presses are delivered in order with a scan hold and release gap.'
				: 'Raw keycaps: exact holds; short presses can miss ROM scanning.'}
			{#if typingCatchUp}Catch-up advances emulated time/RTC faster during buffered input; paused and deterministic
				execution are unchanged.{/if}
		</p>
		<p class="hint" id="keyboard-help">
			Click “Type on device”, then Run for live typing. Paused input does not advance the machine. F9 = device SHIFT ·
			F10 = CAPS · F12 = ON (your keyboard may require Fn). Text fields and Ctrl/Cmd/Alt shortcuts stay with the
			browser. Hover a device key for its host bindings.
		</p>
		<details class="keyboard-help">
			<summary>Keyboard mappings & letter case</summary>
			<p class="hint">
				{#if romModel === 'iq-7000'}
					F1–F5 = Calendar / Schedule / TEL / MEMO / Calc; F6–F8 = Card / World / Home. Page Up/Down = Search; Enter =
					Store; F11 = newline (also Shift+Enter in Letters & symbols).
				{:else}
					F1–F5 = PF1–PF5; F6 = BASIC; F7 = MENU; F8 = Clear; F11 = device CTRL.
				{/if}
				Both: letters, digits, Space, arrows, Backspace, Delete, Insert, Escape (Clear), numeric keypad; keypad Enter = equals.
			</p>
			<p class="hint">
				Letters & symbols follows your host keyboard layout: +, − (minus key), *, /, = and decimal point use device
				operator keys, including symbols typed with Shift. Device CAPS controls letter case, not host Shift. Use F9 for
				device functions (IQ: F9 then A = EDIT). For IQ comma, press F9, release it, then K; a direct comma key is not
				mapped. Unsupported punctuation is reported, not substituted. Paste opens a preview and supports only qualified
				key sequences; IME and automatic case conversion are not supported.
			</p>
			<p class="hint">
				Device keycaps uses physical host key positions and maps host Shift directly to device SHIFT: shifted legends
				belong to the organizer, not your desktop keyboard. Use the numeric keypad or on-screen keys for operators.
			</p>
		</details>
		{#if romLoaded}<p class="hint lcd-meta">
				LCD: {lcdKind ?? '—'} ({lcdCols}×{lcdRows}) · Case proportions, materials and key legends are provisional.
			</p>{/if}
		<details class="session-details">
			<summary>Session & timing details</summary>
			{#if romSource}<p class="hint">Loaded ROM ({romModel}) via {romSource}</p>{/if}
			<p class="hint" data-testid="build-info">WASM: {formatBuildInfo(buildInfo)}</p>
			<p class="hint" data-testid="pacing-status">
				{executionMode}: Interactive pacing uses {pacingStatus?.nominal_timebase_hz ?? '—'} compatibility timing units/s,
				not hardware-calibrated MHz. IQ-7000 currently uses the PC-E500 fallback timebase. The RTC follows emulated elapsed
				time; paused wall time is not simulated. Catch-up is capped at 50 ms; dropped host backlog: {(
					Number(pacingStatus?.dropped_host_ns ?? 0) / 1e6
				).toFixed(1)} ms. Step and Function Runner use explicit, unthrottled budgets in every mode.
			</p>
		</details>
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
			<details
				bind:open={lcdChipsOpen}
				on:toggle={() => {
					if (lcdChipsOpen) refreshAllNow();
				}}
			>
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
		{#if lastInputAck}<p class="hint" data-testid="input-ack">{lastInputAck}</p>{/if}

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
	</details>
</main>

<style>
	.paste-preview {
		padding: 16px;
		border: 1px solid #526169;
		border-radius: 8px;
	}
	.paste-preview textarea {
		display: block;
		box-sizing: border-box;
		width: 100%;
		margin: 10px 0;
		font: inherit;
	}
	.device-toolbar,
	.toolbar-actions,
	.quick-options,
	.queue-status {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}
	.device-toolbar {
		justify-content: space-between;
	}
	.quick-options {
		padding: 14px;
		border: 1px solid #35434a;
		border-radius: 8px;
	}
	.quick-options p {
		flex-basis: 100%;
		margin: 0;
		font-size: 12px;
	}
	.lcd-only {
		max-width: 900px;
		padding: 24px;
		margin: 0 auto;
		background: #bfc0b3;
		border-radius: 6px;
	}
	.advanced {
		border-top: 1px solid #35434a;
	}
	.keyboard-controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px 22px;
		padding: 12px 0 0;
	}
	.keyboard-controls button {
		background: #275a59;
		border-color: #73a9a0;
	}
	.keyboard-help {
		margin-bottom: 14px;
	}
	.keyboard-target:focus {
		outline: 2px solid #73a9a0;
		outline-offset: 4px;
		border-radius: 16px;
	}
	:global(body) {
		margin: 0;
		background: #10191f;
		color: #e5eae9;
		color-scheme: dark;
	}
	main {
		display: flex;
		flex-direction: column;
		gap: 14px;
		padding: 25px 28px 60px;
		max-width: 1200px;
		margin: 0 auto;
		min-width: 0;
		font-family: system-ui, sans-serif;
	}
	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 20px;
		padding-bottom: 8px;
	}
	h1 {
		font-size: 25px;
		margin: 5px 0 0;
		font-weight: 550;
		letter-spacing: -0.8px;
	}
	.eyebrow {
		color: #91b8b2;
		font:
			10px/1.4 ui-monospace,
			monospace;
		letter-spacing: 2px;
		margin: 0;
	}
	.source-controls {
		display: flex;
		gap: 18px;
		flex-wrap: wrap;
		font-size: 11px;
		color: #a9bbbf;
	}
	.source-controls input {
		max-width: 215px;
		font-size: 11px;
	}
	.controls {
		background: #1a282f;
		padding: 12px 14px;
		border: 1px solid #314047;
		border-radius: 9px;
		font-size: 12px;
	}
	.controls input[type='number'] {
		width: 54px;
	}
	.controls .run-button {
		background: #b6d2bd;
		color: #13241e;
		border-color: #b6d2bd;
		font-weight: 650;
		min-width: 70px;
	}
	main > .hint {
		margin: 0;
		font-size: 12px;
	}
	.lcd-meta {
		font-size: 11px;
	}
	:global(summary) {
		cursor: pointer;
		color: #a7bec5;
		padding: 8px 0;
		font-size: 13px;
	}
	:global(button),
	:global(select),
	:global(input) {
		font: inherit;
	}
	:global(button:focus-visible),
	:global(select:focus-visible),
	:global(input:focus-visible),
	:global(summary:focus-visible) {
		outline: 2px solid #b8dfd4;
		outline-offset: 3px;
	}
	@media (max-width: 700px) {
		main {
			padding: 18px 12px 40px;
		}
		h1 {
			font-size: 23px;
		}
		.source-controls {
			gap: 10px;
		}
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
		color: #a6b6bd;
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
		padding: 7px 11px;
		border: 1px solid #526169;
		background: #283940;
		color: #e5eeed;
		border-radius: 5px;
	}
	button:disabled {
		opacity: 0.4;
	}
	select,
	input {
		accent-color: #b6d2bd;
	}
	select {
		padding: 6px;
		border: 1px solid #526169;
		border-radius: 5px;
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
		align-items: center;
		justify-content: center;
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
