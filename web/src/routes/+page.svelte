<script lang="ts">
	import { configuredPvHandwriting, type PvHandwritingAsset } from '$lib/pvf1_bootstrap';
	import { onDestroy, onMount, tick } from 'svelte';
	import { BatterySession, ozBatteryIdentity, type BatteryVersion } from '$lib/emulator/oz_battery_store';
	import {
		capturePvSession,
		encodePvSession,
		decodePvSession,
		pvBatteryIdentity,
		pvRetainedIdentity,
		loadPvSession,
		type PvAssets,
		type PvSession,
	} from '$lib/emulator/pvf1_session';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import { LCD_CHIP_COLS, LCD_CHIP_ROWS, LCD_COLS, LCD_ROWS } from '$lib/lcd';
	import DeviceShell from '$lib/components/DeviceShell.svelte';
	import PvInputStatus from '$lib/components/PvInputStatus.svelte';
	import { lcdPng, downloadBlob } from '$lib/lcd_export';
	import { contactsForKeyEvent, type HostKeyboardMode } from '$lib/keymap';
	import {
		loadPa9500,
		resetPa9500,
		paEditorSequence,
		type PaEditorAction,
		type PaProfile,
		type PaInputMode,
	} from '$lib/emulator/pa9500_loader';
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
	import { runOzPhysicalReplay, type OzProfile, type TabletContact } from '$lib/emulator/oz9600_replay';
	import { piPanelCapture } from '$lib/pi3000_panel';
	import { TabletInputs } from '$lib/emulator/tablet_inputs';
	import { PvReadiness, type PvInputPhase } from '$lib/emulator/pvf1_readiness';
	import { planPaste, MAX_PASTE_CHARACTERS } from '$lib/emulator/paste_plan';
	import { AudioPlayback, AudioDelivery, supportsAudio } from '$lib/emulator/oz_audio';
	import {
		SerialCapture,
		SerialDelivery,
		serialInput,
		supportsSerial,
		queueSerialInput,
		type SerialFormat,
		type SerialEnding,
	} from '$lib/emulator/oz_serial';

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

	const ozAutosaveStore = createPersistedStore<boolean>('sc62015:oz-autosave', true, {
		serialize: (value) => String(value),
		deserialize: (value) => value !== 'false',
	});
	const pvAutosaveStore = createPersistedStore<boolean>('sc62015:pv-autosave', true, {
		serialize: String,
		deserialize: (v) => v !== 'false',
	});
	let pvActiveAssets: PvAssets = {
		cardRom: new Uint8Array(),
		handwritingModel: new Uint8Array(),
		ramCardWritable: true,
	};
	let pvRecoveryVersions: BatteryVersion[] = [];
	let pvRecoveryOverride: Uint8Array | null = null;
	function autosaveEnabled(model: RomModel) {
		return model === 'pv-f1' ? $pvAutosaveStore : model === 'oz-9600' && $ozAutosaveStore;
	}
	let batterySession: BatterySession | null = null;
	let batteryGeneration = -1;
	let batteryStatus = 'Automatic saving is ready';
	let batteryError = '';
	let batteryBusy = false;
	let batterySave: Promise<boolean> | null = null;
	let batteryTimer: ReturnType<typeof setInterval> | null = null;
	let requestedRom: { bytes: Uint8Array; model: RomModel; source: string | null } | null = null;
	let ozProfile: OzProfile = 'provisional-v1';
	let ozLoadedProfile: OzProfile = 'strict';
	let paProfile: PaProfile = 'bootstrap-v3';
	let paLoadedProfile: PaProfile | null = null;
	let paUart = false;
	let paLoadedUart = false;
	let paInput: PaInputMode = 'native';
	let paCardRom = new Uint8Array();
	let paCardSram = new Uint8Array();
	let paCardNames = { rom: '', sram: '' };
	let paCardError = '';
	let ozReplayReport: any = null;
	let ozReplayStatus = '';
	let piBackupError = '';
	let piBackupStatus = '';
	let pvHandwritingModel = new Uint8Array();
	let pvHandwritingName = '';
	let pvHandwritingError = '';
	let pvHandwritingBusy = false;
	let pvBackupStatus = '';
	let pvBackupError = '';
	let pvCardRom = new Uint8Array();
	let pvCardSram = new Uint8Array();
	let pvCardNames = { rom: '', sram: '' };
	let pvCardError = '';
	let pvCardStatus = '';
	let pvRamCard = new Uint8Array();
	let pvRamCardName = '';
	let pvRamCardWritable = true;
	let pvRamCardRequested = false;
	let sessionRecording = false;
	let sessionBusy = false;
	let sessionStatus = '';
	let sessionError = '';
	let sessionProgress: { boundaries: number; events: number; error: string | null } | null = null;
	let sessionStepBudget = 1_000_000;
	let running = false;
	let soundEnabled = false;
	let soundBusy = false;
	let soundError = '';
	let serialFormat: SerialFormat = 'text';
	let serialEnding: SerialEnding = 'crlf';
	let serialText = '';
	let serialBusy = false;
	let serialError = '';
	let serialNotice = '';
	let serialRevision = 0;
	const serialCapture = new SerialCapture();
	let serialStatus = serialCapture.status();
	let serialPreview = '';
	$: {
		serialRevision;
		serialStatus = serialCapture.status();
		serialPreview = serialCapture.preview();
	}
	const fallbackSerial = new SerialDelivery((packet) => {
		serialCapture.push(packet);
		serialRevision++;
		fallbackSerial.consumed(packet.generation, packet.epoch, packet.sequence);
	});
	function resetFallbackSerial() {
		serialCapture.begin(fallbackSerial.reset(romLoadGeneration));
		serialRevision++;
		serialNotice = serialError = '';
	}
	function pumpFallbackSerial() {
		if (!worker && supportsSerial(romModel) && emulator?.has_rom())
			fallbackSerial.pump(() => emulator.sio_drain_tx_bytes());
	}
	async function sendSerial() {
		if (
			serialBusy ||
			loadingRom ||
			stepBusy ||
			functionRunnerBusy ||
			controlPending ||
			!romLoaded ||
			workerHealth !== 'ready'
		)
			return;
		const generation = romLoadGeneration;
		serialBusy = true;
		serialError = '';
		try {
			const bytes = serialInput(serialText, serialFormat, serialEnding);
			const result = worker
				? await workerCall('serial_send', { generation, bytes }, [bytes.buffer])
				: { queued: queueSerialInput(emulator, bytes) };
			if (generation === romLoadGeneration)
				serialNotice = `${result.queued} bytes queued for the device${running ? '' : ' — Resume or Step to receive them'}`;
		} catch (error) {
			if (generation === romLoadGeneration) serialError = String(error);
		} finally {
			serialBusy = false;
		}
	}
	function downloadSerial() {
		const bytes = serialCapture.bytes();
		downloadBlob(
			new Blob([bytes as Uint8Array<ArrayBuffer>], { type: 'application/octet-stream' }),
			`${romModel}-serial.raw`,
		);
	}
	const audioPlayback = new AudioPlayback();
	let fallbackSoundActive = false;
	const fallbackAudio = new AudioDelivery((packet) => {
		audioPlayback.push(packet);
		fallbackAudio.consumed(packet.generation, packet.epoch, packet.sequence);
	});
	function syncFallbackAudio(force = false) {
		const enabled =
			soundEnabled &&
			running &&
			supportsAudio(romModel) &&
			emulator?.execution_mode() === 'interactive' &&
			!document.hidden;
		if (force || enabled !== fallbackSoundActive) {
			if (supportsAudio(romModel) && emulator?.has_rom()) emulator.set_audio_enabled(enabled);
			fallbackSoundActive = enabled;
			const reset = fallbackAudio.reset(romLoadGeneration, enabled);
			audioPlayback.begin(reset.generation, reset.epoch, reset.enabled);
		}
	}
	async function pushSoundPreference() {
		if (worker)
			await workerCall('set_audio', { enabled: soundEnabled && !document.hidden, generation: romLoadGeneration });
		else syncFallbackAudio(true);
	}
	async function toggleSound() {
		if (soundBusy) return;
		soundBusy = true;
		soundError = '';
		try {
			if (soundEnabled) {
				soundEnabled = false;
				audioPlayback.disable();
			} else soundEnabled = await audioPlayback.enable();
			await pushSoundPreference();
		} catch (error) {
			soundEnabled = false;
			audioPlayback.disable();
			soundError = String(error);
		} finally {
			soundBusy = false;
		}
	}
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
	$: pastePlan = planPaste(pasteText, romModel, paInput);
	let typingCatchUp = false;
	let advancedOpen = false;
	let menuOpen = false;
	let lcdOnly = false;
	const pendingVirtualRelease = new Map<number, number>();
	let pvInputPhase: PvInputPhase = 'starting';
	let penStatus = { strokes: 0, points: 0, blocked: false };
	const fallbackReadiness = new PvReadiness();
	function observeFallbackReadiness() {
		if (romModel === 'pv-f1' && emulator)
			pvInputPhase = fallbackReadiness.observe(emulator.power_state(), emulator.halted(), emulator.get_reg('PC'));
	}
	const fallbackCanDeliver = () => romModel !== 'pv-f1' || fallbackReadiness.canDeliver();
	const fallbackInputs = new HostInputs((contact, down) => {
		if (romModel === 'pv-f1' && contact === 'on' && down) {
			fallbackReadiness.noteOn(emulator.power_state() === 'off');
			if (!fallbackReadiness.canDeliver()) fallbackTablet.clear();
		}
		applyContact(emulator, contact, down);
	}, fallbackCanDeliver);
	const fallbackTablet = new TabletInputs(
		(c) =>
			romModel === 'pv-f1'
				? emulator.set_pvf1_tablet_contact(c.raw_x, c.raw_y, c.pressed)
				: romModel === 'pi-3000'
					? emulator.set_pi3000_tablet_contact(c.raw_x, c.raw_y, c.pressed)
					: emulator.set_oz9600_tablet_contact(c.raw_x, c.raw_y, c.pressed),
		() => (['pi-3000', 'pv-f1'].includes(romModel) ? 200_000 : 40_000),
		() => (romModel === 'pv-f1' ? 50_000 : 0),
		() => (romModel === 'pv-f1' ? 20_000 : 0),
		fallbackCanDeliver,
	);
	const fallbackHostInputs = {
		limitBudget: (n: number) => Math.min(fallbackInputs.limitBudget(n), fallbackTablet.limitBudget(n)),
		advance: (n: number) => {
			observeFallbackReadiness();
			fallbackInputs.advance(n);
			fallbackTablet.advance(n);
			penStatus = fallbackTablet.status();
		},
		typingBoostBudget: fallbackInputs.typingBoostBudget,
	};
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
		if (frame.pvInputPhase) pvInputPhase = frame.pvInputPhase;
		if (frame.pen) penStatus = frame.pen;
		if (frame?.model && frame.model !== romModel) return;
		if (frame.recording && sessionRecording) sessionProgress = frame.recording;
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
		const timeoutMs =
			type === 'stop' ? 1_000 : ['eval_js', 'oz_replay', 'session_replay'].includes(type) ? 300_000 : 30_000;
		return workerRequests.request<T>(message, transfer, timeoutMs);
	}

	function failWorker(message: string) {
		audioPlayback.pause();
		lastError = `${message}. Use Reset machine to replace the worker; unsaved emulator state will be lost.`;
		workerHealth = 'faulted';
		if (sessionRecording) sessionError = 'Recording lost when the worker failed; recover the last automatic save.';
		sessionRecording = false;
		sessionProgress = null;
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
			if (data.type === 'audio_reset' && data.generation === romLoadGeneration) {
				audioPlayback.begin(data.generation, data.epoch, data.enabled && soundEnabled && controlPending !== 'stop');
				return;
			}
			if (data.type === 'audio') {
				try {
					if (data.generation === romLoadGeneration) audioPlayback.push(data);
				} catch (error) {
					soundError = String(error);
					soundEnabled = false;
					audioPlayback.disable();
				} finally {
					workerPost({
						id: workerNextId++,
						type: 'audio_consumed',
						generation: data.generation,
						epoch: data.epoch,
						sequence: data.sequence,
					});
				}
				return;
			}
			if (data.type === 'audio_error' && data.generation === romLoadGeneration) {
				soundError = data.error;
				soundEnabled = false;
				audioPlayback.disable();
				return;
			}
			if (data.type === 'serial_reset' && data.generation === romLoadGeneration) {
				serialCapture.begin(data);
				serialRevision++;
				serialNotice = serialError = '';
			}
			if (data.type === 'serial_tx') {
				try {
					if (data.generation === romLoadGeneration && serialCapture.push(data)) serialRevision++;
				} catch (error) {
					serialError = String(error);
				} finally {
					worker?.postMessage({
						id: 0,
						type: 'serial_consumed',
						generation: data.generation,
						epoch: data.epoch,
						sequence: data.sequence,
					});
				}
			}
			if (data.type === 'oz_replay_progress') ozReplayStatus = `${data.step} steps completed`;
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
				if (data.pvInputPhase) pvInputPhase = data.pvInputPhase;
				if (data.pen) penStatus = data.pen;
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
		const minimumHold =
			holdOverride ??
			(source === 'virtual' && assistedTaps ? (romModel === 'pv-f1' && code !== 'on' ? 200_000 : 40_000) : 0);
		const buffered =
			code !== 'on' &&
			((source === 'physical' && hostKeyboardMode === 'symbols') ||
				(source === 'virtual' && romModel === 'pv-f1' && assistedTaps));
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
				fallbackInputs.set({
					source,
					owner,
					contact: code,
					down,
					cancel,
					minimumHold,
					minimumGap: romModel === 'pv-f1' && source === 'virtual' ? 50_000 : undefined,
					buffered,
				});
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
			if (source === 'virtual' && ['oz-9600', 'pi-3000', 'pv-f1'].includes(romModel)) fallbackTablet.clear();
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
	function cancelQueuedInput() {
		releaseAllPhysicalHeldCodes();
		releaseInputSource('virtual');
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
				const result = await workerCall('paste_text', { text: pasteText, generation, paInput });
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
		fallbackHostInputs.advance(stepped);
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
			const configuredModel =
				model === 'pv-f1' && !pvHandwritingModel.length ? await configuredPvHandwriting(fetch) : null;
			if (generation !== romLoadGeneration) return;
			await installRom(
				bytes,
				model,
				res.headers.get('x-rom-source'),
				generation,
				undefined,
				undefined,
				configuredModel,
			);
		} catch (err) {
			if (generation === romLoadGeneration) lastError = `Auto-load failed: ${String(err)}`;
		} finally {
			if (generation === romLoadGeneration) loadingRom = false;
		}
	}

	async function saveBattery(force = false): Promise<boolean> {
		if (batterySave) return batterySave;
		const session = batterySession;
		const generation = batteryGeneration;
		if (!session) return true;
		if (workerHealth !== 'ready' || !installedRom || !['oz-9600', 'pv-f1'].includes(installedRom.model)) return false;
		if (
			!force &&
			(loadingRom || !romLoaded || stepBusy || functionRunnerBusy || controlPending || generation !== romLoadGeneration)
		)
			return true;
		const save = (async () => {
			try {
				const bytes: Uint8Array =
					installedRom?.model === 'pv-f1'
						? await encodePvSession(
								worker
									? await workerCall('pv_session_export', { generation })
									: capturePvSession(emulator, pvActiveAssets),
							)
						: worker
							? await workerCall('oz_retained_export', { generation })
							: emulator.export_oz9600_retained();
				if (session !== batterySession) return false;
				const result = await session.save(bytes);
				if (session !== batterySession) return false;
				batteryError = '';
				batteryStatus = `Saved at ${new Date(result.savedAt).toLocaleTimeString()}`;
				if (installedRom?.model === 'pv-f1') pvRecoveryVersions = session.versions();
				return true;
			} catch (error) {
				if (session === batterySession)
					batteryError = `Automatic save failed: ${String(error)}. Export a backup or retry.`;
				return false;
			}
		})();
		batterySave = save;
		try {
			return await save;
		} finally {
			if (batterySave === save) batterySave = null;
		}
	}
	async function closeBattery() {
		if (!(await saveBattery(true)))
			throw new Error('Current records could not be saved. Export a backup or retry before replacing the session.');
		const session = batterySession;
		batterySession = null;
		await session?.close();
	}
	async function retryRequestedRom() {
		if (!requestedRom || loadingRom) return false;
		const requested = requestedRom;
		if (
			romLoaded &&
			!window.confirm('Recover saved records? Export a backup first if you want to keep current session changes.')
		)
			return false;
		const generation = ++romLoadGeneration;
		loadingRom = true;
		try {
			await ensureWorker();
			await installRom(requested.bytes.slice(), requested.model, requested.source, generation);
			return generation === romLoadGeneration;
		} catch (error) {
			if (generation === romLoadGeneration) lastError = `Load failed: ${String(error)}`;
			return false;
		} finally {
			if (generation === romLoadGeneration) loadingRom = false;
		}
	}
	async function changeAutosave() {
		if (batteryBusy) return;
		const model = installedRom?.model ?? romModel;
		batteryBusy = true;
		try {
			if (!autosaveEnabled(model)) {
				await closeBattery();
				batteryStatus = 'Automatic saving is off; export backups to keep records';
				batteryError = '';
				if (!romLoaded) await retryRequestedRom();
				return;
			}
			if (!romLoaded) {
				await retryRequestedRom();
				return;
			}
			try {
				const generation = romLoadGeneration;
				const bytes: Uint8Array =
					installedRom?.model === 'pv-f1'
						? await encodePvSession(
								worker
									? await workerCall('pv_session_export', { generation })
									: capturePvSession(emulator, pvActiveAssets),
							)
						: worker
							? await workerCall('oz_retained_export', { generation })
							: emulator.export_oz9600_retained();
				const key =
					model === 'pv-f1'
						? worker
							? await workerCall('pv_battery_identity', { bytes: installedRom!.bytes.slice() })
							: pvBatteryIdentity(() => new wasm.Sc62015Emulator(), installedRom!.bytes)
						: 'oz9600:' + Array.from(bytes.subarray(0, 72), (b) => b.toString(16).padStart(2, '0')).join('');
				const session = await BatterySession.open(key);
				if (generation !== romLoadGeneration || !autosaveEnabled(model) || loadingRom) {
					await session.close();
					return;
				}
				if (
					session.loaded &&
					(session.loaded.image.length !== bytes.length || !session.loaded.image.every((b, i) => b === bytes[i]))
				) {
					await session.close();
					throw new Error(
						'Existing saved records are present. Reload to recover them before enabling automatic saving.',
					);
				}
				batterySession = session;
				batteryGeneration = generation;
				await saveBattery(true);
			} catch (error) {
				batteryError = String(error);
			}
		} catch (error) {
			if (model === 'pv-f1') $pvAutosaveStore = true;
			else $ozAutosaveStore = true;
			batteryError = String(error);
		} finally {
			batteryBusy = false;
		}
	}
	function saveOnHidden() {
		if (document.hidden) void pauseForLifecycle('the app was hidden');
	}
	let lifecycleMessage = '';
	let lifecycleStop: Promise<void> | null = null;
	function pauseForLifecycle(reason: string): Promise<void> {
		if (!['pv-f1', 'oz-9600'].includes(romModel)) return saveBattery(true).then(() => {});
		if (!romLoaded || loadingRom || workerHealth !== 'ready') return Promise.resolve();
		if (lifecycleStop) return lifecycleStop;
		cancelQueuedInput();
		const active = running || stepBusy || functionRunnerBusy || controlPending === 'start';
		if (!active) return saveBattery(true).then(() => {});
		const generation = romLoadGeneration;
		lifecycleMessage = 'Pausing for browser suspension…';
		const job = (async () => {
			const paused = await stop();
			if (generation === romLoadGeneration)
				lifecycleMessage = paused
					? `Paused because ${reason}. Device time is frozen; select Resume to continue.`
					: 'Pause was not acknowledged; check the emulator status.';
		})();
		lifecycleStop = job;
		void job.finally(() => {
			if (lifecycleStop === job) lifecycleStop = null;
		});
		return job;
	}
	function onPageFreeze() {
		void pauseForLifecycle('the browser suspended this page');
	}
	async function installRom(
		bytes: Uint8Array,
		model: RomModel,
		source: string | null,
		generation: number,
		paRetained = new Uint8Array(),
		pvRetained = new Uint8Array(),
		configuredModel: PvHandwritingAsset | null = null,
	) {
		const originalBytes = bytes.slice();
		let handwritingModel = configuredModel?.bytes ?? pvHandwritingModel;
		if (generation !== romLoadGeneration) return;
		requestedRom = { bytes: originalBytes, model, source };
		if (!(await stop())) throw new Error('Pause was not acknowledged; ROM was not replaced');
		if (generation !== romLoadGeneration) return;
		releaseAllPhysicalHeldCodes();
		try {
			await closeBattery();
		} catch (error) {
			romLoaded = installedRom !== null;
			throw error;
		}
		if (generation !== romLoadGeneration) return;
		let candidateBattery: BatterySession | null = null;
		if (autosaveEnabled(model)) {
			try {
				let key: string;
				if (worker) {
					const copy = originalBytes.slice();
					key = await workerCall(model === 'pv-f1' ? 'pv_battery_identity' : 'oz_battery_identity', { bytes: copy }, [
						copy.buffer,
					]);
				} else {
					await ensureEmulator();
					key =
						model === 'pv-f1'
							? pvBatteryIdentity(() => new wasm.Sc62015Emulator(), originalBytes)
							: ozBatteryIdentity(() => new wasm.Sc62015Emulator(), originalBytes);
				}
				candidateBattery = await BatterySession.open(key);
			} catch (error) {
				batteryError = `Saved records were preserved: ${String(error)}`;
				throw error;
			}
		}
		if (generation !== romLoadGeneration) {
			await candidateBattery?.close();
			return;
		}
		const retained = candidateBattery?.loaded?.image.slice() ?? new Uint8Array();
		let media: PvSession = {
			retained: pvRetained,
			cardRom: pvCardRom.slice(),
			cardSram: pvCardSram.slice(),
			ramCard: pvRamCardRequested ? pvRamCard.slice() : new Uint8Array(),
			ramCardWritable: pvRamCardWritable,
			handwritingModel,
		};
		resetSymbols();
		try {
			if (model === 'pv-f1') {
				pvRecoveryVersions = candidateBattery?.versions() ?? [];
				const recover = pvRecoveryOverride ?? (!pvRetained.length ? candidateBattery?.loaded?.image : null);
				pvRecoveryOverride = null;
				if (recover) media = await decodePvSession(recover, candidateBattery?.key);
				handwritingModel = Uint8Array.from(media.handwritingModel);
			}

			if (worker) {
				await workerCall(
					'load_rom',
					{
						bytes,
						romSource: source,
						model,
						generation,
						...(model === 'oz-9600' ? { ozProfile, retained } : {}),
						...(model === 'pv-f1'
							? {
									pvHandwritingModel: handwritingModel.slice(),
									pvRetained: media.retained,
									pvCardRom: media.cardRom,
									pvCardSram: media.cardSram,
									pvRamCard: media.ramCard,
									pvRamCardWritable: media.ramCardWritable,
								}
							: {}),
						...(model === 'pa-9500'
							? { paProfile, paUart, paCardRom: paCardRom.slice(), paCardSram: paCardSram.slice(), paRetained }
							: {}),
					},
					[bytes.buffer],
				);
			} else {
				const emu = await ensureEmulator();
				if (generation !== romLoadGeneration) {
					await candidateBattery?.close();
					return;
				}
				if (model === 'oz-9600') emu.load_oz9600(bytes, retained, ozProfile);
				else if (model === 'pa-9500') loadPa9500(emu, bytes, paProfile, paCardRom, paCardSram, paUart, paRetained);
				else if (model === 'pv-f1' && media.ramCard.length)
					emu.load_pvf1_with_ram_card(bytes, media.retained, handwritingModel, media.ramCard, media.ramCardWritable);
				else if (model === 'pv-f1')
					emu.load_pvf1(bytes, media.retained, handwritingModel, media.cardRom, media.cardSram);
				else if (typeof emu.load_rom_with_model === 'function') emu.load_rom_with_model(bytes, model);
				else emu.load_rom(bytes);
				fallbackInputs.clear();
				resetFallbackSerial();
				refreshAllNow();
			}
		} catch (error) {
			await candidateBattery?.close();
			if (['oz-9600', 'pv-f1'].includes(model)) batteryError = `Saved records were preserved: ${String(error)}`;
			throw error;
		}
		if (generation !== romLoadGeneration) {
			await candidateBattery?.close();
			return;
		}
		if (model === 'pv-f1') {
			if (!worker) fallbackReadiness.reset();
			pvHandwritingModel = handwritingModel;
			pvActiveAssets = {
				cardRom: media.cardRom.slice(),
				handwritingModel: handwritingModel.slice(),
				ramCardWritable: media.ramCardWritable,
			};
			pvCardRom = Uint8Array.from(media.cardRom);
			pvCardSram = Uint8Array.from(media.cardSram);
			pvRamCard = Uint8Array.from(media.ramCard);
			pvRamCardRequested = !!media.ramCard.length;
			pvRamCardWritable = media.ramCardWritable;
			if (configuredModel) pvHandwritingName = configuredModel.name;
			if (candidateBattery?.loaded && !pvRetained.length) {
				pvCardNames = {
					rom: media.cardRom.length ? 'Recovered program card' : '',
					sram: media.cardSram.length ? 'Recovered SRAM' : '',
				};
				pvRamCardName = media.ramCard.length ? 'Recovered memory card' : '';
				pvHandwritingName = handwritingModel.length ? 'Recovered recognition model' : '';
			}
		}
		batterySession = candidateBattery;
		batteryGeneration = generation;
		if (['oz-9600', 'pv-f1'].includes(model)) {
			batteryError = '';
			batteryStatus = candidateBattery?.loaded
				? 'Recovered saved records; paused'
				: candidateBattery
					? 'Automatic saving is on'
					: 'Automatic saving is off; export backups to keep records';
		}
		if (model === 'oz-9600') ozLoadedProfile = ozProfile;
		if (model === 'pa-9500') {
			paLoadedProfile = paProfile;
			paLoadedUart = paUart;
		}
		ozReplayReport = null;
		ozReplayStatus = '';
		fallbackTablet.discard();
		romSource = source;
		installedRom = { bytes: originalBytes, model, source };
		romLoaded = true;
		await pushSoundPreference();
		lastError = null;
		if (callStackOpen) void ensureSymbols();
		await saveBattery(true);
	}

	function refreshFast() {
		if (!worker) observeFallbackReadiness();
		pumpFallbackSerial();
		if (worker) return;
		if (!emulator) return;
		updateDeliveredContacts(emulator.input_contacts());
		powerState = emulator.power_state();
		typingStatus = fallbackInputs.typingStatus();
		pasteStatus = fallbackInputs.pasteStatus();
		pacingStatus = emulator.pacing_status?.() ?? null;
		try {
			const rawGeometry = emulator.lcd_capture_if_changed(false);
			const geometry = rawGeometry && romModel === 'pi-3000' ? piPanelCapture(rawGeometry) : rawGeometry;
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
		if (lcdChipsOpen && !['oz-9600', 'pi-3000', 'pv-f1'].includes(romModel)) lcdChipPixels = emulator.lcd_chip_pixels();
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
			const fifoAddresses = ['oz-9600', 'pi-3000', 'pv-f1'].includes(romModel)
				? null
				: resolvePce500KeyboardFifo((address) => emulator.read_u8?.(address));
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
			const fifoAddresses = ['oz-9600', 'pi-3000', 'pv-f1'].includes(romModel)
				? null
				: resolvePce500KeyboardFifo((address) => emulator.read_u8?.(address));
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
				limitBudget: fallbackHostInputs.limitBudget,
				onProgress: (used) => {
					applyVirtualReleaseBudget(used);
					pumpFallbackSerial();
				},
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
		return automaticHostSlice(emulator, count, fallbackHostInputs, typingCatchUp);
	}

	function pumpEmulator(id: number) {
		if (!running || !emulator || id !== runLoopId) return;
		let waitMs: number;
		try {
			waitMs = stepCore(RUN_SLICE_MAX_INSTRUCTIONS);
			pumpFallbackSerial();
		} catch (err) {
			lastError = String(err);
			running = false;
			syncFallbackAudio();
			return;
		}
		try {
			if (fallbackSoundActive) fallbackAudio.pump(() => emulator.take_audio());
		} catch (error) {
			soundError = String(error);
			soundEnabled = false;
			audioPlayback.disable();
			syncFallbackAudio();
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

	async function onSelectPvHandwriting(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !romLoaded || loadingRom || pvHandwritingBusy) return;
		const generation = romLoadGeneration;
		pvHandwritingBusy = true;
		pvHandwritingError = '';
		try {
			if (!(await stop())) throw new Error('Pause was not acknowledged; model was not changed');
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (generation !== romLoadGeneration || romModel !== 'pv-f1') return;
			if (!bytes.length) throw new Error('Empty handwriting model');
			if (worker) {
				const copy = bytes.slice();
				await workerCall('pv_recognizer', { bytes: copy, generation }, [copy.buffer]);
			} else emulator.configure_pvf1_recognizer(bytes);
			if (generation !== romLoadGeneration) return;
			pvHandwritingModel = bytes;
			pvActiveAssets.handwritingModel = bytes.slice();
			pvHandwritingName = file.name;
		} catch (error) {
			if (generation === romLoadGeneration) pvHandwritingError = String(error);
		} finally {
			pvHandwritingBusy = false;
		}
	}

	async function onSelectRom(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		input.value = '';
		if (
			installedRom &&
			!window.confirm(
				installedRom.model === 'oz-9600'
					? batterySession
						? 'Replace the current ROM? OZ-9600 RAM/RTC will be saved before replacement; execution restarts when reloaded.'
						: 'Replace the current ROM and discard unsaved records? Use Export backup first to keep them.'
					: 'Replace the current ROM and discard device RAM changes?',
			)
		) {
			input.value = '';
			return;
		}
		const previousGeneration = romLoadGeneration;
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
			if (generation === romLoadGeneration) {
				lastError = String(err);
				if (installedRom?.model === model) {
					romLoadGeneration = previousGeneration;
					romLoaded = true;
					loadingRom = false;
					releaseAllPhysicalHeldCodes();
					releaseInputSource('virtual');
				}
			}
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
			!window.confirm(
				installedRom.model === 'oz-9600'
					? batterySession
						? 'Change device model? OZ-9600 RAM/RTC will be saved before replacement; execution restarts when reloaded.'
						: 'Change device model and discard unsaved records? Use Export backup first to keep them.'
					: 'Change device model and discard device RAM changes?',
			)
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

	async function selectPaCard(event: Event, kind: 'rom' | 'sram') {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (!file) return;
		try {
			if (file.size !== (kind === 'rom' ? 0x20000 : 0x8000))
				throw new Error(`PA card ${kind} must be ${kind === 'rom' ? '128' : '32'} KiB`);
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (kind === 'rom') paCardRom = bytes;
			else paCardSram = bytes;
			paCardNames = { ...paCardNames, [kind]: file.name };
			paCardError = '';
		} catch (error) {
			paCardError = String(error);
		}
	}

	async function reloadPaCard(remove = false) {
		if (!installedRom || installedRom.model !== 'pa-9500' || loadingRom) return;
		if (remove) {
			paCardRom = new Uint8Array();
			paCardSram = new Uint8Array();
			paCardNames = { rom: '', sram: '' };
		}
		loadingRom = true;
		const previousGeneration = romLoadGeneration;
		const previousInput = paInput;
		const generation = ++romLoadGeneration;
		paInput = 'native';
		try {
			await installRom(installedRom.bytes.slice(), 'pa-9500', installedRom.source, generation);
			paCardError = '';
		} catch (error) {
			paCardError = String(error);
			if (romLoadGeneration === generation) {
				romLoadGeneration = previousGeneration;
				paInput = previousInput;
				releaseAllPhysicalHeldCodes();
				releaseInputSource('virtual');
			}
		} finally {
			loadingRom = false;
		}
	}

	async function exportPaObservation() {
		if (!romLoaded || romModel !== 'pa-9500' || !(await stop())) return;
		try {
			const observation = worker
				? await workerCall('pa_observe', { generation: romLoadGeneration })
				: {
						status: emulator.pa9500_status(),
						capture: emulator.lcd_capture(1),
						card_sram: Array.from(emulator.pa9500_card_sram()),
					};
			observation.capture.pixels = Array.from(observation.capture.pixels);
			downloadBlob(new Blob([safeJson(observation)], { type: 'application/json' }), 'pa9500-observations.json');
		} catch (error) {
			paCardError = String(error);
		}
	}

	async function exportPaRetained() {
		if (!romLoaded || romModel !== 'pa-9500' || !(await stop())) return;
		try {
			const bytes = worker
				? await workerCall('pa_retained_export', { generation: romLoadGeneration })
				: emulator.pa9500_retained_state();
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pa9500-ram-rtc.bin');
		} catch (error) {
			paCardError = String(error);
		}
	}

	async function importPaRetained(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (!file || !installedRom || installedRom.model !== 'pa-9500' || loadingRom || !(await stop())) return;
		const oldGeneration = romLoadGeneration,
			oldInput = paInput,
			oldCard = paCardSram;
		loadingRom = true;
		const generation = ++romLoadGeneration;
		try {
			if (file.size !== 104 + 0x40000 + 32) throw new Error('PA RAM/RTC backing has an invalid length');
			const bytes = new Uint8Array(await file.arrayBuffer());
			paCardSram = Uint8Array.from(
				worker
					? (await workerCall('pa_observe', { generation: oldGeneration })).card_sram
					: emulator.pa9500_card_sram(),
			);
			await installRom(installedRom.bytes.slice(), 'pa-9500', installedRom.source, generation, bytes);
			paInput = 'native';
			paCardError = '';
		} catch (error) {
			paCardError = String(error);
			if (romLoadGeneration === generation) {
				romLoadGeneration = oldGeneration;
				paInput = oldInput;
				paCardSram = oldCard;
			}
		} finally {
			loadingRom = false;
		}
	}

	async function exportPaCard() {
		if (!romLoaded || romModel !== 'pa-9500' || !(await stop())) return;
		try {
			const bytes: Uint8Array = worker
				? await workerCall('pa_card_export', { generation: romLoadGeneration })
				: emulator.pa9500_card_sram();
			if (!bytes.length) throw new Error('No PA card is loaded');
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pa9c3-sram.bin');
		} catch (error) {
			paCardError = String(error);
		}
	}

	async function selectPvCard(event: Event, kind: 'rom' | 'sram') {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file) return;
		try {
			if (file.size !== (kind === 'rom' ? 0x20000 : 0x8000))
				throw new Error(kind === 'rom' ? 'Card ROM must be 128 KiB' : 'Card SRAM must be 32 KiB');
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (kind === 'rom') pvCardRom = bytes;
			else pvCardSram = bytes;
			pvCardNames = { ...pvCardNames, [kind]: file.name };
			pvRamCardRequested = false;
			pvCardError = '';
		} catch (error) {
			pvCardError = String(error);
		}
	}
	async function exportPvCard() {
		if (!romLoaded || romModel !== 'pv-f1' || !(await stop())) return;
		try {
			const bytes = worker
				? await workerCall('pv_card_export', { generation: romLoadGeneration })
				: emulator.pvf1_card_sram();
			if (!bytes.length) throw new Error('No PV-F1 card loaded');
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pa9c3-sram.bin');
		} catch (error) {
			pvCardError = String(error);
		}
	}
	async function reloadPvCard(remove = false) {
		if (!installedRom || installedRom.model !== 'pv-f1' || loadingRom || !(await stop())) return;
		const generation = romLoadGeneration;
		let next = generation;
		const oldRom = pvCardRom,
			oldSram = pvCardSram,
			oldNames = pvCardNames,
			oldRamRequested = pvRamCardRequested;
		pvCardStatus = '';
		loadingRom = true;
		try {
			const retained = Uint8Array.from(
				worker ? await workerCall('pv_retained_export', { generation }) : emulator.export_pvf1_retained(),
			);
			pvRamCardRequested = false;
			if (remove) {
				pvCardRom = new Uint8Array();
				pvCardSram = new Uint8Array();
				pvCardNames = { rom: '', sram: '' };
			}
			next = ++romLoadGeneration;
			await installRom(installedRom.bytes.slice(), 'pv-f1', installedRom.source, next, new Uint8Array(), retained);
			pvCardError = '';
			pvCardStatus = remove ? 'Card removed; organizer paused' : 'Program card imported; organizer paused';
		} catch (error) {
			if (romLoadGeneration === next) {
				pvCardError = String(error);
				romLoadGeneration = generation;
				pvCardRom = oldRom;
				pvCardSram = oldSram;
				pvCardNames = oldNames;
				pvRamCardRequested = oldRamRequested;
			}
		} finally {
			if (romLoadGeneration === generation || romLoadGeneration === next) loadingRom = false;
		}
	}
	async function selectPvRamCard(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file) return;
		try {
			if (file.size !== 0x20000) throw new Error('Memory card image must be 128 KiB');
			pvRamCard = new Uint8Array(await file.arrayBuffer());
			pvRamCardName = file.name;
			pvRamCardRequested = true;
			pvCardRom = new Uint8Array();
			pvCardSram = new Uint8Array();
			pvCardNames = { rom: '', sram: '' };
			pvCardError = '';
		} catch (error) {
			pvCardError = String(error);
		}
	}
	async function exportPvRamCard() {
		if (!romLoaded || romModel !== 'pv-f1' || !(await stop())) return;
		try {
			const bytes = Uint8Array.from(
				worker
					? await workerCall('pv_ram_card_export', { generation: romLoadGeneration })
					: emulator.pvf1_ram_card_image(),
			);
			if (!bytes.length) throw new Error('No memory card is mounted');
			downloadBlob(new Blob([bytes], { type: 'application/octet-stream' }), 'pv-f1-memory-card.bin');
			pvCardError = '';
		} catch (error) {
			pvCardError = String(error);
		}
	}
	async function reloadPvRamCard(remove = false) {
		if (!installedRom || installedRom.model !== 'pv-f1' || loadingRom || !(await stop())) return;
		const generation = romLoadGeneration;
		const previous = {
			rom: pvCardRom,
			sram: pvCardSram,
			names: pvCardNames,
			ram: pvRamCard,
			requested: pvRamCardRequested,
		};
		let next = generation;
		pvCardStatus = '';
		loadingRom = true;
		try {
			const retained = Uint8Array.from(
				worker ? await workerCall('pv_retained_export', { generation }) : emulator.export_pvf1_retained(),
			);
			if (remove) {
				const current = Uint8Array.from(
					worker ? await workerCall('pv_ram_card_export', { generation }) : emulator.pvf1_ram_card_image(),
				);
				if (current.length) pvRamCard = current;
			}
			pvRamCardRequested = !remove;
			pvCardRom = new Uint8Array();
			pvCardSram = new Uint8Array();
			pvCardNames = { rom: '', sram: '' };
			next = ++romLoadGeneration;
			await installRom(installedRom.bytes.slice(), 'pv-f1', installedRom.source, next, new Uint8Array(), retained);
			pvCardError = '';
			pvCardStatus = remove ? 'Memory card removed; organizer paused' : 'Memory card imported; organizer paused';
		} catch (error) {
			if (romLoadGeneration === next) {
				romLoadGeneration = generation;
				pvCardRom = previous.rom;
				pvCardSram = previous.sram;
				pvCardNames = previous.names;
				pvRamCard = previous.ram;
				pvRamCardRequested = previous.requested;
				pvCardError = String(error);
			}
		} finally {
			if (romLoadGeneration === generation || romLoadGeneration === next) loadingRom = false;
		}
	}
	async function resetSession() {
		if (!installedRom || loadingRom) return;
		if (installedRom.model === 'pv-f1' && workerHealth === 'ready') {
			if (!(await stop())) return;
			const generation = romLoadGeneration;
			loadingRom = true;
			releaseAllPhysicalHeldCodes();
			releaseInputSource('virtual');
			romLoadGeneration = generation + 1;
			try {
				const model = pvHandwritingModel.slice();
				if (worker) await workerCall('pv_reset', { model, generation, nextGeneration: generation + 1 }, [model.buffer]);
				else {
					emulator.reset_pvf1(model);
					fallbackReadiness.reset();
					fallbackInputs.clear();
					fallbackTablet.discard();
				}
				if (!worker) resetFallbackSerial();
				refreshAllNow();
				batteryGeneration = romLoadGeneration;
				await saveBattery(true);
				lastError = null;
			} catch (error) {
				if (romLoadGeneration === generation + 1) {
					romLoadGeneration = generation;
					lastError = String(error);
				}
			} finally {
				if (romLoadGeneration === generation || romLoadGeneration === generation + 1) loadingRom = false;
			}
			return;
		}
		if (installedRom.model === 'pa-9500' && workerHealth === 'ready') {
			if (!(await stop())) return;
			try {
				releaseAllPhysicalHeldCodes();
				releaseInputSource('virtual');
				if (worker) await workerCall('pa_reset');
				else {
					fallbackInputs.clear();
					resetPa9500(emulator);
					resetFallbackSerial();
					refreshAllNow();
				}
				lastError = null;
				paInput = 'native';
			} catch (error) {
				lastError = `PA reset failed: ${String(error)}`;
			}
			return;
		}
		if (installedRom.model === 'oz-9600' && workerHealth === 'ready') {
			if (!(await stop())) return;
			try {
				releaseAllPhysicalHeldCodes();
				releaseInputSource('virtual');
				if (worker) await workerCall('oz_reset');
				else {
					emulator.reset();
					fallbackTablet.discard();
					resetFallbackSerial();
					refreshAllNow();
				}
				lastError = null;
			} catch (error) {
				lastError = `OZ reset failed: ${String(error)}`;
			}
			return;
		}

		if (
			!window.confirm(
				autosaveEnabled(installedRom.model)
					? 'Recover from the last automatic save? Changes since that save will be lost.'
					: 'Reset the machine and discard all session changes? This reloads the last successfully loaded ROM, not a saved session.',
			)
		)
			return;
		const savedRom = installedRom;
		if (workerHealth === 'ready' && !(await stop())) return;
		if (!worker && (stepBusy || functionRunnerBusy)) {
			lastError = 'Wait for the cancelled operation to finish before resetting.';
			return;
		}
		// The user accepted recovery from the last durable image. A failed worker
		// cannot supply a final snapshot; release ownership before replacing it.
		const battery = batterySession;
		batterySession = null;
		await battery?.close();
		const generation = ++romLoadGeneration;
		workerRequests?.fail(new Error('Machine reset by user'));
		sessionRecording = false;
		sessionProgress = null;
		sessionBusy = false;
		sessionStatus = '';
		sessionError = '';
		audioPlayback.pause();
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

	async function exportOzRetained() {
		if (!romLoaded || !(await stop())) return;
		try {
			const bytes: Uint8Array = worker ? await workerCall('oz_retained_export') : emulator.export_oz9600_retained();
			downloadBlob(new Blob([bytes as Uint8Array<ArrayBuffer>], { type: 'application/octet-stream' }), 'oz-9600.ozbat');
		} catch (error) {
			lastError = String(error);
		}
	}
	async function recoverPvVersion(version: BatteryVersion) {
		if (loadingRom || batteryBusy) return;
		pvRecoveryOverride = version.image.slice();
		try {
			if (await retryRequestedRom()) pvBackupStatus = 'Recovered selected automatic save; paused';
		} finally {
			pvRecoveryOverride = null;
		}
	}
	async function exportPvBackup() {
		if (!romLoaded || romModel !== 'pv-f1' || !(await stop())) return;
		try {
			const bytes = worker
				? await workerCall('pv_retained_export', { generation: romLoadGeneration })
				: emulator.export_pvf1_retained();
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pv-f1.pvram');
			pvBackupError = '';
			pvBackupStatus = 'Backup saved';
		} catch (error) {
			pvBackupError = String(error);
		}
	}
	function applyPvSessionAssets(a: PvSession, label: string) {
		pvActiveAssets = {
			cardRom: a.cardRom.slice(),
			handwritingModel: a.handwritingModel.slice(),
			ramCardWritable: a.ramCardWritable,
		};
		pvCardRom = a.cardRom.slice();
		pvCardSram = a.cardSram.slice();
		pvRamCard = a.ramCard.slice();
		pvRamCardRequested = !!a.ramCard.length;
		pvRamCardWritable = a.ramCardWritable;
		pvHandwritingModel = a.handwritingModel.slice();
		pvCardNames = {
			rom: a.cardRom.length ? `${label} program card` : '',
			sram: a.cardSram.length ? `${label} SRAM` : '',
		};
		pvRamCardName = a.ramCard.length ? `${label} memory card` : '';
		pvHandwritingName = a.handwritingModel.length ? `${label} recognition model` : '';
	}
	async function exportPvFullBackup() {
		if (!romLoaded || romModel !== 'pv-f1' || loadingRom || !(await stop())) return;
		const generation = romLoadGeneration;
		loadingRom = true;
		try {
			const media: PvSession = worker
				? await workerCall('pv_session_export', { generation })
				: capturePvSession(emulator, pvActiveAssets);
			const bytes = await encodePvSession(media);
			if (generation !== romLoadGeneration) return;
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pv-f1.pvbackup');
			pvBackupError = '';
			pvBackupStatus = 'Full backup saved';
		} catch (error) {
			if (generation === romLoadGeneration) pvBackupError = String(error);
		} finally {
			if (generation === romLoadGeneration) loadingRom = false;
		}
	}
	async function importPvFullBackup(event: Event) {
		const input = event.currentTarget as HTMLInputElement,
			file = input.files?.[0];
		input.value = '';
		if (
			!file ||
			!installedRom ||
			!romLoaded ||
			romModel !== 'pv-f1' ||
			loadingRom ||
			sessionRecording ||
			!(await stop())
		)
			return;
		const generation = romLoadGeneration;
		loadingRom = true;
		try {
			if (file.size > 64 * 1024 * 1024 + 4096) throw new Error('Full backup exceeds size limit');
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (generation !== romLoadGeneration) return;
			let media: PvSession;
			if (worker)
				media = await workerCall('pv_session_restore', { bytes, generation, nextGeneration: generation + 1 }, [
					bytes.buffer,
				]);
			else {
				media = await decodePvSession(bytes, pvRetainedIdentity(emulator.export_pvf1_retained()));
				if (generation !== romLoadGeneration) return;
				loadPvSession(emulator, installedRom.bytes, media);
				fallbackReadiness.reset();
				emulator.set_execution_mode(executionMode);
				fallbackInputs.clear();
				fallbackTablet.discard();
				resetFallbackSerial();
			}
			if (generation !== romLoadGeneration) return;
			romLoadGeneration = generation + 1;
			batteryGeneration = romLoadGeneration;
			applyPvSessionAssets(media, 'Backup');
			releaseAllPhysicalHeldCodes();
			releaseInputSource('virtual');
			await saveBattery(true);
			refreshAllNow();
			pvBackupError = '';
			pvBackupStatus = `Full backup restored from ${file.name}; Resume to boot from saved records`;
		} catch (error) {
			if (generation === romLoadGeneration) pvBackupError = String(error);
		} finally {
			if (romLoadGeneration === generation || romLoadGeneration === generation + 1) loadingRom = false;
		}
	}
	async function beginSessionRecording() {
		if (!worker || sessionBusy || sessionRecording || !romLoaded || !(await stop())) return;
		sessionBusy = true;
		loadingRom = true;
		sessionError = '';
		releaseAllPhysicalHeldCodes();
		releaseInputSource('virtual');
		try {
			sessionProgress = await workerCall('record_start', { generation: romLoadGeneration });
			sessionRecording = true;
			sessionStatus = 'Recording from a fresh restart; Resume or Step to use the organizer';
		} catch (error) {
			sessionError = String(error);
		} finally {
			sessionBusy = false;
			loadingRom = false;
		}
	}
	async function checkpointSession() {
		if (!worker || sessionBusy || !sessionRecording) return;
		sessionBusy = true;
		try {
			sessionProgress = await workerCall('record_checkpoint', { generation: romLoadGeneration });
			sessionStatus = 'Screen checkpoint added';
		} catch (error) {
			sessionError = String(error);
		} finally {
			sessionBusy = false;
		}
	}
	async function finishSessionRecording() {
		if (!worker || sessionBusy || !sessionRecording || !(await stop())) return;
		sessionBusy = true;
		loadingRom = true;
		releaseAllPhysicalHeldCodes();
		releaseInputSource('virtual');
		try {
			const result = await workerCall('record_end', { generation: romLoadGeneration });
			downloadBlob(
				new Blob([Uint8Array.from(result.bytes)], { type: 'application/octet-stream' }),
				`${romModel}.orgsession`,
			);
			sessionRecording = false;
			sessionProgress = null;
			sessionStatus = `Recording exported: ${result.events} events, ${result.boundaries} boundaries`;
		} catch (error) {
			sessionError = String(error);
		} finally {
			sessionBusy = false;
			loadingRom = false;
		}
	}
	async function discardSessionRecording() {
		if (!worker || sessionBusy || !sessionRecording) return;
		try {
			await workerCall('record_discard', { generation: romLoadGeneration });
			sessionRecording = false;
			sessionProgress = null;
			sessionError = '';
			sessionStatus = 'Recording discarded';
		} catch (error) {
			sessionError = String(error);
		}
	}
	async function replaySessionFile(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !worker || sessionBusy || sessionRecording || !romLoaded || !(await stop())) return;
		sessionBusy = true;
		functionRunnerBusy = true;
		sessionError = '';
		sessionStatus = 'Checking and replaying session…';
		releaseAllPhysicalHeldCodes();
		releaseInputSource('virtual');
		try {
			if (file.size > 80 * 1024 * 1024) throw new Error('Recording file exceeds size limit');
			const bytes = new Uint8Array(await file.arrayBuffer());
			const result = await workerCall('session_replay', { bytes, generation: romLoadGeneration }, [bytes.buffer]);
			const context = result.context;
			executionMode = context.settings.executionMode;
			typingCatchUp = context.settings.typingCatchUp;
			targetFps = context.settings.targetFps;
			soundEnabled = context.settings.sound;
			if (result.pvAssets) {
				applyPvSessionAssets(result.pvAssets, 'Recorded');
			} else {
				ozLoadedProfile = context.profile;
				ozProfile = context.profile;
			}
			sessionStatus =
				result.status === 'verified'
					? `Replay verified: ${result.checkpoints} screen checkpoints, ${result.boundaries} boundaries`
					: `Replay ${result.status}; organizer paused`;
			if (result.error) sessionError = result.error;
			await saveBattery(true);
		} catch (error) {
			sessionError = String(error);
			sessionStatus = 'Replay did not complete';
		} finally {
			sessionBusy = false;
			functionRunnerBusy = false;
			running = false;
		}
	}
	async function importPvBackup(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !romLoaded || romModel !== 'pv-f1' || loadingRom || !(await stop())) return;
		const generation = romLoadGeneration;
		loadingRom = true;
		try {
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (generation !== romLoadGeneration) return;
			if (worker) {
				const model = pvHandwritingModel.slice();
				await workerCall('pv_retained_restore', { bytes, model, generation, nextGeneration: generation + 1 }, [
					bytes.buffer,
					model.buffer,
				]);
			} else {
				emulator.restore_pvf1_retained(bytes, pvHandwritingModel);
				fallbackReadiness.reset();
				fallbackInputs.clear();
				fallbackTablet.discard();
			}
			if (generation !== romLoadGeneration) return;
			romLoadGeneration = generation + 1;
			batteryGeneration = romLoadGeneration;
			await saveBattery(true);
			pvBackupError = '';
			pvBackupStatus = `Restored ${file.name}; Resume to boot from saved records`;
			releaseAllPhysicalHeldCodes();
			refreshAllNow();
		} catch (error) {
			if (generation === romLoadGeneration) pvBackupError = String(error);
		} finally {
			loadingRom = false;
		}
	}

	async function exportPiBackup() {
		if (!romLoaded || romModel !== 'pi-3000' || !(await stop())) return;
		try {
			const bytes = worker
				? await workerCall('pi_retained_export', { generation: romLoadGeneration })
				: emulator.export_pi3000_retained();
			downloadBlob(new Blob([Uint8Array.from(bytes)], { type: 'application/octet-stream' }), 'pi-3000.piram');
			piBackupError = '';
			piBackupStatus = 'Backup saved';
		} catch (error) {
			piBackupError = String(error);
		}
	}
	async function importPiBackup(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !romLoaded || romModel !== 'pi-3000' || loadingRom || !(await stop())) return;
		loadingRom = true;
		try {
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (worker) await workerCall('pi_retained_restore', { bytes }, [bytes.buffer]);
			else {
				emulator.restore_pi3000_retained(bytes);
				fallbackInputs.clear();
				fallbackTablet.discard();
				resetFallbackSerial();
				refreshAllNow();
			}
			deliveredContacts = new Set();
			physicalHeldCodes.clear();
			physicalHighlights = new Set();
			piBackupError = '';
			piBackupStatus = `Restored ${file.name}; complete touch calibration to continue`;
			lastError = null;
		} catch (error) {
			piBackupError = String(error);
		} finally {
			loadingRom = false;
		}
	}
	async function importOzRetained(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !romLoaded || !(await stop())) return;
		try {
			const bytes = new Uint8Array(await file.arrayBuffer());
			if (worker) await workerCall('oz_retained_restore', { bytes }, [bytes.buffer]);
			else {
				emulator.restore_oz9600_retained(bytes);
				fallbackInputs.clear();
				fallbackTablet.discard();
				resetFallbackSerial();
				refreshAllNow();
			}
			lastError = null;
			ozReplayStatus = `Restored ${file.name}; fresh CPU`;
			await saveBattery(true);
		} catch (error) {
			lastError = String(error);
		}
	}
	async function replayOz(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || !romLoaded || !(await stop())) return;
		functionRunnerBusy = true;
		ozReplayReport = null;
		ozReplayStatus = 'Validating physical replay…';
		try {
			const document = await file.text();
			if (worker) ozReplayReport = await workerCall('oz_replay', { document });
			else {
				emulator.validate_oz9600_replay(document);
				fallbackInputs.clear();
				fallbackTablet.clear();
				fallbackStepAbort = new AbortController();
				const reports: object[] = [];
				const boundaries = await runOzPhysicalReplay(emulator, document, {
					signal: fallbackStepAbort.signal,
					onProgress: pumpFallbackSerial,
					onStep: async (index, state) => {
						reports.push({ step: index, ...state });
						refreshAllNow();
					},
				});
				ozReplayReport = { boundaries, reports };
			}
			ozReplayStatus = `Completed ${ozReplayReport.reports.length} steps`;
			lastError = null;
		} catch (error) {
			lastError = String(error);
			ozReplayStatus = 'Replay stopped; partial state retained';
		} finally {
			functionRunnerBusy = false;
			fallbackStepAbort = null;
		}
	}
	function lcdTabletContact(contact: import('$lib/pvf1_panel').LcdTabletContact, owner: string, cancel: boolean) {
		if (!romLoaded || romModel !== 'pv-f1' || functionRunnerBusy) return;
		if (worker)
			void workerCall('pv_lcd', { contact, owner, cancel, generation: romLoadGeneration }).catch((error) => {
				lastError = String(error);
			});
		else
			try {
				const [raw_x, raw_y] = contact.pressed ? emulator.pvf1_lcd_to_raw(contact.pixel_x, contact.pixel_y) : [0, 0];
				fallbackTablet.set(owner, { raw_x, raw_y, pressed: contact.pressed }, cancel);
			} catch (error) {
				lastError = String(error);
			}
	}
	function tabletContact(contact: TabletContact, owner: string, cancel: boolean) {
		if (!romLoaded || !['oz-9600', 'pi-3000', 'pv-f1'].includes(romModel) || functionRunnerBusy) return;
		if (worker)
			void workerCall(romModel === 'pv-f1' ? 'pv_tablet' : romModel === 'pi-3000' ? 'pi_tablet' : 'oz_tablet', {
				contact,
				owner,
				cancel,
				generation: romLoadGeneration,
			}).catch((error) => {
				lastError = String(error);
			});
		else {
			try {
				fallbackTablet.set(owner, contact, cancel);
			} catch (error) {
				lastError = String(error);
			}
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
		if (['pv-f1', 'oz-9600'].includes(romModel) && document.hidden) {
			lifecycleMessage = 'Bring this app into view before resuming.';
			return;
		}
		lifecycleMessage = '';
		if (!romLoaded || running || controlPending || stepBusy || functionRunnerBusy || workerHealth !== 'ready') return;
		if (executionMode === 'deterministic') return;
		if (soundEnabled) {
			try {
				await audioPlayback.enable();
			} catch (error) {
				soundError = String(error);
				soundEnabled = false;
				audioPlayback.disable();
				await pushSoundPreference();
			}
		}
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
		syncFallbackAudio();
		runLoopId += 1;
		pumpEmulator(runLoopId);
		pumpRender(runLoopId);
	}

	async function stop(): Promise<boolean> {
		audioPlayback.pause();
		if (worker) {
			controlPending = 'stop';
			try {
				await workerCall('stop');
				running = false;
				workerHealth = 'ready';
				await saveBattery(true);
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
		syncFallbackAudio();
		await saveBattery(true);
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
			loadingRom ||
			(sessionBusy && functionRunnerBusy) ||
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
		if (romModel === 'pa-9500' && ['Delete', 'Insert'].includes(event.key)) {
			event.preventDefault();
			if (physicalKeyboardEnabled) void sendPaEditorKey(event.key === 'Delete' ? 'delete' : 'insert');
			return;
		}

		const contacts = contactsForKeyEvent(event, romModel, hostKeyboardMode, paInput);
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

	async function sendPaEditorKey(action: PaEditorAction) {
		const generation = romLoadGeneration;
		try {
			if (worker) {
				const status = await workerCall('pa_editor_key', { generation, action });
				if (generation !== romLoadGeneration) return;
				pasteStatus = status;
			} else {
				fallbackInputs.startPaste(paEditorSequence(action));
				pasteStatus = fallbackInputs.pasteStatus();
				refreshFast();
			}
			keyboardNotice = '';
		} catch (error) {
			if (generation === romLoadGeneration) keyboardNotice = String(error);
		}
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
	function onAudioVisibilityChange() {
		audioPlayback.pause();
		if (romLoaded)
			void pushSoundPreference().catch((error) => {
				soundError = String(error);
			});
	}
	function onFocusIn(event: FocusEvent) {
		if (isHostControl(event.target)) releaseAllPhysicalHeldCodes();
	}

	onMount(() => {
		mounted = true;
		batteryTimer = setInterval(() => {
			void saveBattery();
		}, 5000);
		document.addEventListener('visibilitychange', saveOnHidden);
		document.addEventListener('freeze', onPageFreeze);
		window.addEventListener('pagehide', onPageFreeze);
		document.addEventListener('visibilitychange', onAudioVisibilityChange);
		try {
			const requestedModel = normalizeRomModel(new URL(window.location.href).searchParams.get('model'));
			if (requestedModel) {
				$romModelStore = requestedModel;
				romModelWasPersisted = true;
			} else {
				romModelWasPersisted = normalizeRomModel(window.localStorage.getItem(ROM_MODEL_STORAGE_KEY)) !== null;
			}
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
		if (batteryTimer) clearInterval(batteryTimer);
		document.removeEventListener('visibilitychange', saveOnHidden);
		document.removeEventListener('freeze', onPageFreeze);
		window.removeEventListener('pagehide', onPageFreeze);
		const battery = batterySession;
		batterySession = null;
		void battery?.close();
		document.removeEventListener('visibilitychange', onAudioVisibilityChange);
		audioPlayback.close();
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
				disabled={!romLoaded || loadingRom || workerHealth !== 'ready' || functionRunnerBusy}
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
			{#if supportsAudio(romModel)}
				<button
					data-testid="sound-toggle"
					aria-pressed={soundEnabled}
					disabled={!romLoaded || loadingRom || soundBusy || !!controlPending}
					title="Organizer sound at normal speed; pitches and speaker response are approximate"
					on:click={toggleSound}>Sound {soundEnabled ? 'on' : 'off'}</button
				>
			{/if}
		</div>
	</header>
	{#if soundError}<p role="status">Sound unavailable: {soundError}</p>{/if}
	{#if ['pv-f1', 'oz-9600'].includes(romModel)}
		<p class="hint" data-testid="lifecycle-policy">
			Hiding or suspending this app pauses the emulator and its clock. Resume explicitly when you return; alarms do not
			fire while paused.
		</p>
		{#if lifecycleMessage}<p role="status" data-testid="lifecycle-status">{lifecycleMessage}</p>{/if}
	{/if}
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
			{#if romModel === 'pv-f1'}
				<p class="hint">
					F12 sends ON; Tab moves focus. Use the tablet sample controls below the display while Resume is active.
				</p>
			{:else if romModel === 'pi-3000'}
				<p class="hint">
					Tap the LCD targets while Resume is active. F12 sends ON. Confirmation and other key mappings remain under
					recovery.
				</p>
			{:else}
				<p class="hint">
					Click the device to type. F9 = device SHIFT · F10 = {romModel === 'pa-9500' ? 'input mode' : 'CAPS'} · {romModel ===
					'oz-9600'
						? 'F11 = 2nd · F12 = ON'
						: 'F12 = ON'}. Device CAPS controls letter case; host Shift selects mapped punctuation.
				</p>
				<p class="hint">
					{romModel === 'iq-7000'
						? 'F1–F8: Calendar, Schedule, TEL, MEMO, Calc, Card, World, Home. Enter stores; F11 or Shift+Enter inserts a newline. Page Up/Down searches.'
						: romModel === 'oz-9600'
							? 'F1: New Entry. F2: Edit. F11: 2nd. Escape: Cancel. Page Up/Down: Prev/Next. Printed mode labels use the tablet.'
							: romModel === 'pa-9500'
								? 'F1–F7: Calendar, Phonebook, Memo, Business, Handwrite, Calculator, Clock. F8: kana. F10: input mode. F11: BASIC Run/Pro. Tap F9 then N for Special or K for SECRET in native bindings.'
								: 'F1–F5: PF1–PF5. F6: BASIC. F7: MENU. F8: Clear. F11: device CTRL. Enter: ENTER.'}
				</p>
				<p class="hint">
					Letters, digits, Space, arrows, Backspace, Delete and Insert use device keys. Hover a key for its host
					binding. Depressed keys show delivered contacts; cyan outlines show host-held keys, not ROM acknowledgement.
					Browser shortcuts and text fields stay with the host.
				</p>
			{/if}
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
				: ''}{typingStatus.blocked ? ' — input buffer blocked' : ''}<button on:click={cancelQueuedInput}
				>Cancel queued keys</button
			>
		</div>
	{/if}

	{#if keyboardNotice}<p class="hint" role="status" data-testid="keyboard-notice">{keyboardNotice}</p>{/if}
	{#if romModel === 'pi-3000'}
		<section class="serial-panel" aria-label="Saved records" data-testid="pi-saved-records">
			<button
				data-testid="pi-backup-export"
				on:click={exportPiBackup}
				disabled={!romLoaded || loadingRom || functionRunnerBusy || stepBusy || !!controlPending}>Save backup</button
			>
			<label
				>Restore backup <input
					type="file"
					accept=".piram,.bin"
					data-testid="pi-backup-file"
					disabled={!romLoaded || loadingRom || functionRunnerBusy || stepBusy || !!controlPending}
					on:change={importPiBackup}
				/></label
			>
			<p class="hint">Restoring restarts the organizer and returns to touch calibration.</p>
			{#if piBackupStatus}<p role="status" data-testid="pi-backup-status">{piBackupStatus}</p>{/if}
			{#if piBackupError}<p class="error" role="alert" data-testid="pi-backup-error">{piBackupError}</p>{/if}
		</section>
	{/if}
	{#if romModel === 'pv-f1'}
		<section class="serial-panel" aria-label="Saved records" data-testid="pv-saved-records">
			<button
				data-testid="pv-full-backup-export"
				on:click={exportPvFullBackup}
				disabled={!romLoaded || loadingRom || functionRunnerBusy || stepBusy || !!controlPending}
				>Save full backup</button
			>
			<label
				>Restore full backup <input
					type="file"
					accept=".pvbackup"
					data-testid="pv-full-backup-file"
					on:change={importPvFullBackup}
					disabled={!romLoaded || loadingRom || sessionRecording || functionRunnerBusy || stepBusy || !!controlPending}
				/></label
			>
			<p class="hint">
				One file includes organizer records, clock, learned handwriting, recognition model and the mounted card. Restore
				restarts the organizer with the matching firmware.
			</p>
			<label
				><input
					type="checkbox"
					bind:checked={$pvAutosaveStore}
					on:change={changeAutosave}
					data-testid="pv-autosave"
					disabled={loadingRom || batteryBusy || !!controlPending}
				/> Automatically save to this browser</label
			>
			<p role="status" data-testid="pv-save-status">{batteryStatus}</p>
			{#if batteryError}<p class="error" role="alert" data-testid="pv-save-error">{batteryError}</p>{/if}
			<button
				on:click={() => saveBattery(true)}
				data-testid="pv-save-now"
				disabled={!romLoaded ||
					!batterySession ||
					!!batterySave ||
					loadingRom ||
					functionRunnerBusy ||
					stepBusy ||
					!!controlPending}>Save now</button
			>
			{#if !loadingRom && !batteryBusy && (!romLoaded || ($pvAutosaveStore && !batterySession))}<button
					on:click={retryRequestedRom}
					data-testid="pv-retry-load">Retry recovery</button
				>{/if}
			{#if pvRecoveryVersions.length > 1}<details data-testid="pv-save-history">
					<summary>Previous saves</summary>
					{#each pvRecoveryVersions as version, i}<button
							on:click={() => recoverPvVersion(version)}
							disabled={loadingRom || batteryBusy || !!controlPending}
							>Recover {i === 0 ? 'latest save' : `previous save ${i}`} ({new Date(
								version.savedAt,
							).toLocaleString()})</button
						>{/each}
				</details>{/if}
			<p class="hint">
				Automatic saves include organizer records, recognition configuration and the mounted card. Recovery restarts the
				organizer. Keep exported backups before clearing browser data.
			</p>

			<button
				data-testid="pv-backup-export"
				on:click={exportPvBackup}
				disabled={!romLoaded || loadingRom || functionRunnerBusy || stepBusy || !!controlPending}>Save backup</button
			>
			<label
				>Restore backup <input
					type="file"
					accept=".pvram,.bin"
					data-testid="pv-backup-file"
					disabled={!romLoaded || loadingRom || functionRunnerBusy || stepBusy || !!controlPending}
					on:change={importPvBackup}
				/></label
			>
			<p class="hint">
				Restoring starts a fresh paused organizer with saved RAM/RTC and the selected handwriting model.
			</p>
			{#if pvBackupStatus}<p role="status">{pvBackupStatus}</p>{/if}
			{#if pvBackupError}<p class="error" role="alert">{pvBackupError}</p>{/if}
		</section>
	{/if}
	{#if romModel === 'oz-9600'}
		<section class="serial-panel" aria-label="Saved records" data-testid="oz-saved-records">
			<label
				><input
					type="checkbox"
					bind:checked={$ozAutosaveStore}
					on:change={changeAutosave}
					data-testid="oz-autosave"
					disabled={loadingRom || batteryBusy || !!controlPending}
				/> Automatically save to this browser</label
			>
			<p role="status" data-testid="oz-save-status">{batteryStatus}</p>
			{#if batteryError}<p class="error" role="alert" data-testid="oz-save-error">{batteryError}</p>{/if}
			<button
				on:click={() => saveBattery(true)}
				data-testid="oz-save-now"
				disabled={!romLoaded || !batterySession || !!batterySave || functionRunnerBusy || stepBusy || !!controlPending}
				>Save now</button
			>
			{#if !loadingRom && !batteryBusy && (!romLoaded || ($ozAutosaveStore && !batterySession))}<button
					on:click={retryRequestedRom}
					data-testid="oz-retry-load">Retry recovery</button
				>{/if}
			<label
				>Import backup <input
					type="file"
					accept=".ozbat,.bin"
					data-testid="oz-retained-file"
					on:change={importOzRetained}
					disabled={!romLoaded || functionRunnerBusy}
				/></label
			>
			<button data-testid="oz-retained-export" on:click={exportOzRetained} disabled={!romLoaded || functionRunnerBusy}
				>Export backup</button
			>
			<p class="hint">
				Records recover with this firmware in the same browser profile. Keep an exported backup before clearing browser
				data or changing computers. Sudden closure can lose changes since the last save.
			</p>
		</section>
	{/if}
	{#if ['pv-f1', 'oz-9600'].includes(romModel)}
		<details class="serial-panel" data-testid="session-panel">
			<summary>Session recording and replay</summary>
			<p class="hint">
				Recording restarts with current saved backing and card configuration. Use the organizer normally, add screen
				checkpoints, then export. Replay restarts with the recorded backing and checks those screens. Finish releases
				held inputs. Pause stops a replay.
			</p>
			<button
				data-testid="session-record-start"
				on:click={beginSessionRecording}
				disabled={!worker ||
					!romLoaded ||
					sessionBusy ||
					sessionRecording ||
					loadingRom ||
					functionRunnerBusy ||
					stepBusy}>Restart and record</button
			>
			<button data-testid="session-checkpoint" on:click={checkpointSession} disabled={!sessionRecording || sessionBusy}
				>Add screen checkpoint</button
			>
			<button
				data-testid="session-record-end"
				on:click={finishSessionRecording}
				disabled={!sessionRecording || sessionBusy}>Finish and export recording</button
			>
			<button
				data-testid="session-record-discard"
				on:click={discardSessionRecording}
				disabled={!sessionRecording || sessionBusy}>Discard recording</button
			>
			<label
				>Replay recording <input
					type="file"
					accept=".orgsession"
					data-testid="session-replay-file"
					on:change={replaySessionFile}
					disabled={!worker || !romLoaded || sessionRecording || sessionBusy || loadingRom}
				/></label
			>
			<label
				>Paused step budget <input
					type="number"
					min="1"
					max="10000000"
					bind:value={sessionStepBudget}
					data-testid="session-step-budget"
				/></label
			>
			<button
				data-testid="session-step"
				on:click={() => stepOnce(sessionStepBudget)}
				disabled={!romLoaded ||
					running ||
					sessionBusy ||
					stepBusy ||
					functionRunnerBusy ||
					!Number.isInteger(sessionStepBudget) ||
					sessionStepBudget < 1 ||
					sessionStepBudget > 10000000}>Advance paused organizer</button
			>
			<p role="status" data-testid="session-status">{sessionStatus}</p>
			{#if sessionProgress}<p data-testid="session-progress">
					{sessionProgress.events} events · {sessionProgress.boundaries} boundaries
				</p>{/if}
			{#if sessionError || sessionProgress?.error}<p role="alert" data-testid="session-error">
					{sessionError || sessionProgress?.error}
				</p>{/if}
		</details>
	{/if}
	{#if supportsSerial(romModel)}
		<details class="serial-panel" data-testid="serial-panel">
			<summary>Serial</summary>
			<label
				>Send format <select data-testid="serial-format" bind:value={serialFormat}>
					<option value="text">Text (UTF-8)</option><option value="hex">Hex bytes</option>
				</select></label
			>
			<label
				>Line ending <select data-testid="serial-ending" bind:value={serialEnding}>
					<option value="none">None</option><option value="cr">CR</option>
					<option value="lf">LF</option><option value="crlf">CR + LF</option>
				</select></label
			>
			<label
				>Send to device <textarea data-testid="serial-input" bind:value={serialText} rows="3" maxlength="12288"
				></textarea></label
			>
			<button
				data-testid="serial-send"
				on:click={sendSerial}
				disabled={!romLoaded ||
					loadingRom ||
					serialBusy ||
					stepBusy ||
					functionRunnerBusy ||
					!!controlPending ||
					workerHealth !== 'ready'}>Send to device</button
			>
			{#if serialNotice}<p class="hint" role="status" data-testid="serial-notice">{serialNotice}</p>{/if}
			{#if serialError}<p class="error" role="alert" data-testid="serial-error">{serialError}</p>{/if}
			<label
				>From device <textarea data-testid="serial-output" readonly value={serialPreview} rows="4"></textarea></label
			>
			<p class="hint" data-testid="serial-count">
				{serialStatus.received} bytes received; {serialStatus.retained} bytes retained
			</p>
			{#if serialStatus.captureDropped || serialStatus.transportDropped}<p class="error" role="status">
					{serialStatus.captureDropped} bytes exceeded capture capacity; {serialStatus.transportDropped} bytes lost in transport
				</p>{/if}
			<button data-testid="serial-download" on:click={downloadSerial} disabled={!serialStatus.retained}
				>Save raw bytes</button
			>
			<button
				data-testid="serial-clear"
				on:click={() => {
					serialCapture.clear();
					serialRevision++;
				}}>Clear display</button
			>
			<p class="hint">
				{romModel === 'pa-9500'
					? 'Enable serial on the next PA load, then OPEN COM in BASIC.'
					: 'Open Terminal and choose Connect on the device.'} Send at most 4096 bytes at once. The preview shows the last
				2048 bytes; downloads retain up to 4 MiB. Device time is frozen while paused.
			</p>
		</details>
	{/if}
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
				disabled={!romLoaded || loadingRom || workerHealth !== 'ready' || functionRunnerBusy}
				onTablet={tabletContact}
				onLcdTablet={lcdTabletContact}
				{hostKeyboardMode}
				{paInput}
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
				<svelte:fragment slot="input-status"
					>{#if romModel === 'pv-f1' && romLoaded}<PvInputStatus
							phase={pvInputPhase}
							pen={penStatus}
							paused={!running}
							onCancel={cancelQueuedInput}
						/>{/if}</svelte:fragment
				>
			</DeviceShell>
		{/if}
	</div>
	<details class="advanced" bind:open={advancedOpen} data-testid="advanced-panel">
		<summary>Advanced</summary>
		<details data-testid="session-safety">
			<summary>Session safety & recovery</summary>
			<p class="hint">
				{#if romModel === 'oz-9600'}
					Automatic saving preserves OZ-9600 RAM/RTC in this browser when enabled. Check Saved records for the last
					completed save. Reloading the same ROM recovers that save into a fresh paused CPU; Reset preserves RAM/RTC.
					With automatic saving off, use Export backup before reload or replacement, then restore it with Import backup.
					Keep exported backups of records you want to retain. Saved RAM/RTC restores records and settings; device
					execution starts afresh.
				{:else}
					Device RAM is held in memory only. Reload or ROM/model replacement loses device RAM changes.
				{/if}
				Full running-session snapshots are not supported.
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
						<option value="oz-9600">OZ-9600 (experimental hardware)</option>
						<option value="pv-f1">PV-F1 (experimental hardware)</option>
						<option value="pi-3000">PI-3000 (experimental hardware)</option>
						<option value="pa-9500">PA-9500 / DB-Z (experimental hardware)</option>
					</select>
				</label>

				<label>
					Load ROM file:
					<input
						type="file"
						accept=".bin,.rom,.img,.ozrom,.parom,.pirom,.pvrom"
						data-testid="rom-file"
						on:change={onSelectRom}
					/>
				</label>
			</div>
		</header>

		{#if romModel === 'pv-f1'}
			<section aria-label="PV-F1 handwriting">
				<label
					>Handwriting model:
					<input
						type="file"
						accept=".model,.bin"
						data-testid="pv-handwriting-model"
						disabled={!romLoaded ||
							loadingRom ||
							pvHandwritingBusy ||
							stepBusy ||
							functionRunnerBusy ||
							Boolean(controlPending)}
						on:change={onSelectPvHandwriting}
					/>
				</label>
				<p class="hint">
					{pvHandwritingName
						? `Loaded ${pvHandwritingName}; retained for ROM reloads in this session.`
						: 'Load a handwriting model to recognize pen strokes.'} Loading pauses the emulator; Resume to continue.
				</p>
				{#if pvHandwritingError}<p role="alert">{pvHandwritingError}</p>{/if}
			</section>
			<section aria-label="PV-F1 card" data-testid="pv-card">
				{#if pvCardStatus}<p role="status" data-testid="pv-card-status">{pvCardStatus}</p>{/if}
				<label
					>Memory card image (128 KiB) <input
						type="file"
						accept=".bin"
						data-testid="pv-ram-card-image"
						disabled={loadingRom}
						on:change={selectPvRamCard}
					/></label
				>
				<label
					><input type="checkbox" bind:checked={pvRamCardWritable} disabled={loadingRom} /> Allow writes on load</label
				>
				<p class="hint">
					{pvRamCardName || 'No memory card image selected'}. Loading a card restarts the emulator and keeps organizer
					data. Card images are saved separately.
				</p>
				<button
					data-testid="pv-ram-card-load"
					disabled={!romLoaded || loadingRom || !pvRamCard.length}
					on:click={() => reloadPvRamCard()}>Load memory card</button
				>
				<button
					data-testid="pv-ram-card-save"
					disabled={!romLoaded || loadingRom || functionRunnerBusy}
					on:click={exportPvRamCard}>Save memory card</button
				>
				<button
					data-testid="pv-ram-card-remove"
					disabled={!romLoaded || loadingRom}
					on:click={() => reloadPvRamCard(true)}>Remove memory card</button
				>
				<label
					>PA-9C3 ROM <input
						type="file"
						accept=".bin"
						data-testid="pv-card-rom"
						disabled={loadingRom}
						on:change={(e) => selectPvCard(e, 'rom')}
					/></label
				>
				<label
					>PA-9C3 SRAM <input
						type="file"
						accept=".bin"
						data-testid="pv-card-sram"
						disabled={loadingRom}
						on:change={(e) => selectPvCard(e, 'sram')}
					/></label
				>
				<p class="hint">
					Selected for next load: {pvCardNames.rom || 'no ROM'} / {pvCardNames.sram || 'no SRAM'}. Card SRAM is saved
					separately from organizer backups.
				</p>
				<button
					data-testid="pv-card-load"
					disabled={!romLoaded || loadingRom || !pvCardRom.length || !pvCardSram.length}
					on:click={() => reloadPvCard()}>Import card and restart</button
				>
				<button data-testid="pv-card-remove" disabled={!romLoaded || loadingRom} on:click={() => reloadPvCard(true)}
					>Restart without card</button
				>
				<button
					data-testid="pv-card-export"
					disabled={!romLoaded || loadingRom || functionRunnerBusy}
					on:click={exportPvCard}>Export current card SRAM</button
				>
				{#if pvCardError}<p role="alert">{pvCardError}</p>{/if}
			</section>
		{/if}

		{#if romModel === 'pa-9500'}
			<section aria-label="PA configuration" data-testid="pa-configuration">
				<p class="hint">Loaded profile: {paLoadedProfile ?? 'none'}.</p>
				<p class="hint" data-testid="pa-uart-loaded">
					Loaded serial transport: {paLoadedUart ? 'enabled' : 'disabled'}.
				</p>
				<label
					><input type="checkbox" bind:checked={paUart} data-testid="pa-uart" />Enable experimental serial on next load</label
				>
				<label
					>Profile for next ROM load: <select bind:value={paProfile} data-testid="pa-profile">
						<option value="bootstrap-v4">Bootstrap v4 (OFF/ON)</option><option value="bootstrap-v3">Bootstrap v3</option
						><option value="bootstrap-v2">Bootstrap v2</option><option value="bootstrap-v1">Bootstrap v1</option>
					</select></label
				>
				<label
					>Letter bindings: <select
						bind:value={paInput}
						data-testid="pa-input"
						on:change={() => {
							releaseAllPhysicalHeldCodes();
							releaseInputSource('virtual');
						}}
					>
						<option value="native">Native kana layout</option><option value="basic">Latin layout / BASIC</option>
					</select></label
				>
				<p class="hint">
					The ROM translates native and BASIC letters differently. These bindings send ordinary contacts; the device's
					Mode key still controls its input mode. Reset starts a fresh organizer and keeps the current card SRAM.
					Physical geometry and timing remain provisional.
				</p>
				<label
					>PA-9C3 ROM (128 KiB): <input
						type="file"
						accept=".bin"
						data-testid="pa-card-rom"
						on:change={(event) => selectPaCard(event, 'rom')}
					/></label
				>
				<label
					>PA-9C3 SRAM (32 KiB): <input
						type="file"
						accept=".bin"
						data-testid="pa-card-sram"
						on:change={(event) => selectPaCard(event, 'sram')}
					/></label
				>
				<p class="hint">
					Selected: {paCardNames.rom || 'no ROM'} / {paCardNames.sram || 'no SRAM'}. Import restarts the organizer with
					this card image. Export saves current card SRAM; Reset preserves it in memory.
				</p>
				<button
					data-testid="pa-card-load"
					disabled={!romLoaded || loadingRom || !paCardRom.length || !paCardSram.length}
					on:click={() => reloadPaCard()}>Import card and restart</button
				>
				<button data-testid="pa-card-remove" disabled={!romLoaded || loadingRom} on:click={() => reloadPaCard(true)}
					>Restart without card</button
				>
				<button
					data-testid="pa-card-export"
					disabled={!romLoaded || loadingRom || functionRunnerBusy}
					on:click={exportPaCard}>Export card SRAM</button
				>
				<button
					data-testid="pa-observe"
					disabled={!romLoaded || loadingRom || functionRunnerBusy}
					on:click={exportPaObservation}>Export PA observations</button
				>
				{#if paCardError}<p role="alert" data-testid="pa-card-error">{paCardError}</p>{/if}
				<label
					>Import organizer RAM/RTC and restart
					<input
						type="file"
						data-testid="pa-retained-file"
						disabled={!romLoaded || loadingRom || functionRunnerBusy}
						on:change={importPaRetained}
					/>
				</label>
				<button
					data-testid="pa-retained-export"
					disabled={!romLoaded || loadingRom || functionRunnerBusy}
					on:click={exportPaRetained}>Export organizer RAM/RTC</button
				>
				<p>
					Card SRAM is saved separately. Import starts a fresh CPU; choose No at the ROM's erase-data prompt to keep
					organizer records.
				</p>
			</section>
		{/if}
		{#if romModel === 'oz-9600'}
			<section aria-label="OZ configuration" data-testid="oz-configuration">
				<label
					>Profile for next ROM load:
					<select bind:value={ozProfile} data-testid="oz-profile"
						><option value="provisional-v1">OZ-9600 v1 (provisional hardware)</option>
						<option value="strict">Strict</option><option value="experimental">Experimental ROM workflows</option>
						<option value="experimental-isr-clear-only">ISR experiment: clear-only</option>
						<option value="experimental-isr-mti-writable">ISR experiment: writable timer pending</option>
						<option value="experimental-on-edge">ON experiment: one interrupt per press</option>
						<option value="experimental-irq-imr">IRQ experiment: IMR controls acceptance</option>
						<option value="experimental-rtc">RTC experiment: clock and ordinary alarms</option>
						<option value="experimental-rtc-irq-imr">RTC and IRQ experiment: clock with resumable input</option></select
					>
				</label>
				<p class="hint">
					Loaded profile: {ozLoadedProfile}. v1 uses provisional power-on and timing assumptions. Load an OZROM01
					bundle; Save RAM/RTC preserves organizer entries for later restoration.
				</p>
				<label
					>Physical replay <input
						type="file"
						accept=".json"
						data-testid="oz-replay-file"
						on:change={replayOz}
						disabled={!romLoaded || functionRunnerBusy}
					/></label
				>
				<p role="status" data-testid="oz-replay-status">{ozReplayStatus}</p>
				{#if ozReplayReport}<button
						data-testid="oz-replay-report"
						on:click={() =>
							downloadBlob(new Blob([safeJson(ozReplayReport)], { type: 'application/json' }), 'oz-9600-replay.json')}
						>Download replay observations</button
					>{/if}
			</section>
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
		{#if romModel === 'pi-3000'}
			<p class="hint" id="keyboard-help">
				Tap the LCD calibration targets while Run is active. F12 sends ON. Confirmation-key wiring is still under
				recovery.
			</p>
		{:else}
			<p class="hint" id="keyboard-help">
				Click “Type on device”, then Run for live typing. Paused input does not advance the machine. F9 = device SHIFT ·
				F10 = {romModel === 'pa-9500' ? 'input mode' : 'CAPS'} · {romModel === 'oz-9600'
					? 'F11 = 2nd · F12 = ON'
					: 'F12 = ON'} (your keyboard may require Fn). Text fields and Ctrl/Cmd/Alt shortcuts stay with the browser. Hover
				a device key for its host bindings.
			</p>
			<details class="keyboard-help">
				<summary>Keyboard mappings & letter case</summary>
				<p class="hint">
					{#if romModel === 'iq-7000'}
						F1–F5 = Calendar / Schedule / TEL / MEMO / Calc; F6–F8 = Card / World / Home. Page Up/Down = Search; Enter =
						Store; F11 = newline (also Shift+Enter in Letters & symbols).
					{:else if romModel === 'oz-9600'}
						F1 = New Entry; F2 = Edit; Escape = Cancel; Page Up/Down = Prev/Next. Printed application labels and the LCD
						send tablet contacts.
					{:else if romModel === 'pa-9500'}
						F1–F7 = Calendar / Phonebook / Memo / Business / Handwrite / Calculator / Clock. F8 = kana, F10 = input
						mode, F11 = BASIC Run/Pro. Native-app and BASIC English bindings are selected in PA configuration. Tap F9
						then native N for Special or K for SECRET.
					{:else}
						F1–F5 = PF1–PF5; F6 = BASIC; F7 = MENU; F8 = Clear; F11 = device CTRL.
					{/if}
					Letters, digits, Space, arrows, Backspace, Delete, Insert, Escape (Clear), numeric keypad; keypad Enter = equals.
				</p>
				<p class="hint">
					Letters & symbols follows your host keyboard layout: +, − (minus key), *, /, = and decimal point use device
					operator keys, including symbols typed with Shift. Device CAPS controls letter case, not host Shift. Use F9
					for device functions (IQ: F9 then A = EDIT). For IQ comma, press F9, release it, then K; a direct comma key is
					not mapped. Unsupported punctuation is reported, not substituted. Paste opens a preview and supports only
					qualified key sequences; IME and automatic case conversion are not supported.
				</p>
				<p class="hint">
					Device keycaps uses physical host key positions and maps host Shift directly to device SHIFT: shifted legends
					belong to the organizer, not your desktop keyboard. Use the numeric keypad or on-screen keys for operators.
				</p>
			</details>
		{/if}
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
	.serial-panel {
		margin: 0.75rem 0;
		padding: 0.75rem;
		border: 1px solid #32414b;
		border-radius: 8px;
		background: #0d1920;
	}
	.serial-panel summary {
		cursor: pointer;
		font-weight: 600;
	}
	.serial-panel label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0.75rem 0;
	}
	.serial-panel textarea {
		width: min(70ch, 75%);
		font-family: monospace;
	}
	.serial-panel button {
		margin-right: 0.5rem;
	}
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
