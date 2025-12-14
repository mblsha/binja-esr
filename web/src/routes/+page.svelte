<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import VirtualKeyboard from '$lib/components/VirtualKeyboard.svelte';
	import { matrixCodeForKeyEvent } from '$lib/keymap';

	let wasm: any = null;
	let emulator: any = null;

	let lcdPixels: Uint8Array | null = null;
	let lcdText: string[] | null = null;
	let regs: any = null;
	let callStack: number[] | null = null;
	let debugState: any = null;
	let lastError: string | null = null;
	let romSource: string | null = null;
	let pcReg: number | null = null;

	let running = false;
	let instructionsPerFrame = 20_000;
	const pressedCodes = new Set<number>();
	const pendingVirtualRelease = new Map<number, number>();
	const MIN_VIRTUAL_HOLD_INSTRUCTIONS = 40_000;
	const IMEM_BASE = 0x100000;
	const FIFO_BASE_ADDR = 0x00bfc96;
	const FIFO_HEAD_ADDR = 0x00bfc9d;
	const FIFO_TAIL_ADDR = 0x00bfc9e;
	const debugLog: string[] = [];

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

	function hex(value: number | null | undefined, width = 5): string {
		if (value === null || value === undefined) return '—';
		return `0x${value.toString(16).toUpperCase().padStart(width, '0')}`;
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
				return Array.from(input.entries()).filter(
					([k, v]) => typeof k === 'string' && typeof v === 'number'
				);
			}
			return Object.entries(input).filter(
				([k, v]) => typeof k === 'string' && typeof v === 'number'
			) as [string, number][];
		} catch {
			return [];
		}
	}

	function logDebug(line: string) {
		debugLog.unshift(line);
		if (debugLog.length > 50) debugLog.pop();
	}

	function setMatrixCode(code: number, down: boolean) {
		if (!emulator) return;
		if (down) {
			if (pressedCodes.has(code)) return;
			pressedCodes.add(code);
			// Inject directly so short taps are observable even if firmware polling is sparse.
			emulator.inject_matrix_event?.(code, false);
			emulator.press_matrix_code?.(code);
			logDebug(`press ${hex(code, 2)}`);
		} else {
			if (!pressedCodes.has(code)) return;
			pressedCodes.delete(code);
			pendingVirtualRelease.delete(code);
			emulator.inject_matrix_event?.(code, true);
			emulator.release_matrix_code?.(code);
			logDebug(`release ${hex(code, 2)}`);
		}
	}

	function virtualPress(code: number) {
		setMatrixCode(code, true);
		pendingVirtualRelease.delete(code);
	}

	function virtualRelease(code: number) {
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
			wasm = await import('$lib/wasm/pce500_wasm/pce500_wasm.js');
			if (typeof wasm.default === 'function') {
				await wasm.default();
			}
		}
		if (!emulator) {
			emulator = new wasm.Pce500Emulator();
		}
		return emulator;
	}

	async function tryAutoLoadRom() {
		if (emulator?.has_rom?.()) return;
		try {
			const res = await fetch('/api/rom');
			if (!res.ok) return;
			romSource = res.headers.get('x-rom-source');
			const bytes = new Uint8Array(await res.arrayBuffer());
			const emu = await ensureEmulator();
			emu.load_rom(bytes);
			refresh();
		} catch (err) {
			lastError = `Auto-load failed: ${String(err)}`;
		}
	}

	function refresh() {
		if (!emulator) return;
		lcdPixels = emulator.lcd_pixels();
		lcdText = emulator.lcd_text?.() ?? null;
		regs = emulator.regs?.() ?? null;
		try {
			pcReg = emulator.get_reg?.('PC') ?? null;
		} catch {
			pcReg = null;
		}
		callStack = emulator.call_stack?.() ?? null;
		debugState = emulator.debug_state?.() ?? null;
	}

	function snapshotKeyboardState() {
		if (!emulator) {
			debugKio = null;
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
		} catch {
			debugKio = null;
		}
	}

	function dumpKeyboardState(tag = 'dump') {
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
				pendingVirtualRelease: Array.from(pendingVirtualRelease.entries())
			});
		} catch (err) {
			console.log(`[pce500] ${tag}: dump failed`, err);
		}
	}

	function installDevtoolsDebugHelpers() {
		if (!isDevBuild()) return;
		(globalThis as any).__pce500 = {
			get emulator() {
				return emulator;
			},
			dump: dumpKeyboardState,
			step: (n: number) => stepOnce(n),
			read: (addr: number) => emulator?.read_u8?.(addr),
			readInternal: (offset: number) => emulator?.read_u8?.(IMEM_BASE + offset),
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
			}
		};
		console.log(
			'[pce500] devtools helpers installed: __pce500.dump(), __pce500.tapPF1(), __pce500.readInternal(0xF2)'
		);
	}

	$: halted = Boolean(debugState?.halted);
	$: statusLabel = running ? 'RUNNING' : halted ? 'HALTED' : 'STOPPED';
	$: pc = pcReg;

	function sortedRegs(input: any): [string, number][] {
		return regsEntries(input).sort(([a], [b]) => a.localeCompare(b));
	}

	function stepOnce(count: number) {
		if (!emulator) return;
		try {
			emulator.step(count);
			applyVirtualReleaseBudget(count);
			refresh();
			snapshotKeyboardState();
		} catch (err) {
			lastError = String(err);
			running = false;
		}
	}

	function tick() {
		if (!running || !emulator) return;
		stepOnce(instructionsPerFrame);
		if (!running) return;
		requestAnimationFrame(tick);
	}

	async function onSelectRom(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		lastError = null;
		try {
			const emu = await ensureEmulator();
			const bytes = new Uint8Array(await file.arrayBuffer());
			emu.load_rom(bytes);
			refresh();
		} catch (err) {
			lastError = String(err);
		}
	}

	function start() {
		if (!emulator) return;
		if (running) return;
		running = true;
		requestAnimationFrame(tick);
	}

	function stop() {
		running = false;
	}

	function onKeyDown(event: KeyboardEvent) {
		if (!emulator) return;
		if (event.repeat) return;
		const code = matrixCodeForKeyEvent(event);
		if (code === null) return;
		setMatrixCode(code, true);
		event.preventDefault();
	}

	function onKeyUp(event: KeyboardEvent) {
		if (!emulator) return;
		const code = matrixCodeForKeyEvent(event);
		if (code === null) return;
		setMatrixCode(code, false);
		event.preventDefault();
	}

	onMount(() => {
		void tryAutoLoadRom();
		installDevtoolsDebugHelpers();
		window.addEventListener('keydown', onKeyDown, { passive: false });
		window.addEventListener('keyup', onKeyUp, { passive: false });
		snapshotKeyboardState();
	});

	onDestroy(() => {
		window.removeEventListener('keydown', onKeyDown);
		window.removeEventListener('keyup', onKeyUp);
	});
</script>

<main>
	<h1>PC-E500 Web Emulator (LLAMA/WASM)</h1>

	<label>
		Load ROM:
		<input type="file" accept=".bin,.rom,.img" on:change={onSelectRom} />
	</label>

	{#if romSource}
		<p class="hint">Auto-loaded ROM via {romSource}</p>
	{/if}

	<div class="controls">
		<button on:click={() => stepOnce(1_000)} disabled={!emulator}>Step 1k</button>
		<button on:click={() => stepOnce(20_000)} disabled={!emulator}>Step 20k</button>
		<button on:click={start} disabled={!emulator || running}>Run</button>
		<button on:click={stop} disabled={!running}>Stop</button>
		<label>
			Instr/frame:
			<input type="number" min="1" step="1000" bind:value={instructionsPerFrame} />
		</label>
	</div>

	<p class="hint" data-testid="emu-status">Status: {statusLabel} • PC: {hex(pc)} • Instr: {debugState?.instruction_count ?? '—'}</p>

	<LcdCanvas pixels={lcdPixels} />

	<VirtualKeyboard
		disabled={!emulator}
		onPress={(code) => virtualPress(code)}
		onRelease={(code) => virtualRelease(code)}
	/>

	{#if emulator}
		<details>
			<summary>Debug (keyboard)</summary>
			<div class="debug-row">
				<button type="button" on:click={() => snapshotKeyboardState()}>Refresh</button>
				<button type="button" on:click={() => dumpKeyboardState('ui')}>Dump to console</button>
				<button type="button" on:click={() => (debugLog.length = 0)}>Clear log</button>
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
							<td class="val">{Array.from(pressedCodes).map((c) => hex(c, 2)).join(' ') || '—'}</td>
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

	{#if callStack}
		<details open>
			<summary>Call stack</summary>
			{#if callStack.length === 0}
				<p class="hint" data-testid="call-stack-empty">No frames</p>
			{:else}
				<ol class="stack" data-testid="call-stack">
					{#each callStack as frame}
						<li>{formatFunction(frame)} ({hex(frame)})</li>
					{/each}
				</ol>
			{/if}
		</details>
	{/if}

	{#if regs}
		<details>
			<summary>Registers</summary>
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
		</details>
	{/if}

	{#if lcdText}
		<details open>
			<summary>LCD (decoded text)</summary>
			<pre data-testid="lcd-text">{lcdText.join('\n')}</pre>
		</details>
	{/if}

	{#if lastError}
		<p class="error">{lastError}</p>
	{/if}

	{#if debugState}
		<details>
			<summary>Debug state</summary>
			<pre>{safeJson(debugState)}</pre>
		</details>
	{/if}

	<p class="hint">
		Keyboard: F1/F2 (PF1/PF2), arrows (cursor keys). Virtual keyboard supports PF1/PF2 + arrows.
	</p>
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
	</style>
