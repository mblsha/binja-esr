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

	function setMatrixCode(code: number, down: boolean) {
		if (!emulator) return;
		if (down) {
			if (pressedCodes.has(code)) return;
			pressedCodes.add(code);
			// Inject directly so short taps are observable even if firmware polling is sparse.
			emulator.inject_matrix_event?.(code, false);
			emulator.press_matrix_code?.(code);
		} else {
			if (!pressedCodes.has(code)) return;
			pressedCodes.delete(code);
			pendingVirtualRelease.delete(code);
			emulator.inject_matrix_event?.(code, true);
			emulator.release_matrix_code?.(code);
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
		window.addEventListener('keydown', onKeyDown, { passive: false });
		window.addEventListener('keyup', onKeyUp, { passive: false });
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
	</style>
