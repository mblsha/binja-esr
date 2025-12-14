<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import LcdCanvas from '$lib/components/LcdCanvas.svelte';
	import { matrixCodeForKeyEvent } from '$lib/keymap';

	let wasm: any = null;
	let emulator: any = null;

	let lcdPixels: Uint8Array | null = null;
	let debugState: any = null;
	let lastError: string | null = null;

	let running = false;
	let instructionsPerFrame = 20_000;

	function safeJson(value: any): string {
		return JSON.stringify(value, (_key, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
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

	function refresh() {
		if (!emulator) return;
		lcdPixels = emulator.lcd_pixels();
		debugState = emulator.debug_state();
	}

	function stepOnce(count: number) {
		if (!emulator) return;
		try {
			emulator.step(count);
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
		emulator.press_matrix_code(code);
		event.preventDefault();
	}

	function onKeyUp(event: KeyboardEvent) {
		if (!emulator) return;
		const code = matrixCodeForKeyEvent(event);
		if (code === null) return;
		emulator.release_matrix_code(code);
		event.preventDefault();
	}

	onMount(() => {
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

	<LcdCanvas pixels={lcdPixels} />

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
		Keyboard: F1/F2 (PF1/PF2), arrows (cursor keys).
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
	</style>
