<script lang="ts">
	type Example = {
		id: string;
		title: string;
		hint?: string;
		code: string;
	};

	const EXAMPLES: Example[] = [
		{
			id: 'reset',
			title: 'Reset + warmup',
			code: `await e.reset({ fresh: true, warmupTicks: 20_000 });`,
		},
		{
			id: 'step-read-regs',
			title: 'Step + read registers',
			code: `
await e.step(50_000);
e.print({ PC: e.reg('PC'), A: e.reg('A') });
`.trim(),
		},
		{
			id: 'call-address',
			title: 'Call a ROM function by address',
			hint: 'Uses the call harness (runs until return or timeout).',
			code: `await e.call(0x00F2A87, undefined, { maxInstructions: 200_000 });`,
		},
		{
			id: 'stub',
			title: 'Stub a ROM routine',
			hint: 'Intercepts execution, patches state, then returns (or jumps).',
			code: `
e.stub(0x00F1234, 'demo_stub', (mem, regs, flags) => ({
  mem_writes: { 0x2000: mem.read8(0x2000) ^ 0xff },
  regs: { A: 0x42 },
  flags: { Z: 0, C: 1 },
  ret: { kind: 'ret' },
}));
await e.call(0x00F1234, undefined, { maxInstructions: 5_000 });
`.trim(),
		},
		{
			id: 'virtual-key',
			title: 'Tap a virtual key',
			hint: 'Example uses PF1 (PC‑E500 default keymap).',
			code: `await e.keyboard.tap(0x56);`,
		},
		{
			id: 'iocs-putcxy',
			title: 'Print a character at X/Y (IOCS 0x0041)',
			hint: 'PC‑E500‑shaped; other models may not implement character output.',
			code: `await e.iocs.putcXY('A', { bl: 0, bh: 0, cl: 0, ch: 0 });`,
		},
		{
			id: 'perfetto-trace',
			title: 'Capture a long Perfetto trace',
			hint: 'Creates a named trace item in the UI; nesting is unsupported.',
			code: `
await e.perfetto.trace('boot-menu', async () => {
  await e.step(500_000);
  await e.keyboard.tap(0x56);
});
`.trim(),
		},
		{
			id: 'probe',
			title: 'Probe a hot PC and print register snapshots',
			code: `
await e.withProbe(0x00F299C, (s) => e.print({ hit: s.count, pc: s.pc, A: s.regs.A }), async () => {
  await e.step(200_000);
});
`.trim(),
		},
	];

	let copyError: string | null = null;
	let copiedId: string | null = null;
	let copiedTimeout: ReturnType<typeof setTimeout> | null = null;

	async function copyToClipboard(example: Example) {
		copyError = null;
		try {
			const clipboard = navigator?.clipboard;
			if (!clipboard?.writeText) {
				throw new Error('Clipboard API unavailable');
			}
			await clipboard.writeText(example.code.trimEnd() + '\n');
			copiedId = example.id;
			if (copiedTimeout) clearTimeout(copiedTimeout);
			copiedTimeout = setTimeout(() => {
				if (copiedId === example.id) copiedId = null;
			}, 1200);
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			copyError = `Copy failed: ${msg}`;
		}
	}
</script>

<details data-testid="fnr-examples">
	<summary>Function runner examples</summary>
	<p class="hint">Copy/paste these into the Function Runner editor (async JS).</p>
	{#if copyError}
		<p class="error">{copyError}</p>
	{/if}

	{#each EXAMPLES as example (example.id)}
		<div class="example">
			<div class="row">
				<strong>{example.title}</strong>
				<button type="button" on:click={() => copyToClipboard(example)}>
					{copiedId === example.id ? 'Copied' : 'Copy'}
				</button>
			</div>
			{#if example.hint}
				<div class="hint">{example.hint}</div>
			{/if}
			<pre class="code"><code>{example.code}</code></pre>
		</div>
	{/each}
</details>

<style>
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		justify-content: space-between;
		margin: 8px 0;
	}

	.example {
		border: 1px solid #243041;
		border-radius: 8px;
		padding: 10px;
		margin: 10px 0;
	}

	.hint {
		color: #9aa4b2;
	}

	.error {
		color: #ff5c5c;
	}

	.code {
		margin: 8px 0 0;
		overflow: auto;
		background: #0c0f12;
		color: #dbe7ff;
		padding: 12px;
		border-radius: 8px;
	}
</style>
