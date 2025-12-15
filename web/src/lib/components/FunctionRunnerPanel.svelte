<script lang="ts">
	import { createPersistedStore } from '$lib/stores/persisted';
	import FunctionRunnerResults from '$lib/components/FunctionRunnerResults.svelte';
	import type { FunctionRunnerOutput } from '$lib/debug/function_runner_types';

	export let disabled = false;
	export let busy = false;
	export let onRun: (source: string) => Promise<FunctionRunnerOutput>;
	export let onBeforeRun: () => void = () => {};

	const openStore = createPersistedStore('pce500:function-runner:open', false);
	const sourceStore = createPersistedStore(
		'pce500:function-runner:source',
		`// Examples:
// await e.reset({ fresh: true, warmupTicks: 20_000 });
// await e.keyboard.tap(0x56); // PF1 (virtual injection), holds ~40k instr by default
// await e.call(0x00F2A87, undefined, { maxInstructions: 200_000, trace: true });
//
// await e.withProbe(0x00F299C, (s) => e.print({ hit: s.count, pc: s.pc, A: s.regs.A }), async () => {
//   await e.call(0x00F2A87, undefined, { maxInstructions: 200_000 });
// });
`
	);

	let output: FunctionRunnerOutput | null = null;

	function downloadTrace(call: any) {
		const events = call?.artifacts?.traceEvents ?? [];
		if (!events.length) return;
		const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `pce500_call_${call.index}.trace.json`;
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
	}

	async function runNow() {
		if (disabled || busy) return;
		onBeforeRun();
		output = null;
		output = await onRun($sourceStore);
	}
</script>

<details bind:open={$openStore} data-testid="fnr-panel">
	<summary>Debug (function runner)</summary>
	<div class="row">
		<button type="button" on:click={runNow} disabled={disabled || busy} data-testid="fnr-run">
			{busy ? 'Running…' : 'Run script'}
		</button>
		<button type="button" on:click={() => (output = null)} disabled={busy} data-testid="fnr-clear">
			Clear output
		</button>
	</div>
	<textarea
		class="editor"
		rows="10"
		bind:value={$sourceStore}
		placeholder="Write async JS here. Available: e (Eval API), Reg, Flag."
		data-testid="fnr-editor"
	></textarea>

	<FunctionRunnerResults {output} onDownloadTrace={downloadTrace} />
</details>

<style>
	.row {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 8px 0;
	}

	.editor {
		width: 100%;
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
		background: #0c0f12;
		color: #dbe7ff;
		border: 1px solid #243041;
		border-radius: 8px;
		padding: 12px;
	}
</style>
