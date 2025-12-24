<script lang="ts">
	import type { CallHandle, EvalEvent, PerfettoTraceHandle } from '$lib/debug/sc62015_eval_api';
	import type { FunctionRunnerOutput } from '$lib/debug/function_runner_types';
	import { formatAddress } from '$lib/debug/memory_write_blocks';

	export let output: FunctionRunnerOutput | null = null;
	export let onDownloadTrace: (call: CallHandle) => void;
	export let onDownloadPerfettoTrace: (trace: PerfettoTraceHandle) => void;

	function perfettoTraces(events: EvalEvent[] | null | undefined): PerfettoTraceHandle[] {
		if (!events?.length) return [];
		const out: PerfettoTraceHandle[] = [];
		for (const event of events) {
			if (event.kind === 'perfetto_trace') out.push(event.trace);
		}
		return out;
	}

	function hex(value: number | null | undefined, width = 2): string {
		if (value === null || value === undefined) return '—';
		return `0x${value.toString(16).toUpperCase().padStart(width, '0')}`;
	}

	function formatFunction(pc: number): string {
		return `sub_${pc.toString(16).toUpperCase().padStart(5, '0')}`;
	}
</script>

{#if output?.error}
	<p class="error" data-testid="fnr-error">Script error: {output.error}</p>
{/if}

{#if perfettoTraces(output?.events).length}
	<details open>
		<summary>Perfetto traces ({perfettoTraces(output?.events).length})</summary>
		{#each perfettoTraces(output?.events) as trace (trace.index)}
			<div class="call" data-testid="fnr-trace">
				<div class="call-title">
					<strong>{trace.name}</strong>
				</div>
				<div class="debug-row">
					<span class="hint">{trace.byteLength.toLocaleString()} bytes</span>
					<button type="button" on:click={() => onDownloadPerfettoTrace(trace)}>Download</button>
				</div>
			</div>
		{/each}
	</details>
{/if}

{#if output?.prints?.length}
	<details open>
		<summary>Prints ({output.prints.length})</summary>
		<pre class="log" data-testid="fnr-prints">{JSON.stringify(
				output.prints.map((p) => p.value),
				null,
				2,
			)}</pre>
	</details>
{/if}

{#if output?.calls?.length}
	<details open>
		<summary>Calls ({output.calls.length})</summary>
		{#each output.calls as call (call.index)}
			<div class="call" data-testid="fnr-call">
				<div class="call-title">
					<strong>{call.name ?? formatFunction(call.address)}</strong> ({formatAddress(call.address)})
				</div>
				{#if call.artifacts.infoLog?.length}
					<div class="hint">{call.artifacts.infoLog.join(' • ')}</div>
				{/if}
				{#if call.artifacts.result?.fault}
					<div class="error">
						Fault: {call.artifacts.result.fault.kind}: {call.artifacts.result.fault.message}
					</div>
				{/if}
				{#if call.artifacts.changed?.length}
					<div class="hint">Changed regs: {call.artifacts.changed.join(', ')}</div>
				{/if}

				{#if call.artifacts.memoryBlocks?.length}
					<details>
						<summary>Memory writes ({call.artifacts.memoryBlocks.length} block(s))</summary>
						{#each call.artifacts.memoryBlocks as block (block.start)}
							<pre class="log">{formatAddress(block.start)}:
{block.lines.join('\n')}</pre>
						{/each}
					</details>
				{/if}

				{#if call.artifacts.lcdWrites?.length}
					<details>
						<summary>LCD writes ({call.artifacts.lcdWrites.length})</summary>
						<pre class="log">
{call.artifacts.lcdWrites
								.slice(0, 64)
								.map((w) => `page=${w.page} col=${w.col} val=${hex(w.value, 2)} pc=${hex(w.trace.pc, 6)}`)
								.join('\n')}{call.artifacts.lcdWrites.length > 64 ? '\n…' : ''}</pre>
					</details>
				{/if}

				{#if call.artifacts.perfettoTraceB64}
					<div class="debug-row">
						<span class="hint">Perfetto trace captured.</span>
						<button type="button" on:click={() => onDownloadTrace(call)}>Download</button>
					</div>
				{/if}
			</div>
		{/each}
	</details>
{:else if output}
	{#if !output.prints?.length && !output.resultJson}
		<p class="hint">Script executed (no calls).</p>
	{/if}
{/if}

{#if output?.resultJson}
	<details>
		<summary>Return value (JSON)</summary>
		<pre class="log">{output.resultJson}</pre>
	</details>
{/if}

<style>
	.error {
		color: #ff5c5c;
		white-space: pre-wrap;
	}

	.hint {
		color: #9aa4b2;
	}

	.log {
		margin: 8px 0 0;
		max-height: 200px;
		overflow: auto;
		background: #0c0f12;
		color: #dbe7ff;
		padding: 12px;
		border-radius: 8px;
	}

	.call {
		border: 1px solid #243041;
		border-radius: 8px;
		padding: 10px;
		margin: 10px 0;
	}

	.call-title {
		font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
	}

	.debug-row {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 8px 0;
	}
</style>
