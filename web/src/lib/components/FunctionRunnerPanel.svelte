<script lang="ts">
	import { createPersistedStore } from '$lib/stores/persisted';
	import FunctionRunnerResults from '$lib/components/FunctionRunnerResults.svelte';
	import type { FunctionRunnerOutput } from '$lib/debug/function_runner_types';
	import { tick } from 'svelte';

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
// // IOCS experimentation (public entry is CALLF 0xFFFE8):
// await e.iocs.putc('~', { trace: true });
// await e.iocs.text('HELLO', { trace: true });
//
// await e.withProbe(0x00F299C, (s) => e.print({ hit: s.count, pc: s.pc, A: s.regs.A }), async () => {
//   await e.call(0x00F2A87, undefined, { maxInstructions: 200_000 });
// });
`
	);

	let output: FunctionRunnerOutput | null = null;
	let editor: HTMLTextAreaElement | null = null;
	const INDENT = '  ';

	function downloadTrace(call: any) {
		const b64 = call?.artifacts?.perfettoTraceB64 ?? null;
		if (!b64) return;
		const binary = atob(b64);
		const bytes = new Uint8Array(binary.length);
		for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
		const blob = new Blob([bytes], { type: 'application/octet-stream' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `pce500_call_${call.index}.perfetto-trace`;
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

	function resolveLineRange(text: string, start: number, end: number): { start: number; end: number } {
		const clamp = (n: number) => Math.max(0, Math.min(text.length, n));
		const s = clamp(start);
		const e = clamp(end);
		const left = Math.min(s, e);
		const right = Math.max(s, e);

		const lineStart = text.lastIndexOf('\n', left - 1) + 1;
		const nl = text.indexOf('\n', right);
		const lineEnd = nl === -1 ? text.length : nl;
		return { start: lineStart, end: lineEnd };
	}

	async function toggleLineComments() {
		if (!editor) return;
		const text = editor.value ?? '';
		const selection = resolveLineRange(text, editor.selectionStart ?? 0, editor.selectionEnd ?? 0);
		const block = text.slice(selection.start, selection.end);
		const lines = block.split('\n');

		const nonEmpty = lines.filter((l) => l.trim().length > 0);
		const allCommented =
			nonEmpty.length > 0 && nonEmpty.every((line) => /^\s*\/\//.test(line));

		const updatedLines = lines.map((line) => {
			if (line.trim().length === 0) return line;
			if (allCommented) return line.replace(/^(\s*)\/\/ ?/, '$1');
			const m = /^(\s*)(.*)$/.exec(line);
			if (!m) return `// ${line}`;
			return `${m[1]}// ${m[2]}`;
		});
		const updatedBlock = updatedLines.join('\n');
		const updated = text.slice(0, selection.start) + updatedBlock + text.slice(selection.end);

		$sourceStore = updated;
		await tick();
		editor.selectionStart = selection.start;
		editor.selectionEnd = selection.start + updatedBlock.length;
	}

	async function indentSelection(delta: 1 | -1) {
		if (!editor) return;
		const text = editor.value ?? '';
		const selStart = editor.selectionStart ?? 0;
		const selEnd = editor.selectionEnd ?? 0;

		// When nothing is selected, treat Tab as inserting indentation at the cursor.
		if (selStart === selEnd) {
			if (delta === 1) {
				const updated = text.slice(0, selStart) + INDENT + text.slice(selEnd);
				$sourceStore = updated;
				await tick();
				editor.selectionStart = selStart + INDENT.length;
				editor.selectionEnd = selStart + INDENT.length;
			} else {
				const lineStart = text.lastIndexOf('\n', selStart - 1) + 1;
				const prefix = text.slice(lineStart, Math.min(lineStart + INDENT.length, text.length));
				const remove = prefix.startsWith(INDENT) ? INDENT.length : prefix.startsWith(' ') ? 1 : 0;
				if (remove > 0) {
					const updated = text.slice(0, lineStart) + text.slice(lineStart + remove);
					$sourceStore = updated;
					await tick();
					const newCursor = Math.max(lineStart, selStart - remove);
					editor.selectionStart = newCursor;
					editor.selectionEnd = newCursor;
				}
			}
			return;
		}

		const selection = resolveLineRange(text, selStart, selEnd);
		const block = text.slice(selection.start, selection.end);
		const lines = block.split('\n');

		const updatedLines = lines.map((line) => {
			if (delta === 1) return INDENT + line;
			if (line.startsWith(INDENT)) {
				return line.slice(INDENT.length);
			}
			if (line.startsWith(' ')) {
				return line.slice(1);
			}
			return line;
		});
		const updatedBlock = updatedLines.join('\n');
		const updated = text.slice(0, selection.start) + updatedBlock + text.slice(selection.end);

		$sourceStore = updated;
		await tick();

		// Keep selection spanning the same logical lines after editing.
		editor.selectionStart = selection.start;
		editor.selectionEnd = selection.start + updatedBlock.length;
	}

	function onEditorKeydown(ev: KeyboardEvent) {
		if (ev.key === 'Tab') {
			ev.preventDefault();
			void indentSelection(ev.shiftKey ? -1 : 1);
			return;
		}

		const mod = ev.metaKey || ev.ctrlKey;
		if (!mod) return;
		if (ev.key === 'Enter') {
			ev.preventDefault();
			void runNow();
			return;
		}
		if (ev.key === '/') {
			ev.preventDefault();
			void toggleLineComments();
		}
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
		bind:this={editor}
		bind:value={$sourceStore}
		on:keydown={onEditorKeydown}
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
