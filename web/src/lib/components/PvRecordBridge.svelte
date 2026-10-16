<script lang="ts">
	import {
		parseNativeRecords,
		RECORD_JSON_LIMIT,
		recordsJson,
		importRecordsJson,
		recordsVcard,
		recordsCalendar,
		recordsMemoText,
		type NativeRecords,
	} from '../emulator/pvf1_records';
	import { downloadBlob } from '../lcd_export';
	export let capture: () => Uint8Array;
	export let onImport: (bytes: Uint8Array) => Promise<string>;
	export let canImport = false;
	export let progress = '';
	let records: NativeRecords | null = null,
		query = '',
		error = '',
		notice = '',
		busy = false;
	$: indexed = (records?.rows ?? []).map((row) => ({ row, text: row.join('\n').normalize('NFKC').toLowerCase() }));
	$: needle = query.normalize('NFKC').toLowerCase().trim();
	$: matching = needle ? indexed.filter(({ text }) => text.includes(needle)) : indexed;
	async function load(event: Event) {
		const input = event.currentTarget as HTMLInputElement,
			file = input.files?.[0];
		input.value = '';
		if (!file) return;
		try {
			const json = file.name.toLowerCase().endsWith('.json');
			if (file.size > (json ? RECORD_JSON_LIMIT : 4 * 1024 * 1024)) throw new Error('Record file exceeds size limit');
			records = json
				? await importRecordsJson(await file.text())
				: parseNativeRecords(new Uint8Array(await file.arrayBuffer()));
			error = '';
			notice = `Opened ${records.rows.filter((row) => row[0] === 'D').length} records`;
		} catch (e) {
			error = String(e);
		}
	}
	function captured() {
		try {
			records = parseNativeRecords(capture());
			error = '';
			notice = 'Opened native serial capture';
		} catch (e) {
			error = String(e);
		}
	}
	async function save(kind: 'json' | 'native' | 'contacts' | 'calendar' | 'memo') {
		if (!records) return;
		try {
			const content =
				kind === 'json'
					? await recordsJson(records)
					: kind === 'native'
						? records.bytes
						: kind === 'contacts'
							? await recordsVcard(records)
							: kind === 'calendar'
								? await recordsCalendar(records)
								: recordsMemoText(records);
			const suffix = { json: 'json', native: 'native.csv', contacts: 'vcf', calendar: 'ics', memo: 'txt' }[kind];
			downloadBlob(
				new Blob([typeof content === 'string' ? content : Uint8Array.from(content)], {
					type: 'application/octet-stream',
				}),
				`pv-f1-${records.family}.${suffix}`,
			);
			error = '';
		} catch (e) {
			error = String(e);
		}
	}
	async function send() {
		if (!records || busy) return;
		busy = true;
		error = '';
		try {
			notice = await onImport(records.bytes.slice());
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}
</script>

<details data-testid="pv-record-bridge">
	<summary>Record explorer and transfer</summary>
	<p>
		Open a native record transfer or a lossless JSON export to browse records and save contacts, calendar entries or
		memos.
	</p>
	<label
		>Open record file <input
			data-testid="pv-record-file"
			type="file"
			accept=".csv,.json"
			on:change={load}
			disabled={busy}
		/></label
	>
	<button data-testid="pv-record-capture" on:click={captured} disabled={busy}>Read serial capture</button>
	{#if records}
		<p data-testid="pv-record-summary">
			{records.family} · {records.rows.filter((row) => row[0] === 'D').length} records
		</p>
		<label>Search every field <input data-testid="pv-record-search" type="search" bind:value={query} /></label>
		<p data-testid="pv-record-matches">{matching.length} matching rows</p>
		<button data-testid="pv-record-json" on:click={() => save('json')}>Save lossless JSON</button>
		<button data-testid="pv-record-native" on:click={() => save('native')}>Save native transfer</button>
		{#if ['contacts', 'business'].includes(records.family)}<button
				data-testid="pv-record-vcard"
				on:click={() => save('contacts')}>Save contacts (.vcf)</button
			>{/if}
		{#if records.family === 'schedule'}<button data-testid="pv-record-calendar" on:click={() => save('calendar')}
				>Save calendar (.ics)</button
			>{/if}
		{#if records.family === 'memo'}<button data-testid="pv-record-memo" on:click={() => save('memo')}
				>Save memo text</button
			>{/if}
		<p>
			For import, open Receive in the matching application on the organizer, then select Import to Receive. Native
			transfer files and lossless JSON preserve original Japanese bytes and all fields. Calendar exports keep local
			device times.
		</p>
		<button data-testid="pv-record-import" disabled={!canImport || busy} on:click={send}>Import to Receive</button>
		{#each records.warnings as warning}<p>{warning}</p>{/each}
		<div class="records">
			<table>
				<thead
					><tr
						>{#each records.titles.slice(1) as title}<th>{title}</th>{/each}</tr
					></thead
				><tbody
					>{#each matching.slice(0, 100) as { row }}<tr
							>{#each row.slice(1) as value}<td>{value}</td>{/each}</tr
						>{/each}</tbody
				>
			</table>
		</div>
		{#if matching.length > 100}<p>Showing the first 100 matches. Exports include every row.</p>{/if}
	{/if}
	{#if error}<p role="alert">{error}</p>{/if}
	{#if notice}<p role="status" data-testid="pv-record-notice">{notice}</p>{/if}
	{#if busy && progress}<p role="status" data-testid="pv-record-progress">{progress}</p>{/if}
</details>

<style>
	.records {
		max-height: 24rem;
		overflow: auto;
		margin-block: 1rem;
	}
	table {
		border-collapse: collapse;
	}
	th,
	td {
		border: 1px solid #888;
		padding: 0.3rem;
		white-space: pre-wrap;
		min-width: 6rem;
		text-align: left;
	}
	details {
		margin-block: 1rem;
	}
	button,
	label {
		margin: 0.25rem;
	}
</style>
