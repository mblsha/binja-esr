<script lang="ts">
	import { tick } from 'svelte';
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
	import { previewModernImport, type ImportOptions, type ImportPreview } from '../emulator/pvf1_modern_import';
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
	let modernSource: { bytes: Uint8Array; name: string } | null = null;
	let modern: ImportPreview | null = null;
	let reviewed = false;
	let options: ImportOptions = {
		target: 'memo',
		duplicates: 'skip-identical',
		memoCode: '１',
		memoClassification: '',
		deviceTimeZone: '',
	};
	function rebuildPreview() {
		reviewed = false;
		records = null;
		modern = null;
		if (!modernSource) return;
		try {
			modern = previewModernImport(modernSource.bytes, modernSource.name, options);
			records = modern.records;
			error = '';
			notice = records
				? `${records.rows.length} converted records ready for review`
				: 'Resolve the conversion errors before importing';
		} catch (e) {
			error = String(e);
		}
	}
	async function updatePreview() {
		reviewed = false;
		await tick();
		rebuildPreview();
	}
	function saveOriginal() {
		if (modernSource)
			downloadBlob(
				new Blob([Uint8Array.from(modernSource.bytes)], { type: 'application/octet-stream' }),
				modernSource.name,
			);
	}
	function saveReview() {
		if (!modern) return;
		const report = {
			schema: 'pvf1-modern-review-1',
			sourceName: modern.sourceName,
			options: modern.options,
			rows: modern.rows,
			issues: modern.issues,
			skippedDuplicates: modern.skippedDuplicates,
			originalSourceSavedSeparately: true,
		};
		downloadBlob(
			new Blob([JSON.stringify(report, null, 2) + '\n'], { type: 'application/json' }),
			'pv-f1-conversion-review.json',
		);
	}
	$: indexed = (records?.rows ?? []).map((row) => ({ row, text: row.join('\n').normalize('NFKC').toLowerCase() }));
	$: needle = query.normalize('NFKC').toLowerCase().trim();
	$: matching = needle ? indexed.filter(({ text }) => text.includes(needle)) : indexed;
	async function load(event: Event) {
		if (busy) return;
		const input = event.currentTarget as HTMLInputElement,
			file = input.files?.[0];
		input.value = '';
		if (!file) return;
		busy = true;
		records = null;
		modernSource = null;
		modern = null;
		reviewed = false;
		error = '';
		try {
			const json = file.name.toLowerCase().endsWith('.json');
			if (file.size > (json ? RECORD_JSON_LIMIT : 4 * 1024 * 1024)) throw new Error('Record file exceeds size limit');
			if (/\.(vcf|vcard|ics|ical|txt)$/i.test(file.name)) {
				modernSource = { bytes: new Uint8Array(await file.arrayBuffer()), name: file.name };
				options = {
					...options,
					target: /\.(ics|ical)$/i.test(file.name) ? 'schedule' : /\.txt$/i.test(file.name) ? 'memo' : 'telephone1',
				};
				rebuildPreview();
				return;
			}
			records = json
				? await importRecordsJson(await file.text())
				: parseNativeRecords(new Uint8Array(await file.arrayBuffer()));
			error = '';
			notice = `Opened ${records.rows.filter((row) => row[0] === 'D').length} records`;
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}
	function captured() {
		records = null;
		modern = null;
		modernSource = null;
		reviewed = false;
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
		if (!records || busy || (modernSource && !reviewed)) return;
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
		Open native CSV or lossless JSON to browse records, or preview a UTF-8 vCard, calendar or Memo text file for import.
	</p>
	<label
		>Open record file <input
			data-testid="pv-record-file"
			type="file"
			accept=".csv,.json,.vcf,.vcard,.ics,.ical,.txt"
			on:change={load}
			disabled={busy}
		/></label
	>
	<button data-testid="pv-record-capture" on:click={captured} disabled={busy}>Read serial capture</button>
	{#if modernSource}
		<fieldset disabled={busy} data-testid="pv-modern-preview">
			<legend>Review modern import: {modernSource.name}</legend>
			<label
				>Destination <select data-testid="pv-modern-target" bind:value={options.target} on:change={updatePreview}>
					<option value="telephone1">Telephone 1</option><option value="telephone2">Telephone 2</option><option
						value="business">Business</option
					><option value="schedule">Schedule</option><option value="memo">Memo</option>
				</select></label
			>
			<label
				>Identical records within this file <select
					data-testid="pv-modern-duplicates"
					bind:value={options.duplicates}
					on:change={updatePreview}
				>
					<option value="skip-identical">Skip later identical records</option><option value="keep"
						>Keep every record</option
					>
				</select></label
			>
			{#if options.target === 'memo'}
				<label
					>Classification code (１–９) <input
						data-testid="pv-modern-code"
						bind:value={options.memoCode}
						on:input={updatePreview}
						maxlength="1"
					/></label
				>
				<label
					>Classification name <input
						data-testid="pv-modern-classification"
						bind:value={options.memoClassification}
						on:input={updatePreview}
					/></label
				>
			{/if}
			{#if options.target === 'schedule'}
				<label
					>Organizer time zone for fixed-time events <input
						data-testid="pv-modern-zone"
						bind:value={options.deviceTimeZone}
						on:input={updatePreview}
						placeholder="Australia/Sydney"
					/></label
				>
				<p>
					Floating times remain local. UTC and named-zone events require the organizer's intended time zone. Recurrence,
					custom zone rules and embedded alarms are rejected.
				</p>
			{/if}
			<p>
				Existing organizer records are not compared here. Native Receive controls how they are added or matched.
				Unsupported properties stay in the original source, which you can save below.
			</p>
			<button data-testid="pv-modern-original" on:click={saveOriginal}>Save original source</button>
			{#if modern}
				<button data-testid="pv-modern-review" on:click={saveReview}>Save complete conversion review</button>
				<p data-testid="pv-modern-count">
					{modern.rows.length} source records · {modern.skippedDuplicates} identical records skipped · {modern.issues.filter(
						(x) => x.severity === 'error',
					).length} errors
				</p>
				<ul data-testid="pv-modern-issues">
					{#each modern.issues.slice(0, 100) as issue}<li>
							{issue.severity} · row {issue.row || 'all'} · {issue.field}: {issue.message}
						</li>{/each}
				</ul>
				{#if modern.issues.length > 100}<p>
						Showing the first 100 issues. The saved review includes every issue and source row.
					</p>{/if}
				<div class="records">
					<table>
						<tbody
							>{#each modern.rows.slice(0, 100) as row, i}<tr
									><th>Source row {i + 1}</th><td
										>{Object.entries(row)
											.filter(([key]) => key !== 'SCRT')
											.map(([key, value]) => `${key}: ${value}`)
											.join('\n')}</td
									></tr
								>{/each}</tbody
						>
					</table>
				</div>
				<label
					><input data-testid="pv-modern-ack" type="checkbox" bind:checked={reviewed} disabled={!records} /> I reviewed the
					converted fields and omissions.</label
				>
			{/if}
		</fieldset>
	{/if}
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
		<button
			data-testid="pv-record-import"
			disabled={!canImport || busy || (modernSource !== null && !reviewed)}
			on:click={send}>Import to Receive</button
		>
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
