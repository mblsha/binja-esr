<script lang="ts">
	import OzTabletSurface from './OzTabletSurface.svelte';
	import type { TabletContact } from '../emulator/oz9600_replay';
	import { PV_APPLICATIONS } from '../pvf1_panel';
	export let disabled = false;
	export let onContact: (contact: TabletContact, owner: string, cancel: boolean) => void;
	let x = 362;
	let y = 585;
	$: valid = Number.isInteger(x) && Number.isInteger(y) && x >= 0 && x <= 1023 && y >= 0 && y <= 1023;
</script>

<div class="applications" aria-label="PV-F1 applications">
	{#each PV_APPLICATIONS as app}
		<OzTabletSurface label={app.label} fixed={app.raw} testPrefix="pv-app" {disabled} {onContact}
			><span>{app.label}</span></OzTabletSurface
		>
	{/each}
</div>

<div class="tablet" aria-label="PV-F1 tablet samples" data-testid="pv-tablet-controls">
	<label
		>X <input type="number" min="0" max="1023" step="1" bind:value={x} {disabled} data-testid="pv-tablet-x" /></label
	>
	<label
		>Y <input type="number" min="0" max="1023" step="1" bind:value={y} {disabled} data-testid="pv-tablet-y" /></label
	>
	<div class="pen">
		<OzTabletSurface label="Pen" fixed={[x, y]} disabled={disabled || !valid} testPrefix="pv-tablet" {onContact}
			>Hold pen down</OzTabletSurface
		>
	</div>
	<p>Click the LCD or use raw tablet samples, 0–1023. Pen capture is experimental.</p>
</div>

<style>
	.applications {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
		margin-top: 12px;
	}
	.applications :global(.surface) {
		padding: 10px;
		border: 1px solid #84918b;
		border-radius: 6px;
		background: #394942;
		color: #f2f4eb;
		text-align: center;
		cursor: pointer;
	}
	.tablet {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		align-items: center;
		margin-top: 12px;
	}
	label {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	input {
		width: 6em;
	}
	.pen {
		border: 1px solid #84918b;
		border-radius: 6px;
		padding: 8px 14px;
		background: #394942;
		color: #f2f4eb;
	}
	p {
		width: 100%;
		font-size: 0.85em;
		margin: 0;
	}
</style>
