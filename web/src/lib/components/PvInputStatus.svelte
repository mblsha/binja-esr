<script lang="ts">
	import type { PvInputPhase } from '../emulator/pvf1_readiness';
	export let phase: PvInputPhase;
	export let pen: { strokes: number; points: number; blocked: boolean };
	export let paused: boolean;
	export let onCancel: () => void;
</script>

<p role="status" data-testid="pv-input-readiness" data-phase={phase}>
	{phase === 'waking'
		? `Waking up; assisted input waits in order.${paused ? ' Resume to continue.' : ''}`
		: phase === 'starting'
			? 'Follow the device display; complete initialization if prompted. Pen input is available.'
			: phase === 'off'
				? 'Device OFF; use Power / ON to wake.'
				: 'Ready for input.'}
</p>
{#if pen.strokes || pen.blocked}<p role="status" data-testid="pv-pen-queue">
		{pen.strokes} pen strokes active or pending · {pen.points} movement points pending{pen.blocked
			? ' — buffer blocked'
			: ''}<button on:click={onCancel}>Cancel pending input</button>
	</p>{/if}
