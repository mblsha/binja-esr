<script lang="ts">
	import { onDestroy } from 'svelte';
	import { ozLcdTablet } from '../oz9600_panel';
	import type { TabletContact } from '../emulator/oz9600_replay';
	export let disabled = false;
	export let label: string;
	export let fixed: readonly [number, number] | null = null;
	export let onContact: (contact: TabletContact, owner: string, cancel: boolean) => void;
	let held: { x: number; y: number; owner: string; pointer?: number } | null = null;
	let element: HTMLDivElement;
	function press(owner: string, x: number, y: number, pointer?: number) {
		if (disabled || held) return;
		held = { owner, x, y, pointer };
		onContact({ raw_x: x, raw_y: y, pressed: true }, owner, false);
	}
	function release(cancel = false) {
		const previous = held;
		held = null;
		if (!previous) return;
		onContact({ raw_x: previous.x, raw_y: previous.y, pressed: false }, previous.owner, cancel);
		if (previous.pointer !== undefined && element.hasPointerCapture?.(previous.pointer))
			element.releasePointerCapture(previous.pointer);
	}
	function down(event: PointerEvent) {
		if (event.button !== 0 || disabled || held) return;
		event.preventDefault();
		const box = element.getBoundingClientRect();
		const [x, y] = fixed ?? ozLcdTablet((event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height);
		press(`tablet:${label}:${event.pointerId}`, x, y, event.pointerId);
		element.setPointerCapture?.(event.pointerId);
	}
	$: if (disabled) release(true);
	onDestroy(() => release(true));
</script>

<svelte:window
	on:blur={() => release(true)}
	on:pointerup={(e) => {
		if (held?.pointer === e.pointerId) release();
	}}
	on:pointercancel={(e) => {
		if (held?.pointer === e.pointerId) release(true);
	}}
/>
<svelte:document
	on:visibilitychange={() => {
		if (document.hidden) release(true);
	}}
/>
<div
	bind:this={element}
	role="button"
	tabindex={disabled ? -1 : 0}
	aria-label={label}
	title={label === 'Clock' ? 'Hold to view Clock; release to close' : undefined}
	aria-disabled={disabled}
	data-testid={`oz-tablet-${label.toLowerCase().replaceAll(' ', '-')}`}
	class="surface"
	on:pointerdown={down}
	on:lostpointercapture={() => release(true)}
	on:keydown={(e) => {
		if (fixed && ['Enter', ' '].includes(e.key)) {
			e.preventDefault();
			e.stopPropagation();
			if (!e.repeat) press(`tablet:key:${label}`, fixed[0], fixed[1]);
		}
	}}
	on:keyup={(e) => {
		if (fixed && ['Enter', ' '].includes(e.key)) {
			e.preventDefault();
			e.stopPropagation();
			release();
		}
	}}
	on:blur={() => release(true)}
>
	<slot />
</div>

<style>
	.surface {
		width: 100%;
		height: 100%;
		touch-action: none;
		cursor: crosshair;
	}
	.surface:focus-visible {
		outline: 2px solid #126266;
		outline-offset: 1px;
	}
</style>
