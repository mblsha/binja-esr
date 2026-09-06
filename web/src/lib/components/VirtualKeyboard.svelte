<script lang="ts">
	import { onDestroy } from 'svelte';
	import { virtualKeysForModel, hostKeyHints, type HostKeyboardMode } from '../keymap';
	import type { InputContact } from '../emulator/host_inputs';
	import type { RomModel } from '../rom_model';
	import { rectStyle, type DeviceLayout, type DeviceKey } from '../device_layout';
	import ModeGlyph from './ModeGlyph.svelte';

	export let disabled = false;
	export let model: RomModel = 'pc-e500';
	export let layout: DeviceLayout | null = null;
	export let hostKeyboardMode: HostKeyboardMode = 'symbols';
	export let physicalHighlights = new Set<InputContact>();
	export let onPress: (code: InputContact, owner: string) => void;
	export let onRelease: (code: InputContact, owner: string, cancel: boolean) => void;
	export let onCancelAll: () => void = () => {};
	const held = new Map<string, { code: InputContact; element: HTMLButtonElement; pointerId?: number }>();
	$: keys = layout?.keys ?? virtualKeysForModel(model);
	let activeModel = model;
	$: if (disabled || model !== activeModel) {
		cancelAll();
		activeModel = model;
	}

	function press(code: InputContact, owner: string, element: HTMLButtonElement, pointerId?: number) {
		if (disabled || held.has(owner)) return;
		held.set(owner, { code, element, pointerId });
		onPress(code, `${owner}:${code}`);
	}
	function release(owner: string, cancel = false) {
		const state = held.get(owner);
		if (!state) return;
		held.delete(owner); // lostpointercapture may fire synchronously.
		onRelease(state.code, `${owner}:${state.code}`, cancel); // Also release after disabling.
		if (state.pointerId !== undefined && state.element.hasPointerCapture?.(state.pointerId))
			state.element.releasePointerCapture(state.pointerId);
	}
	function cancelAll() {
		for (const owner of held.keys()) release(owner, true);
		onCancelAll(); // Includes assisted releases whose pointers already went UP.
	}
	function pointerDown(code: InputContact, event: PointerEvent) {
		if (event.button !== undefined && event.button !== 0) return;
		event.preventDefault();
		const element = event.currentTarget as HTMLButtonElement;
		press(code, `pointer:${event.pointerId}`, element, event.pointerId);
		if (!disabled) element.setPointerCapture?.(event.pointerId);
	}
	function keyDown(code: InputContact, event: KeyboardEvent) {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		event.stopPropagation();
		if (!event.repeat) press(code, `key:${event.code}`, event.currentTarget as HTMLButtonElement);
	}
	function keyUp(event: KeyboardEvent) {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		event.stopPropagation();
		release(`key:${event.code}`);
	}
	onDestroy(cancelAll);
</script>

<svelte:window
	on:pointerup={(event) => release(`pointer:${event.pointerId}`)}
	on:pointercancel={(event) => release(`pointer:${event.pointerId}`, true)}
	on:blur={cancelAll}
/>
<svelte:document
	on:visibilitychange={() => {
		if (document.hidden) cancelAll();
	}}
/>

<section class="vk" class:physical={layout !== null} aria-label="Virtual keyboard">
	<div class="grid" role="group" aria-label="Keys">
		{#each keys as key (key.testId)}
			<button
				type="button"
				class="key"
				class:unmapped={key.code === null}
				class:host-held={key.code !== null && physicalHighlights.has(key.code)}
				class:mode={'tone' in key && key.tone === 'mode'}
				class:shift={'tone' in key && key.tone === 'shift'}
				class:clear={'tone' in key && key.tone === 'clear'}
				class:long={'face' in key && key.face.length >= 4}
				style={layout ? rectStyle(key as DeviceKey, layout) : undefined}
				data-testid={key.testId}
				data-key-id={'id' in key ? key.id : undefined}
				aria-label={key.code === null ? `${key.label} (unmapped)` : key.label}
				title={key.code === null
					? `${key.label}: physical key shown; input mapping not yet qualified`
					: `${key.label}${hostKeyHints(key.code, model, hostKeyboardMode) ? ' — keyboard: ' + hostKeyHints(key.code, model, hostKeyboardMode) : ''}`}
				disabled={disabled || key.code === null}
				on:pointerdown={(event) => {
					if (key.code !== null) pointerDown(key.code, event);
				}}
				on:lostpointercapture={(event) => release(`pointer:${event.pointerId}`, true)}
				on:keydown={(event) => {
					if (key.code !== null) keyDown(key.code, event);
				}}
				on:keyup={keyUp}
				on:blur={() => {
					for (const owner of held.keys()) if (owner.startsWith('key:')) release(owner, true);
				}}
				on:click={(event) => {
					// Assistive activation has no pointer DOWN/UP pair.
					if (event.detail === 0 && key.code !== null) {
						press(key.code, 'activation', event.currentTarget);
						release('activation');
					}
				}}
			>
				{#if 'legend' in key && key.legend}<span class="legend" aria-hidden="true">{key.legend}</span>{/if}
				{#if layout?.model === 'iq-7000' && 'tone' in key && key.tone === 'mode' && !['ON', 'OFF'].includes(key.id)}
					<span class="legend" aria-hidden="true">{key.face}</span><ModeGlyph name={key.id} />
				{:else}
					<span class="face" aria-hidden="true">{'face' in key ? key.face : key.label}</span>
				{/if}
			</button>
		{/each}
	</div>
</section>

<style>
	.vk {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(6, minmax(44px, 1fr));
		gap: 8px;
	}
	.key {
		padding: 10px 12px;
		border-radius: 10px;
		border: 1px solid #243041;
		background: #0c0f12;
		color: #dbe7ff;
		touch-action: none;
		user-select: none;
	}
	.key:disabled {
		opacity: 0.5;
	}
	.key:active,
	.key.host-held {
		background: #121722;
	}
	.key:focus-visible {
		outline: 3px solid #67cde3;
		outline-offset: 3px;
		z-index: 2;
	}
	.physical {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.physical .grid {
		display: contents;
	}
	.physical .key {
		position: absolute;
		pointer-events: auto;
		padding: 0;
		border-radius: 5px;
		border: 1px solid #101416;
		background: linear-gradient(#3b3c3c, #161b1d 40%, #24292b);
		box-shadow:
			0 2px 1px #111,
			0 0 0 2px #6669,
			inset 0 1px 1px #afb7b966;
		color: #f1f2eb;
		font:
			600 clamp(10px, 1.55cqw, 20px) / 1 Arial,
			sans-serif;
	}
	.physical .key:active:not(:disabled),
	.physical .key.host-held {
		transform: translateY(2px);
		box-shadow: inset 0 2px 4px #070c0e;
		background: #101e24;
	}
	.physical .key.host-held {
		outline: 2px solid #67cde3;
		outline-offset: 2px;
	}
	.physical .key:disabled {
		opacity: 0.65;
	}
	.physical .unmapped {
		opacity: 0.82;
		box-shadow:
			0 2px 1px #111,
			inset 0 1px 1px #afb7b944;
	}
	.physical .unmapped::after {
		content: '';
		position: absolute;
		right: 3px;
		bottom: 3px;
		width: 3px;
		height: 3px;
		border-radius: 50%;
		background: #c89260;
	}
	.legend {
		position: absolute;
		bottom: calc(100% + 5px);
		left: -15%;
		width: 130%;
		color: #c6b77f;
		font:
			500 0.92cqw / 1 Arial,
			sans-serif;
		white-space: nowrap;
	}
	.physical .mode .legend {
		color: #9ac9b1;
	}
	.physical .shift {
		background: linear-gradient(#e9d89b, #b7a668);
		color: #282b28;
	}
	.physical .clear {
		background: linear-gradient(#d46d53, #a54432);
	}
	.physical .mode {
		font-size: 1.05cqw;
	}
	.physical .long {
		font-size: 1.15cqw;
	}
	:global(.device.pc-e500) .physical .mode {
		font-size: 1.55cqw;
	}
	:global(.device.pc-e500) .physical [data-key-id='MENU'],
	:global(.device.pc-e500) .physical [data-key-id='BASIC'] {
		font-size: 1.05cqw;
		color: #e5ffec;
		background: linear-gradient(#70a792, #447963);
	}
	:global(.device.iq-7000) .physical .key {
		border-radius: 7px;
		font-size: 1.65cqw;
	}
	:global(.device.iq-7000) .physical .mode {
		font-size: 0.91cqw;
	}
	:global(.device.iq-7000) .physical .long {
		font-size: 1.1cqw;
	}
	:global(.device.iq-7000) .physical .legend {
		color: #a4d6ed;
		font-size: 0.85cqw;
	}
	:global(.device.iq-7000) .physical .shift {
		color: #8cd3ee;
		background: #273841;
		border-color: #8cd3ee;
	}
	:global(.device.iq-7000) .physical .clear {
		color: #ef8a73;
		background: #252627;
	}
	:global(.device.iq-7000) .physical .mode {
		border-color: #b9b2a0;
		box-shadow:
			0 0 0 2px #252729,
			0 0 0 3px #8e8d81;
	}
</style>
