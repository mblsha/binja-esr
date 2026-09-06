<script lang="ts">
	import { onDestroy } from 'svelte';
	import { virtualKeysForModel } from '../keymap';
	import type { InputContact } from '../emulator/host_inputs';
	import type { RomModel } from '../rom_model';

	export let disabled = false;
	export let model: RomModel = 'pc-e500';
	export let onPress: (code: InputContact, owner: string) => void;
	export let onRelease: (code: InputContact, owner: string, cancel: boolean) => void;
	export let onCancelAll: () => void = () => {};
	const held = new Map<string, { code: InputContact; element: HTMLButtonElement; pointerId?: number }>();
	$: keys = virtualKeysForModel(model);
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

<section class="vk" aria-label="Virtual keyboard">
	<div class="grid" role="group" aria-label="Keys">
		{#each keys as key (key.testId)}
			<button
				type="button"
				class="key"
				data-testid={key.testId}
				{disabled}
				on:pointerdown={(event) => pointerDown(key.code, event)}
				on:lostpointercapture={(event) => release(`pointer:${event.pointerId}`, true)}
				on:keydown={(event) => keyDown(key.code, event)}
				on:keyup={keyUp}
				on:blur={() => {
					for (const owner of held.keys()) if (owner.startsWith('key:')) release(owner, true);
				}}
				on:click={(event) => {
					// Assistive activation has no pointer DOWN/UP pair.
					if (event.detail === 0) {
						press(key.code, 'activation', event.currentTarget);
						release('activation');
					}
				}}
			>
				{key.label}
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
	.key:active {
		background: #121722;
	}
</style>
