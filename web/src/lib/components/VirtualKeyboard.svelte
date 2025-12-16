<script lang="ts">
	type VirtualKey = {
		label: string;
		code: number;
		testId: string;
	};

	export let disabled = false;
	export let onPress: (code: number) => void;
	export let onRelease: (code: number) => void;

	const keys: VirtualKey[] = [
		{ label: 'PF1', code: 0x56, testId: 'vk-pf1' },
		{ label: 'PF2', code: 0x55, testId: 'vk-pf2' },
		{ label: '↑', code: 0x1e, testId: 'vk-up' },
		{ label: '←', code: 0x27, testId: 'vk-left' },
		{ label: '→', code: 0x26, testId: 'vk-right' },
		{ label: '↓', code: 0x17, testId: 'vk-down' },
	];

	function press(code: number, event: Event) {
		event.preventDefault();
		if (disabled) return;
		onPress?.(code);
	}

	function release(code: number, event: Event) {
		event.preventDefault();
		if (disabled) return;
		onRelease?.(code);
	}
</script>

<section class="vk" aria-label="Virtual keyboard">
	<div class="grid" role="group" aria-label="Keys">
		{#each keys as key (key.testId)}
			<button
				type="button"
				class="key"
				data-testid={key.testId}
				{disabled}
				on:pointerdown={(e) => press(key.code, e)}
				on:pointerup={(e) => release(key.code, e)}
				on:pointerleave={(e) => release(key.code, e)}
				on:pointercancel={(e) => release(key.code, e)}
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
