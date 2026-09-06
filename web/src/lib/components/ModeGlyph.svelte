<script lang="ts">
	export let name: string;
</script>

<!-- Original vector approximations of the IQ mode-key pictograms, not scans. -->
<svg viewBox="0 0 24 20" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">
	{#if name === 'CALENDAR'}
		<rect x="3" y="3" width="18" height="14" rx="2" /><path
			d="M3 7h18M7 1v4m5-4v4m5-4v4M7 10h1m3 0h1m3 0h1m-9 3h1m3 0h1m3 0h1"
		/>
	{:else if name === 'SCHEDULE'}
		<path d="M2 3q5-2 10 0 5-2 10 0v14q-5-2-10 0-5-2-10 0zM12 3v14M5 6h4M5 9h4M15 6h4M15 9h4" />
	{:else if name === 'TEL'}
		<path d="M4 8V5q8-5 16 0v3h-5V6H9v2zM8 9h8l4 8H4z" /><circle cx="12" cy="12" r="2" />
	{:else if name === 'MEMO'}
		<path d="M14 3H3v14h17V8M8 13l2-5 8-7 3 3-8 8zM16 3l3 3" />
	{:else if name === 'CALC'}
		<rect x="5" y="1" width="14" height="18" rx="2" /><path d="M8 4h8v4H8zM8 11h2m4 0h2m-8 4h2m4 0h2" />
	{:else if name === 'CARD'}
		<rect x="4" y="1" width="16" height="18" rx="2" /><path d="M7 4h10v12H7zM9 7h6M9 10h6M9 13h6" />
	{:else if name === 'WORLD'}
		<circle cx="12" cy="10" r="9" /><ellipse cx="12" cy="10" rx="4" ry="9" /><path d="M3 10h18M5 5h14M5 15h14" />
	{:else if name === 'HOME'}
		<circle cx="12" cy="10" r="9" /><path d="M12 3v7h6" />
	{/if}
</svg>

<style>
	svg {
		width: 62%;
		height: 78%;
		vertical-align: middle;
	}
</style>
