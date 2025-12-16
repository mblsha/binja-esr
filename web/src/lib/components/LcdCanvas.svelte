<script lang="ts">
	import { afterUpdate } from 'svelte';
	import { LCD_COLS, LCD_ROWS, pixelsToRgba } from '$lib/lcd';

	export let pixels: Uint8Array | null = null;
	export let cols = LCD_COLS;
	export let rows = LCD_ROWS;
	export let scale = 4;

	let canvas: HTMLCanvasElement | null = null;

	function draw(p: Uint8Array) {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const rgba = pixelsToRgba(p, cols, rows, [20, 255, 150, 255], [0, 0, 0, 255]);
		const image = ctx.createImageData(cols, rows);
		image.data.set(rgba);
		ctx.putImageData(image, 0, 0);
	}

	afterUpdate(() => {
		if (pixels) draw(pixels);
	});
</script>

<canvas
	bind:this={canvas}
	width={cols}
	height={rows}
	style={`width:${cols * scale}px;height:${rows * scale}px;image-rendering:pixelated;`}
></canvas>
