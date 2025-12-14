<script lang="ts">
	import { afterUpdate } from 'svelte';
	import { LCD_COLS, LCD_ROWS, pixelsToRgba } from '$lib/lcd';

	export let pixels: Uint8Array | null = null;
	export let scale = 4;

	let canvas: HTMLCanvasElement | null = null;

	function draw(p: Uint8Array) {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const rgba = pixelsToRgba(p, LCD_COLS, LCD_ROWS, [20, 255, 150, 255], [0, 0, 0, 255]);
		const image = ctx.createImageData(LCD_COLS, LCD_ROWS);
		image.data.set(rgba);
		ctx.putImageData(image, 0, 0);
	}

	afterUpdate(() => {
		if (pixels) draw(pixels);
	});
</script>

<canvas
	bind:this={canvas}
	width={LCD_COLS}
	height={LCD_ROWS}
	style={`width:${LCD_COLS * scale}px;height:${LCD_ROWS * scale}px;image-rendering:pixelated;`}
></canvas>
