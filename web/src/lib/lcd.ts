export const LCD_ROWS = 32;
export const LCD_COLS = 240;
export const LCD_CHIP_ROWS = 64;
export const LCD_CHIP_COLS = 64;

export type Rgba = readonly [number, number, number, number];

export function pixelsToRgba(
	pixels: Uint8Array,
	cols = LCD_COLS,
	rows = LCD_ROWS,
	on: Rgba = [255, 255, 255, 255],
	off: Rgba = [0, 0, 0, 255]
): Uint8ClampedArray {
	if (pixels.length !== rows * cols) {
		throw new Error(`expected ${rows * cols} pixels, got ${pixels.length}`);
	}

	const out = new Uint8ClampedArray(rows * cols * 4);
	for (let i = 0; i < pixels.length; i++) {
		const src = pixels[i] ? on : off;
		const dst = i * 4;
		out[dst] = src[0];
		out[dst + 1] = src[1];
		out[dst + 2] = src[2];
		out[dst + 3] = src[3];
	}
	return out;
}
