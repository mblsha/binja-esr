import { describe, expect, it } from 'vitest';
import { LCD_COLS, LCD_ROWS, pixelsToRgba } from './lcd';

describe('pixelsToRgba', () => {
	it('throws on wrong length', () => {
		expect(() => pixelsToRgba(new Uint8Array([0, 1, 2]))).toThrow(/expected/i);
	});

	it('maps off/on pixels to RGBA', () => {
		const pixels = new Uint8Array(LCD_COLS * LCD_ROWS);
		pixels[0] = 0;
		pixels[1] = 1;

		const rgba = pixelsToRgba(pixels, LCD_COLS, LCD_ROWS, [10, 20, 30, 40], [1, 2, 3, 4]);

		expect(rgba.slice(0, 4)).toEqual(new Uint8ClampedArray([1, 2, 3, 4]));
		expect(rgba.slice(4, 8)).toEqual(new Uint8ClampedArray([10, 20, 30, 40]));
	});
});

