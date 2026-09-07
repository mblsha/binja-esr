import { describe, expect, it } from 'vitest';
import { LCD_COLS, LCD_ROWS, pixelsToRgba, grayscaleToRgba } from './lcd';

describe('pixelsToRgba', () => {
	it('reuses supplied RGBA storage and validates its geometry', () => {
		const destination = new Uint8ClampedArray(8);
		expect(grayscaleToRgba(new Uint8Array([73, 192]), 2, 1, destination)).toBe(destination);
		expect(Array.from(destination)).toEqual([73, 73, 73, 255, 192, 192, 192, 255]);
		expect(() => grayscaleToRgba(new Uint8Array(2), 2, 1, new Uint8ClampedArray(4))).toThrow('RGBA geometry');
	});
	it('preserves grayscale antialiasing and opaque LCD background', () => {
		expect(Array.from(grayscaleToRgba(new Uint8Array([0, 73, 192]), 3, 1))).toEqual([
			0, 0, 0, 255, 73, 73, 73, 255, 192, 192, 192, 255,
		]);
		expect(() => grayscaleToRgba(new Uint8Array(1), 2, 1)).toThrow('geometry');
	});
	it('keeps inactive fixed segments faint instead of treating them as on', () => {
		const rgba = pixelsToRgba(new Uint8Array([0, 1, 2, 3]), 4, 1, [200, 200, 200, 255]);
		expect(Array.from(rgba.filter((_, i) => i % 4 === 0))).toEqual([0, 200, 100, 26]);
		expect(Array.from(rgba.filter((_, i) => i % 4 === 3))).toEqual([255, 255, 255, 255]);
	});
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
