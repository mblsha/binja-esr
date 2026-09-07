import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import { LCD_COLS, LCD_ROWS } from '../lcd';
import LcdCanvas from './LcdCanvas.svelte';

describe('LcdCanvas', () => {
	it('skips status/layout-only redraws and reuses ImageData until geometry changes', async () => {
		const ctx = {
			createImageData: vi.fn((width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) })),
			putImageData: vi.fn(),
		};
		const spy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as any);
		try {
			const pixels = new Uint8Array([0, 192]);
			const view = render(LcdCanvas, { pixels, cols: 2, rows: 1, pixelFormat: 'gray8' });
			expect(ctx.putImageData).toHaveBeenCalledTimes(1);
			await view.rerender({ pixels, cols: 2, rows: 1, pixelFormat: 'gray8', scale: 8, fit: true });
			expect(ctx.putImageData).toHaveBeenCalledTimes(1);
			await view.rerender({ pixels: new Uint8Array([73, 0]), cols: 2, rows: 1, pixelFormat: 'gray8' });
			expect(ctx.putImageData).toHaveBeenCalledTimes(2);
			expect(ctx.createImageData).toHaveBeenCalledTimes(1);
			expect(Array.from(ctx.putImageData.mock.calls[1][0].data)).toEqual([73, 73, 73, 255, 0, 0, 0, 255]);
			await view.rerender({ pixels: new Uint8Array([0, 0]), cols: 1, rows: 2, pixelFormat: 'gray8' });
			expect(ctx.createImageData).toHaveBeenCalledTimes(2);
		} finally {
			spy.mockRestore();
		}
	});
	it('renders a canvas sized to the LCD dimensions', () => {
		const { container } = render(LcdCanvas, { pixels: null, scale: 2 });
		const canvas = container.querySelector('canvas') as HTMLCanvasElement | null;
		expect(canvas).not.toBeNull();
		expect(canvas?.width).toBe(LCD_COLS);
		expect(canvas?.height).toBe(LCD_ROWS);
	});

	it('supports custom dimensions', () => {
		const { container } = render(LcdCanvas, { pixels: null, cols: 64, rows: 64, scale: 1 });
		const canvas = container.querySelector('canvas') as HTMLCanvasElement | null;
		expect(canvas).not.toBeNull();
		expect(canvas?.width).toBe(64);
		expect(canvas?.height).toBe(64);
	});
});
