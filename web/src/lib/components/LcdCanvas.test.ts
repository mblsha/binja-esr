import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { LCD_COLS, LCD_ROWS } from '../lcd';
import LcdCanvas from './LcdCanvas.svelte';

describe('LcdCanvas', () => {
	it('renders a canvas sized to the LCD dimensions', () => {
		const { container } = render(LcdCanvas, { pixels: null, scale: 2 });
		const canvas = container.querySelector('canvas') as HTMLCanvasElement | null;
		expect(canvas).not.toBeNull();
		expect(canvas?.width).toBe(LCD_COLS);
		expect(canvas?.height).toBe(LCD_ROWS);
	});
});

