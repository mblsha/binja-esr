import { describe, expect, it } from 'vitest';
import { matrixCodeForKeyEvent } from './keymap';

describe('matrixCodeForKeyEvent', () => {
	it('maps PF keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'F1' } as KeyboardEvent)).toBe(0x56);
		expect(matrixCodeForKeyEvent({ code: 'F2' } as KeyboardEvent)).toBe(0x55);
	});

	it('returns null for unmapped keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'KeyA' } as KeyboardEvent)).toBeNull();
	});
});
