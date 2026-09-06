import { describe, expect, it } from 'vitest';
import { matrixCodeForKeyEvent } from './keymap';

describe('matrixCodeForKeyEvent', () => {
	it('maps PF keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'F1' } as KeyboardEvent)).toBe(0x56);
		expect(matrixCodeForKeyEvent({ code: 'F2' } as KeyboardEvent)).toBe(0x55);
	});
	it('uses model-specific physical contacts rather than translated IQ events', () => {
		expect(matrixCodeForKeyEvent({ code: 'ArrowLeft' } as KeyboardEvent)).toBe(0x1f);
		expect(matrixCodeForKeyEvent({ code: 'Enter' } as KeyboardEvent)).toBe(0x27);
		expect(matrixCodeForKeyEvent({ code: 'F4' } as KeyboardEvent, 'iq-7000')).toBe(0x08);
		expect(matrixCodeForKeyEvent({ code: 'CapsLock' } as KeyboardEvent, 'iq-7000')).toBe(0x24);
		expect(matrixCodeForKeyEvent({ code: 'ArrowLeft' } as KeyboardEvent, 'iq-7000')).toBeNull();
		for (const model of ['pc-e500', 'iq-7000'] as const)
			expect(matrixCodeForKeyEvent({ code: 'F12' } as KeyboardEvent, model)).toBe('on');
	});

	it('returns null for unmapped keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'KeyA' } as KeyboardEvent)).toBeNull();
	});
});
