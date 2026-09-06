import { describe, expect, it } from 'vitest';
import { matrixCodeForKeyEvent, physicalKey, virtualKeysForModel } from './keymap';

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
		expect(matrixCodeForKeyEvent({ code: 'ArrowLeft' } as KeyboardEvent, 'iq-7000')).toBe(0x14);
		expect(matrixCodeForKeyEvent({ code: 'ArrowRight' } as KeyboardEvent, 'iq-7000')).toBe(0x13);
		expect(matrixCodeForKeyEvent({ code: 'ArrowUp' } as KeyboardEvent, 'iq-7000')).toBe(0x1b);
		expect(matrixCodeForKeyEvent({ code: 'ArrowDown' } as KeyboardEvent, 'iq-7000')).toBe(0x0c);
		expect(matrixCodeForKeyEvent({ code: 'Delete' } as KeyboardEvent, 'iq-7000')).toBe(0x0a);
		expect(matrixCodeForKeyEvent({ code: 'Insert' } as KeyboardEvent, 'iq-7000')).toBe(0x12);
		for (const model of ['pc-e500', 'iq-7000'] as const)
			expect(matrixCodeForKeyEvent({ code: 'F12' } as KeyboardEvent, model)).toBe('on');
	});

	it('returns null for unmapped keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'Unidentified' } as KeyboardEvent)).toBeNull();
		expect(matrixCodeForKeyEvent({ code: 'Comma' } as KeyboardEvent, 'iq-7000')).toBeNull();
	});

	it('shares complete letter/digit contacts between native data, browser host keys and buttons', () => {
		for (const model of ['pc-e500', 'iq-7000'] as const) {
			const virtual = virtualKeysForModel(model);
			expect(new Set(virtual.map((key) => key.testId)).size).toBe(virtual.length);
			for (const char of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') {
				const code = physicalKey(model, char);
				expect(code).not.toBeNull();
				expect(code).toBeLessThan(128);
				const host = /[A-Z]/.test(char) ? 'Key' + char : 'Digit' + char;
				expect(matrixCodeForKeyEvent({ code: host } as KeyboardEvent, model)).toBe(code);
				expect(virtual.find((key) => key.label === char)?.code).toBe(code);
			}
		}
		expect(physicalKey('iq-7000', 'A')).toBe(0x1c);
		expect(physicalKey('iq-7000', 'B')).toBe(0x04);
		expect(physicalKey('iq-7000', '2')).toBe(0x29);
		expect(physicalKey('pc-e500', 'A')).toBe(0x03);
	});
});
