import { describe, expect, it } from 'vitest';
import { contactsForKeyEvent, hostKeyHints, matrixCodeForKeyEvent, physicalKey, virtualKeysForModel } from './keymap';

describe('host letters and symbols', () => {
	const event = (code: string, key: string, shiftKey = false) => new KeyboardEvent('keydown', { code, key, shiftKey });
	for (const model of ['pc-e500', 'iq-7000'] as const) {
		it(`${model}: follows host layout and shifted symbols without guest Shift`, () => {
			for (const [code, key] of [
				['KeyQ', 'a'],
				['KeyY', 'z'],
				['Digit1', '1'],
				['Equal', '+'],
				['Digit8', '*'],
				['Slash', '/'],
				['NumpadDecimal', '.'],
			])
				expect(contactsForKeyEvent(event(code, key, true), model, 'symbols')).toEqual([
					physicalKey(model, key.toUpperCase()),
				]);
			expect(contactsForKeyEvent(event('ShiftLeft', 'Shift'), model, 'symbols')).toEqual([]);
			expect(contactsForKeyEvent(event('F9', 'F9'), model, 'symbols')).toEqual([physicalKey(model, 'SHIFT')]);
			expect(contactsForKeyEvent(event('Digit1', '!'), model, 'symbols')).toEqual([]);
			expect(contactsForKeyEvent(event('Quote', 'Dead'), model, 'symbols')).toEqual([]);
			expect(contactsForKeyEvent(event('KeyA', 'é'), model, 'symbols')).toEqual([]);
		});
		it(`${model}: retains positional device-key mapping in raw mode`, () => {
			expect(contactsForKeyEvent(event('ShiftLeft', 'Shift'), model, 'keycaps')).toEqual([physicalKey(model, 'SHIFT')]);
			expect(contactsForKeyEvent(event('KeyQ', 'a'), model, 'keycaps')).toEqual([physicalKey(model, 'Q')]);
			expect(contactsForKeyEvent(event('Equal', '+', true), model, 'keycaps')).toEqual([physicalKey(model, '=')]);
		});
		it(`${model}: respects keypad navigation with Num Lock off`, () => {
			expect(contactsForKeyEvent(event('Numpad2', 'ArrowDown'), model, 'symbols')).toEqual([
				physicalKey(model, 'DOWN'),
			]);
			expect(contactsForKeyEvent(event('Numpad0', 'Insert'), model, 'symbols')).toEqual([physicalKey(model, 'INS')]);
			expect(contactsForKeyEvent(event('NumpadEnter', 'Enter'), model, 'symbols')).toEqual([physicalKey(model, '=')]);
		});
	}
	it('rejects unqualified IQ punctuation and offers laptop newline', () => {
		expect(contactsForKeyEvent(event('Comma', ','), 'iq-7000', 'symbols')).toEqual([]);
		expect(contactsForKeyEvent(event('Semicolon', ':', true), 'iq-7000', 'symbols')).toEqual([]);
		expect(contactsForKeyEvent(event('Enter', 'Enter', true), 'iq-7000', 'symbols')).toEqual([0x3d]);
		expect(contactsForKeyEvent(event('Enter', 'Enter'), 'iq-7000', 'symbols')).toEqual([0x45]);
	});
	it('maps PC parentheses and provides a browser-safe CTRL alias', () => {
		expect(contactsForKeyEvent(event('Digit9', '(', true), 'pc-e500', 'symbols')).toEqual([0x4b]);
		expect(contactsForKeyEvent(event('Digit0', ')', true), 'pc-e500', 'symbols')).toEqual([0x48]);
		expect(contactsForKeyEvent(event('F11', 'F11'), 'pc-e500', 'symbols')).toEqual([0x07]);
	});
	it('explains usable host shortcuts without claiming host Control or raw Shift in symbol mode', () => {
		expect(hostKeyHints(0x06, 'pc-e500', 'symbols')).toBe('F9');
		expect(hostKeyHints(0x06, 'pc-e500', 'keycaps')).toBe('Shift / F9');
		expect(hostKeyHints(0x07, 'pc-e500', 'symbols')).toBe('F11');
		expect(hostKeyHints(0x1f, 'pc-e500', 'symbols')).toBe('ArrowLeft');
		expect(hostKeyHints('on', 'iq-7000', 'symbols')).toBe('F12');
	});
});

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

	it('uses OZ matrix contacts and a separate ON input without inheriting PC mode keys', () => {
		expect(physicalKey('oz-9600', 'SPACE')).toBe(38);
		expect(physicalKey('oz-9600', 'ENTER')).toBe(78);
		expect(physicalKey('oz-9600', 'Q')).toBe(3);
		expect(physicalKey('oz-9600', 'CALENDAR')).toBeNull();
		expect(physicalKey('oz-9600', 'OFF')).toBe(1);
		expect(matrixCodeForKeyEvent({ code: 'F12' } as KeyboardEvent, 'oz-9600')).toBe('on');
		const expected = {
			F1: 48,
			F2: 49,
			F9: 5,
			F10: 6,
			F11: 86,
			Escape: 9,
			PageUp: 72,
			PageDown: 81,
			NumpadEnter: 78,
			ArrowLeft: 56,
			ArrowDown: 57,
			ArrowUp: 64,
			ArrowRight: 65,
		};
		for (const [code, value] of Object.entries(expected))
			expect(matrixCodeForKeyEvent({ code } as KeyboardEvent, 'oz-9600')).toBe(value);
		for (const code of ['F3', 'F4', 'F5', 'F6', 'F7', 'F8'])
			expect(matrixCodeForKeyEvent({ code } as KeyboardEvent, 'oz-9600')).toBeNull();
		expect(virtualKeysForModel('oz-9600').filter((k) => k.code === 'on')).toHaveLength(1);
	});
	it('returns null for unmapped keys', () => {
		expect(matrixCodeForKeyEvent({ code: 'Unidentified' } as KeyboardEvent)).toBeNull();
		expect(matrixCodeForKeyEvent({ code: 'Comma' } as KeyboardEvent, 'iq-7000')).toBeNull();
	});

	it('shares complete letter/digit contacts between native data, browser host keys and buttons', () => {
		for (const model of ['pc-e500', 'iq-7000', 'oz-9600'] as const) {
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
