import { describe, expect, it } from 'vitest';
import { physicalKey } from '../keymap';
import { planPaste, MAX_PASTE_CHARACTERS } from './paste_plan';
import { HostInputs, TYPING_GAP, TYPING_HOLD } from './host_inputs';

describe('qualified paste', () => {
	it('uses actual model-specific contacts, sequential IQ comma, and explicit newline keys', () => {
		for (const model of ['pc-e500', 'iq-7000'] as const) {
			const names =
				model === 'iq-7000'
					? ['A', 'A', 'SHIFT', 'K', 'RETURN', '1', '+', '2']
					: ['A', 'A', ',', 'ENTER', '1', '+', '2'];
			expect(planPaste('Aa,\r\n1+2', model)).toEqual({
				contacts: names.map((name) => physicalKey(model, name)),
				unsupported: [],
				error: null,
			});
		}
	});
	it('rejects the whole plan for unknown characters and oversized text', () => {
		const result = planPaste('A\tB:😀', 'iq-7000');
		expect(result.contacts).toEqual([]);
		expect(result.unsupported.map((entry) => entry.character)).toEqual(['\t', ':', '😀']);
		expect(planPaste('A'.repeat(MAX_PASTE_CHARACTERS + 1), 'pc-e500').error).toContain('limited');
	});
	it('feeds a large paste in bounded chunks and preserves every repeated contact edge', () => {
		const edges: [number, boolean][] = [];
		const inputs = new HostInputs((code, down) => edges.push([code as number, down]));
		const plan = Array.from({ length: 300 }, (_, index) => index % 2);
		inputs.startPaste(plan);
		expect(inputs.pasteStatus()).toEqual({ pending: 300, total: 300 });
		expect(inputs.typingStatus().pending).toBe(128);
		while (inputs.pasteStatus().pending) {
			inputs.advance(inputs.limitBudget(TYPING_HOLD + TYPING_GAP));
			expect(inputs.typingStatus().pending).toBeLessThanOrEqual(128);
		}
		expect(edges).toEqual(
			plan.flatMap((contact) => [
				[contact, true],
				[contact, false],
			]),
		);
		expect(inputs.snapshot().pressedCodes).toEqual([]);
	});
	it('rejects interleaving and invalid plans atomically; ON and cleanup cancel even unsubmitted keys', () => {
		const inputs = new HostInputs(() => {});
		expect(() => inputs.startPaste([0, 128])).toThrow('Invalid');
		expect(inputs.typingStatus().pending).toBe(0);
		inputs.startPaste(Array(300).fill(1));
		expect(() => inputs.set({ source: 'physical', owner: 'A', contact: 2, down: true })).toThrow('Paste in progress');
		expect(inputs.pasteStatus().pending).toBe(300);
		inputs.set({ source: 'physical', owner: 'on', contact: 'on', down: true });
		expect(inputs.pasteStatus().pending).toBe(0);
		expect(inputs.snapshot().pressedCodes).toEqual([]);
		expect(inputs.snapshot().onHeld).toBe(true);
		inputs.releaseSource('physical');
		inputs.startPaste(Array(300).fill(1));
		inputs.releaseSource('physical');
		inputs.advance(200_000);
		expect(inputs.pasteStatus().pending).toBe(0);
		expect(inputs.snapshot().pressedCodes).toEqual([]);
	});
});
