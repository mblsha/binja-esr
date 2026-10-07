import { describe, expect, it } from 'vitest';
import { deviceLayout, rectStyle } from './device_layout';
import { physicalKey, virtualKeysForModel } from './keymap';

describe('reference-based device layouts', () => {
	for (const model of ['pc-e500', 'iq-7000', 'oz-9600'] as const) {
		it(`${model}: preserves every supported key exactly once and never guesses an unmapped contact`, () => {
			const layout = deviceLayout(model);
			expect(layout.provenance).toBe('photo-reference-approximation');
			expect(new Set(layout.keys.map((key) => key.id)).size).toBe(layout.keys.length);
			expect(new Set(layout.keys.map((key) => key.testId)).size).toBe(layout.keys.length);
			for (const key of layout.keys) {
				expect(key.code).toBe(key.id === 'ON' ? 'on' : physicalKey(model, key.id));
				expect(key.x).toBeGreaterThanOrEqual(0);
				expect(key.y).toBeGreaterThanOrEqual(0);
				expect(key.x + key.w).toBeLessThanOrEqual(layout.width);
				expect(key.y + key.h).toBeLessThanOrEqual(layout.height);
				for (const other of layout.keys.filter((other) => other.id !== key.id)) {
					const overlap =
						key.x < other.x + other.w &&
						key.x + key.w > other.x &&
						key.y < other.y + other.h &&
						key.y + key.h > other.y;
					expect(overlap, `${key.id} overlaps ${other.id}`).toBe(false);
				}
			}
			for (const key of virtualKeysForModel(model)) {
				expect(layout.keys.filter((candidate) => candidate.testId === key.testId)).toHaveLength(1);
			}
			expect(layout.keys.find((key) => key.id === 'OFF')?.code).toBe(model === 'oz-9600' ? 1 : null);
		});
	}
	it('keeps the original alphabet groupings and IQ screen left of keyboard', () => {
		const iq = deviceLayout('iq-7000');
		const pc = deviceLayout('pc-e500');
		expect(iq.keys.filter((key) => key.y === iq.keys.find((key) => key.id === 'A')!.y).map((key) => key.id)).toEqual([
			'A',
			'B',
			'C',
			'D',
			'E',
			'F',
		]);
		expect(
			pc.keys
				.filter((key) => /^[A-Z]$/.test(key.id) && key.y === pc.keys.find((key) => key.id === 'Q')!.y)
				.map((key) => key.id),
		).toEqual('QWERTYUIOP'.split(''));
		expect(iq.lcd.x + iq.lcd.w).toBeLessThan(Math.min(...iq.keys.map((key) => key.x)));
		expect(rectStyle({ x: 10, y: 20, w: 30, h: 40 }, { width: 100, height: 100 })).toBe(
			'left:10%;top:20%;width:30%;height:40%;',
		);
	});
});
