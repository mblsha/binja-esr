import { describe, expect, it, vi } from 'vitest';
import { TabletInputs } from './tablet_inputs';

describe('buffered pen strokes', () => {
	it('preserves a released stroke before a rapid repress with the same pointer owner', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
		);
		pen.set('LCD:1', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('LCD:1', { raw_x: 10, raw_y: 20, pressed: false });
		pen.set('LCD:1', { raw_x: 30, raw_y: 40, pressed: true });
		pen.set('LCD:1', { raw_x: 30, raw_y: 40, pressed: false });
		expect(apply.mock.calls).toEqual([[{ raw_x: 10, raw_y: 20, pressed: true }]]);
		expect(pen.limitBudget(100)).toBe(2);
		pen.advance(2);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 10, raw_y: 20, pressed: false });
		expect(pen.limitBudget(100)).toBe(2);
		pen.advance(1);
		expect(apply).toHaveBeenCalledTimes(2);
		pen.advance(1);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 30, raw_y: 40, pressed: true });
		expect(pen.limitBudget(100)).toBe(2);
		pen.advance(2);
		expect(apply.mock.calls).toEqual([
			[{ raw_x: 10, raw_y: 20, pressed: true }],
			[{ raw_x: 10, raw_y: 20, pressed: false }],
			[{ raw_x: 30, raw_y: 40, pressed: true }],
			[{ raw_x: 30, raw_y: 40, pressed: false }],
		]);
	});
});

describe('buffered stroke lifecycle', () => {
	it('cancels pending strokes while the ADC is already up between strokes', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
		);
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: false });
		pen.set('one', { raw_x: 30, raw_y: 40, pressed: true });
		pen.advance(2);
		pen.set('one', { raw_x: 30, raw_y: 40, pressed: false }, true);
		pen.advance(100);
		expect(apply).toHaveBeenCalledTimes(2);
		expect(pen.limitBudget(100)).toBe(100);
	});
	it('keeps movement on the physical stroke and ignores a competing pen', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
		);
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('other', { raw_x: 900, raw_y: 900, pressed: true });
		pen.set('one', { raw_x: 11, raw_y: 21, pressed: true });
		expect(apply.mock.calls).toEqual([
			[{ raw_x: 10, raw_y: 20, pressed: true }],
			[{ raw_x: 11, raw_y: 21, pressed: true }],
		]);
		pen.set('one', { raw_x: 11, raw_y: 21, pressed: false });
		pen.set('one', { raw_x: 30, raw_y: 40, pressed: true });
		pen.set('one', { raw_x: 31, raw_y: 41, pressed: true });
		pen.advance(2);
		pen.advance(2);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 31, raw_y: 41, pressed: true });
		pen.advance(2); // A physical drag remains down after its assisted hold expires.
		expect(apply).toHaveBeenCalledTimes(4);
		pen.set('one', { raw_x: 31, raw_y: 41, pressed: false });
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 31, raw_y: 41, pressed: false });
	});
	it.each(['cancel', 'clear', 'discard'])('drops pending contacts and gaps on %s', (mode) => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
		);
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: false });
		pen.set('one', { raw_x: 30, raw_y: 40, pressed: true });
		if (mode === 'cancel') pen.set('one', { raw_x: 30, raw_y: 40, pressed: false }, true);
		else if (mode === 'clear') pen.clear();
		else pen.discard();
		const calls = apply.mock.calls.length;
		pen.advance(100);
		expect(apply).toHaveBeenCalledTimes(calls);
		expect(calls).toBe(mode === 'discard' ? 1 : 2);
		expect(pen.limitBudget(100)).toBe(100);
		pen.set('two', { raw_x: 50, raw_y: 60, pressed: true });
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 50, raw_y: 60, pressed: true });
	});
	it('bounds an unexecuted burst and cancels it on overflow until explicitly cleared', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
		);
		const tap = () => {
			pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
			pen.set('one', { raw_x: 10, raw_y: 20, pressed: false });
		};
		for (let i = 0; i < 128; i++) tap();
		expect(apply).toHaveBeenCalledOnce();
		expect(tap).toThrow('overflow');
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 10, raw_y: 20, pressed: false });
		expect(tap).toThrow('overflow');
		pen.advance(100);
		expect(apply).toHaveBeenCalledTimes(2);
		pen.clear();
		tap();
		expect(apply).toHaveBeenCalledTimes(3);
	});
});

describe('deferred handwriting motion', () => {
	it('replays every distinct point of two paused strokes before either release', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 8,
			() => 2,
			() => 2,
		);
		for (const x of [10, 11, 12]) pen.set('one', { raw_x: x, raw_y: 20, pressed: true });
		pen.set('one', { raw_x: 12, raw_y: 20, pressed: false });
		for (const x of [30, 31, 32]) pen.set('one', { raw_x: x, raw_y: 40, pressed: true });
		pen.set('one', { raw_x: 32, raw_y: 40, pressed: false });
		expect(apply).toHaveBeenCalledOnce();
		let budget = 20;
		while (budget) {
			const n = pen.limitBudget(budget);
			expect(n).toBeGreaterThan(0);
			pen.advance(n);
			budget -= n;
		}
		expect(apply.mock.calls.map(([c]) => [c.raw_x, c.raw_y, c.pressed])).toEqual([
			[10, 20, true],
			[11, 20, true],
			[12, 20, true],
			[12, 20, false],
			[30, 40, true],
			[31, 40, true],
			[32, 40, true],
			[32, 40, false],
		]);
	});
	it('keeps a live drag down after assisted hold and releases only after its final point', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 1,
			() => 2,
			() => 2,
		);
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
		pen.advance(1);
		pen.advance(1);
		pen.set('one', { raw_x: 11, raw_y: 21, pressed: true });
		pen.set('other', { raw_x: 900, raw_y: 900, pressed: true });
		pen.set('one', { raw_x: 12, raw_y: 22, pressed: true });
		pen.set('one', { raw_x: 12, raw_y: 22, pressed: false });
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 11, raw_y: 21, pressed: true });
		pen.advance(2);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 12, raw_y: 22, pressed: true });
		pen.advance(2);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 12, raw_y: 22, pressed: false });
	});
	it.each(['cancel', 'clear', 'discard'])('removes held and pending motion on %s', (mode) => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
			() => 2,
		);
		pen.set('one', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('one', { raw_x: 11, raw_y: 21, pressed: true });
		pen.set('one', { raw_x: 11, raw_y: 21, pressed: false });
		pen.set('one', { raw_x: 30, raw_y: 40, pressed: true });
		pen.set('one', { raw_x: 31, raw_y: 41, pressed: true });
		if (mode === 'cancel') pen.set('one', { raw_x: 31, raw_y: 41, pressed: false }, true);
		else if (mode === 'clear') pen.clear();
		else pen.discard();
		const calls = apply.mock.calls.length;
		pen.advance(100);
		expect(apply).toHaveBeenCalledTimes(calls);
		expect(calls).toBe(mode === 'discard' ? 1 : 2);
		if (mode !== 'discard') expect(apply).toHaveBeenLastCalledWith({ raw_x: 10, raw_y: 20, pressed: false });
		pen.set('new', { raw_x: 50, raw_y: 60, pressed: true });
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 50, raw_y: 60, pressed: true });
	});
	it('bounds an unexecuted path, releases the delivered contact and recovers after clear', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
			undefined,
			undefined,
			() => 2,
			() => 2,
			() => 2,
		);
		const point = (n: number) => pen.set('one', { raw_x: n % 1024, raw_y: Math.floor(n / 1024), pressed: true });
		point(0);
		for (let n = 1; n <= 4096; n++) point(n);
		expect(apply).toHaveBeenCalledOnce();
		expect(() => point(4097)).toThrow('movement buffer overflow');
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 0, raw_y: 0, pressed: false });
		expect(() => point(1)).toThrow('overflow');
		pen.clear();
		point(1);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 1, raw_y: 0, pressed: true });
	});
});
