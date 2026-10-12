import { describe, expect, it, vi } from 'vitest';
import { TabletInputs } from './tablet_inputs';

describe('buffered pen strokes', () => {
	it('preserves a released stroke before a rapid repress with the same pointer owner', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(
			apply,
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
