import { expect, it } from 'vitest';
import { PvReadiness } from './pvf1_readiness';
import { HostInputs } from './host_inputs';
import { TabletInputs } from './tablet_inputs';

it('admits calibration, observes native input wait, latches ordinary IRQs and queues only during wake', () => {
	const r = new PvReadiness();
	expect(r.canDeliver()).toBe(true);
	expect(r.observe('on', true, 0xf0b54)).toBe('starting');
	expect(r.observe('on', true, 0xf0b6c)).toBe('ready');
	expect(r.observe('on', false, 0xf19a7)).toBe('ready');
	expect(r.observe('off', true, 0xfabcc)).toBe('off');
	r.noteOn(true);
	expect(r.observe('off', true, 0xfabcc)).toBe('waking');
	expect(r.status()).toBe('waking');
	expect(r.canDeliver()).toBe(false);
	expect(r.observe('on', false, 0xf6e68)).toBe('waking');
	expect(r.observe('on', true, 0xf0b6c)).toBe('ready');
	r.reset();
	expect(r.status()).toBe('starting');
});
it('does not consume holds while waiting; preserves queued matrix and complete pen stroke after readiness', () => {
	const r = new PvReadiness();
	r.noteOn(true);
	const keys: any[] = [],
		pen: any[] = [];
	const k = new HostInputs(
		(...e) => keys.push(e),
		() => r.canDeliver(),
	);
	const p = new TabletInputs(
		(e) => pen.push(e),
		() => 20,
		() => 5,
		() => 2,
		() => r.canDeliver(),
	);
	for (const down of [true, false, true, false])
		k.set({ source: 'virtual', owner: 'same', contact: 1, down, buffered: true, minimumHold: 20, minimumGap: 5 });
	p.set('pen', { raw_x: 1, raw_y: 2, pressed: true });
	p.set('pen', { raw_x: 3, raw_y: 4, pressed: true });
	p.set('pen', { raw_x: 3, raw_y: 4, pressed: false });
	k.advance(100);
	p.advance(100);
	expect(keys).toEqual([]);
	expect(pen).toEqual([]);
	expect(p.status()).toEqual({ strokes: 1, points: 1, blocked: false });
	r.observe('on', true, 0xf0b6c);
	k.advance(0);
	p.advance(0);
	expect(keys).toEqual([[1, true]]);
	expect(pen[0]).toEqual({ raw_x: 1, raw_y: 2, pressed: true });
	p.advance(2);
	expect(pen[1]).toEqual({ raw_x: 3, raw_y: 4, pressed: true });
	p.advance(2);
	p.advance(16);
	expect(pen.at(-1).pressed).toBe(false);
	k.advance(20);
	k.advance(5);
	k.advance(20);
	expect(keys).toEqual([
		[1, true],
		[1, false],
		[1, true],
		[1, false],
	]);
});
it('cancels waiting input immediately without executing or releasing an undelivered pen', () => {
	const r = new PvReadiness();
	r.noteOn(true);
	const events: any[] = [];
	const p = new TabletInputs(
		(e) => events.push(e),
		() => 20,
		() => 5,
		() => 2,
		() => r.canDeliver(),
	);
	p.set('pen', { raw_x: 1, raw_y: 2, pressed: true });
	p.set('pen', { raw_x: 1, raw_y: 2, pressed: false });
	p.clear();
	r.observe('on', true, 0xf0b6c);
	p.advance(100);
	expect(events).toEqual([]);
	expect(p.status().strokes).toBe(0);
});
