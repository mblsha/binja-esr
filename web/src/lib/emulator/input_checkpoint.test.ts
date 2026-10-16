import { describe, expect, it } from 'vitest';
import { HostInputs } from './host_inputs';
import { TabletInputs } from './tablet_inputs';
import { PvReadiness } from './pvf1_readiness';

describe('live input checkpoints', () => {
	it('preserves buffered ownership, remaining holds/gaps and cancellation without re-emitting native levels', () => {
		const aEvents: unknown[] = [],
			bEvents: unknown[] = [];
		const a = new HostInputs((...e) => aEvents.push(e)),
			b = new HostInputs((...e) => bEvents.push(e));
		for (const owner of ['first', 'second', 'third']) {
			a.set({ source: 'virtual', owner, contact: 1, down: true, buffered: true });
			a.set({ source: 'virtual', owner, contact: 1, down: false, buffered: true });
		}
		a.advance(123);
		b.restoreCheckpoint(a.checkpoint());
		expect(bEvents).toEqual([]);
		aEvents.length = 0;
		for (const requested of [50, 40_000, 60_000, 40_000, 80_000]) {
			const step = a.limitBudget(requested);
			expect(b.limitBudget(requested)).toBe(step);
			a.advance(step);
			b.advance(step);
			expect(b.checkpoint()).toEqual(a.checkpoint());
			expect(bEvents).toEqual(aEvents);
		}
		a.releaseSource('virtual');
		b.releaseSource('virtual');
		expect(bEvents).toEqual(aEvents);
		const prior = b.checkpoint();
		const invalid = structuredClone(prior);
		invalid.gap = -1;
		expect(() => b.restoreCheckpoint(invalid)).toThrow();
		expect(b.checkpoint()).toEqual(prior);
	});
	it('retains shared pen references, movement cadence, pending strokes and live physical ownership', () => {
		const aEvents: unknown[] = [],
			bEvents: unknown[] = [];
		const make = (events: unknown[]) =>
			new TabletInputs(
				(c) => events.push({ ...c }),
				() => 200_000,
				() => 50_000,
				() => 20_000,
			);
		const a = make(aEvents),
			b = make(bEvents);
		a.set('p1', { raw_x: 100, raw_y: 100, pressed: true });
		a.set('p1', { raw_x: 200, raw_y: 200, pressed: true });
		a.set('p1', { raw_x: 200, raw_y: 200, pressed: false });
		a.set('p2', { raw_x: 300, raw_y: 300, pressed: true });
		a.advance(123);
		b.restoreCheckpoint(a.checkpoint());
		expect(bEvents).toEqual([]);
		aEvents.length = 0;
		// Moving the still-down second pointer must mutate the queued stroke owner.
		for (const item of [a, b]) {
			item.set('p2', { raw_x: 350, raw_y: 350, pressed: true });
			item.set('p2', { raw_x: 350, raw_y: 350, pressed: false });
		}
		for (let i = 0; i < 50; i++) {
			const n = a.limitBudget(20_000);
			expect(b.limitBudget(20_000)).toBe(n);
			a.advance(n);
			b.advance(n);
			expect(b.checkpoint()).toEqual(a.checkpoint());
		}
		expect(bEvents).toEqual(aEvents);
		expect(a.status().strokes).toBe(0);
		const prior = b.checkpoint(),
			invalid = structuredClone(prior);
		invalid.physical = 2;
		expect(() => b.restoreCheckpoint(invalid)).toThrow();
		expect(b.checkpoint()).toEqual(prior);
	});
	it('preserves wake gating before ON has executed', () => {
		const a = new PvReadiness(),
			b = new PvReadiness();
		a.noteOn(true);
		b.restoreCheckpoint(a.checkpoint());
		for (const [power, halt, pc] of [
			['off', false, 0],
			['running', false, 0],
			['halted', true, 0xf0b6c],
		] as const)
			expect(b.observe(power, halt, pc)).toBe(a.observe(power, halt, pc));
		expect(b.status()).toBe('ready');
	});
});
