import { describe, expect, it, vi } from 'vitest';
import { HostInputs, type InputContact, type InputSource } from './host_inputs';
import { stepBounded } from './bounded_step';

function setup() {
	const sink = vi.fn();
	const inputs = new HostInputs(sink);
	const set = (source: InputSource, down: boolean, hold = 0, contact: InputContact = 0x56, owner = 'key') =>
		inputs.set({ source, owner, contact, down, minimumHold: hold });
	return { sink, inputs, set };
}

describe('host contact ownership', () => {
	it('releases only the last owner, for matrix and ON contacts', () => {
		for (const contact of [0x56, 'on'] as const) {
			const { sink, inputs, set } = setup();
			set('physical', true, 0, contact, 'left-shift');
			set('physical', true, 0, contact, 'right-shift');
			set('virtual', true, 0, contact);
			set('script', true, 0, contact);
			set('physical', false, 0, contact, 'left-shift');
			inputs.releaseSource('physical');
			inputs.releaseSource('script');
			expect(sink.mock.calls).toEqual([[contact, true]]);
			inputs.releaseSource('virtual');
			expect(sink.mock.calls).toEqual([
				[contact, true],
				[contact, false],
			]);
		}
	});

	it('counts the minimum from DOWN, not UP, and cancels stale releases on repress', () => {
		const { sink, inputs, set } = setup();
		set('virtual', true, 40);
		inputs.advance(30);
		set('virtual', false);
		expect(inputs.limitBudget(100)).toBe(10);
		set('virtual', true, 40);
		inputs.advance(20);
		expect(sink).toHaveBeenCalledTimes(1);
		set('virtual', false);
		expect(inputs.limitBudget(100)).toBe(20);
		inputs.advance(20);
		expect(sink.mock.calls).toEqual([
			[0x56, true],
			[0x56, false],
		]);
	});

	it('long holds release immediately, repeated UP never extends a pending hold', () => {
		const { sink, inputs, set } = setup();
		set('virtual', true, 40);
		inputs.advance(80);
		set('virtual', false);
		expect(sink).toHaveBeenCalledTimes(2);
		set('virtual', true, 40);
		set('virtual', false);
		inputs.advance(20);
		set('virtual', false);
		expect(inputs.limitBudget(100)).toBe(20);
	});

	it('raw release and lifecycle cancellation work without executing the guest', () => {
		const { sink, inputs, set } = setup();
		set('physical', true);
		set('physical', false);
		set('virtual', true, 40);
		inputs.set({ source: 'virtual', owner: 'key', contact: 0x56, down: false, cancel: true });
		set('script', true, 0, 'on');
		inputs.clear();
		expect(inputs.snapshot().owners).toEqual([]);
		expect(sink.mock.calls).toEqual([
			[0x56, true],
			[0x56, false],
			[0x56, true],
			[0x56, false],
			['on', true],
			['on', false],
		]);
	});

	it('ignores duplicate DOWN and unknown UP, validates before mutation', () => {
		const { sink, inputs, set } = setup();
		set('virtual', false);
		set('virtual', true, 40);
		inputs.advance(20);
		set('virtual', true, 40);
		set('virtual', false);
		expect(inputs.limitBudget(100)).toBe(20);
		for (const contact of [-1, 128, 1.5, NaN]) expect(() => set('script', true, 0, contact)).toThrow('Physical matrix');
		expect(() => set('virtual', true, 40, 1)).toThrow('old contact');
		expect(() => inputs.advance(21)).toThrow('deadline');
		expect(sink).toHaveBeenCalledTimes(1);
	});

	it('does not forget a contact when the Rust transition fails', () => {
		const { sink, inputs, set } = setup();
		set('script', true);
		sink.mockImplementationOnce(() => {
			throw new Error('poisoned');
		});
		expect(() => inputs.clear()).toThrow('poisoned');
		expect(inputs.snapshot().pressedCodes).toEqual([0x56]);
	});

	it('schedules the same release boundary across host chunk sizes, including inert OFF budgets', async () => {
		for (const chunk of [1, 7, 64, 200_000]) {
			let boundaries = 0;
			const edges: [number, boolean][] = [];
			const inputs = new HostInputs((_contact, down) => edges.push([boundaries, down]));
			inputs.set({ source: 'virtual', owner: 'pointer1', contact: 0x56, down: true, minimumHold: 40 });
			inputs.set({ source: 'virtual', owner: 'pointer1', contact: 0x56, down: false });
			await stepBounded(
				{
					run_slice: (requested) => {
						const used = Math.min(chunk, requested);
						boundaries += used;
						return {
							reason: 'host_yield',
							progress: { boundary_budget_used: used, instructions_retired: 0, timing_units_advanced: 0 },
						};
					},
				},
				100,
				{ limitBudget: inputs.limitBudget, onProgress: inputs.advance, yieldHost: async () => {} },
			);
			expect(edges).toEqual([
				[0, true],
				[40, false],
			]);
			expect(boundaries).toBe(100);
		}
	});
});
