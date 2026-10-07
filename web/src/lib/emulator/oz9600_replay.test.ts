import { describe, expect, it, vi } from 'vitest';
import { runOzPhysicalReplay, ozPbm } from './oz9600_replay';
import { TabletInputs } from './tablet_inputs';

function emulator() {
	return {
		validate_oz9600_replay: vi.fn(),
		press_matrix_code: vi.fn(),
		release_matrix_code: vi.fn(),
		press_on_key: vi.fn(),
		release_on_key: vi.fn(),
		set_oz9600_tablet_contact: vi.fn(),
		oz9600_state: vi.fn(() => ({ pc: 123 })),
		run_slice: vi.fn((n: number) => ({
			reason: 'boundary_budget' as const,
			progress: { boundary_budget_used: Math.min(n, 2), instructions_retired: 1, timing_units_advanced: 3 },
		})),
	};
}
describe('OZ physical replay through shared bounded stepping', () => {
	it('validates before any input or CPU mutation', async () => {
		const e = emulator();
		e.validate_oz9600_replay.mockImplementation(() => {
			throw new Error('invalid contact');
		});
		await expect(runOzPhysicalReplay(e, 'invalid')).rejects.toThrow('invalid contact');
		expect(e.press_matrix_code).not.toHaveBeenCalled();
		expect(e.run_slice).not.toHaveBeenCalled();
	});
	it('observes completed boundaries and uses column/row contacts', async () => {
		const e = emulator();
		const onStep = vi.fn(async (_index: number, _state: object) => {});
		const document = JSON.stringify({
			steps: [
				{ boundaries: 5, contact: { column: 4, row: 6, pressed: true } },
				{ boundaries: 1, contact: { column: 4, row: 6, pressed: false } },
			],
		});
		expect(await runOzPhysicalReplay(e, document, { onStep, yieldHost: async () => {} })).toBe(6);
		expect(e.press_matrix_code).toHaveBeenCalledWith(38);
		expect(e.release_matrix_code).toHaveBeenCalledWith(38);
		expect(e.run_slice.mock.calls.map(([n]) => n)).toEqual([5, 3, 1, 1]);
		expect(onStep.mock.calls.map(([n]) => n)).toEqual([0, 1]);
	});
	it('Stop releases physical contacts without executing remaining boundaries', async () => {
		const e = emulator();
		const abort = new AbortController();
		const document = JSON.stringify({
			steps: [
				{
					boundaries: 50,
					on_key: true,
					contact: { column: 4, row: 6, pressed: true },
					tablet: { raw_x: 61, raw_y: 288, pressed: true },
				},
			],
		});
		await expect(
			runOzPhysicalReplay(e, document, { signal: abort.signal, yieldHost: async () => abort.abort() }),
		).rejects.toThrow('cancelled');
		expect(e.run_slice).toHaveBeenCalledTimes(1);
		expect(e.release_matrix_code).toHaveBeenCalledWith(38);
		expect(e.press_on_key).toHaveBeenCalledOnce();
		expect(e.release_on_key).toHaveBeenCalledOnce();
		expect(e.set_oz9600_tablet_contact).toHaveBeenLastCalledWith(61, 288, false);
	});
	it('exports full PBM with the right first and final controller pixels', () => {
		const pixels = new Uint8Array(336 * 240).fill(255);
		pixels[0] = 0;
		pixels[pixels.length - 1] = 0;
		const pbm = ozPbm({ cols: 336, rows: 240, pixels, pixel_format: 'gray8', pixel_scale: 1 });
		const offset = new TextEncoder().encode('P4\n336 240\n').length;
		expect(pbm.length).toBe(offset + 10080);
		expect(pbm[offset]).toBe(128);
		expect(pbm[pbm.length - 1]).toBe(1);
	});
});
describe('one pen with assisted release', () => {
	it('holds short taps until their scheduler deadline, never wall time', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(apply);
		pen.set('p1', { raw_x: 61, raw_y: 288, pressed: true });
		pen.set('p1', { raw_x: 61, raw_y: 288, pressed: false });
		expect(pen.limitBudget(100_000)).toBe(40_000);
		pen.advance(39_999);
		expect(apply).toHaveBeenCalledTimes(1);
		pen.advance(1);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 61, raw_y: 288, pressed: false });
	});
	it('ignores competing pointers, cancels immediately, discards only on fresh core replacement', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(apply);
		pen.set('p1', { raw_x: 10, raw_y: 20, pressed: true });
		pen.set('p2', { raw_x: 30, raw_y: 40, pressed: true });
		expect(apply).toHaveBeenCalledTimes(1);
		pen.set('p1', { raw_x: 10, raw_y: 20, pressed: false }, true);
		expect(apply).toHaveBeenLastCalledWith({ raw_x: 10, raw_y: 20, pressed: false });
		pen.set('p3', { raw_x: 10, raw_y: 20, pressed: true });
		pen.discard();
		expect(pen.limitBudget(100_000)).toBe(100_000);
		expect(apply).toHaveBeenCalledTimes(3);
	});
});
