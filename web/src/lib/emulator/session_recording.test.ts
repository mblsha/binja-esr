import { describe, expect, it } from 'vitest';
import {
	SessionRecorder,
	decodeRecordedSession,
	encodeRecordedSession,
	replayRecordedSession,
	sessionHash,
	sessionPbm,
	type SessionContext,
	type DeliveredInput,
} from './session_recording';

const context: SessionContext = {
	model: 'pv-f1',
	profile: 'firmware',
	firmwareSha256: 'a'.repeat(64),
	coreIdentity: 'test-core',
	settings: { executionMode: 'interactive', typingCatchUp: false, targetFps: 30, sound: false },
};
describe('delivered-input recordings', () => {
	it('records actual boundary progress including zero yields, preserves checkpoint order across hashing and owns inputs', async () => {
		const initial = new Uint8Array([3, 4]);
		const recorder = new SessionRecorder(context, initial);
		initial.fill(0);
		const pbm = new Uint8Array([5]);
		const pending = recorder.checkpoint(pbm);
		pbm.fill(0);
		recorder.input({ kind: 'on', pressed: true });
		recorder.advance(0);
		recorder.advance(9);
		recorder.input({ kind: 'on', pressed: false });
		await pending;
		await recorder.checkpoint(new Uint8Array([7]));
		const session = await recorder.finish();
		expect(session.initial).toEqual(new Uint8Array([3, 4]));
		expect(session.events.map((e) => [e.at, e.kind])).toEqual([
			[0, 'frame'],
			[0, 'on'],
			[9, 'on'],
			[9, 'frame'],
		]);
		expect(session.events[0]).toMatchObject({ sha256: await sessionHash(new Uint8Array([5])) });
		expect(await decodeRecordedSession(await encodeRecordedSession(session))).toEqual(session);
	});
	it('rejects corruption, unsupported fields, invalid contacts and nonmonotonic boundaries before replay', async () => {
		const recorder = new SessionRecorder(context, new Uint8Array([1]));
		recorder.input({ kind: 'matrix', code: 6, pressed: true });
		await recorder.checkpoint(new Uint8Array([0]));
		const session = await recorder.finish();
		const image = await encodeRecordedSession(session);
		image[image.length - 1] ^= 1;
		await expect(decodeRecordedSession(image)).rejects.toThrow('checksum');
		for (const event of [
			{ kind: 'matrix', code: 2, pressed: true, at: 0 },
			{ kind: 'tablet', raw_x: 1024, raw_y: 0, pressed: true, at: 0 },
			{ kind: 'on', pressed: true, at: 1 },
			{ kind: 'on', pressed: true, at: 0, PC: 5 },
		]) {
			await expect(encodeRecordedSession({ ...session, events: [event as any, session.events[1]] })).rejects.toThrow();
		}
	});
	it.each(['pv-f1', 'oz-9600'] as const)(
		'replays %s through bounded execution and physical adapters with checked frames',
		async (model) => {
			const recorder = new SessionRecorder(
				{ ...context, model, profile: model === 'pv-f1' ? 'firmware' : 'provisional-v1' },
				new Uint8Array([1]),
			);
			const first = new Uint8Array([0, 1]);
			await recorder.checkpoint(first);
			recorder.input({ kind: 'tablet', raw_x: 900, raw_y: 120, pressed: true });
			recorder.advance(7);
			recorder.input({ kind: 'tablet', raw_x: 900, raw_y: 120, pressed: false });
			recorder.advance(4);
			await recorder.checkpoint(new Uint8Array([11, 0]));
			const session = await recorder.finish();
			let position = 0;
			let yields = 0;
			const events: [number, DeliveredInput][] = [];
			const result = await replayRecordedSession(
				{
					input: (e) => events.push([position, e]),
					frame: () => (position ? new Uint8Array([position, 0]) : first),
					run_slice: (n) => {
						const used = ++yields === 1 ? 0 : Math.min(n, 3);
						position += used;
						return {
							reason: used === n ? 'boundary_budget' : 'host_yield',
							progress: { boundary_budget_used: used, instructions_retired: used, timing_units_advanced: used },
						};
					},
				},
				session,
				{ yieldHost: async () => {} },
			);
			expect(result).toEqual({ boundaries: 11, checkpoints: 2 });
			expect(events.map(([at, e]) => [at, e.pressed])).toEqual([
				[0, true],
				[7, false],
			]);
		},
	);
	it('releases held contacts on cancellation and reports a changed checkpoint', async () => {
		const recorder = new SessionRecorder(context, new Uint8Array([1]));
		recorder.input({ kind: 'matrix', code: 6, pressed: true });
		recorder.advance(10);
		await recorder.checkpoint(new Uint8Array([0]));
		const events: DeliveredInput[] = [];
		const abort = new AbortController();
		const target = {
			input: (e: DeliveredInput) => events.push(e),
			frame: () => new Uint8Array([1]),
			run_slice: (n: number) => {
				abort.abort();
				return {
					reason: 'host_yield' as const,
					progress: { boundary_budget_used: Math.min(n, 1), instructions_retired: 1, timing_units_advanced: 1 },
				};
			},
		};
		await expect(
			replayRecordedSession(target, await recorder.finish(), { signal: abort.signal, yieldHost: async () => {} }),
		).rejects.toThrow('cancelled');
		expect(events.map((e) => e.pressed)).toEqual([true, false]);
		const zero = new SessionRecorder(context, new Uint8Array([1]));
		await zero.checkpoint(new Uint8Array([0]));
		await expect(replayRecordedSession(target, await zero.finish())).rejects.toThrow('checkpoint');
	});
	it('packs native matrix pixels without viewport cropping', () => {
		const pixels = new Uint8Array(336 * 292);
		pixels[0] = pixels[pixels.length - 1] = 1;
		const pbm = sessionPbm('pv-f1', pixels);
		expect(new TextDecoder().decode(pbm.subarray(0, 11))).toBe('P4\n336 292\n');
		expect(pbm[11]).toBe(128);
		expect(pbm.at(-1)).toBe(1);
	});
});
