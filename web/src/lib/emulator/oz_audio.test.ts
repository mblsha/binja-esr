import { describe, it, expect, vi } from 'vitest';
import { AudioDelivery, AudioPlayback, MAX_AUDIO_SAMPLES, type AudioPacket } from './oz_audio';

function packet(sequence = 1, first = 0, length = 24): AudioPacket {
	return {
		generation: 2,
		epoch: 1,
		sequence,
		first_sample: first,
		total_samples: first + length,
		sample_rate: 48000,
		dropped_samples: 0,
		samples: new Int16Array(length).fill(8192).buffer,
	};
}
function context() {
	const nodes: any[] = [];
	const filter = { type: '', frequency: { value: 0 }, connect: vi.fn() };
	const ctx = {
		state: 'running',
		currentTime: 0,
		destination: {},
		resume: vi.fn(async () => {}),
		close: vi.fn(async () => {}),
		createBiquadFilter: () => filter,
		createBuffer: (_channels: number, length: number, rate: number) => {
			const data = new Float32Array(length);
			return { length, sampleRate: rate, getChannelData: () => data };
		},
		createBufferSource: () => {
			const node = {
				buffer: null as any,
				onended: null as any,
				connect: vi.fn(),
				disconnect: vi.fn(),
				stop: vi.fn(),
				start: vi.fn(),
			};
			nodes.push(node);
			return node;
		},
	};
	return { ctx: ctx as unknown as AudioContext, raw: ctx, nodes, filter };
}

describe('audio delivery credit independent of LCD presentation', () => {
	it('does not harvest without credit; stale/duplicate credit cannot release a current block', () => {
		const send = vi.fn(),
			take = vi.fn(() => ({ ...packet(), samples: new Int16Array([8192, 0]) }));
		const delivery = new AudioDelivery(send);
		delivery.pump(take);
		expect(take).not.toHaveBeenCalled();
		const r = delivery.reset(2, true);
		delivery.pump(take);
		delivery.pump(take);
		expect(take).toHaveBeenCalledTimes(1);
		const sent = send.mock.calls[0][0];
		delivery.consumed(1, r.epoch, sent.sequence);
		delivery.pump(take);
		expect(take).toHaveBeenCalledTimes(1);
		delivery.consumed(2, r.epoch, sent.sequence);
		delivery.pump(take);
		expect(take).toHaveBeenCalledTimes(2);
		delivery.consumed(2, r.epoch, sent.sequence);
		delivery.pump(take);
		expect(take).toHaveBeenCalledTimes(2);
		delivery.reset(2, false);
		delivery.pump(take);
		expect(take).toHaveBeenCalledTimes(2);
	});
	it('keeps only a bounded current tail and reports exact discarded indices', () => {
		const sent: AudioPacket[] = [];
		const d = new AudioDelivery((p) => sent.push(p));
		d.reset(2, true);
		const source = new Int16Array(MAX_AUDIO_SAMPLES * 3).map((_, i) => i);
		d.pump(() => ({
			sample_rate: 48000,
			first_sample: 100,
			total_samples: 14500,
			dropped_samples: 7,
			samples: source,
		}));
		expect(new Int16Array(sent[0].samples)).toEqual(source.slice(-MAX_AUDIO_SAMPLES));
		expect(sent[0].first_sample).toBe(100 + MAX_AUDIO_SAMPLES * 2);
		expect(d.snapshot().discarded_samples).toBe(MAX_AUDIO_SAMPLES * 2);
		expect(sent[0].dropped_samples).toBe(7);
	});
});

describe('WebAudio playback ownership', () => {
	it('requires a gesture enable, routes normalized PCM through DC removal, and ignores stale packets', async () => {
		const c = context(),
			create = vi.fn(() => c.ctx),
			p = new AudioPlayback(create);
		p.begin(2, 1, true);
		p.push(packet());
		expect(create).not.toHaveBeenCalled();
		await p.enable();
		p.push({ ...packet(), generation: 1 });
		p.push(packet());
		expect(c.nodes).toHaveLength(1);
		expect(c.nodes[0].buffer.getChannelData(0)).toEqual(new Float32Array(24).fill(0.25));
		expect(c.nodes[0].connect).toHaveBeenCalledWith(c.filter);
		expect(c.filter.connect).toHaveBeenCalledWith(c.raw.destination);
		expect(c.filter.frequency.value).toBe(40);
		p.push(packet());
		expect(c.nodes).toHaveLength(1); // Duplicate sequence.
		p.begin(1, 99, true);
		p.push(packet(2, 24));
		expect(c.nodes).toHaveLength(2);
		p.pause();
		expect(c.nodes.every((n) => n.stop.mock.calls.length === 1)).toBe(true);
		p.push(packet(3, 48));
		expect(c.nodes).toHaveLength(2);
		p.close();
		expect(c.raw.close).toHaveBeenCalled();
	});
	it('bounds queued time, drops discontinuous history and cancels immediately on mute', async () => {
		const c = context(),
			p = new AudioPlayback(() => c.ctx);
		await p.enable();
		p.begin(2, 1, true);
		p.push(packet(1, 0, 4800));
		p.push(packet(2, 4800, 4800));
		expect(c.nodes[0].stop).toHaveBeenCalledOnce();
		expect(p.snapshot().queued_seconds).toBeLessThanOrEqual(0.15);
		p.push(packet(3, 19000, 24));
		expect(c.nodes[1].stop).toHaveBeenCalledOnce();
		p.disable();
		expect(c.nodes[2].stop).toHaveBeenCalledOnce();
		expect(p.snapshot().active_sources).toBe(0);
		p.push(packet(4, 19024));
		expect(c.nodes).toHaveLength(3);
	});
	it('does not resurrect audio when an outstanding enable resolves after mute', async () => {
		const c = context();
		let resume!: () => void;
		c.raw.resume = vi.fn(
			() =>
				new Promise<void>((r) => {
					resume = r;
				}),
		);
		const p = new AudioPlayback(() => c.ctx);
		p.begin(2, 1, true);
		const enabling = p.enable();
		p.disable();
		resume();
		expect(await enabling).toBe(false);
		p.push(packet());
		expect(c.nodes).toHaveLength(0);
	});
	it('rejects invalid PCM before changing the playback graph', async () => {
		const c = context(),
			p = new AudioPlayback(() => c.ctx);
		await p.enable();
		p.begin(2, 1, true);
		expect(() => p.push({ ...packet(), sample_rate: 44100 })).toThrow('Invalid');
		expect(() => p.push(packet(1, 0, 4801))).toThrow('Invalid');
		expect(c.nodes).toHaveLength(0);
	});
});
