import { describe, expect, it, vi } from 'vitest';
import { LatestFrame } from './latest_frame';

function setup() {
	const send = vi.fn();
	const onError = vi.fn();
	const jobs: (() => void)[] = [];
	const frames = new LatestFrame<number>(send, onError, (job) => {
		jobs.push(job);
	});
	const flush = () => {
		while (jobs.length) jobs.shift()!();
	};
	return { send, onError, jobs, frames, flush };
}

describe('latest-only frame delivery', () => {
	it('can discard deferred work on a machine fault without creating phantom credit', () => {
		const { frames, send, flush } = setup();
		const stale = vi.fn(() => 2);
		frames.request(() => 1);
		flush();
		frames.request(stale);
		frames.discardPending();
		frames.consumed(1);
		flush();
		expect(stale).not.toHaveBeenCalled();
		expect(send).toHaveBeenCalledTimes(1);
		frames.request(() => 3);
		flush();
		expect(send).toHaveBeenLastCalledWith(3, 2);
	});
	it('captures lazily, coalesces before capture, and holds at most one frame in flight', () => {
		const { send, frames, flush, jobs } = setup();
		const capture = vi.fn(() => 1);
		frames.request(capture);
		expect(capture).not.toHaveBeenCalled();
		frames.request(() => 2);
		expect(jobs).toHaveLength(1);
		flush();
		expect(capture).not.toHaveBeenCalled();
		expect(send.mock.calls).toEqual([[2, 1]]);
		for (let i = 0; i < 10_000; i++) frames.request(() => i);
		flush();
		expect(send).toHaveBeenCalledTimes(1);
		frames.consumed(1);
		flush();
		expect(send.mock.calls).toEqual([
			[2, 1],
			[9999, 2],
		]);
	});

	it('captures current state after credit, not the state when a refresh was requested', () => {
		const { frames, send, flush } = setup();
		let state = 1;
		frames.request(() => state);
		flush();
		frames.request(() => state);
		state = 2;
		frames.consumed(1);
		flush();
		expect(send.mock.calls).toEqual([
			[1, 1],
			[2, 2],
		]);
	});

	it('ignores stale, duplicate, malformed and unsolicited acknowledgements', () => {
		const { frames, send, flush } = setup();
		frames.request(() => 1);
		flush();
		frames.request(() => 2);
		for (const bad of [0, 2, -1, NaN, Infinity]) frames.consumed(bad);
		flush();
		expect(send).toHaveBeenCalledTimes(1);
		frames.consumed(1);
		frames.consumed(1);
		flush();
		frames.request(() => 3);
		frames.consumed(1);
		flush();
		expect(send).toHaveBeenCalledTimes(2);
		frames.consumed(2);
		flush();
		expect(send).toHaveBeenCalledTimes(3);
	});

	it('reports capture/transport failures without freezing future refreshes or leaking credit', () => {
		const { frames, send, onError, flush } = setup();
		frames.request(() => {
			throw new Error('capture');
		});
		flush();
		expect(onError).toHaveBeenCalledTimes(1);
		expect(frames.snapshot().inFlight).toBeNull();
		send.mockImplementationOnce(() => {
			throw new Error('transport');
		});
		frames.request(() => 2);
		flush();
		expect(onError).toHaveBeenCalledTimes(2);
		frames.request(() => 3);
		flush();
		expect(send).toHaveBeenLastCalledWith(3, 2);
		expect(frames.snapshot().failures).toBe(2);
	});
});
