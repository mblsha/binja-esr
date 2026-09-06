import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WorkerRequests } from './worker_requests';

describe('worker RPC lifecycle', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('settles replies and removes their timeouts', async () => {
		const post = vi.fn();
		const timeout = vi.fn();
		const requests = new WorkerRequests(post, timeout);
		const promise = requests.request({ id: 1, type: 'stop' });
		expect(requests.pendingCount).toBe(1);
		requests.reply({ id: 1, ok: true, result: 42 });
		expect(await promise).toBe(42);
		expect(requests.pendingCount).toBe(0);
		await vi.runAllTimersAsync();
		expect(timeout).not.toHaveBeenCalled();
	});

	it('times out without claiming stop succeeded; late replies cannot settle a newer request', async () => {
		const timeout = vi.fn();
		const requests = new WorkerRequests(vi.fn(), timeout);
		const old = requests.request({ id: 1, type: 'stop' }, undefined, 100).catch((error) => error);
		await vi.advanceTimersByTimeAsync(100);
		await expect(old).resolves.toMatchObject({ message: expect.stringContaining('execution state is unconfirmed') });
		expect(requests.pendingCount).toBe(0);
		expect(timeout).toHaveBeenCalledTimes(1);
		const retry = requests.request({ id: 2, type: 'stop' });
		requests.reply({ id: 1, ok: true });
		expect(requests.pendingCount).toBe(1);
		requests.reply({ id: 2, ok: true });
		await retry;
	});

	it('fails every pending and future request when the worker fails or is destroyed', async () => {
		const requests = new WorkerRequests(vi.fn(), vi.fn());
		const first = requests.request({ id: 1, type: 'step' }).catch((error) => error);
		const second = requests.request({ id: 2, type: 'eval_js' }).catch((error) => error);
		requests.fail(new Error('worker crashed'));
		await expect(first).resolves.toMatchObject({ message: 'worker crashed' });
		await expect(second).resolves.toMatchObject({ message: 'worker crashed' });
		await expect(requests.request({ id: 3, type: 'start' })).rejects.toThrow('worker crashed');
		expect(requests.pendingCount).toBe(0);
		expect(vi.getTimerCount()).toBe(0);
	});

	it('cleans up synchronous postMessage failures', async () => {
		const requests = new WorkerRequests(() => {
			throw new Error('clone failed');
		}, vi.fn());
		await expect(requests.request({ id: 1, type: 'step' })).rejects.toThrow('clone failed');
		expect(requests.pendingCount).toBe(0);
		expect(vi.getTimerCount()).toBe(0);
	});
});
