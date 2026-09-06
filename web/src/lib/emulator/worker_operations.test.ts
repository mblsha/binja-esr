import { describe, expect, it } from 'vitest';
import { WorkerOperations } from './worker_operations';

describe('machine operation ownership', () => {
	it('rejects overlapping mutations across await points and acknowledges stop only after release', async () => {
		const operations = new WorkerOperations();
		let release!: () => void;
		let signal!: AbortSignal;
		const job = operations.run(async (s) => {
			signal = s;
			await new Promise<void>((resolve) => {
				release = resolve;
			});
			return 42;
		});
		expect(operations.busy).toBe(true);
		await expect(operations.run(async () => 1)).rejects.toThrow('busy');
		let acknowledged = false;
		const stopped = operations.stop().then(() => {
			acknowledged = true;
		});
		expect(signal.aborted).toBe(true);
		await Promise.resolve();
		expect(acknowledged).toBe(false);
		await expect(operations.run(async () => 2)).rejects.toThrow('busy');
		release();
		expect(await job).toBe(42);
		await stopped;
		expect(acknowledged).toBe(true);
		expect(operations.busy).toBe(false);
		expect(await operations.run(async () => 3)).toBe(3);
	});

	it('releases ownership on failure and permits concurrent stop requests', async () => {
		const operations = new WorkerOperations();
		const job = operations.run(async () => {
			throw new Error('fault');
		});
		const stops = Promise.all([operations.stop(), operations.stop()]);
		await expect(job).rejects.toThrow('fault');
		await stops;
		expect(operations.busy).toBe(false);
	});
});
