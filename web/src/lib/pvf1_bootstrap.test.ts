import { describe, expect, it, vi } from 'vitest';
import { configuredPvHandwriting } from './pvf1_bootstrap';

describe('configured PV-F1 handwriting startup asset', () => {
	it('keeps manual loading available when the server has no configured model', async () => {
		const fetcher = vi.fn().mockResolvedValue(new Response('', { status: 404 }));
		expect(await configuredPvHandwriting(fetcher)).toBeNull();
	});
	it('stages bytes and name without accepting them as a validated model', async () => {
		const bytes = new Uint8Array([1, 2, 3]); // Rust must reject this at installation.
		const fetcher = vi
			.fn()
			.mockResolvedValue(new Response(bytes, { headers: { 'x-model-name': 'pen%20model.model' } }));
		expect(await configuredPvHandwriting(fetcher)).toEqual({ bytes, name: 'pen model.model' });
	});
	it('surfaces missing or unreadable configured assets rather than omitting them', async () => {
		const fetcher = vi.fn().mockResolvedValue(new Response('', { status: 500 }));
		await expect(configuredPvHandwriting(fetcher)).rejects.toThrow('fetch failed (500)');
	});
	it('rejects empty successful responses', async () => {
		const fetcher = vi.fn().mockResolvedValue(new Response(new Uint8Array()));
		await expect(configuredPvHandwriting(fetcher)).rejects.toThrow('is empty');
	});
});
