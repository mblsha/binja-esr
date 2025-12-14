/// <reference types="node" />

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const isVitest = process.env.VITEST === 'true' || process.env.VITEST === '1';

function parseAllowedHosts(): string[] | true | undefined {
	const raw = process.env.VITE_ALLOWED_HOSTS;
	if (!raw) return true;
	if (raw === 'true' || raw === '1') return true;
	const hosts = raw
		.split(',')
		.map((h) => h.trim())
		.filter(Boolean);
	return hosts.length ? hosts : true;
}

export default defineConfig({
	plugins: [sveltekit()],
	resolve: {
		conditions: isVitest ? ['browser'] : undefined
	},
	server: {
		host: true,
		allowedHosts: parseAllowedHosts()
	},
	preview: {
		host: true,
		allowedHosts: parseAllowedHosts()
	},
	test: {
		environment: 'jsdom',
		exclude: ['**/node_modules/**', 'e2e/**']
	}
});
