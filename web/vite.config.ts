/// <reference types="node" />

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const isVitest = process.env.VITEST === 'true' || process.env.VITEST === '1';

export default defineConfig({
	plugins: [sveltekit()],
	resolve: {
		conditions: isVitest ? ['browser'] : undefined
	},
	test: {
		environment: 'jsdom'
	}
});
