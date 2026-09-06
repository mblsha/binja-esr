/// <reference types="node" />

import { sveltekit } from '@sveltejs/kit/vite';
import path from 'node:path';
import { defineConfig, type Plugin } from 'vite';

const isVitest = process.env.VITEST === 'true' || process.env.VITEST === '1';
const isolationHeaders = {
	'Cross-Origin-Opener-Policy': 'same-origin',
	'Cross-Origin-Embedder-Policy': 'require-corp'
};

function scriptIsolationHeaders(): Plugin {
	const install = (server: { middlewares: { use: Function } }) => {
		server.middlewares.use((_req: unknown, res: { setHeader(name: string, value: string): void }, next: () => void) => {
			for (const [name, value] of Object.entries(isolationHeaders)) res.setHeader(name, value);
			next();
		});
	};
	// SvelteKit preview serves immutable worker assets before Vite's normal
	// header middleware. Worker responses must carry COEP too, not only HTML.
	return { name: 'script-isolation-headers', configureServer: install, configurePreviewServer: install };
}

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

function wasmPackageReload(): Plugin {
	const wasmPkgDir = path.resolve('src/lib/wasm/pce500_wasm');
	const wasmPkgPrefix = `${wasmPkgDir}${path.sep}`;

	return {
		name: 'wasm-package-reload',
		configureServer(server) {
			server.watcher.add(wasmPkgDir);
		},
		handleHotUpdate(ctx) {
			const file = ctx.file;
			if (file === wasmPkgDir || file.startsWith(wasmPkgPrefix)) {
				ctx.server.ws.send({ type: 'full-reload' });
				return [];
			}
		}
	};
}

export default defineConfig({
	plugins: [scriptIsolationHeaders(), sveltekit(), wasmPackageReload()],
	worker: {
		format: 'es'
	},
	resolve: {
		conditions: isVitest ? ['browser'] : undefined
	},
	server: {
		host: true,
		headers: isolationHeaders,
		allowedHosts: parseAllowedHosts()
	},
	preview: {
		host: true,
		headers: isolationHeaders,
		allowedHosts: parseAllowedHosts()
	},
	test: {
		environment: 'jsdom',
		exclude: ['**/node_modules/**', 'e2e/**']
	}
});
