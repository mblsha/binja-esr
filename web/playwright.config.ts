import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

const port = Number(process.env.PCE500_E2E_PORT ?? 4173);
const realRomMode = process.env.PCE500_E2E_REAL_ROM === '1';
const syntheticRom = resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom');
const configuredRom = process.env.PCE500_ROM_PATH ? resolve(process.cwd(), process.env.PCE500_ROM_PATH) : null;
const romPath = realRomMode ? configuredRom : syntheticRom;
// Opt-in compatibility checks reuse the same WASM and focused scenarios.
const engines = (process.env.PCE500_E2E_ENGINES ?? 'chromium').split(',');
const engineDevices = { chromium: 'Desktop Chrome', firefox: 'Desktop Firefox', webkit: 'Desktop Safari' } as const;
for (const engine of engines) {
	if (!(engine in engineDevices)) throw new Error(`Unknown browser engine: ${engine}`);
}
// CI already builds WASM and Vite before browser checks. Local runs still
// rebuild by default, so a stale binary cannot silently pass a regression.
const previewCommand = `npm run preview -- --host 127.0.0.1 --port ${port}`;

export default defineConfig({
	testDir: './e2e',
	testMatch: /.*\.spec\.ts/,
	timeout: 60_000,
	expect: {
		timeout: 15_000,
	},
	use: {
		baseURL: `http://127.0.0.1:${port}`,
		trace: 'retain-on-failure',
		headless: process.env.PCE500_E2E_HEADED !== '1',
		launchOptions: { timeout: 30_000 },
	},
	projects: engines.map((name) => ({ name, use: { ...devices[engineDevices[name as keyof typeof engineDevices]] } })),
	webServer: {
		command: process.env.PCE500_E2E_USE_EXISTING_BUILD === '1' ? previewCommand : `npm run build && ${previewCommand}`,
		port,
		reuseExistingServer: !process.env.CI,
		env: {
			...(romPath ? { PCE500_ROM_PATH: romPath } : {}),
		},
	},
});
