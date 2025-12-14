import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

const port = 4173;
const realRomMode = process.env.PCE500_E2E_REAL_ROM === '1';
const syntheticRom = resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom');
const configuredRom = process.env.PCE500_ROM_PATH ? resolve(process.cwd(), process.env.PCE500_ROM_PATH) : null;
const romPath = realRomMode ? configuredRom : syntheticRom;

export default defineConfig({
	testDir: './e2e',
	testMatch: /.*\.spec\.ts/,
	timeout: 60_000,
	expect: {
		timeout: 15_000
	},
	use: {
		baseURL: `http://127.0.0.1:${port}`,
		trace: 'retain-on-failure'
	},
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] }
		}
	],
	webServer: {
		command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${port}`,
		port,
		reuseExistingServer: !process.env.CI,
		env: {
			...(romPath ? { PCE500_ROM_PATH: romPath } : {})
		}
	}
});
