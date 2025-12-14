import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

const port = 4173;

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
			PCE500_ROM_PATH: resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom')
		}
	}
});

