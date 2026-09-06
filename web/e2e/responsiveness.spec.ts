import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { cpus, release } from 'node:os';

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: measure acknowledged Pause while running`, async ({ page }, info) => {
		if (process.env.PCE500_E2E_REAL_ROM !== '1') {
			// Public control test, not a real-app/screenshots claim.
			const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
			await page.route(`**/api/rom?model=${model}`, (route) => route.fulfill({ status: 200, body: rom }));
		}
		await page.addInitScript((selectedModel) => {
			localStorage.setItem('sc62015:rom-model', selectedModel);
			const samples: number[] = [];
			(window as any).__pauseSamples = samples;
			const NativeWorker = window.Worker;
			window.Worker = class extends NativeWorker {
				posted = new Map<number, number>();
				constructor(url: string | URL, options?: WorkerOptions) {
					super(url, options);
					this.addEventListener('message', (event) => {
						const started = this.posted.get(event.data.id);
						if (event.data.type === 'reply' && started !== undefined) {
							this.posted.delete(event.data.id);
							if (event.data.ok) samples.push(performance.now() - started);
						}
					});
				}
				postMessage(message: any, transfer: Transferable[] = []) {
					if (message.type === 'stop') this.posted.set(message.id, performance.now());
					super.postMessage(message, transfer);
				}
			};
		}, model);
		await page.goto('/');
		const run = page.getByRole('button', { name: 'Run', exact: true });
		await expect(run).toBeEnabled();
		await page.evaluate(() => {
			(window as any).__pauseSamples.length = 0;
		});
		for (let i = 0; i < 20; i++) {
			await run.click();
			await expect(page.getByTestId('emu-status')).toContainText('RUNNING');
			await page.getByRole('button', { name: 'Stop', exact: true }).click();
			await expect(page.getByTestId('emu-status')).toContainText(/STOPPED|HALTED/);
			await expect(run).toBeEnabled();
		}
		const samples = await page.evaluate(() => (window as any).__pauseSamples as number[]);
		expect(samples).toHaveLength(20);
		const sorted = [...samples].sort((a, b) => a - b);
		const report = {
			model,
			realRom: process.env.PCE500_E2E_REAL_ROM === '1',
			host: {
				platform: process.platform,
				arch: process.arch,
				osRelease: release(),
				cpu: cpus()[0]?.model,
				browser: page.context().browser()?.version(),
			},
			samplesMs: samples,
			p99Ms: sorted[Math.ceil(sorted.length * 0.99) - 1],
			maxMs: sorted.at(-1),
			scope:
				'Foreground Chromium Stop request -> actual worker acknowledgement; final frame capture is separate; not input-to-firmware latency',
		};
		console.log(JSON.stringify(report));
		await info.attach('pause-latency.json', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
	});
}

test('Pause stays pending until the actual worker acknowledgement arrives', async ({ page }) => {
	await page.addInitScript(() => {
		const NativeWorker = window.Worker;
		const harness = { delayStops: false, release: null as (() => void) | null };
		(window as any).__controlTest = harness;
		// The real compiled worker and WASM still execute. Only its Stop reply
		// delivery is delayed, to expose optimistic UI acknowledgement bugs.
		(window as any).Worker = class {
			native: Worker;
			onmessage: ((event: MessageEvent) => void) | null = null;
			onerror: ((event: ErrorEvent) => void) | null = null;
			onmessageerror: ((event: MessageEvent) => void) | null = null;
			stops = new Set<number>();
			constructor(url: string | URL, options?: WorkerOptions) {
				this.native = new NativeWorker(url, options);
				this.native.onmessage = (event) => {
					if (event.data.type === 'reply' && this.stops.delete(event.data.id) && harness.delayStops) {
						harness.release = () => this.onmessage?.(event);
					} else this.onmessage?.(event);
				};
				this.native.onerror = (event) => this.onerror?.(event);
				this.native.onmessageerror = (event) => this.onmessageerror?.(event);
			}
			postMessage(message: any, transfer: Transferable[] = []) {
				if (message.type === 'stop') this.stops.add(message.id);
				this.native.postMessage(message, transfer);
			}
			terminate() {
				this.native.terminate();
			}
		};
	});
	await page.goto('/');
	const run = page.getByRole('button', { name: 'Run', exact: true });
	await expect(run).toBeEnabled();
	await run.click();
	await expect(page.getByTestId('emu-status')).toContainText('RUNNING');
	await page.evaluate(() => {
		(window as any).__controlTest.delayStops = true;
	});
	await page.getByRole('button', { name: 'Stop', exact: true }).click();
	await expect(page.getByTestId('emu-status')).toContainText('PAUSE REQUESTED');
	await expect(run).toBeDisabled();
	await page.waitForFunction(() => (window as any).__controlTest.release !== null);
	await page.evaluate(() => {
		(window as any).__controlTest.release();
	});
	await expect(page.getByTestId('emu-status')).toContainText(/STOPPED|HALTED/);
	await expect(run).toBeEnabled();
});

test('Stop cancels a huge Function Runner step and the machine remains usable', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
	await page.getByTestId('fnr-panel').evaluate((panel: HTMLDetailsElement) => {
		panel.open = true;
	});
	await page.getByTestId('fnr-editor').fill('await e.step(4_000_000_000);');
	await page.getByTestId('fnr-run').click();
	await expect(page.getByTestId('emu-status')).toContainText('EXECUTING SCRIPT');
	await page.getByRole('button', { name: 'Stop', exact: true }).click();
	await expect(page.getByTestId('emu-status')).toContainText(/STOPPED|HALTED/);
	await expect(page.getByTestId('fnr-error')).toContainText('Execution cancelled');
	await expect(page.getByTestId('fnr-run')).toBeEnabled();
	const step = page.getByRole('button', { name: 'Step 1k' });
	await step.click();
	await expect(step).toBeEnabled();
	await expect(page.getByTestId('emu-status')).not.toContainText(/FAULTED|UNRESPONSIVE/);
});

test('a delayed previous ROM fetch cannot replace the newly selected model', async ({ page }) => {
	const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
	let releaseOld!: () => void;
	const oldResponse = new Promise<void>((resolve) => {
		releaseOld = resolve;
	});
	let oldRequested = false;
	await page.route('**/api/rom?model=pc-e500', async (route) => {
		oldRequested = true;
		await oldResponse;
		await route.fulfill({ status: 200, body: rom });
	});
	await page.route('**/api/rom?model=iq-7000', (route) => route.fulfill({ status: 200, body: rom }));
	await page.goto('/');
	await expect.poll(() => oldRequested).toBe(true);
	await page.getByTestId('rom-model').selectOption('iq-7000');
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
	await expect(page.getByText('LCD: iq7000-vram', { exact: false })).toBeVisible();
	releaseOld();
	await page.waitForLoadState('networkidle');
	await expect(page.getByTestId('rom-model')).toHaveValue('iq-7000');
	await expect(page.getByText('LCD: iq7000-vram', { exact: false })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
});
