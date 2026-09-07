import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__pacingRpc(type, payload), { type, payload });

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: acknowledged modes preserve state, explicit budgets and pause ownership`, async ({ page }, info) => {
		// Synthetic control qualification using the real compiled Rust/WASM worker.
		const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
		await page.route('**/api/rom?model=*', (route) => route.fulfill({ status: 200, body: rom }));
		await page.addInitScript((model) => {
			localStorage.setItem('sc62015:rom-model', model);
			const NativeWorker = window.Worker;
			window.Worker = class extends NativeWorker {
				constructor(url: string | URL, options?: WorkerOptions) {
					super(url, options);
					let nextId = 3_000_000;
					(window as any).__pacingRpc = (type: string, payload: any = {}) =>
						new Promise((resolve, reject) => {
							const id = nextId++;
							const timer = setTimeout(() => {
								this.removeEventListener('message', receive);
								reject(new Error('Pacing RPC timed out'));
							}, 5000);
							const receive = ({ data }: MessageEvent) => {
								if (data.type !== 'reply' || data.id !== id) return;
								clearTimeout(timer);
								this.removeEventListener('message', receive);
								if (data.ok) resolve(data.result);
								else reject(new Error(data.error));
							};
							this.addEventListener('message', receive);
							this.postMessage({ id, type, ...payload });
						});
				}
			};
		}, model);
		await page.goto('/');
		await page.getByTestId('advanced-panel').locator('> summary').click();
		const mode = page.getByTestId('execution-mode');
		const run = page.getByRole('button', { name: 'Run', exact: true });
		const stop = page.getByRole('button', { name: 'Stop', exact: true });
		await expect(run).toBeEnabled();
		const initial = await rpc(page, 'pacing_status');
		expect(initial.mode).toBe('interactive');
		expect(initial.calibration).toContain('not hardware calibrated');
		await mode.selectOption('deterministic');
		await expect(mode).toBeEnabled();
		await expect(run).toBeDisabled();
		await expect(page.getByTestId('pacing-status')).toContainText('not hardware-calibrated');
		expect((await rpc(page, 'pacing_status')).instructions_retired).toBe(initial.instructions_retired);
		await expect(rpc(page, 'start')).rejects.toThrow('explicit Step');
		await rpc(page, 'step', { instructions: 1000 });
		const stepped = await rpc(page, 'pacing_status');
		expect(Number(stepped.instructions_retired)).toBeGreaterThan(Number(initial.instructions_retired));
		await expect(rpc(page, 'set_execution_mode', { mode: 'realtime' })).rejects.toThrow('execution mode must');
		expect((await rpc(page, 'pacing_status')).mode).toBe('deterministic');

		const measurements = [];
		for (const selected of ['interactive', 'turbo']) {
			await mode.selectOption(selected);
			await expect(mode).toBeEnabled();
			const before = await rpc(page, 'pacing_status');
			const started = Date.now();
			await run.click();
			await expect(mode).toBeDisabled();
			await expect(rpc(page, 'set_execution_mode', { mode: 'deterministic' })).rejects.toThrow('Pause before');
			await expect
				.poll(() => rpc(page, 'pacing_status').then((s) => Number(s.elapsed_timing_units)))
				.toBeGreaterThan(Number(before.elapsed_timing_units));
			await stop.click();
			await expect(mode).toBeEnabled();
			const after = await rpc(page, 'pacing_status');
			measurements.push({ selected, hostMs: Date.now() - started, before, after });
			// State remains stable across later host turns; observation adds no execution.
			for (let i = 0; i < 10; i++) expect(await rpc(page, 'pacing_status')).toEqual(after);
		}
		await mode.selectOption('deterministic');
		await expect(mode).toBeEnabled();
		// A long explicit request retains ownership across JS yields; mode RPC
		// cannot race it or bypass Stop's cancellation handshake.
		const outcomes = await page.evaluate(async () => {
			const rpc = (window as any).__pacingRpc;
			const work = rpc('step', { instructions: 4_000_000_000 }).then(
				() => 'finished',
				(e: Error) => e.message,
			);
			const mode = await rpc('set_execution_mode', { mode: 'turbo' }).then(
				() => 'changed',
				(e: Error) => e.message,
			);
			await rpc('stop');
			return { work: await work, mode, status: await rpc('pacing_status') };
		});
		expect(outcomes.mode).toContain('busy');
		expect(outcomes.work).toContain('cancelled');
		expect(outcomes.status.mode).toBe('deterministic');
		await info.attach('pacing-control-observations.json', {
			body: JSON.stringify(measurements, (_, value) => (typeof value === 'bigint' ? value.toString() : value), 2),
			contentType: 'application/json',
		});
	});
}
