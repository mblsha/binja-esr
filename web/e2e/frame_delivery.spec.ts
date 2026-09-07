import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__displayHarness.rpc(type, payload), { type, payload });

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: delayed presentation bounds frames without blocking Pause, input or model replacement`, async ({
		page,
	}) => {
		const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
		await page.route('**/api/rom?model=*', (route) => route.fulfill({ status: 200, body: rom }));
		await page.addInitScript((model) => {
			localStorage.setItem('sc62015:rom-model', model);
			const NativeWorker = window.Worker;
			window.Worker = class extends NativeWorker {
				harness: any;
				constructor(url: string | URL, options?: WorkerOptions) {
					super(url, options);
					let nextId = 2_000_000;
					const send = (message: any) => super.postMessage(message);
					this.harness = {
						holdCredit: false,
						heldCredit: null,
						received: [] as { sequence: number; model: string; generation: number }[],
						release: () => {
							this.harness.holdCredit = false;
							if (this.harness.heldCredit) send(this.harness.heldCredit);
							this.harness.heldCredit = null;
						},
						sendCredit: (sequence: number) => send({ id: nextId++, type: 'frame_consumed', sequence }),
						rpc: (type: string, payload: any) =>
							new Promise((resolve, reject) => {
								const id = nextId++;
								const timer = setTimeout(() => {
									this.removeEventListener('message', receive);
									reject(new Error('Display test RPC timed out'));
								}, 5000);
								const receive = ({ data }: MessageEvent) => {
									if (data.type !== 'reply' || data.id !== id) return;
									clearTimeout(timer);
									this.removeEventListener('message', receive);
									if (data.ok) resolve(data.result);
									else reject(new Error(data.error));
								};
								this.addEventListener('message', receive);
								send({ id, type, ...payload });
							}),
					};
					(window as any).__displayHarness = this.harness;
					this.addEventListener('message', ({ data }) => {
						if (data.type === 'frame')
							this.harness.received.push({
								sequence: data.sequence,
								model: data.frame.model,
								generation: data.frame.generation,
								lcdText: data.frame.lcdText,
								callStack: data.frame.callStack,
							});
					});
				}
				postMessage(message: any, transfer: Transferable[] = []) {
					if (message.type === 'frame_consumed' && this.harness.holdCredit) this.harness.heldCredit = message;
					else super.postMessage(message, transfer);
				}
			};
		}, model);
		await page.goto('/');
		await page.getByTestId('advanced-panel').locator('> summary').click();
		await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
		await expect.poll(() => rpc(page, 'frame_delivery_state').then((s) => s.inFlight)).toBeNull();
		const frames = await page.evaluate(() => (window as any).__displayHarness.received);
		expect(frames.length).toBeGreaterThan(0);
		for (const frame of frames) {
			expect(frame.lcdText).toBeNull();
			expect(frame.callStack).toBeNull();
		}
		await page.evaluate(() => {
			const h = (window as any).__displayHarness;
			h.holdCredit = true;
			h.received.length = 0;
		});
		await page.getByRole('button', { name: 'Run', exact: true }).click();
		await page.waitForFunction(() => (window as any).__displayHarness.heldCredit !== null);
		const first = await rpc(page, 'frame_delivery_state');
		// The real worker receives 200 refresh requests while consumer credit is withheld.
		await page.evaluate(() =>
			Promise.all(Array.from({ length: 200 }, () => (window as any).__displayHarness.rpc('snapshot', {}))),
		);
		const blocked = await rpc(page, 'frame_delivery_state');
		expect(blocked.sequence).toBe(first.sequence);
		expect(blocked.inFlight).toBe(first.sequence);
		expect(blocked.pending).toBe(true);
		expect(blocked.coalesced - first.coalesced).toBeGreaterThanOrEqual(199);
		expect(await page.evaluate(() => (window as any).__displayHarness.received.length)).toBe(1);
		await page.evaluate((sequence) => (window as any).__displayHarness.sendCredit(sequence - 1), first.sequence);
		expect((await rpc(page, 'frame_delivery_state')).sequence).toBe(first.sequence);
		await page.getByRole('button', { name: 'Stop', exact: true }).click();
		await expect(page.getByTestId('emu-status')).toContainText(/STOPPED|HALTED/);
		await rpc(page, 'physical_key', { owner: 'display-test', code: 'on', down: true });
		expect((await rpc(page, 'input_state')).rust.on).toBe(true);
		await rpc(page, 'physical_key', { owner: 'display-test', code: 'on', down: false });
		const replacement = model === 'pc-e500' ? 'iq-7000' : 'pc-e500';
		page.once('dialog', (dialog) => dialog.accept());
		await page.getByTestId('rom-model').selectOption(replacement);
		await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
		const generation = (await rpc(page, 'input_state')).generation;
		expect((await rpc(page, 'frame_delivery_state')).sequence).toBe(first.sequence);
		await page.evaluate(() => (window as any).__displayHarness.release());
		await expect
			.poll(() => page.evaluate(() => (window as any).__displayHarness.received.at(-1)))
			.toMatchObject({ model: replacement, generation });
		await expect.poll(() => rpc(page, 'frame_delivery_state').then((s) => s.inFlight)).toBeNull();
		expect(await page.evaluate(() => (window as any).__displayHarness.received.length)).toBe(2);
	});
}
