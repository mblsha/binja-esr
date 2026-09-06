import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function open(page: Page, model: string, realRom = false) {
	if (!realRom) {
		const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
		await page.route(`**/api/rom?model=${model}`, (route) => route.fulfill({ status: 200, body: rom }));
	}
	await page.addInitScript((model) => {
		localStorage.setItem('sc62015:rom-model', model);
		const NativeWorker = window.Worker;
		window.Worker = class extends NativeWorker {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				let nextId = 1_000_000;
				// Inspect the actual compiled machine worker, never a fake emulator.
				(window as any).__inputRequest = (type: string, payload: any = {}) =>
					new Promise((resolve, reject) => {
						const id = nextId++;
						const timer = setTimeout(() => {
							this.removeEventListener('message', receive);
							reject(new Error('Input test RPC timeout'));
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
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
}

const request = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__inputRequest(type, payload), { type, payload });
const contacts = (page: Page) => request(page, 'input_state').then((state) => state.rust);

async function pointDown(page: Page, id: string) {
	const button = page.getByTestId(id);
	await button.scrollIntoViewIfNeeded();
	const box = (await button.boundingBox())!;
	await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
	await page.mouse.down();
}

async function script(page: Page, source: string, wait = true) {
	await page.getByTestId('fnr-panel').evaluate((panel: HTMLDetailsElement) => {
		panel.open = true;
	});
	await page.getByTestId('fnr-editor').fill(source);
	await page.getByTestId('fnr-run').click();
	if (wait) {
		await expect(page.getByTestId('fnr-run')).toBeEnabled();
		await expect(page.getByTestId('fnr-error')).toHaveCount(0);
	}
}

for (const model of ['pc-e500', 'iq-7000']) {
	const code = model === 'pc-e500' ? 0x56 : 0x18;
	const virtual = model === 'pc-e500' ? 'vk-pf1' : 'vk-calendar';
	const shift = model === 'pc-e500' ? 0x06 : 0x02;

	test(`${model}: physical, pointer and modifier owners do not release each other`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('assisted-taps-toggle').uncheck();
		await page.getByTestId('physical-keyboard-toggle').check();
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: false });
		await page.getByTestId(virtual).click();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: false });
		await page.keyboard.up('F1');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('ShiftLeft');
		await page.keyboard.down('ShiftRight');
		await page.keyboard.up('ShiftLeft');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [shift], on: false });
		await page.keyboard.up('ShiftRight');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
	});

	test(`${model}: assisted release is bounded, repress-safe, and cancelled on blur while paused`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId(virtual).click();
		let state = await request(page, 'input_state');
		expect(state.rust.matrix).toEqual([code]);
		expect(state.pendingVirtualRelease).toEqual([[code, 40_000]]);
		await request(page, 'step', { instructions: 10_000 });
		state = await request(page, 'input_state');
		expect(state.pendingVirtualRelease).toEqual([[code, 30_000]]);
		await pointDown(page, virtual);
		await request(page, 'step', { instructions: 40_000 });
		expect(await contacts(page)).toEqual({ matrix: [code], on: false });
		await page.mouse.up();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId(virtual).click();
		await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		expect((await request(page, 'input_state')).pendingVirtualRelease).toEqual([]);
	});

	test(`${model}: focus loss and text entry release host keys without stealing editor input`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('physical-keyboard-toggle').check();
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F1');
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await page.getByTestId('fnr-panel').evaluate((panel: HTMLDetailsElement) => {
			panel.open = true;
		});
		await page.getByTestId('fnr-editor').focus();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F1');
		await page.keyboard.press('ShiftLeft');
		await page.keyboard.press('Enter');
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F12');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: true });
		await page.getByTestId('physical-keyboard-toggle').uncheck();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F12');
	});

	test(`${model}: isolated script cleanup releases its raw/ON keys and preserves another owner`, async ({ page }) => {
		await open(page, model);
		// A distinct raw host owner persists while the script owns the same matrix contact.
		await request(page, 'physical_key', { code, down: true, owner: 'test-host' });
		await script(
			page,
			`await e.keys.phys.press(${code}); await e.onKey.press(); e.print('contacts held'); while (true) {}`,
			false,
		);
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: true });
		await page.getByRole('button', { name: 'Stop', exact: true }).click();
		await expect(page.getByTestId('fnr-run')).toBeEnabled();
		expect(await contacts(page)).toEqual({ matrix: [code], on: false });
		await request(page, 'physical_key', { code, down: false, owner: 'test-host' });
		await script(page, `await e.keys.phys.press(${code}); await e.onKey.press();`);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await script(page, `await e.keys.event.press(${code});`);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});

	test(`${model}: stale input cannot affect a replacement machine`, async ({ page }) => {
		await open(page, model);
		const { generation } = await request(page, 'input_state');
		await expect(request(page, 'physical_key', { code, down: true, generation: generation - 1 })).rejects.toThrow(
			'Stale input',
		);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});
}

test('real IQ ROM consumes the browser physical MEMO and CAPS controls', async ({ page }, info) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await request(page, 'step', { instructions: 500_000 });
	await page.getByTestId('vk-memo').click();
	await request(page, 'step', { instructions: 200_000 });
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await script(page, `e.assert((await e.lcd.capture()).annunciators.caps, 'reset starts with CAPS');`);
	await page.getByTestId('vk-caps').click();
	await request(page, 'step', { instructions: 200_000 });
	await script(page, `e.assert(!(await e.lcd.capture()).annunciators.caps, 'physical CAPS must change ROM state');`);
	await page.locator('.lcd-display').screenshot({ path: info.outputPath('real-iq-physical-memo-caps.png') });
});
