import { test, expect, type Page } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

const bundlePath = process.env.PI3000_ROM_PATH;
const output = process.env.PI3000_PROOF_DIR;
test.skip(!bundlePath, 'Supply verified private PI bundle with PI3000_ROM_PATH');
const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__piRpc(type, payload), { type, payload });

test('PI fresh browser touch completes initialization and enters the calendar', async ({ page }) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pi-3000');
		const Original = window.Worker;
		window.Worker = class extends Original {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				if (!String(url).includes('pce500.worker')) return;
				let sequence = 4_000_000;
				(window as any).__piRpc = (type: string, payload: any = {}) =>
					new Promise((resolve, reject) => {
						const id = sequence++;
						const timer = setTimeout(() => {
							this.removeEventListener('message', receive);
							reject(new Error('PI RPC timeout'));
						}, 120_000);
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
	});
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(String(error)));
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await expect(page.getByTestId('device-shell')).toHaveClass(/pi-3000/);
	const canvas = page.getByTestId('pi-tablet-lcd').locator('canvas');
	await expect(canvas).toHaveAttribute('width', '239');
	await expect(canvas).toHaveAttribute('height', '168');
	await rpc(page, 'step', { instructions: 400_000 });
	const snapshots: any[] = [];
	for (const [x, y] of [
		[32, 32],
		[205, 32],
		[32, 135],
		[205, 135],
	]) {
		await canvas.scrollIntoViewIfNeeded();
		const box = (await canvas.boundingBox())!;
		await page.mouse.click(box.x + ((x + 0.5) / 239) * box.width, box.y + ((y + 0.5) / 168) * box.height);
		await expect.poll(() => rpc(page, 'pi_state').then((s) => s.tablet[2])).toBe(1);
		await rpc(page, 'step', { instructions: 300_000 });
		snapshots.push(await rpc(page, 'pi_state'));
	}
	const state = snapshots.at(-1);
	expect(state.fault).toBeNull();
	expect(state.tablet.slice(0, 3)).toEqual([700, 700, 0]);
	expect(state.tablet[5]).toBe(995);
	expect(state.native_instructions).toBe(1071065);
	const observed = await rpc(page, 'eval_js', {
		source:
			'return { capture: await e.lcd.capture(1), completed: await Promise.all(Array.from({length:4}, (_,i) => e.memory.read(0xBB294 + 0x21E + i))), result: await e.memory.read(0xBB294 + 0x222) };',
	});
	expect(observed.error).toBeNull();
	const result = JSON.parse(observed.resultJson);
	expect(result.completed).toEqual([1, 1, 1, 1]);
	expect(result.result).toBe(2);
	const header = Buffer.from('P4\n240 336\n');
	const pbm = Buffer.alloc(header.length + 30 * 336);
	header.copy(pbm);
	for (let i = 0; i < result.capture.pixels.length; i++)
		if (result.capture.pixels[i] === 0) pbm[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
	const digest = createHash('sha256').update(pbm).digest('hex');
	expect(digest).toBe('e3f12a2322d3d48f158b13d8f80035ea8452d274cea7119131c2632ed4dbd39e');
	expect(errors).toEqual([]);
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(
			resolve(output, 'browser-calibration.json'),
			JSON.stringify({ snapshots, state, digest, completed: result.completed, result: result.result }),
		);
		await canvas.screenshot({ path: resolve(output, 'browser-calibration.png') });
		await rpc(page, 'step', { instructions: 500_000 });
		await canvas.screenshot({ path: resolve(output, 'browser-completion.png') });
	}
	// The ROM waits four A2 ticks after completing calibration. Keyboard input
	// cannot close that wait; exercise the real periodic RTC and next touch UI.
	await rpc(page, 'step', { instructions: output ? 2_500_000 : 3_000_000 });
	async function controllerDigest() {
		const observed = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
		expect(observed.error).toBeNull();
		const pixels = JSON.parse(observed.resultJson).pixels;
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < pixels.length; i++)
			if (pixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		return createHash('sha256').update(frame).digest('hex');
	}
	const contrastState = await rpc(page, 'pi_state');
	const contrastDigest = await controllerDigest();
	expect(contrastState.fault).toBeNull();
	expect(contrastDigest).toBe('0338fff05952b2379a4c1347711fa277387ffefbc6a4bd9dfc5d36f196584fa9');
	if (output) await canvas.screenshot({ path: resolve(output, 'browser-contrast.png') });
	async function holdTouch(x: number, y: number, held: number, released: number) {
		await canvas.scrollIntoViewIfNeeded();
		const box = (await canvas.boundingBox())!;
		await page.mouse.move(box.x + ((x + 0.5) / 239) * box.width, box.y + ((y + 0.5) / 168) * box.height);
		await page.mouse.down();
		await expect.poll(() => rpc(page, 'pi_state').then((s) => s.tablet[2])).toBe(1);
		await rpc(page, 'step', { instructions: held });
		await page.mouse.up();
		await rpc(page, 'step', { instructions: released });
	}
	await holdTouch(219, 155, 250_000, 1_000_000);
	const initializationState = await rpc(page, 'pi_state');
	const initializationDigest = await controllerDigest();
	expect(initializationState.fault).toBeNull();
	expect(initializationDigest).toBe('f25cdbc1bbfddf62ffcefb56f0f07dfb98ae8e9f273ee55a9014794a8a08eb4e');
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-initialization.png') });
	}
	// Confirm fresh emulated RAM initialization using the actual Yes control.
	await holdTouch(74, 112, 250_000, 6_500_000);
	const calendarState = await rpc(page, 'pi_state');
	const calendarDigest = await controllerDigest();
	expect(calendarState.fault).toBeNull();
	expect(calendarState.selector).toBe(0xf1);
	expect(calendarState.unmapped_low_read).toBe(0xe0);
	expect(calendarState.native_instructions).toBe(5127802);
	expect(calendarDigest).toBe('72d7fc2b314eb7d805a64a68a28e85193d5558a51e24d3083da8defa5e704138');
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-calendar.png') });
	}
	// Normal pointer input reaches a numeric entry response. The physical and
	// application hit mapping remains a recovery gap; this does not select day16.
	await holdTouch(76, 72, 250_000, 500_000);
	const contactState = await rpc(page, 'pi_state');
	const contactDigest = await controllerDigest();
	expect(contactState.fault).toBeNull();
	expect(contactState.tablet.slice(0, 3)).toEqual([432, 419, 0]);
	expect(contactState.native_instructions).toBe(5378834);
	expect(contactDigest).toBe('48e46b64f27429fa090f6e630da7e390caf9821e0f6f8b65297c3e018789a0e9');
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-calendar-contact.png') });
		await writeFile(
			resolve(output, 'browser-startup.json'),
			JSON.stringify({
				contrastState,
				contrastDigest,
				initializationState,
				initializationDigest,
				calendarState,
				calendarDigest,
				contactState,
				contactDigest,
			}),
		);
	}
});

test('PI no-Worker browser fallback shares panel and assisted contact behavior', async ({ page }) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pi-3000');
		(window as any).Worker = undefined;
	});
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(String(error)));
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('advanced-panel').locator('> summary').click();
	const step = page.getByRole('button', { name: 'Step 20k', exact: true });
	const canvas = page.getByTestId('pi-tablet-lcd').locator('canvas');
	async function advance(count: number) {
		for (let i = 0; i < count; i++) {
			await step.click();
			await expect(step).toBeEnabled();
		}
	}
	await advance(20);
	for (const [x, y] of [
		[32, 32],
		[205, 32],
		[32, 135],
		[205, 135],
	]) {
		await canvas.scrollIntoViewIfNeeded();
		const box = (await canvas.boundingBox())!;
		await page.mouse.click(box.x + ((x + 0.5) / 239) * box.width, box.y + ((y + 0.5) / 168) * box.height);
		await advance(15);
	}
	await expect(page.getByTestId('emu-status')).toContainText('1071065');
	const pixels = await canvas.evaluate((node: HTMLCanvasElement) => {
		const rgba = node.getContext('2d')!.getImageData(0, 0, node.width, node.height).data;
		return Array.from({ length: node.width * node.height }, (_, i) => (rgba[i * 4] === 0 ? 1 : 0));
	});
	expect(createHash('sha256').update(Buffer.from(pixels)).digest('hex')).toBe(
		'f8e11c6e47532440c3d1bd6fa97d3ced9cb657e083837beec7a3d432c75616bc',
	);
	expect(errors).toEqual([]);
	if (output) {
		await mkdir(output, { recursive: true });
		await canvas.screenshot({ path: resolve(output, 'browser-fallback.png') });
		await writeFile(
			resolve(output, 'browser-fallback.json'),
			JSON.stringify({
				boundaries: 1_600_000,
				native_instructions: 1071065,
				panel_sha256: 'f8e11c6e47532440c3d1bd6fa97d3ced9cb657e083837beec7a3d432c75616bc',
				errors,
			}),
		);
	}
});
