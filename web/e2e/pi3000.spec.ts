import { test, expect, type Page } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

const bundlePath = process.env.PI3000_ROM_PATH;
const output = process.env.PI3000_PROOF_DIR;
test.skip(!bundlePath, 'Supply verified private PI bundle with PI3000_ROM_PATH');
const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__piRpc(type, payload), { type, payload });

test('PI browser initializes, handles input and power, and reopens a saved Schedule record', async ({ page }) => {
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
	expect(state.tablet[5]).toBe(1055);
	expect(state.native_instructions).toBe(1110227);
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
	expect(calendarState.native_instructions).toBe(5197729);
	expect(calendarDigest).toBe('72d7fc2b314eb7d805a64a68a28e85193d5558a51e24d3083da8defa5e704138');
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-calendar.png') });
	}
	// Normal pointer input selects day16. The previous selection (day1) becomes
	// visible again; idle execution blinks only the newly selected date.
	await holdTouch(76, 72, 250_000, 500_000);
	const contactState = await rpc(page, 'pi_state');
	const contactDigest = await controllerDigest();
	expect(contactState.fault).toBeNull();
	expect(contactState.tablet.slice(0, 3)).toEqual([432, 419, 0]);
	expect(contactState.native_instructions).toBe(5463195);
	expect(contactDigest).toBe('48e46b64f27429fa090f6e630da7e390caf9821e0f6f8b65297c3e018789a0e9');
	expect(errors).toEqual([]);
	const blinkSnapshots = [{ state: contactState, digest: contactDigest }];
	for (let i = 0; i < 32; i++) {
		await rpc(page, 'step', { instructions: 100_000 });
		blinkSnapshots.push({ state: await rpc(page, 'pi_state'), digest: await controllerDigest() });
	}
	expect([...new Set(blinkSnapshots.map((s) => s.digest))]).toEqual([
		'48e46b64f27429fa090f6e630da7e390caf9821e0f6f8b65297c3e018789a0e9',
		'b5c34cd004f6732bb14f18c76d9e5054cdbd5719fc414b9c3c58bce696212875',
	]);
	expect(blinkSnapshots.at(-1)!.state.native_instructions).toBe(5474438);
	expect(blinkSnapshots.every((s) => s.state.fault === null)).toBe(true);
	expect(errors).toEqual([]);
	async function holdKey(id: string, held: number, released: number) {
		const key = page.getByTestId(id);
		await key.scrollIntoViewIfNeeded();
		const box = (await key.boundingBox())!;
		await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
		await page.mouse.down();
		await expect(key).toHaveClass(/delivered/);
		await rpc(page, 'step', { instructions: held });
		await page.mouse.up();
		await expect(key).not.toHaveClass(/delivered/);
		await rpc(page, 'step', { instructions: released });
	}
	// Functional key controls deliver raw matrix contacts. Date opens the ROM's
	// dialog; its on-screen Cancel returns to the selected calendar date.
	await holdKey('vk-date', 200_000, 750_000);
	const dateDialogDigest = await controllerDigest();
	expect(blinkSnapshots.some((s) => s.digest === dateDialogDigest)).toBe(false);
	await holdTouch(180, 147, 50_000, 750_000);
	const cancelledDigest = await controllerDigest();
	expect(blinkSnapshots.some((s) => s.digest === cancelledDigest)).toBe(true);
	const cancelledState = await rpc(page, 'pi_state');
	expect(cancelledState.fault).toBeNull();
	await holdKey('vk-off', 200_000, 750_000);
	const offState = await rpc(page, 'pi_state');
	expect(offState.pc).toBe(0xf8ed5);
	await rpc(page, 'step', { instructions: 100_000 });
	const stillOff = await rpc(page, 'pi_state');
	expect(stillOff.native_instructions).toBe(offState.native_instructions);
	await holdKey('vk-on', 250_000, 4_770_000);
	const resumedState = await rpc(page, 'pi_state');
	expect(resumedState.fault).toBeNull();
	expect(resumedState.pc).toBe(0xf06ba);
	expect(resumedState.native_instructions).toBeGreaterThan(offState.native_instructions);
	const resumedPixels = await canvas.evaluate((node: HTMLCanvasElement) => {
		const rgba = node.getContext('2d')!.getImageData(0, 0, node.width, node.height).data;
		return Array.from({ length: node.width * node.height }, (_, i) => (rgba[i * 4] === 0 ? 1 : 0));
	});
	const resumedPanelDigest = createHash('sha256').update(Buffer.from(resumedPixels)).digest('hex');
	// Wake recreates the calendar focus at day1. Check the whole visible panel
	// against the independently captured native initial-date phases.
	expect([
		'd68c2d349da4125a920c3042093e5bb7c80369f2cd6c12ec3d78a8985a99f9f5',
		'a67f66c45f0b535d6a65b68f69fdab0778591b647992bc618f080aee9e5c64d8',
	]).toContain(resumedPanelDigest);
	expect(errors).toEqual([]);
	await holdTouch(159, 7, 250_000, 1_000_000);
	const scheduleState = await rpc(page, 'pi_state');
	expect(scheduleState.fault).toBeNull();
	expect(scheduleState.pc).toBe(0xf06ba);
	const scheduleCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(scheduleCapture.error).toBeNull();
	const scheduleNativePixels = JSON.parse(scheduleCapture.resultJson).pixels;
	const scheduleNativePanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					scheduleNativePixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	const scheduleHeaderDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 16 }, (_, i) =>
					scheduleNativePixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	if (output) {
		await writeFile(
			resolve(output, 'browser-schedule-diagnostic.json'),
			JSON.stringify({ scheduleState, scheduleNativePanelDigest, scheduleHeaderDigest }),
		);
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < scheduleNativePixels.length; i++)
			if (scheduleNativePixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		await writeFile(resolve(output, 'browser-schedule-controller.pbm'), frame);
	}
	// The entry header is invariant across cursor/input phases. Retain the whole
	// body for recovery; this check does not qualify a stored Schedule record.
	expect(scheduleHeaderDigest).toBe('0ca64e87f424d98f24da564986b95555c20ac2082fff988c5a7b630366c8cf2b');
	async function renderedPanelDigest() {
		const pixels = await canvas.evaluate((node: HTMLCanvasElement) => {
			const rgba = node.getContext('2d')!.getImageData(0, 0, node.width, node.height).data;
			return Array.from({ length: node.width * node.height }, (_, i) => (rgba[i * 4] === 0 ? 1 : 0));
		});
		return createHash('sha256').update(Buffer.from(pixels)).digest('hex');
	}
	await expect.poll(renderedPanelDigest).toBe(scheduleNativePanelDigest);
	const schedulePanelDigest = await renderedPanelDigest();
	expect(errors).toEqual([]);
	if (output) await canvas.screenshot({ path: resolve(output, 'browser-schedule-entry.png') });
	await holdTouch(10, 54, 50_000, 1_000_000);
	const contentState = await rpc(page, 'pi_state');
	expect(contentState.fault).toBeNull();
	expect(contentState.pc).toBe(0xf06ba);
	expect(contentState.auxiliary_accepted.slice(-4)).toEqual([0x00f9, 0x01f9, 0x04fe, 0x03fe]);
	const contentCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(contentCapture.error).toBeNull();
	const contentPixels = JSON.parse(contentCapture.resultJson).pixels;
	const contentPanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					contentPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	// Native ROM output supplies the input controls; exclude the blinking text cursor.
	const contentControlsDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 58 }, (_, i) =>
					contentPixels[110 + Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	await expect.poll(renderedPanelDigest).toBe(contentPanelDigest);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-schedule-content.png') });
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < contentPixels.length; i++)
			if (contentPixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		await writeFile(resolve(output, 'browser-content-controller.pbm'), frame);
		await writeFile(
			resolve(output, 'browser-content-diagnostic.json'),
			JSON.stringify({ contentState, contentPanelDigest, contentControlsDigest }),
		);
	}
	expect(contentControlsDigest).toBe('6bf16689d09e4ef64ae62cbcb635a2faa53279709da73f392af8d5a5e3fe20dc');
	expect(errors).toEqual([]);
	await holdTouch(223, 93, 50_000, 300_000);
	await holdTouch(16, 142, 50_000, 300_000);
	await holdTouch(219, 155, 50_000, 1_000_000);
	const savedState = await rpc(page, 'pi_state');
	expect(savedState.fault).toBeNull();
	if (output) await canvas.screenshot({ path: resolve(output, 'browser-schedule-saved.png') });
	await holdTouch(120, 84, 50_000, 1_000_000);
	await holdTouch(198, 7, 50_000, 1_000_000);
	async function holdApplication(id: string) {
		const control = page.getByTestId(id);
		await control.scrollIntoViewIfNeeded();
		await control.hover();
		await page.mouse.down();
		await expect.poll(() => rpc(page, 'pi_state').then((s) => s.tablet[2])).toBe(1);
		await rpc(page, 'step', { instructions: 200_000 });
		await page.mouse.up();
		await rpc(page, 'step', { instructions: 1_000_000 });
	}
	await holdApplication('pi-tablet-phonebook');
	const phonebookState = await rpc(page, 'pi_state');
	expect(phonebookState.fault).toBeNull();
	expect(phonebookState.selector).toBe(0xf9);
	const phonebookCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(phonebookCapture.error).toBeNull();
	const phonebookPixels = JSON.parse(phonebookCapture.resultJson).pixels;
	const phonebookHeaderDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 16 }, (_, i) =>
					phonebookPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(phonebookHeaderDigest).toBe('d989767fc00b3380ab73d5ede75a1dc17279bfb253eac63ccde11676c43b9b86');
	if (output) await canvas.screenshot({ path: resolve(output, 'browser-record-phonebook.png') });
	await holdApplication('pi-tablet-schedule');
	await holdTouch(198, 7, 50_000, 1_000_000);
	const reopenedState = await rpc(page, 'pi_state');
	expect(reopenedState.fault).toBeNull();
	expect(reopenedState.pc).toBe(0xf06ba);
	const reopenedCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(reopenedCapture.error).toBeNull();
	const reopenedPixels = JSON.parse(reopenedCapture.resultJson).pixels;
	const reopenedContentDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 30 * 16 }, (_, i) =>
					reopenedPixels[49 + Math.floor(i / 30) + (10 + (i % 30)) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	const reopenedPanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					reopenedPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	await expect.poll(renderedPanelDigest).toBe(reopenedPanelDigest);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-schedule-reopened.png') });
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < reopenedPixels.length; i++)
			if (reopenedPixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		await writeFile(resolve(output, 'browser-reopened-controller.pbm'), frame);
		await writeFile(
			resolve(output, 'browser-record-diagnostic.json'),
			JSON.stringify({
				savedState,
				phonebookState,
				phonebookHeaderDigest,
				reopenedState,
				reopenedContentDigest,
				reopenedPanelDigest,
			}),
		);
	}
	expect(reopenedContentDigest).toBe('7a5f6055da909a1b7b9fa0da10724478f8c7575e7c8ce761f3b34489c50de0e0');
	expect(reopenedPanelDigest).toBe('9caec32856f67c8ebffff4f202a227807917adf46451004e07aa510e817d96f5');
	expect(errors).toEqual([]);
	if (output) {
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
				blinkSnapshots,
				dateDialogDigest,
				cancelledDigest,
				cancelledState,
				offState,
				stillOff,
				resumedState,
				resumedPanelDigest,
				scheduleState,
				schedulePanelDigest,
				scheduleNativePanelDigest,
				scheduleHeaderDigest,
				contentState,
				contentPanelDigest,
				contentControlsDigest,
				savedState,
				phonebookState,
				phonebookHeaderDigest,
				reopenedState,
				reopenedContentDigest,
				reopenedPanelDigest,
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
	await expect(page.getByTestId('emu-status')).toContainText('1110227');
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
