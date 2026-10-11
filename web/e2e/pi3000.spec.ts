import { test, expect, type Page } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

const bundlePath = process.env.PI3000_ROM_PATH;
const output = process.env.PI3000_PROOF_DIR;
test.skip(!bundlePath, 'Supply verified private PI bundle with PI3000_ROM_PATH');
const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__piRpc(type, payload), { type, payload });

async function installPiRpc(page: Page) {
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
}

test('PI browser initializes and reopens a saved Schedule record after power and restart', async ({
	page,
}, testInfo) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await installPiRpc(page);
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
	// Ordered window protection changes one indicator-strip pixel (38,239).
	// The visible calibration panel remains identical to the selected-only control.
	expect(digest).toBe('609df92844a1c6b065210f0a7a8c79ddc130b2d00da167c151c502ec9f5ddde2');
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
	expect(contrastDigest).toBe('4d4e70ad6ae2204f0b4cb522acec4cefc676ef3e2c33c0d1c3f32db73da22549');
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
	expect(initializationDigest).toBe('c4a3195c582bd4f4bc3e65456ce5da1f07fc64d6f8ab531022f485f0c6c77dd6');
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
	expect(calendarDigest).toBe('316007e39fb2600274c94cd7f9c0a62de8c2b1d0aa5d31c3bba837f3c4f2c4d4');
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
	expect(contactDigest).toBe('0e6349a75f611f12bd250111bb7097428c422a37350656844b0dfbb188d5f58e');
	expect(errors).toEqual([]);
	const blinkSnapshots = [{ state: contactState, digest: contactDigest }];
	for (let i = 0; i < 32; i++) {
		await rpc(page, 'step', { instructions: 100_000 });
		blinkSnapshots.push({ state: await rpc(page, 'pi_state'), digest: await controllerDigest() });
	}
	expect([...new Set(blinkSnapshots.map((s) => s.digest))]).toEqual([
		'0e6349a75f611f12bd250111bb7097428c422a37350656844b0dfbb188d5f58e',
		'9dc0ca3831b5a4c12282a81bf08df071258f59fe312f1cceac40964f400ce2cf',
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
	// The entry's lower-priority clear must preserve the foreground keypad.
	// Its complete lower panel matches the independently captured numeric mode.
	const reopenedKeypadDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 84 }, (_, i) =>
					reopenedPixels[84 + Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(reopenedKeypadDigest).toBe('8d33de968a2f9e23e2eedf8b66b051ebcdaabacc1df9fc8ae2e78844d4a5807f');
	expect(reopenedPanelDigest).toBe('d10f60acd356dce92b83d891794bb22e02013aa6d98186c2a80f62bc53904638');
	expect(errors).toEqual([]);
	await holdKey('vk-off', 200_000, 750_000);
	const recordOffState = await rpc(page, 'pi_state');
	expect(recordOffState.pc).toBe(0xf8ed5);
	await rpc(page, 'step', { instructions: 100_000 });
	const recordStillOff = await rpc(page, 'pi_state');
	expect(recordStillOff.native_instructions).toBe(recordOffState.native_instructions);
	await holdKey('vk-on', 250_000, 4_770_000);
	const recordWakeState = await rpc(page, 'pi_state');
	expect(recordWakeState.fault).toBeNull();
	expect(recordWakeState.pc).toBe(0xf06ba);
	// Calendar -> selected day -> weekly view -> day view -> record -> Edit.
	await holdTouch(222, 34, 50_000, 300_000);
	await holdTouch(222, 34, 50_000, 1_000_000);
	await holdTouch(14, 146, 50_000, 1_000_000);
	await holdTouch(100, 70, 50_000, 300_000);
	await holdTouch(198, 7, 50_000, 1_000_000);
	const retainedRecordState = await rpc(page, 'pi_state');
	expect(retainedRecordState.fault).toBeNull();
	expect(retainedRecordState.pc).toBe(0xf06ba);
	const retainedCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(retainedCapture.error).toBeNull();
	const retainedPixels = JSON.parse(retainedCapture.resultJson).pixels;
	const retainedPanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					retainedPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-schedule-power-reopened.png') });
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < retainedPixels.length; i++)
			if (retainedPixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		await writeFile(resolve(output, 'browser-power-reopened-controller.pbm'), frame);
		await writeFile(
			resolve(output, 'browser-power-record-diagnostic.json'),
			JSON.stringify({ recordOffState, recordStillOff, recordWakeState, retainedRecordState, retainedPanelDigest }),
		);
	}
	expect(retainedPanelDigest).toBe(reopenedPanelDigest);
	await expect.poll(renderedPanelDigest).toBe(retainedPanelDigest);
	expect(errors).toEqual([]);
	const downloadPromise = page.waitForEvent('download');
	await page.getByTestId('pi-backup-export').click();
	const download = await downloadPromise;
	const backupPath = output ? resolve(output, 'browser-record.piram') : testInfo.outputPath('browser-record.piram');
	await download.saveAs(backupPath);
	const backup = await readFile(backupPath);
	expect(backup.length).toBe(104 + 0x40000 + 0x4000 + 32);
	expect(backup.subarray(0, 8)).toEqual(Buffer.from('PIRAM01\0'));
	const beforeInvalidRestore = await rpc(page, 'pi_state');
	const corrupt = Buffer.from(backup);
	corrupt[corrupt.length - 1] ^= 1;
	await page
		.getByTestId('pi-backup-file')
		.setInputFiles({ name: 'corrupt.piram', mimeType: 'application/octet-stream', buffer: corrupt });
	await expect(page.getByTestId('pi-backup-error')).toContainText('checksum');
	expect(await rpc(page, 'pi_state')).toEqual(beforeInvalidRestore);
	await expect.poll(renderedPanelDigest).toBe(retainedPanelDigest);
	await page
		.getByTestId('pi-backup-file')
		.setInputFiles({ name: 'saved.piram', mimeType: 'application/octet-stream', buffer: backup });
	await expect(page.getByTestId('pi-backup-status')).toContainText('Restored saved.piram');
	const restoredState = await rpc(page, 'pi_state');
	expect(restoredState.native_instructions).toBe(0);
	expect(restoredState.cycles).toBe(0);
	expect(restoredState.fault).toBeNull();
	await rpc(page, 'step', { instructions: 400_000 });
	for (const [x, y] of [
		[32, 32],
		[205, 32],
		[32, 135],
		[205, 135],
	])
		await holdTouch(x, y, 200_000, 100_000);
	await rpc(page, 'step', { instructions: 3_000_000 });
	await holdTouch(219, 155, 250_000, 1_000_000);
	await rpc(page, 'step', { instructions: 5_000_000 });
	await holdTouch(222, 34, 50_000, 300_000);
	await holdTouch(222, 34, 50_000, 1_000_000);
	await holdTouch(14, 146, 50_000, 1_000_000);
	await holdTouch(100, 70, 50_000, 300_000);
	await holdTouch(198, 7, 50_000, 1_000_000);
	const restartedRecordState = await rpc(page, 'pi_state');
	expect(restartedRecordState.fault).toBeNull();
	expect(restartedRecordState.pc).toBe(0xf06ba);
	const restartedCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(restartedCapture.error).toBeNull();
	const restartedPixels = JSON.parse(restartedCapture.resultJson).pixels;
	const restartedPanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					restartedPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(restartedPanelDigest).toBe(retainedPanelDigest);
	await expect.poll(renderedPanelDigest).toBe(restartedPanelDigest);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-schedule-restarted.png') });
		const frame = Buffer.alloc(header.length + 30 * 336);
		header.copy(frame);
		for (let i = 0; i < restartedPixels.length; i++)
			if (restartedPixels[i] === 0) frame[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
		await writeFile(resolve(output, 'browser-restarted-controller.pbm'), frame);
		await writeFile(
			resolve(output, 'browser-restart-diagnostic.json'),
			JSON.stringify({
				beforeInvalidRestore,
				restoredState,
				restartedRecordState,
				restartedPanelDigest,
				backupSha256: createHash('sha256').update(backup).digest('hex'),
				invalidRestorePreservedState: true,
			}),
		);
	}
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
	// Clock is the second printed column at raw240,572. It remains open after
	// release; the OZ hold-to-view convention does not apply to this control.
	await holdApplication('pi-tablet-clock');
	await rpc(page, 'step', { instructions: 500_000 });
	const clockState = await rpc(page, 'pi_state');
	expect(clockState.fault).toBeNull();
	await expect.poll(renderedPanelDigest).toBe('28e58649ba1e5e1f499cecbad6b5e69eca6910c38694c0e4cd3753d67d026c77');
	const clockDigest = await renderedPanelDigest();
	expect(clockDigest).toBe('28e58649ba1e5e1f499cecbad6b5e69eca6910c38694c0e4cd3753d67d026c77');
	await holdTouch(38, 14, 250_000, 1_500_000);
	const clockDialogState = await rpc(page, 'pi_state');
	expect(clockDialogState.fault).toBeNull();
	await expect.poll(renderedPanelDigest).toBe('907c04c5345fbdd18946d32c317fb9fc83a949779ff2954ad12c3bb761ea3078');
	const clockDialogDigest = await renderedPanelDigest();
	expect(clockDialogDigest).toBe('907c04c5345fbdd18946d32c317fb9fc83a949779ff2954ad12c3bb761ea3078');
	// Clear uses column1,row6 (scanner byte1C). It closes the rejected-hour
	// popup and clears the hour field through the ROM's matrix scanner.
	await holdTouch(16, 142, 50_000, 300_000);
	await holdTouch(219, 49, 50_000, 1_000_000);
	await expect.poll(renderedPanelDigest).toBe('59927dd08f7183fb50ceaf44296b00cad20c9289a0a0f85f666e79a32c6086af');
	await holdKey('vk-clear', 50_000, 3_000_000);
	const clearedClockError = await rpc(page, 'eval_js', {
		source:
			'return { capture: await e.lcd.capture(1), time: await Promise.all(Array.from({length:5}, (_,i) => e.memory.read(0x8000+i))) };',
	});
	expect(clearedClockError.error).toBeNull();
	const clearedClockResult = JSON.parse(clearedClockError.resultJson);
	expect(clearedClockResult.time).toEqual([0x00, 0xc0, 0x08, 0xe1, 0x05]);
	const clearedClockDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					clearedClockResult.capture.pixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(clearedClockDigest).toBe('8e86fbeb187388db18991a3323257a4ad27779dc5c04db24c4ed9171fe619820');
	await expect.poll(renderedPanelDigest).toBe(clearedClockDigest);
	if (output)
		await writeFile(
			resolve(output, 'browser-clock-error-clear.json'),
			JSON.stringify({ clearedClockDigest, time: clearedClockResult.time, state: await rpc(page, 'pi_state') }),
		);
	await holdTouch(219, 69, 50_000, 3_000_000);
	const cancelledClockCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(cancelledClockCapture.error).toBeNull();
	const cancelledClockPixels = JSON.parse(cancelledClockCapture.resultJson).pixels;
	const cancelledClockDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					cancelledClockPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	if (output)
		await writeFile(
			resolve(output, 'browser-clock-cancel.json'),
			JSON.stringify({
				cancelledClockDigest,
				renderedDigest: await renderedPanelDigest(),
				state: await rpc(page, 'pi_state'),
			}),
		);
	expect(cancelledClockDigest).toBe(clockDigest);
	await expect.poll(renderedPanelDigest).toBe(clockDigest);
	await holdTouch(84, 14, 50_000, 1_500_000);
	const dailyAlarmState = await rpc(page, 'pi_state');
	expect(dailyAlarmState.fault).toBeNull();
	const dailyCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(dailyCapture.error).toBeNull();
	const dailyPixels = JSON.parse(dailyCapture.resultJson).pixels;
	const dailyNativeDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) => (dailyPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0)),
			),
		)
		.digest('hex');
	if (output)
		await writeFile(
			resolve(output, 'browser-daily-native.json'),
			JSON.stringify({ dailyNativeDigest, renderedDigest: await renderedPanelDigest() }),
		);
	// Native idle replay establishes two phases differing only in the focused
	// hour digit's 12x18 rectangle; all other pixels are identical.
	expect([
		'ce421b2abf4dfdf87bc80188229f4b5b47fd8c831f8f3a8769ca6006c193f0d9',
		'1e8fc8ef30ec57cc7b31ee95adfde2e4397789a89823d8311d4f110e030a79b2',
	]).toContain(dailyNativeDigest);
	await expect.poll(renderedPanelDigest).toBe(dailyNativeDigest);
	const dailyAlarmDigest = await renderedPanelDigest();
	expect(dailyAlarmDigest).toBe(dailyNativeDigest);
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-daily-alarm-dialog.png') });
		await writeFile(
			resolve(output, 'browser-clock.json'),
			JSON.stringify({
				clockState,
				clockDigest,
				clockDialogState,
				clockDialogDigest,
				dailyAlarmState,
				dailyAlarmDigest,
			}),
		);
	}
	// Ordinary numeric entry accepts both hour digits. The earlier single-digit
	// candidate is rejected by the ROM's hour-field status check.
	await holdTouch(219, 69, 50_000, 3_000_000);
	await holdTouch(38, 14, 250_000, 1_500_000);
	await holdTouch(16, 156, 50_000, 300_000);
	await holdTouch(16, 142, 50_000, 300_000);
	await holdTouch(219, 49, 50_000, 3_000_000);
	const editedClock = await rpc(page, 'eval_js', {
		source:
			'return { capture: await e.lcd.capture(1), time: await Promise.all(Array.from({length:5}, (_,i) => e.memory.read(0x8000+i))) };',
	});
	expect(editedClock.error).toBeNull();
	const editedClockResult = JSON.parse(editedClock.resultJson);
	expect(editedClockResult.time).toEqual([0x00, 0x00, 0x09, 0xe1, 0x05]);
	const editedClockDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					editedClockResult.capture.pixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(editedClockDigest).toBe('5e4f2ceed9b92fe52d94cb31c85f19b51145a1e62a9ac7389e750e78ca888d91');
	await expect.poll(renderedPanelDigest).toBe(editedClockDigest);
	await holdTouch(84, 14, 50_000, 1_500_000);
	for (const y of [156, 142, 156, 142]) await holdTouch(16, y, 50_000, 300_000);
	await holdTouch(82, 35, 50_000, 300_000);
	await holdTouch(219, 49, 50_000, 3_000_000);
	const savedAlarm = await rpc(page, 'eval_js', {
		source:
			'return { capture: await e.lcd.capture(1), alarm: await Promise.all(Array.from({length:6}, (_,i) => e.memory.read(0x8010+i))) };',
	});
	expect(savedAlarm.error).toBeNull();
	const savedAlarmResult = JSON.parse(savedAlarm.resultJson);
	expect(savedAlarmResult.alarm).toEqual([0x00, 0x01, 0x09, 0xe1, 0x05, 0x07]);
	const savedAlarmDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					savedAlarmResult.capture.pixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(savedAlarmDigest).toBe('8d8413e74cf75231927455dd8bf51a81bd99d2d296d604fd67a5c7ac3e3a23bf');
	await expect.poll(renderedPanelDigest).toBe(savedAlarmDigest);
	expect((await rpc(page, 'pi_state')).fault).toBeNull();
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-clock-edited-alarm.png') });
		await writeFile(
			resolve(output, 'browser-clock-edited-alarm.json'),
			JSON.stringify({
				editedClockDigest,
				time: editedClockResult.time,
				savedAlarmDigest,
				alarm: savedAlarmResult.alarm,
				state: await rpc(page, 'pi_state'),
			}),
		);
	}
	// Resource-backed Calculator controls: the tall middle button is plus,
	// and the tall button to its right is equals. Ordinary contacts compute 5.
	await holdApplication('pi-tablet-calculator');
	await rpc(page, 'step', { instructions: 500_000 });
	await expect.poll(renderedPanelDigest).toBe('7de4ae38df39c618ff54687e14a373736ba39bf1ba4ef2fa0c425b49536ffbc6');
	for (const [x, y] of [
		[87, 118],
		[156, 118],
		[116, 118],
		[187, 132],
	])
		await holdTouch(x, y, 50_000, 300_000);
	const calculatorState = await rpc(page, 'pi_state');
	expect(calculatorState.fault).toBeNull();
	const calculatorCapture = await rpc(page, 'eval_js', { source: 'return await e.lcd.capture(1);' });
	expect(calculatorCapture.error).toBeNull();
	const calculatorPixels = JSON.parse(calculatorCapture.resultJson).pixels;
	const calculatorPanelDigest = createHash('sha256')
		.update(
			Buffer.from(
				Array.from({ length: 239 * 168 }, (_, i) =>
					calculatorPixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
				),
			),
		)
		.digest('hex');
	expect(calculatorPanelDigest).toBe('7cd4db8f68fdb0eaa7211a85986bb333c7e849dc701d08f49327401b44a48d15');
	await expect.poll(renderedPanelDigest).toBe(calculatorPanelDigest);
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-calculator-add.png') });
		await writeFile(
			resolve(output, 'browser-calculator-add.json'),
			JSON.stringify({ calculatorState, calculatorPanelDigest, expression: '2 + 3 = 5' }),
		);
	}
	const calculatorContacts: Record<string, [number, number]> = {
		'0': [71, 146],
		'2': [86, 118],
		'3': [116, 118],
		'4': [56, 90],
		'6': [116, 90],
		'7': [56, 62],
		'8': [86, 62],
		'9': [116, 62],
		C: [218, 62],
		'-': [188, 90],
		'*': [158, 62],
		'/': [188, 62],
		'=': [188, 132],
		T: [20, 62],
		R: [20, 90],
		M: [20, 146],
	};
	const calculatorOperations = [
		{ keys: 'C9-4=', result: '5', digest: '7cd4db8f68fdb0eaa7211a85986bb333c7e849dc701d08f49327401b44a48d15' },
		{ keys: 'C6*7=', result: '42', digest: 'e1b01d24183f8fc56da380dab938a1ea9016429c097d6c435149f3f474cd4d41' },
		{ keys: 'C8/2=', result: '4', digest: 'c7e7baeff84d46cf59e2e208208a5cf94bca9e7c3f52d8b0d40dc27bce51eba7' },
		{ keys: 'C2/0=', result: '0 E', digest: '45c3306cc66d2638650fd8c9f710340ea6036f9002aa117d2a0c9fb10b81a203' },
		{ keys: 'C', result: '0', digest: '7de4ae38df39c618ff54687e14a373736ba39bf1ba4ef2fa0c425b49536ffbc6' },
		{ keys: 'T2MC3MCR', result: '5 M', digest: '37a365544627c8a797797b95c52bcaab78e9dca8fc17ab22829f8e2a85bc0d53' },
	];
	for (const operation of calculatorOperations) {
		for (const key of operation.keys) {
			const [x, y] = calculatorContacts[key];
			await holdTouch(x, y, 50_000, 300_000);
		}
		expect((await rpc(page, 'pi_state')).fault).toBeNull();
		await expect.poll(renderedPanelDigest).toBe(operation.digest);
	}
	expect(errors).toEqual([]);
	if (output)
		await writeFile(
			resolve(output, 'browser-calculator-operations.json'),
			JSON.stringify({ calculatorOperations, state: await rpc(page, 'pi_state') }),
		);
});

test('PI daily alarm returns to Clock through ordinary C/CE input', async ({ page }) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await installPiRpc(page);
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(String(error)));
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	const initial = await rpc(page, 'pi_state');
	expect(initial.native_instructions).toBe(0);
	expect(initial.rtc_second_period).toBeNull();
	expect(initial.rtc_event_candidate).toBe(false);
	await expect(rpc(page, 'pi_rtc_candidate', { secondPeriod: 0, events: true })).rejects.toThrow(/positive uint32/);
	const configured = await rpc(page, 'pi_rtc_candidate', { secondPeriod: 1_024_000, events: true });
	expect(configured.rtc_event_candidate).toBe(true);
	const { generation } = await rpc(page, 'input_state');
	// Exact ordinary ADC contacts from the independent native alarm replay.
	// No RAM, event, deadline or framebuffer is installed by this consumer.
	const replay = JSON.parse(await readFile(resolve('e2e/fixtures/pi3000-clock-alarm-cce.json'), 'utf8'));
	for (const step of replay.steps.slice(0, 52)) {
		if (step.tablet)
			await rpc(page, 'pi_tablet', {
				contact: step.tablet,
				owner: 'alarm-raw-prefix',
				generation,
				// The supplied replay owns exact raw hold/release budgets.
				cancel: !step.tablet.pressed,
			});
		await rpc(page, 'step', { instructions: step.boundaries });
	}
	await expect(rpc(page, 'pi_rtc_candidate', { secondPeriod: 1_024_000, events: false })).rejects.toThrow(
		/before execution/,
	);
	const canvas = page.getByTestId('pi-tablet-lcd').locator('canvas');
	async function snapshot() {
		const observed = await rpc(page, 'eval_js', {
			source:
				'return { capture: await e.lcd.capture(1), rtc: await Promise.all(Array.from({length:32}, (_,i) => e.memory.read(0x8000+i))) };',
		});
		expect(observed.error).toBeNull();
		const data = JSON.parse(observed.resultJson);
		const panel = Buffer.from(
			Array.from({ length: 239 * 168 }, (_, i) =>
				data.capture.pixels[Math.floor(i / 239) + (i % 239) * 240] === 0 ? 1 : 0,
			),
		);
		return { panelSha256: createHash('sha256').update(panel).digest('hex'), rtc: data.rtc };
	}
	async function renderedDigest() {
		return canvas.evaluate(async (element: HTMLCanvasElement) => {
			const rgba = element.getContext('2d')!.getImageData(0, 0, 239, 168).data;
			const bytes = new Uint8Array(239 * 168);
			for (let i = 0; i < bytes.length; i++) bytes[i] = rgba[i * 4] < 128 ? 1 : 0;
			const digest = await crypto.subtle.digest('SHA-256', bytes);
			return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
		});
	}
	const popup = await snapshot();
	expect(popup.panelSha256).toBe('2da5d7a1c65dfba1f6d5b60fad0781fe992d532d6686087a8d088edfcfaa3ee3');
	await expect.poll(renderedDigest).toBe(popup.panelSha256);
	if (output) {
		await mkdir(output, { recursive: true });
		await canvas.screenshot({ path: resolve(output, 'browser-alarm-cce-before.png') });
	}
	// C/CE uses the existing contact1. Its calendar Date action remains covered.
	const key = page.getByTestId('vk-date');
	await expect(key).toHaveAttribute('aria-label', 'C/CE');
	await key.scrollIntoViewIfNeeded();
	const box = (await key.boundingBox())!;
	await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
	await page.mouse.down();
	await expect(key).toHaveClass(/delivered/);
	await rpc(page, 'step', { instructions: 50_000 });
	await page.mouse.up();
	await expect(key).not.toHaveClass(/delivered/);
	await rpc(page, 'step', { instructions: 2_000_000 });
	const cleared = await snapshot();
	expect(cleared.panelSha256).toBe('2e60b0d0e68bea00373b3084e874fda77d381e380a9fb34de58cb563d614ce4b');
	await expect.poll(renderedDigest).toBe(cleared.panelSha256);
	await rpc(page, 'step', { instructions: 3_000_000 });
	const persistent = await snapshot();
	expect(persistent.panelSha256).toBe(cleared.panelSha256);
	expect(persistent.rtc.slice(0, 5)).toEqual([0x2f, 0x01, 0x09, 0xe1, 0x05]);
	expect(persistent.rtc.slice(16, 22)).toEqual([0x00, 0x01, 0x11, 0xe1, 0x05, 0x07]);
	const state = await rpc(page, 'pi_state');
	expect(state.fault).toBeNull();
	expect(state.native_instructions).toBe(22_242_927);
	expect(errors).toEqual([]);
	if (output) {
		await canvas.screenshot({ path: resolve(output, 'browser-alarm-cce-after.png') });
		await writeFile(
			resolve(output, 'browser-alarm-cce.json'),
			JSON.stringify({ initial, configured, popup, cleared, persistent, state, errors }),
		);
	}
});

test('PI no-Worker browser fallback shares panel, contacts and backup restore', async ({ page }, testInfo) => {
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
	const downloadPromise = page.waitForEvent('download');
	await page.getByTestId('pi-backup-export').click();
	const download = await downloadPromise;
	const backupPath = output ? resolve(output, 'browser-fallback.piram') : testInfo.outputPath('browser-fallback.piram');
	await download.saveAs(backupPath);
	const backup = await readFile(backupPath);
	expect(backup.subarray(0, 8)).toEqual(Buffer.from('PIRAM01\0'));
	const beforeInvalidRestore = await page.getByTestId('emu-status').innerText();
	const bad = Buffer.from(backup);
	bad[bad.length - 1] ^= 1;
	await page
		.getByTestId('pi-backup-file')
		.setInputFiles({ name: 'bad.piram', mimeType: 'application/octet-stream', buffer: bad });
	await expect(page.getByTestId('pi-backup-error')).toContainText('checksum');
	expect(await page.getByTestId('emu-status').innerText()).toBe(beforeInvalidRestore);
	await page
		.getByTestId('pi-backup-file')
		.setInputFiles({ name: 'fallback.piram', mimeType: 'application/octet-stream', buffer: backup });
	await expect(page.getByTestId('pi-backup-status')).toContainText('Restored fallback.piram');
	await expect(page.getByTestId('emu-status')).toContainText('Instr: 0');
	expect(errors).toEqual([]);
	if (output)
		await writeFile(
			resolve(output, 'browser-fallback-restore.json'),
			JSON.stringify({
				backupSha256: createHash('sha256').update(backup).digest('hex'),
				invalidRestorePreservedState: true,
				restoredNativeInstructions: 0,
			}),
		);
});
