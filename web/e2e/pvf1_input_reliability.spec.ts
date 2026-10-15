import { test, expect, type Page, type TestInfo } from '@playwright/test';
import { readFile, copyFile } from 'node:fs/promises';
import { decodeRecordedSession } from '../src/lib/emulator/session_recording';

async function open(page: Page, handwriting = false) {
	test.skip(!process.env.PVF1_ROM_PATH || !process.env.PVF1_RETAINED_PATH, 'Requires acquired PV-F1 firmware/backing');
	page.on('dialog', (d) => d.accept());
	await page.addInitScript(() => {
		const NativeWorker = window.Worker;
		window.Worker = class extends NativeWorker {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				let id = 1_000_000;
				(window as any).__observe = (type: string) =>
					new Promise((resolve, reject) => {
						const n = id++;
						const receive = ({ data }: MessageEvent) => {
							if (data.type !== 'reply' || data.id !== n) return;
							this.removeEventListener('message', receive);
							data.ok ? resolve(data.result) : reject(new Error(data.error));
						};
						this.addEventListener('message', receive);
						this.postMessage({ id: n, type });
					});
			}
		};
	});
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByText(/^Restored .*Resume to boot from saved records$/)).toBeVisible();
	if (handwriting) {
		test.skip(!process.env.PVF1_HANDWRITING_PATH, 'Requires the qualified handwriting model');
		await page.getByTestId('advanced-panel').locator('> summary').click();
		await page.getByTestId('pv-handwriting-model').setInputFiles(process.env.PVF1_HANDWRITING_PATH!);
		await expect(page.getByRole('region', { name: 'PV-F1 handwriting' })).toContainText('Loaded');
	}
	await page.getByTestId('session-panel').locator('summary').click();
	await page.getByTestId('session-record-start').click();
	await expect(page.getByTestId('session-status')).toContainText('Recording from a fresh restart');
	await advance(page, 6_000_000);
}
async function advance(page: Page, n = 2_000_000) {
	await page.getByTestId('session-step-budget').fill(String(n));
	await page.getByTestId('session-step').click();
	await expect(page.getByTestId('session-step')).toBeEnabled();
}
async function point(page: Page, x: number, y: number) {
	const lcd = page.getByTestId('pv-tablet-lcd');
	await lcd.scrollIntoViewIfNeeded();
	const b = (await lcd.boundingBox())!;
	return { x: b.x + ((x + 0.5) * b.width) / 336, y: b.y + ((y + 0.5) * b.height) / 292 };
}
async function tap(page: Page, x: number, y: number) {
	const p = await point(page, x, y);
	await page.mouse.click(p.x, p.y);
}
async function finish(page: Page, info: TestInfo) {
	const pending = page.waitForEvent('download');
	await page.getByTestId('session-record-end').click();
	const file = info.outputPath('pv-f1.orgsession');
	await copyFile((await (await pending).path())!, file);
	const backup = page.waitForEvent('download');
	await page.getByTestId('pv-backup-export').click();
	await copyFile((await (await backup).path())!, info.outputPath('final-retained.bin'));
	return decodeRecordedSession(await readFile(file));
}
const observe = (page: Page, type: string) => page.evaluate((type) => (window as any).__observe(type), type);

test('quick ON wakes normal OFF and short pen clicks work while Resume is running', async ({ page }, info) => {
	await open(page);
	await page.getByTestId('vk-off').click();
	await advance(page);
	await expect(page.getByTestId('device-status')).toContainText('device OFF');
	await page.getByTestId('power-on').click();
	// Same wake-settling budget as the qualified native OFF/ON consumer. The
	// firmware can ignore contacts while it is still restoring the foreground.
	await advance(page, 6_000_000);
	await expect(page.getByTestId('device-status')).not.toContainText('device OFF');
	await page.getByRole('button', { name: 'Calculator', exact: true }).click();
	await advance(page);
	const boundaries = async () =>
		Number((await page.getByTestId('session-progress').innerText()).match(/(\d+) boundaries/)![1]);
	const before = await boundaries();
	await page.getByTestId('pause-resume').click();
	for (const [x, y] of [
		[135, 167],
		[195, 176],
		[164, 167],
		[224, 188],
	])
		await tap(page, x, y);
	await expect.poll(boundaries, { timeout: 20_000 }).toBeGreaterThan(before + 4_000_000);
	await page.getByTestId('pause-resume').click();
	await page.getByTestId('pv-tablet-lcd').screenshot({ path: info.outputPath('live-calculator-five.png') });
	const s = await finish(page, info);
	expect(s.events.filter((e) => e.kind === 'on').map((e) => e.pressed)).toEqual([true, false]);
	// Independently observed normal Calendar-background Calculator5 frame.
	expect(s.events.at(-1)).toMatchObject({ sha256: '82a38e7abb006feef7df9dad76977f88b71a6b76c6568db83c54e65bcb45dacf' });
});

test('paused quick app changes and repeated virtual contacts preserve separate taps', async ({ page }, info) => {
	await open(page);
	for (const name of ['Calendar', 'Memo', 'Calculator']) await page.getByRole('button', { name, exact: true }).click();
	await advance(page, 6_000_000);
	// All four clicks are queued before execution, including the repeated digit.
	for (const [x, y] of [
		[135, 167],
		[135, 167],
		[195, 176],
		[164, 167],
		[224, 188],
	])
		await tap(page, x, y);
	await advance(page, 4_000_000);
	await page.getByTestId('pv-tablet-lcd').screenshot({ path: info.outputPath('calculator-25.png') });
	await page.getByTestId('vk-clear').click();
	await page.getByTestId('vk-clear').click();
	await advance(page);
	const s = await finish(page, info);
	expect(
		s.events.filter((e) => e.kind === 'tablet' && e.pressed && e.raw_x >= 900).map((e) => [e.raw_x, e.raw_y]),
	).toEqual([
		[900, 120],
		[980, 380],
		[900, 640],
	]);
	expect(s.events.filter((e) => e.kind === 'matrix' && e.pressed)).toHaveLength(2);
	expect(s.events.filter((e) => e.kind === 'matrix' && !e.pressed)).toHaveLength(2);
});

test('paused handwriting preserves two complete pointer trajectories through normal ROM recognition', async ({
	page,
}, info) => {
	await open(page, true);
	for (const [x, y] of [
		[277, 62],
		[222, 16],
		[100, 90],
		[298, 130],
		[290, 109],
	]) {
		await tap(page, x, y);
		await advance(page);
	}
	const strokes = [
		[
			[186, 210],
			[143, 710],
			[223, 863],
			[313, 810],
		],
		[
			[710, 220],
			[770, 570],
			[693, 723],
		],
	];
	for (const stroke of strokes) {
		for (let i = 0; i < stroke.length; i++) {
			const previous = stroke[Math.max(0, i - 1)],
				next = stroke[i];
			for (let j = i ? 1 : 12; j <= 12; j++) {
				const t = j / 12,
					p = await point(
						page,
						Math.round(18 + (previous[0] + (next[0] - previous[0]) * t) * 0.06),
						Math.round(174 + (previous[1] + (next[1] - previous[1]) * t) * 0.061),
					);
				await page.mouse.move(p.x, p.y);
				if (i === 0) await page.mouse.down();
			}
		}
		await page.mouse.up();
	}
	await advance(page, 6_000_000);
	const status = await observe(page, 'pv_state');
	expect(status.auxiliary.classifications).toBe(1);
	expect(status.auxiliary.recognized_labels[0]).toBe('い');
	await page.getByTestId('pv-tablet-lcd').screenshot({ path: info.outputPath('recognized-i.png') });
	const s = await finish(page, info);
	// Initial five menu taps and two full strokes; motion is not coalesced away.
	const pen = s.events.filter((e) => e.kind === 'tablet');
	expect(pen.filter((e) => !e.pressed)).toHaveLength(7);
	expect(pen.filter((e) => e.pressed).length).toBeGreaterThan(50);
});

test('pointer cancel, lost capture, blur and hidden/resume discard queued contacts and permit fresh input', async ({
	page,
}, info) => {
	await open(page);
	const lcd = page.getByTestId('pv-tablet-lcd');
	// Actual browser key events exercise matrix and ON cleanup while paused.
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.press('F7');
	await page.keyboard.press('F7');
	await page.keyboard.down('F12');
	await page.evaluate(() => window.dispatchEvent(new Event('blur')));
	await page.keyboard.up('F12');
	expect((await observe(page, 'input_state')).rust).toEqual({ matrix: [], on: false });
	for (const cancel of ['pointercancel', 'lostpointercapture', 'blur', 'visibilitychange']) {
		// The first contact is delivered, the next is queued while paused.
		await tap(page, 50, 50);
		await tap(page, 70, 70);
		const p = await point(page, 90, 90);
		await page.mouse.move(p.x, p.y);
		await page.mouse.down();
		if (cancel === 'pointercancel')
			await page.evaluate(() => window.dispatchEvent(new PointerEvent('pointercancel', { pointerId: 1 })));
		else if (cancel === 'lostpointercapture') await lcd.dispatchEvent('lostpointercapture', { pointerId: 1 });
		else if (cancel === 'blur') await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		else
			await page.evaluate(() => {
				Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
				document.dispatchEvent(new Event('visibilitychange'));
				delete (document as any).hidden;
				document.dispatchEvent(new Event('visibilitychange'));
			});
		await page.mouse.up();
		await advance(page);
	}
	await page.getByRole('button', { name: 'Calculator', exact: true }).click();
	await advance(page);
	for (const [x, y] of [
		[135, 167],
		[195, 176],
		[164, 167],
		[224, 188],
	]) {
		await tap(page, x, y);
		await advance(page);
	}
	await lcd.screenshot({ path: info.outputPath('resumed-calculator-five.png') });
	const s = await finish(page, info);
	const pen = s.events.filter((e) => e.kind === 'tablet');
	// Each cancellation releases the first contact and drops the other two.
	expect(pen.filter((e) => !e.pressed)).toHaveLength(9);
	expect(pen.filter((e) => e.pressed)).toHaveLength(9);
	expect(s.events.filter((e) => e.kind === 'on').map((e) => e.pressed)).toEqual([true, false]);
});
