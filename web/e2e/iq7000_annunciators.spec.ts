import { test, expect, type Page } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';

async function runScript(page: Page, source: string) {
	await page.getByTestId('fnr-panel').evaluate((el: HTMLDetailsElement) => (el.open = true));
	await page.getByTestId('fnr-editor').fill(source);
	await page.getByTestId('fnr-run').click();
	await expect(page.getByTestId('fnr-run')).toBeEnabled({ timeout: 120_000 });
	expect(await page.getByTestId('fnr-error').allTextContents()).toEqual([]);
}

async function openIq(page: Page) {
	// Start with the requested preset, not a concurrent preset change while
	// the initial ROM/WASM load is still in flight.
	await page.addInitScript(() => localStorage.setItem('sc62015:rom-model', 'iq-7000'));
	await page.goto('/');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
	await expect(page.locator('.lcd-display canvas')).toHaveAttribute('width', '488');
}

async function rowHasInk(page: Page, y: number) {
	return page.locator('.lcd-display canvas').evaluate((el: HTMLCanvasElement, y) => {
		const rgba = el.getContext('2d')!.getImageData(98 * 4, y * 4, 24 * 4, 7 * 4).data;
		return rgba.some((shade, i) => i % 4 === 0 && shade < 128);
	}, y);
}

async function saveCanvasPixels(page: Page, path: string) {
	// Serialize the actual live canvas, without the browser screenshot's
	// fractional CSS-position rounding adding an extra row at its edge.
	const png = await page.locator('.lcd-display canvas').evaluate((el: HTMLCanvasElement) => el.toDataURL('image/png'));
	await writeFile(path, Buffer.from(png.split(',')[1], 'base64'));
}

test('IQ full-glass capture and live worker frame agree, including clear and unknown bits', async ({ page }, info) => {
	// Diagnostic fixture only. No ROM UI claim: write one-hot LCD bytes explicitly.
	const rom = Buffer.alloc(0x40000);
	rom[rom.length - 1] = 0x0c;
	await page.route('**/api/rom?model=iq-7000', (route) =>
		route.fulfill({
			status: 200,
			body: rom,
			contentType: 'application/octet-stream',
		}),
	);
	await openIq(page);
	await runScript(
		page,
		`
		await e.memory.write(0x1FDA3, 1, 0x08); // stale workspace CAPS
		await e.memory.write(0x6160, 1, 0x10); // physical SHIFT only
		const frame = await e.lcd.capture();
		e.assert(frame.cols === 488 && frame.rows === 256 && frame.pixel_scale === 4 && frame.pixel_format === 'gray8');
		e.assert((await e.lcd.pixels()).length === 96 * 64);
		e.assert(frame.annunciators.shift && !frame.annunciators.caps);
		const smaller = await e.lcd.capture({ scale: 3 });
		e.assert(smaller.cols === 366 && smaller.rows === 192);
	`,
	);
	await expect(page.locator('.lcd-display canvas')).toHaveAttribute('width', '488');
	await expect.poll(() => rowHasInk(page, 22)).toBe(true);
	await expect.poll(() => rowHasInk(page, 29)).toBe(false);

	await runScript(
		page,
		`
		await e.memory.write(0x6160, 1, 0);
		await e.memory.write(0x6161, 1, 0x80); // unknown: no named segment
		const frame = await e.lcd.capture();
		e.assert(!frame.annunciators.shift && !frame.annunciators.caps);
		e.assert(frame.annunciators.unmapped_shadow_bytes[1] === 0x80);
	`,
	);
	await expect.poll(() => rowHasInk(page, 22)).toBe(false);
	await runScript(
		page,
		`
		for (const [a, v] of [[0x6160, 0xff], [0x6161, 7], [0x61e0, 0x80], [0x61e1, 0x80]])
			await e.memory.write(a, 1, v);
	`,
	);
	await expect.poll(() => rowHasInk(page, 0)).toBe(true);
	await saveCanvasPixels(page, info.outputPath('diagnostic-all-segments.png'));
});

test('real IQ ROM drives CAPS, SHIFT and key-beep segments without framebuffer injection', async ({ page }, info) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await openIq(page);
	await runScript(
		page,
		`
		await e.step(500000);
		await e.keys.app.tap('memo');
		await e.step(160000);
		const frame = await e.lcd.capture();
		e.assert(frame.annunciators.caps && frame.annunciators.key_beep);
		e.assert(!frame.annunciators.shift);
		return { ...frame, pixels: undefined, text: await e.lcd.text(), pc: e.reg(Reg.PC) };
	`,
	);
	await expect.poll(() => rowHasInk(page, 29)).toBe(true);
	await saveCanvasPixels(page, info.outputPath('rom-memo.png'));
	await runScript(
		page,
		`
		await e.keys.app.tap('shift'); // scanner event 02 -> logical keycode 01
		await e.step(80000);
		const frame = await e.lcd.capture();
		e.assert(frame.annunciators.shift, JSON.stringify(frame.annunciators));
		return { ...frame, pixels: undefined, text: await e.lcd.text(), pc: e.reg(Reg.PC) };
	`,
	);
	await expect.poll(() => rowHasInk(page, 22)).toBe(true);
	await saveCanvasPixels(page, info.outputPath('rom-memo-shift.png'));
});

test('private ROM demo comparison capture', async ({ page }, info) => {
	test.skip(!process.env.IQ7000_DEMO_PROOF_SCRIPT, 'Requires private ROM demo harness');
	await openIq(page);
	const source = await readFile(process.env.IQ7000_DEMO_PROOF_SCRIPT!, 'utf8');
	await runScript(page, source);
	const proof = JSON.parse((await page.getByTestId('fnr-panel').locator('pre.log').last().textContent())!);
	expect(proof.setup.reason).toBe('returned');
	expect(proof.weekly.reason).toBe('returned');
	await expect
		.poll(() =>
			page.locator('.lcd-display canvas').evaluate((el: HTMLCanvasElement) => {
				const pixels = el.getContext('2d')!.getImageData(0, 0, el.width, el.height).data;
				let hash = 0x811c9dc5;
				for (let i = 0; i < pixels.length; i += 4) {
					if (pixels[i] !== pixels[i + 1] || pixels[i] !== pixels[i + 2] || pixels[i + 3] !== 255)
						throw new Error('Live canvas is not opaque grayscale');
					hash = Math.imul(hash ^ pixels[i], 0x01000193) >>> 0;
				}
				return hash;
			}),
		)
		.toBe(proof.rasterHash);
	await writeFile(info.outputPath('rom-weekly-demo.json'), JSON.stringify(proof, null, 2));
	await saveCanvasPixels(page, info.outputPath('rom-weekly-demo.png'));
	await page.locator('.lcd-display canvas').screenshot({ path: info.outputPath('rom-weekly-demo-live.png') });
});
