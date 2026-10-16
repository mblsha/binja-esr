import { test, expect, type Page } from '@playwright/test';
import { readFile, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { inflateSync, deflateSync } from 'node:zlib';
import { decodePvLiveState, encodePvLiveState } from '../src/lib/emulator/pvf1_live_state';
test.skip(!process.env.PVF1_ROM_PATH || !process.env.PVF1_RETAINED_PATH, 'Requires acquired firmware/backing');
const hash = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
function parts(b: Buffer) {
	const n = b.readUInt32LE(40),
		header = JSON.parse(b.subarray(44, 44 + n).toString());
	const native = b.subarray(44 + n + header.lengths[0]);
	return { header, native, document: JSON.parse(inflateSync(native.subarray(40)).toString()) };
}
async function open(page: Page) {
	page.on('dialog', (d) => d.accept());
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('session-panel').locator('summary').click();
}
async function advance(page: Page, n = 6_000_000) {
	await page.getByTestId('session-step-budget').fill(String(n));
	await page.getByTestId('session-step').click();
	await expect(page.getByTestId('session-step')).toBeEnabled();
}
async function save(page: Page, path?: string) {
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-live-export').click();
	const file = (await (await pending).path())!;
	if (path) await copyFile(file, path);
	return readFile(file);
}
async function tap(page: Page, x: number, y: number) {
	const lcd = page.getByTestId('pv-tablet-lcd');
	await lcd.scrollIntoViewIfNeeded();
	const b = (await lcd.boundingBox())!;
	await page.mouse.click(b.x + ((x + 0.5) * b.width) / 336, b.y + ((y + 0.5) * b.height) / 292);
}

test('fresh browser live state preserves pending pen work and continues exactly to Calculator5', async ({
	page,
	browser,
}, info) => {
	await open(page);
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await advance(page);
	await page.getByRole('button', { name: 'Calculator', exact: true }).click();
	for (const [x, y] of [
		[135, 167],
		[195, 176],
		[164, 167],
		[224, 188],
	])
		await tap(page, x, y);
	const path = info.outputPath('pending.pvstate'),
		original = await save(page, path);
	expect(parts(original).header.tablet.strokes).toHaveLength(5);
	await advance(page);
	const reference = await save(page, info.outputPath('continued.pvstate'));
	const pixels = parts(reference).document.hardware.lcd.pixels;
	const pbm = Buffer.alloc(42 * 292);
	for (let i = 0; i < pixels.length; i++) if (pixels[i]) pbm[i >> 3] |= 1 << (7 - (i & 7));
	expect(hash(Buffer.concat([Buffer.from('P4\n336 292\n'), pbm]))).toBe(
		'82a38e7abb006feef7df9dad76977f88b71a6b76c6568db83c54e65bcb45dacf',
	);
	const fresh = await browser.newContext(),
		other = await fresh.newPage();
	try {
		await open(other);
		await other.getByTestId('pv-live-file').setInputFiles(path);
		await expect(other.getByTestId('pv-saved-records')).toContainText('Live state restored');
		expect(hash(await save(other))).toBe(hash(original));
		await advance(other);
		expect(hash(await save(other, info.outputPath('restored-continued.pvstate')))).toBe(hash(reference));
		await other.getByTestId('pv-tablet-lcd').screenshot({ path: info.outputPath('restored-calculator-five.png') });
	} finally {
		await fresh.close();
	}
});

test('bad live states fail atomically before replacing the running machine', async ({ page }) => {
	await open(page);
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await advance(page);
	const original = await save(page),
		{ header, native, document } = parts(original),
		source = await decodePvLiveState(original, header.context);
	const corrupt = Buffer.from(original);
	corrupt[corrupt.length - 1] ^= 1;
	const foreign = structuredClone(source);
	foreign.context.coreIdentity = 'wasm-sha256:' + '0'.repeat(64);
	document.state.pc = 0x200000;
	const payload = deflateSync(Buffer.from(JSON.stringify(document))),
		badNative = Buffer.concat([native.subarray(0, 8), createHash('sha256').update(payload).digest(), payload]);
	const malformed = { ...source, native: badNative };
	for (const [name, buffer, message] of [
		['corrupt', corrupt, 'checksum'],
		['foreign', Buffer.from(await encodePvLiveState(foreign)), 'matching firmware and emulator build'],
		['native', Buffer.from(await encodePvLiveState(malformed)), 'invalid live register'],
	] as const) {
		await page
			.getByTestId('pv-live-file')
			.setInputFiles({ name: name + '.pvstate', mimeType: 'application/octet-stream', buffer });
		await expect(page.getByTestId('pv-saved-records').getByRole('alert')).toContainText(message);
		expect(hash(await save(page))).toBe(hash(original));
	}
});
