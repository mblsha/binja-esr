import { test, expect, type Page } from '@playwright/test';
import { readFile, copyFile } from 'node:fs/promises';
import { decodePvSession, encodePvSession } from '../src/lib/emulator/pvf1_session';
import { createHash } from 'node:crypto';
const hash = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
test.skip(!process.env.PVF1_ROM_PATH || !process.env.PVF1_RETAINED_PATH, 'Requires acquired firmware/backing');
async function open(page: Page) {
	page.on('dialog', (d) => d.accept());
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
}
async function full(page: Page, path?: string) {
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-full-backup-export').click();
	const file = (await (await pending).path())!;
	if (path) await copyFile(file, path);
	return readFile(file);
}

test('portable backup restores records, program-card bytes and recognition model in a fresh browser profile', async ({
	page,
	browser,
}, info) => {
	const rom = process.env.PVF1_PROGRAM_CARD_ROM,
		sram = process.env.PVF1_PROGRAM_CARD_SRAM,
		model = process.env.PVF1_HANDWRITING_PATH;
	test.skip(!rom || !sram || !model, 'Requires acquired program card and qualified recognition model');
	await open(page);
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('pv-card-rom').setInputFiles(rom!);
	await page.getByTestId('pv-card-sram').setInputFiles(sram!);
	await page.getByTestId('pv-card-load').click();
	await expect(page.getByTestId('pv-card-status')).toContainText('Program card imported');
	await page.getByTestId('pv-handwriting-model').setInputFiles(model!);
	await expect(page.getByRole('region', { name: 'PV-F1 handwriting' })).toContainText('Loaded');
	const path = info.outputPath('original.pvbackup'),
		before = await full(page, path),
		media = await decodePvSession(before);
	expect(hash(media.retained)).toBe(hash(await readFile(process.env.PVF1_RETAINED_PATH!)));
	expect(hash(media.cardRom)).toBe(hash(await readFile(rom!)));
	expect(hash(media.cardSram)).toBe(hash(await readFile(sram!)));
	expect(hash(media.handwritingModel)).toBe(hash(await readFile(model!)));
	const fresh = await browser.newContext(),
		other = await fresh.newPage();
	try {
		await open(other);
		await other.getByTestId('pv-full-backup-file').setInputFiles(path);
		await expect(other.getByTestId('pv-saved-records')).toContainText('Full backup restored');
		expect(hash(await full(other, info.outputPath('restored.pvbackup')))).toBe(hash(before));
		await other.reload();
		await expect(other.getByTestId('pv-save-status')).toContainText('Saved at');
		expect(hash(await full(other))).toBe(hash(before));
	} finally {
		await fresh.close();
	}
});

test('portable read-only memory card survives import; corrupt, foreign and malformed native assets leave the active machine unchanged', async ({
	page,
}, info) => {
	await open(page);
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page
		.getByTestId('pv-ram-card-image')
		.setInputFiles({ name: 'readonly.bin', mimeType: 'application/octet-stream', buffer: Buffer.alloc(0x20000, 0x6d) });
	await page.getByRole('checkbox', { name: 'Allow writes on load' }).uncheck();
	await page.getByTestId('pv-ram-card-load').click();
	await expect(page.getByTestId('pv-card-status')).toContainText('Memory card imported');
	const path = info.outputPath('readonly.pvbackup'),
		original = await full(page, path),
		media = await decodePvSession(original);
	await page.getByTestId('pv-ram-card-remove').click();
	await expect(page.getByTestId('pv-card-status')).toContainText('removed');
	await page.getByTestId('pv-full-backup-file').setInputFiles(path);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Full backup restored');
	expect(await page.getByRole('checkbox', { name: 'Allow writes on load' }).isChecked()).toBe(false);
	expect(hash(await full(page))).toBe(hash(original));
	const corrupt = Buffer.from(original);
	corrupt[corrupt.length - 1] ^= 1;
	const foreign = structuredClone(media);
	foreign.retained[8] ^= 1;
	const malformed = structuredClone(media);
	malformed.handwritingModel = new Uint8Array([1, 2, 3]);
	for (const [name, buffer, message] of [
		['corrupt', corrupt, 'checksum'],
		['foreign', Buffer.from(await encodePvSession(foreign)), 'another firmware'],
		['native-model', Buffer.from(await encodePvSession(malformed)), 'model'],
	] as const) {
		await page
			.getByTestId('pv-full-backup-file')
			.setInputFiles({ name: name + '.pvbackup', mimeType: 'application/octet-stream', buffer });
		await expect(page.getByTestId('pv-saved-records').getByRole('alert')).toContainText(message);
		expect(hash(await full(page))).toBe(hash(original));
	}
});
