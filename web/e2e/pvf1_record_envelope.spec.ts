import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { parseNativeRecords, recordsJson } from '../src/lib/emulator/pvf1_records';
test.skip(!process.env.PVF1_ROM_PATH, 'Requires acquired firmware');

test('large escaped lossless JSON reopens through the file picker without losing opaque bytes', async ({ page }) => {
	const start = Buffer.from('"i","FUTURE"\r\n"T","Unknown"\r\n"D","'),
		end = Buffer.from('"\r\n\x1a');
	const bytes = Buffer.concat([start, Buffer.alloc(2 * 1024 * 1024), end]);
	const text = await recordsJson(parseNativeRecords(bytes));
	expect(Buffer.byteLength(text)).toBeGreaterThan(12 * 1024 * 1024);
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
	await page
		.getByTestId('pv-record-file')
		.setInputFiles({ name: 'opaque.json', mimeType: 'application/json', buffer: Buffer.from(text) });
	await expect(page.getByTestId('pv-record-summary')).toHaveText('unknown · 1 records', { timeout: 30_000 });
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-record-native').click();
	expect(await readFile((await (await pending).path())!)).toEqual(bytes);
});
