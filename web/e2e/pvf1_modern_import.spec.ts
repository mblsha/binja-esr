import { test, expect } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { decodePvLiveState, encodePvLiveState } from '../src/lib/emulator/pvf1_live_state';
import { parseNativeRecords } from '../src/lib/emulator/pvf1_records';
test.skip(!process.env.PVF1_ROM_PATH, 'Requires acquired firmware');
const hash = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
test('modern preview preserves source and requires review of the current conversion', async ({ page }, info) => {
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
	const source = Buffer.from(
		'BEGIN:VCARD\r\nVERSION:4.0\r\nFN:ＩＭＰＯＲＴ\r\nORG:会社\r\nTEL;VALUE=text:１２３\r\nNOTE:日本語\\nSecond line\r\nEMAIL:original@example.com\r\nEND:VCARD\r\n',
	);
	await page.getByTestId('pv-record-file').setInputFiles({ name: 'book.vcf', mimeType: 'text/vcard', buffer: source });
	await expect(page.getByTestId('pv-record-summary')).toHaveText('contacts · 1 records');
	await expect(page.getByTestId('pv-modern-issues')).toContainText('EMAIL');
	await expect(page.getByTestId('pv-record-import')).toBeDisabled();
	let download = page.waitForEvent('download');
	await page.getByTestId('pv-modern-original').click();
	expect(await readFile((await (await download).path())!)).toEqual(source);
	await page.getByTestId('pv-modern-ack').check();
	await expect(page.getByTestId('pv-record-import')).toBeEnabled();
	await page.getByTestId('pv-modern-target').selectOption('business');
	await expect(page.getByTestId('pv-record-summary')).toHaveText('business · 1 records');
	await expect(page.getByTestId('pv-modern-ack')).not.toBeChecked();
	download = page.waitForEvent('download');
	await page.getByTestId('pv-record-native').click();
	const native = parseNativeRecords(new Uint8Array(await readFile((await (await download).path())!)));
	expect(native.rows[0][native.columns.indexOf('RMRK')]).toBe('MEM1:日本語\nSecond line\x1e\x0c');
	await page.getByTestId('pv-modern-preview').screenshot({ path: info.outputPath('modern-preview.png') });
	await page
		.getByTestId('pv-record-file')
		.setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('😀') });
	await expect(page.getByTestId('pv-modern-issues')).toContainText('U+1F600');
	await expect(page.getByTestId('pv-record-import')).toHaveCount(0);
	await expect(page.getByTestId('pv-record-summary')).toHaveCount(0);
});

test('calendar conversion waits for the chosen organizer zone and rejects recurrence', async ({ page }) => {
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
	const source =
		'BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\nDTSTART:20261015T140000Z\r\nSUMMARY:Meeting\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n';
	await page
		.getByTestId('pv-record-file')
		.setInputFiles({ name: 'event.ics', mimeType: 'text/calendar', buffer: Buffer.from(source) });
	await expect(page.getByTestId('pv-modern-issues')).toContainText('Choose the organizer time zone');
	await page.getByTestId('pv-modern-zone').fill('Australia/Sydney');
	await expect(page.getByTestId('pv-modern-count')).toContainText('0 errors');
	await expect(page.getByTestId('pv-modern-preview')).toContainText('20261016010000');
	await page.getByTestId('pv-record-file').setInputFiles({
		name: 'repeat.ics',
		mimeType: 'text/calendar',
		buffer: Buffer.from(source.replace('SUMMARY:', 'RRULE:FREQ=DAILY\r\nSUMMARY:')),
	});
	await expect(page.getByTestId('pv-modern-issues')).toContainText('Recurrence');
	await expect(page.getByTestId('pv-record-import')).toHaveCount(0);
});

test('previewed Memo traverses native Receive and matches the independent Rust retained result', async ({
	page,
}, info) => {
	test.skip(
		!process.env.PVF1_MODERN_MEMO_READY || !process.env.PVF1_MODERN_MEMO_EXPECTED,
		'Requires qualified native transaction',
	);
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
	const download = page.waitForEvent('download');
	await page.getByTestId('pv-live-export').click();
	const before = await readFile((await (await download).path())!),
		size = before.readUInt32LE(40),
		header = JSON.parse(before.subarray(44, 44 + size).toString());
	const state = await decodePvLiveState(before, header.context);
	state.native = new Uint8Array(await readFile(process.env.PVF1_MODERN_MEMO_READY!));
	const ready = info.outputPath('ready.pvstate');
	await writeFile(ready, await encodePvLiveState(state));
	await page.getByTestId('pv-live-file').setInputFiles(ready);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Live state restored');
	await page.getByTestId('pv-record-file').setInputFiles({
		name: 'memo.txt',
		mimeType: 'text/plain',
		buffer: Buffer.from('ＩＭＰＯＲＴ\n日本語\nsecond line'),
	});
	await page.getByTestId('pv-modern-ack').check();
	await page.getByTestId('pv-record-import').click();
	await expect(page.getByTestId('pv-record-notice')).toContainText('bytes transferred', { timeout: 40_000 });
	const received = page.waitForEvent('download');
	await page.getByTestId('pv-backup-export').click();
	const actual = await readFile((await (await received).path())!);
	await writeFile(info.outputPath('received-retained.bin'), actual);
	expect(hash(actual)).toBe(hash(await readFile(process.env.PVF1_MODERN_MEMO_EXPECTED!)));
});
