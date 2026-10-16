import { test, expect, type Page } from '@playwright/test';
import { readFile, writeFile, copyFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { decodePvLiveState, encodePvLiveState } from '../src/lib/emulator/pvf1_live_state';
test.skip(
	!process.env.PVF1_ROM_PATH || !process.env.PVF1_RECORD_FIXTURE_DIR,
	'Requires acquired firmware and original native exports',
);
const hash = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
async function open(page: Page) {
	page.on('dialog', (d) => d.accept());
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
}
async function download(page: Page, testid: string, path?: string) {
	const pending = page.waitForEvent('download');
	await page.getByTestId(testid).click();
	const file = (await (await pending).path())!;
	if (path) await copyFile(file, path);
	return readFile(file);
}
for (const [name, family, count] of [
	['schedule', 'schedule', 5],
	['telephone1', 'contacts', 2],
	['telephone2', 'contacts', 2],
	['business', 'business', 3],
	['memo', 'memo', 2],
	['actionlist', 'tasks', 2],
] as const)
	test(`browses native ${name} and exports readable and byte-preserving files`, async ({ page }, info) => {
		await open(page);
		const path = join(process.env.PVF1_RECORD_FIXTURE_DIR!, name + '.csv');
		await page.getByTestId('pv-record-file').setInputFiles(path);
		await expect(page.getByTestId('pv-record-summary')).toHaveText(`${family} · ${count} records`);
		expect(hash(await download(page, 'pv-record-native'))).toBe(hash(await readFile(path)));
		const json = info.outputPath(name + '.json');
		await download(page, 'pv-record-json', json);
		await page.getByTestId('pv-record-file').setInputFiles(json);
		await expect(page.getByTestId('pv-record-summary')).toContainText(`${count} records`);
		if (family === 'contacts' || family === 'business')
			expect((await download(page, 'pv-record-vcard', info.outputPath(name + '.vcf'))).toString()).toContain(
				'BEGIN:VCARD',
			);
		if (family === 'schedule')
			expect((await download(page, 'pv-record-calendar', info.outputPath('schedule.ics'))).toString()).toContain(
				'DTSTART;VALUE=DATE:19930101',
			);
		if (family === 'memo')
			expect((await download(page, 'pv-record-memo', info.outputPath('memo.txt'))).toString()).toContain('ＡＴＥＳＴ');
		await page.getByTestId('pv-record-bridge').screenshot({ path: info.outputPath('record-explorer.png') });
	});
test('imports through the real ROM Receive context and preserves native fresh-restart results', async ({
	page,
}, info) => {
	test.skip(
		!process.env.PVF1_RECEIVE_READY_PATH || !process.env.PVF1_RECEIVE_EXPECTED_RETAINED,
		'Requires the qualified native Receive context',
	);
	await open(page);
	const original = await download(page, 'pv-live-export'),
		size = original.readUInt32LE(40),
		header = JSON.parse(original.subarray(44, 44 + size).toString());
	const state = await decodePvLiveState(original, header.context);
	state.native = new Uint8Array(await readFile(process.env.PVF1_RECEIVE_READY_PATH!));
	const ready = info.outputPath('receive-ready.pvstate');
	await writeFile(ready, await encodePvLiveState(state));
	await page.getByTestId('pv-live-file').setInputFiles(ready);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Live state restored');
	await page.getByTestId('pv-record-file').setInputFiles(join(process.env.PVF1_RECORD_FIXTURE_DIR!, 'schedule.csv'));
	await page.getByTestId('pv-record-import').click();
	await expect(page.getByTestId('pv-record-notice')).toContainText('366 bytes transferred', { timeout: 40_000 });
	const retained = await download(page, 'pv-backup-export', info.outputPath('received-retained.bin'));
	expect(hash(retained)).toBe(hash(await readFile(process.env.PVF1_RECEIVE_EXPECTED_RETAINED!)));
	await page.getByTestId('pv-tablet-lcd').screenshot({ path: info.outputPath('native-receive-result.png') });
});
test('a disabled Receive transaction fails and restores the exact prior machine', async ({ page }) => {
	await open(page);
	const before = await download(page, 'pv-live-export');
	await page.getByTestId('pv-record-file').setInputFiles(join(process.env.PVF1_RECORD_FIXTURE_DIR!, 'schedule.csv'));
	await page.getByTestId('pv-record-import').click();
	await expect(page.getByTestId('pv-record-bridge').getByRole('alert')).toContainText('prior organizer state restored');
	expect(hash(await download(page, 'pv-live-export'))).toBe(hash(before));
});
