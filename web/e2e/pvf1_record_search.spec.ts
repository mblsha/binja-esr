import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
test.skip(!process.env.PVF1_ROM_PATH || !process.env.PVF1_PRESSURE_RECORDS, 'Requires native near-capacity export');

test('search finds records beyond the displayed page and keeps all original export bytes', async ({
	page,
	browserName,
}, info) => {
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-record-bridge').locator('summary').click();
	const started = performance.now();
	await page.getByTestId('pv-record-file').setInputFiles(process.env.PVF1_PRESSURE_RECORDS!);
	await expect(page.getByTestId('pv-record-matches')).toHaveText('160 matching rows');
	const opened = performance.now() - started;
	const searches: number[] = [];
	for (const query of ['PRESSURE-0159', 'ｐｒｅｓｓｕｒｅ－０１５９']) {
		const at = performance.now();
		await page.getByTestId('pv-record-search').fill(query);
		await expect(page.getByTestId('pv-record-matches')).toHaveText('1 matching rows');
		await expect(page.getByTestId('pv-record-bridge').locator('tbody tr')).toHaveCount(1);
		searches.push(performance.now() - at);
	}
	await page.getByTestId('pv-record-search').fill('missing-record');
	await expect(page.getByTestId('pv-record-matches')).toHaveText('0 matching rows');
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-record-native').click();
	expect(await readFile((await (await pending).path())!)).toEqual(await readFile(process.env.PVF1_PRESSURE_RECORDS!));
	await info.attach('explorer-latency', {
		body: JSON.stringify({
			browserName,
			bytes: 165207,
			rows: 160,
			opened_ms: opened,
			search_ms: searches,
			includes_automation_overhead: true,
		}),
		contentType: 'application/json',
	});
});
