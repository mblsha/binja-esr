import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

const seedA = process.env.PVF1_RETAINED_PATH;
const seedB = process.env.PVF1_SECOND_RETAINED_PATH;
test.skip(
	!process.env.PVF1_ROM_PATH || !seedA || !seedB,
	'Requires acquired PV-F1 firmware and two native retained fixtures',
);

async function backup(page: Page) {
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-backup-export').click();
	const file = await pending;
	return readFile((await file.path())!);
}
async function ready(page: Page) {
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
}

test('PV-F1 automatic reload, previous-version recovery and exclusive tab ownership', async ({ page, context }) => {
	expect(hash(await readFile(seedA!))).not.toBe(hash(await readFile(seedB!)));
	page.on('dialog', (dialog) => dialog.accept());
	await ready(page);
	await page.getByTestId('pv-backup-file').setInputFiles(seedA!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	expect(hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
	await page.getByTestId('pv-backup-file').setInputFiles(seedB!);
	await expect.poll(async () => hash(await backup(page))).toEqual(hash(await readFile(seedB!)));
	await page.reload();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	expect(hash(await backup(page))).toEqual(hash(await readFile(seedB!)));
	const other = await context.newPage();
	await other.goto('/?model=pv-f1');
	await expect(other.getByTestId('pv-save-error')).toContainText('Another tab');
	await other.close();
	await page.getByTestId('pv-save-history').locator('summary').click();
	await page.getByRole('button', { name: /Recover previous save 1 / }).click();
	await expect(page.getByTestId('pv-saved-records')).toContainText('Recovered selected automatic save');
	await expect.poll(async () => hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
	await page.reload();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	expect(hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
});

test('PV-F1 active program-card SRAM survives reload without applying pending selections', async ({ page }) => {
	const rom = process.env.PVF1_PROGRAM_CARD_ROM;
	const sram = process.env.PVF1_PROGRAM_CARD_SRAM;
	test.skip(!rom || !sram, 'Requires acquired PA-9C3 backing');
	await ready(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('pv-card-rom').setInputFiles(rom!);
	await page.getByTestId('pv-card-sram').setInputFiles(sram!);
	// Saving before Import must capture the active no-card machine, not selections.
	await page.getByTestId('pv-save-now').click();
	await page.getByTestId('pv-card-load').click();
	await expect(page.getByTestId('pv-card-status')).toContainText('Program card imported');
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	await page.reload();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await expect(page.getByTestId('advanced-panel')).toContainText('Recovered program card');
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-card-export').click();
	expect(await readFile((await (await pending).path())!)).toEqual(await readFile(sram!));
});

test('PV-F1 write-protected memory card survives reload with its contents and protection', async ({ page }) => {
	await ready(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	const image = Buffer.alloc(0x20000, 0x6d);
	await page
		.getByTestId('pv-ram-card-image')
		.setInputFiles({ name: 'readonly.bin', mimeType: 'application/octet-stream', buffer: image });
	await page.getByRole('checkbox', { name: 'Allow writes on load' }).uncheck();
	await page.getByTestId('pv-ram-card-load').click();
	await expect(page.getByTestId('pv-card-status')).toContainText('Memory card imported');
	await page.reload();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await expect(page.getByRole('checkbox', { name: 'Allow writes on load' })).not.toBeChecked();
	const pending = page.waitForEvent('download');
	await page.getByTestId('pv-ram-card-save').click();
	expect(hash(await readFile((await (await pending).path())!))).toEqual(hash(image));
});

test('PV-F1 preserves a corrupt latest save and recovers a selected prior native save', async ({ page }) => {
	await ready(page);
	await page.getByTestId('pv-backup-file').setInputFiles(seedA!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await page.getByTestId('pv-backup-file').setInputFiles(seedB!);
	await expect.poll(async () => hash(await backup(page))).toEqual(hash(await readFile(seedB!)));
	await page.evaluate(async () => {
		const db = await new Promise<IDBDatabase>((resolve, reject) => {
			const request = indexedDB.open('sc62015-oz9600-battery', 1);
			request.onsuccess = () => resolve(request.result);
			request.onerror = () => reject(request.error);
		});
		try {
			await new Promise<void>((resolve, reject) => {
				const tx = db.transaction('images', 'readwrite');
				const store = tx.objectStore('images');
				const cursor = store.openCursor();
				cursor.onsuccess = () => {
					const row = cursor.result;
					if (!row) return;
					if (String(row.key).startsWith('pvf1:')) {
						const record = row.value;
						record.image[record.image.length - 1] ^= 1;
						row.update(record);
					} else row.continue();
				};
				tx.oncomplete = () => resolve();
				tx.onerror = () => reject(tx.error);
			});
		} finally {
			db.close();
		}
	});
	await page.reload();
	await expect(page.getByTestId('pv-save-error')).toContainText('checksum');
	await expect(page.getByTestId('pv-backup-export')).toBeDisabled();
	await page.getByTestId('pv-save-history').locator('summary').click();
	await page.getByRole('button', { name: /Recover previous save 1 / }).click();
	await expect(page.getByTestId('pv-saved-records')).toContainText('Recovered selected automatic save');
	expect(hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
});

test('PV-F1 disabling autosave preserves stored records and re-enabling requires recovery before replacement', async ({
	page,
}) => {
	page.on('dialog', (dialog) => dialog.accept());
	await ready(page);
	await page.getByTestId('pv-backup-file').setInputFiles(seedA!);
	await expect.poll(async () => hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
	await page.getByTestId('pv-autosave').uncheck();
	await expect(page.getByTestId('pv-save-status')).toContainText('Automatic saving is off');
	await page.getByTestId('pv-backup-file').setInputFiles(seedB!);
	await expect.poll(async () => hash(await backup(page))).toEqual(hash(await readFile(seedB!)));
	await page.getByTestId('pv-autosave').check();
	await expect(page.getByTestId('pv-save-error')).toContainText('Existing saved records are present');
	await page.getByTestId('pv-retry-load').click();
	await expect(page.getByTestId('pv-save-status')).toContainText('Saved at');
	expect(hash(await backup(page))).toEqual(hash(await readFile(seedA!)));
});
