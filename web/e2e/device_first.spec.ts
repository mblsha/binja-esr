import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: device-first controls preserve the LCD and expose tools only on demand`, async ({ page }) => {
		// Synthetic ROM tests host UI structure, not application correctness.
		const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
		await page.route('**/api/rom?model=*', (route) => route.fulfill({ status: 200, body: rom }));
		await page.addInitScript((model) => localStorage.setItem('sc62015:rom-model', model), model);
		await page.goto('/');
		await expect(page.getByTestId('pause-resume')).toBeEnabled();
		await expect(page.getByTestId('device-shell')).toBeVisible();
		await expect(page.getByTestId('rom-model')).toBeHidden();
		await expect(page.getByTestId('fnr-editor')).toBeHidden();
		await expect(page.getByTestId('device-status')).toContainText('Paused');
		await expect(page.getByText('Cancel queued keys', { exact: true })).toHaveCount(0);
		await page.getByRole('button', { name: 'More options' }).click();
		await expect(page.getByTestId('pace-preset')).toHaveValue('device');
		await page.getByTestId('pace-preset').selectOption('responsive');
		await expect(page.getByTestId('typing-catch-up')).toBeChecked();
		await page.getByTestId('pace-preset').selectOption('device');
		await expect(page.getByTestId('typing-catch-up')).not.toBeChecked();
		const readPixels = (canvas: HTMLCanvasElement) =>
			Array.from(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data);
		const pixels = await page.locator('.keyboard-target canvas').evaluate(readPixels);
		const pngDownload = page.waitForEvent('download');
		await page.getByTestId('export-lcd').click();
		const png = await pngDownload;
		expect(png.suggestedFilename()).toMatch(new RegExp(`^${model}-lcd-.*\\.png$`));
		const pngBytes = await readFile((await png.path())!);
		const decoded = await page.evaluate(async (bytes) => {
			const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: 'image/png' }));
			const canvas = document.createElement('canvas');
			canvas.width = bitmap.width;
			canvas.height = bitmap.height;
			const context = canvas.getContext('2d')!;
			context.drawImage(bitmap, 0, 0);
			bitmap.close();
			return {
				cols: canvas.width,
				rows: canvas.height,
				pixels: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data),
			};
		}, Array.from(pngBytes));
		expect(decoded.pixels).toEqual(pixels);
		const metadataDownload = page.waitForEvent('download');
		await page.getByTestId('export-lcd-metadata').click();
		const metadata = await metadataDownload;
		expect(metadata.suggestedFilename()).toBe(png.suggestedFilename().replace('.png', '.json'));
		const provenance = JSON.parse(await readFile((await metadata.path())!, 'utf8'));
		expect(provenance.model).toBe(model);
		expect([provenance.cols, provenance.rows]).toEqual([decoded.cols, decoded.rows]);
		expect(provenance.accuracy).toContain('Not a machine snapshot');
		await page.getByTestId('display-view').selectOption({ label: 'LCD only' });
		await expect(page.getByTestId('device-shell')).toHaveCount(0);
		expect(await page.locator('.lcd-only canvas').evaluate(readPixels)).toEqual(pixels);
		await page.getByTestId('display-view').selectOption({ label: 'Device' });
		await expect(page.getByTestId('device-shell')).toBeVisible();
		await page.getByRole('button', { name: 'Advanced', exact: true }).click();
		await expect(page.getByTestId('rom-model')).toBeVisible();
		await expect(page.getByTestId('execution-mode')).toHaveValue('interactive');
	});
}
