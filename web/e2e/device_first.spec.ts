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
