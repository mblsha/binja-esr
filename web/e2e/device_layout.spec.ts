import { test, expect } from '@playwright/test';

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: device geometry is legible and horizontally pans on narrow screens`, async ({ page }) => {
		await page.goto('/');
		await page.getByTestId('rom-model').selectOption(model);
		const shell = page.getByTestId('device-shell');
		await expect(shell).toHaveClass(new RegExp(model));
		await expect(shell.getByRole('button', { name: 'OFF (unmapped)' })).toBeDisabled();
		await expect(page.getByText('REFERENCE-BASED · NOT SCAN-DERIVED')).toBeVisible();
		const screen = await page.locator('.lcd-window').boundingBox();
		const a = await page.getByTestId('vk-a').boundingBox();
		expect(screen).not.toBeNull();
		expect(a).not.toBeNull();
		if (model === 'iq-7000') expect(screen!.x + screen!.width).toBeLessThan(a!.x);
		else expect(screen!.y + screen!.height).toBeLessThan(a!.y);
		await page.setViewportSize({ width: 390, height: 844 });
		expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
		const scroll = await page.locator('.pan').evaluate((el) => ({ width: el.clientWidth, content: el.scrollWidth }));
		expect(scroll.content).toBeGreaterThan(scroll.width);
		await page.getByTestId('physical-keyboard-toggle').check();
		await page.locator('.pan').focus();
		const before = await page.locator('.pan').evaluate((el) => el.scrollLeft);
		await page.locator('.pan').press('ArrowRight');
		await expect.poll(() => page.locator('.pan').evaluate((el) => el.scrollLeft)).toBeGreaterThan(before);
		await page.getByTestId('vk-enter').scrollIntoViewIfNeeded();
		await expect(page.getByTestId('vk-enter')).toBeVisible();
	});
}
