import { test, expect } from '@playwright/test';

test('PF1 click changes decoded LCD text', async ({ page }) => {
	await page.goto('/');

	const step20k = page.getByRole('button', { name: 'Step 20k' });
	await expect(step20k).toBeEnabled();

	await step20k.click();
	await expect(page.getByTestId('lcd-text')).toContainText('BOOT');

	const pf1 = page.getByTestId('vk-pf1');
	await pf1.dispatchEvent('pointerdown');
	await pf1.dispatchEvent('pointerup');

	await step20k.click();
	await step20k.click();
	await step20k.click();

	await expect(page.getByTestId('lcd-text')).toContainText('MENU');
});

