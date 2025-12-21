import { test, expect } from '@playwright/test';

test('physical keyboard: holding F1 changes decoded LCD text', async ({ page }) => {
	await page.goto('/');

	await page.getByTestId('rom-model').selectOption('pc-e500');

	const step20k = page.getByRole('button', { name: 'Step 20k' });
	await expect(step20k).toBeEnabled();

	const keyboardToggle = page.getByTestId('physical-keyboard-toggle');
	await expect(keyboardToggle).not.toBeChecked();
	await keyboardToggle.check();

	// Ensure the page has focus so keyboard events hit the window handler.
	// (Avoid clicking "body": layout changes can place the checkbox at the center of the page,
	// toggling it back off.)
	await page.getByTestId('emu-status').click();

	await step20k.click();
	await expect(page.getByTestId('lcd-text')).toContainText('BOOT');

	// Hold F1 while stepping so firmware polling sees the pressed state.
	await page.keyboard.down('F1');
	await step20k.click();
	await step20k.click();
	await step20k.click();
	await page.keyboard.up('F1');

	await expect(page.getByTestId('lcd-text')).toContainText('MENU');
});
