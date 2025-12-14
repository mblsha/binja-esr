import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';

async function clickStep20k(page: any, times: number) {
	const step20k = page.getByRole('button', { name: 'Step 20k' });
	await expect(step20k).toBeEnabled();
	for (let i = 0; i < times; i++) {
		await step20k.click();
	}
}

test('real ROM: PF1 changes boot menu', async ({ page }) => {
	test.skip(process.env.PCE500_E2E_REAL_ROM !== '1', 'Set PCE500_E2E_REAL_ROM=1 to run real-ROM test');
	test.skip(!process.env.PCE500_ROM_PATH, 'Set PCE500_ROM_PATH to the real pc-e500.bin path');
	test.skip(!existsSync(process.env.PCE500_ROM_PATH), 'PCE500_ROM_PATH does not exist on disk');

	await page.goto('/');

	// Boot: the ROM should render the S2(CARD) header after initial init.
	await clickStep20k(page, 1);
	await expect(page.getByTestId('lcd-text')).toContainText('S2(CARD):NEW CARD', { timeout: 30_000 });

	// Hold PF1 while we step a while (mirrors the private harness behavior).
	const pf1 = page.getByTestId('vk-pf1');
	await pf1.dispatchEvent('pointerdown');
	await clickStep20k(page, 40); // 800k instructions total is the harness scale; start with 40*20k while held.
	await pf1.dispatchEvent('pointerup');

	// Continue stepping until the menu header changes.
	await clickStep20k(page, 40);
	await expect(page.getByTestId('lcd-text')).toContainText('S1(MAIN):NEW CARD', { timeout: 30_000 });
});

