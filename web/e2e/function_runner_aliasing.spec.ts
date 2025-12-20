import { test, expect } from '@playwright/test';

test('function runner: reset + PF1 tap + traced call does not trigger wasm-bindgen aliasing error', async ({
	page
}) => {
	await page.goto('/');

	await page.getByTestId('rom-model').selectOption('pc-e500');

	// Wait for auto-ROM-load to complete (enables controls), but don't "start" the emulator.
	const step20k = page.getByRole('button', { name: 'Step 20k' });
	await expect(step20k).toBeEnabled();

	// Open function runner panel.
	const panel = page.getByTestId('fnr-panel');
	await panel.evaluate((el: any) => (el.open = true));

	const editor = page.getByTestId('fnr-editor');

	const realRomMode = process.env.PCE500_E2E_REAL_ROM === '1';
	const addr = realRomMode ? '0x00F2A87' : 'e.reg(Reg.PC)';

	const script = `
await e.reset({ fresh: true, warmupTicks: 100_000 });

await e.keyboard.tap(0x56); // PF1 (virtual injection), holds ~40k instr by default
await e.call(${addr}, undefined, { maxInstructions: 200_000, trace: true });
`;
	await editor.fill(script);

	const run = page.getByTestId('fnr-run');
	await run.click();

	// Wait for script completion (the run button re-enables).
	await expect(run).toBeEnabled({ timeout: 120_000 });

	// If we have a script error, assert it's not the wasm-bindgen aliasing guard.
	const error = page.getByTestId('fnr-error');
	if ((await error.count()) > 0) {
		const text = await error.textContent();
		expect(text ?? '').not.toContain('recursive use of an object detected');
		expect(text ?? '').not.toContain('unsafe aliasing');
	}

	// Ensure the script actually executed the call.
	await expect(page.getByTestId('fnr-call').first()).toBeVisible({ timeout: 30_000 });
});
