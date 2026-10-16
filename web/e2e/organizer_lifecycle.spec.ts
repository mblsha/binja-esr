import { test, expect, type Page } from '@playwright/test';
import { readFile, copyFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { nativeLifecycle } from './native_lifecycle';
const hash = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
let lifecycle: ReturnType<typeof nativeLifecycle>;
test.beforeAll(() => {
	lifecycle = nativeLifecycle();
});
test.afterAll(() => lifecycle.restore());
async function backup(page: Page, model: string) {
	const pending = page.waitForEvent('download');
	await page.getByTestId(model === 'pv-f1' ? 'pv-full-backup-export' : 'oz-retained-export').click();
	return readFile((await (await pending).path())!);
}
for (const model of ['pv-f1', 'oz-9600'] as const)
	test(`${model} pauses on browser-generated hide/freeze and resumes explicitly without clock catch-up or stuck contacts`, async ({
		page,
		context,
	}, info) => {
		const seed = process.env[model === 'pv-f1' ? 'PVF1_RETAINED_PATH' : 'OZ9600_RETAINED_PATH'];
		test.skip(
			!seed || !process.env[model === 'pv-f1' ? 'PVF1_ROM_PATH' : 'OZ9600_ROM_PATH'],
			'Requires acquired firmware/backing',
		);
		page.on('dialog', (d) => d.accept());
		await page.goto('/?model=' + model);
		await expect(page.getByTestId('pause-resume')).toBeEnabled();
		await page.getByTestId(model === 'pv-f1' ? 'pv-backup-file' : 'oz-retained-file').setInputFiles(seed!);
		if (model === 'pv-f1') await expect(page.getByText(/^Restored /)).toBeVisible();
		else await expect.poll(async () => hash(await backup(page, model))).toBe(hash(await readFile(seed!)));
		await page.getByTestId('session-panel').locator('summary').click();
		await page.getByTestId('session-record-start').click();
		await expect(page.getByTestId('session-status')).toContainText('Recording from a fresh restart');
		await page.getByTestId('session-step-budget').fill('6000000');
		await page.getByTestId('session-step').click();
		await expect(page.getByTestId('session-step')).toBeEnabled();
		await page.evaluate(() => {
			(window as any).__lifecycle = [];
			for (const name of ['visibilitychange', 'freeze', 'resume'])
				document.addEventListener(name, () => (window as any).__lifecycle.push({ name, hidden: document.hidden }));
		});
		const cdp = await context.newCDPSession(page);
		await lifecycle.foreground(cdp);
		const boundaries = async () =>
			Number((await page.getByTestId('session-progress').innerText()).match(/(\d+) boundaries/)![1]);
		let last: Buffer | undefined;
		for (let cycle = 0; cycle < 3; cycle++) {
			const before = await boundaries();
			await page.getByTestId('pause-resume').click();
			await expect(page.getByTestId('pause-resume')).toHaveText('Pause');
			await expect.poll(boundaries).toBeGreaterThan(before + 100000);
			const key = page.locator('.vk button:not([disabled]):not([aria-label="ON"])').first();
			await key.scrollIntoViewIfNeeded();
			const b = (await key.boundingBox())!;
			await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
			await page.mouse.down();
			await lifecycle.freeze(cdp);
			await new Promise((r) => setTimeout(r, 100));
			await cdp.send('Page.setWebLifecycleState', { state: 'active' });
			await lifecycle.foreground(cdp);
			await page.mouse.up();
			await expect(page.getByTestId('lifecycle-status')).toContainText('Paused because');
			await expect(page.getByTestId('pause-resume')).toHaveText('Resume');
			await expect(page.locator('.vk button.delivered')).toHaveCount(0);
			last = await backup(page, model);
			const at = await boundaries();
			await page.waitForTimeout(150);
			expect(await boundaries()).toBe(at);
			expect(hash(await backup(page, model))).toBe(hash(last));
		}
		const events = await page.evaluate(() => (window as any).__lifecycle);
		expect(events.filter((e: any) => e.name === 'freeze')).toHaveLength(3);
		expect(events.filter((e: any) => e.name === 'resume')).toHaveLength(3);
		expect(events.some((e: any) => e.name === 'visibilitychange' && e.hidden)).toBe(true);
		await writeFile(info.outputPath('lifecycle-events.json'), JSON.stringify(events, null, 2));
		const pending = page.waitForEvent('download');
		await page.getByTestId('session-record-end').click();
		await copyFile((await (await pending).path())!, info.outputPath(model + '.orgsession'));
		await writeFile(info.outputPath('paused-backup.bin'), last!);
		// Crash the isolated PV test renderer: recovery must use the durable save,
		// without relying on pagehide/unload or a final snapshot from the old CPU.
		let recovered = page;
		if (model === 'pv-f1') {
			const crashed = page.waitForEvent('crash');
			// Chromium can leave this command unresolved after the renderer dies.
			// Await its crash notification rather than blocking recovery on the reply.
			void cdp.send('Page.crash').catch(() => {});
			await crashed;
			recovered = await context.newPage();
			await recovered.goto('/?model=' + model);
		} else await page.reload();
		await expect(recovered.getByTestId(model === 'pv-f1' ? 'pv-save-status' : 'oz-save-status')).toContainText(
			'Saved at',
		);
		expect(hash(await backup(recovered, model))).toBe(hash(last!));
	});
