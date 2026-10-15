import { test, expect, type Page } from '@playwright/test';
import { readFile, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { decodeRecordedSession, encodeRecordedSession } from '../src/lib/emulator/session_recording';

const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
async function advance(page: Page, boundaries: number) {
	await page.getByTestId('session-step-budget').fill(String(boundaries));
	await page.getByTestId('session-step').click();
	await expect(page.getByTestId('session-step')).toBeEnabled();
}
async function pixels(page: Page, x: number, y: number) {
	const lcd = page.getByTestId('pv-tablet-lcd');
	await lcd.scrollIntoViewIfNeeded();
	const box = (await lcd.boundingBox())!;
	await page.mouse.click(box.x + ((x + 0.5) * box.width) / 336, box.y + ((y + 0.5) * box.height) / 292);
}
async function backing(page: Page, model: string, output?: string) {
	const download = page.waitForEvent('download');
	await page.getByTestId(model === 'pv-f1' ? 'pv-backup-export' : 'oz-retained-export').click();
	const path = (await (await download).path())!;
	if (output) await copyFile(path, output);
	return hash(await readFile(path));
}

for (const model of ['pv-f1', 'oz-9600'] as const) {
	test(`${model} exports a physical-input session and replays every screen and retained byte`, async ({
		page,
	}, info) => {
		const seed = process.env[model === 'pv-f1' ? 'PVF1_RETAINED_PATH' : 'OZ9600_RETAINED_PATH'];
		test.skip(
			!seed || !process.env[model === 'pv-f1' ? 'PVF1_ROM_PATH' : 'OZ9600_ROM_PATH'],
			'Requires acquired firmware and native retained backing',
		);
		page.on('dialog', (dialog) => dialog.accept());
		await page.goto(`/?model=${model}`);
		await expect(page.getByTestId('pause-resume')).toBeEnabled();
		await page.getByTestId(model === 'pv-f1' ? 'pv-backup-file' : 'oz-retained-file').setInputFiles(seed!);
		await expect.poll(() => backing(page, model)).toBe(hash(await readFile(seed!)));
		await page.getByTestId('session-panel').locator('summary').click();
		await page.getByTestId('session-record-start').click();
		await expect(page.getByTestId('session-status')).toContainText('Recording from a fresh restart');
		await advance(page, 6_000_000);
		await page.getByRole('button', { name: model === 'pv-f1' ? 'Calculator' : 'Calendar', exact: true }).click();
		await advance(page, 2_000_000);
		if (model === 'pv-f1') {
			for (const [x, y] of [
				[135, 167],
				[195, 176],
				[164, 167],
				[224, 188],
			]) {
				await pixels(page, x, y);
				await advance(page, 2_000_000);
			}
			await page.screenshot({ path: info.outputPath('calculator-five.png') });
		}
		await page.getByTestId('session-checkpoint').click();
		await expect(page.getByTestId('session-status')).toContainText('Screen checkpoint added');
		await page.locator('.vk button:not([disabled]):not([aria-label="ON"])').first().click();
		await advance(page, 2_000_000);
		await page.getByRole('button', { name: 'Power / ON', exact: true }).click();
		await advance(page, 2_000_000);
		const pending = page.waitForEvent('download');
		await page.getByTestId('session-record-end').click();
		const downloaded = (await (await pending).path())!;
		const path = info.outputPath(`${model}.orgsession`);
		await copyFile(downloaded, path);
		const session = await decodeRecordedSession(await readFile(path));
		expect(session.context.model).toBe(model);
		expect(session.events.some((e) => e.kind === 'tablet' && e.pressed)).toBe(true);
		expect(session.events.some((e) => e.kind === 'matrix' && e.pressed)).toBe(true);
		expect(session.events.some((e) => e.kind === 'on' && e.pressed)).toBe(true);
		const saved = await backing(page, model, info.outputPath('final-retained.bin'));
		await page.getByTestId('session-replay-file').setInputFiles(path);
		await expect(page.getByTestId('session-status')).toContainText('Replay verified');
		expect(await backing(page, model)).toBe(saved);
		await expect(page.getByTestId('session-error')).toHaveCount(0);
		// A damaged import must not replace the current machine or backing.
		const damaged = Buffer.from(await readFile(path));
		damaged[damaged.length - 1] ^= 1;
		await page
			.getByTestId('session-replay-file')
			.setInputFiles({ name: 'damaged.orgsession', mimeType: 'application/octet-stream', buffer: damaged });
		await expect(page.getByTestId('session-error')).toContainText('checksum');
		expect(await backing(page, model)).toBe(saved);
		for (const [field, expected] of [
			['coreIdentity', 'matching emulator build'],
			['firmwareSha256', 'matching organizer firmware'],
		] as const) {
			const changed = structuredClone(session);
			changed.context[field] = field === 'coreIdentity' ? 'other-core' : '0'.repeat(64);
			await page.getByTestId('session-replay-file').setInputFiles({
				name: 'mismatched.orgsession',
				mimeType: 'application/octet-stream',
				buffer: Buffer.from(await encodeRecordedSession(changed)),
			});
			await expect(page.getByTestId('session-error')).toContainText(expected);
			expect(await backing(page, model)).toBe(saved);
		}
		if (model === 'pv-f1') {
			const long = structuredClone(session);
			long.duration = 1_000_000_000;
			long.events = [
				session.events[0],
				{ kind: 'matrix', code: 1, pressed: true, at: 0 },
				{ kind: 'frame', at: long.duration, sha256: (session.events[0] as { sha256: string }).sha256 },
			];
			await page.getByTestId('session-replay-file').setInputFiles({
				name: 'cancellable.orgsession',
				mimeType: 'application/octet-stream',
				buffer: Buffer.from(await encodeRecordedSession(long)),
			});
			await expect(page.getByTestId('session-status')).toContainText('Checking and replaying');
			await page.getByTestId('pause-resume').click();
			await expect(page.getByTestId('session-status')).toContainText('Replay cancelled');
			await expect(page.locator('.vk button.delivered')).toHaveCount(0);
			await page.getByTestId('session-replay-file').setInputFiles(path);
			await expect(page.getByTestId('session-status')).toContainText('Replay verified');
			expect(await backing(page, model)).toBe(saved);
		}
	});
}
