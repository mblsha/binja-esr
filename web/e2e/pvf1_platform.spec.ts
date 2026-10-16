import { test, expect } from '@playwright/test';
test.skip(!process.env.PVF1_ROM_PATH || !process.env.PVF1_RETAINED_PATH, 'Requires acquired firmware/backing');

test('native organizer sound reaches real WebAudio and stops when disabled', async ({ page }, info) => {
	await page.addInitScript(() => {
		const Base = window.AudioContext;
		(window as any).__audioProbe = { starts: 0, nonzero: 0, states: [] };
		window.AudioContext = class extends Base {
			constructor(options?: AudioContextOptions) {
				super(options);
				this.addEventListener('statechange', () => (window as any).__audioProbe.states.push(this.state));
			}
			createBufferSource() {
				const source = super.createBufferSource(),
					start = source.start.bind(source);
				source.start = (...args: Parameters<AudioBufferSourceNode['start']>) => {
					const p = (window as any).__audioProbe;
					p.starts++;
					if (source.buffer?.getChannelData(0).some((v) => v !== 0)) p.nonzero++;
					start(...args);
				};
				return source;
			}
		};
	});
	page.on('dialog', (d) => d.accept());
	await page.goto('/?model=pv-f1');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('pv-backup-file').setInputFiles(process.env.PVF1_RETAINED_PATH!);
	await expect(page.getByTestId('pv-saved-records')).toContainText('Restored');
	await page.getByTestId('sound-toggle').click();
	await expect(page.getByTestId('sound-toggle')).toHaveAttribute('aria-pressed', 'true');
	await page.getByTestId('pause-resume').click();
	await expect(page.getByTestId('pv-input-readiness')).toHaveAttribute('data-phase', 'ready');
	await page.getByRole('button', { name: 'Calculator', exact: true }).click();
	await expect.poll(() => page.evaluate(() => (window as any).__audioProbe.nonzero)).toBeGreaterThan(0);
	await page.getByTestId('sound-toggle').click();
	await expect(page.getByTestId('sound-toggle')).toHaveAttribute('aria-pressed', 'false');
	const count = await page.evaluate(() => (window as any).__audioProbe.starts);
	await page.waitForTimeout(200);
	expect(await page.evaluate(() => (window as any).__audioProbe.starts)).toBe(count);
	await info.attach('web-audio-observations', {
		body: JSON.stringify(await page.evaluate(() => (window as any).__audioProbe)),
		contentType: 'application/json',
	});
});
