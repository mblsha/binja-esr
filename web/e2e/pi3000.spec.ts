import { test, expect, type Page } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

const bundlePath = process.env.PI3000_ROM_PATH;
const output = process.env.PI3000_PROOF_DIR;
test.skip(!bundlePath, 'Supply verified private PI bundle with PI3000_ROM_PATH');
const rpc = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__piRpc(type, payload), { type, payload });

test('PI fresh browser reset and LCD clicks complete ROM calibration', async ({ page }) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pi-3000');
		const Original = window.Worker;
		window.Worker = class extends Original {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				if (!String(url).includes('pce500.worker')) return;
				let sequence = 4_000_000;
				(window as any).__piRpc = (type: string, payload: any = {}) =>
					new Promise((resolve, reject) => {
						const id = sequence++;
						const timer = setTimeout(() => {
							this.removeEventListener('message', receive);
							reject(new Error('PI RPC timeout'));
						}, 120_000);
						const receive = ({ data }: MessageEvent) => {
							if (data.type !== 'reply' || data.id !== id) return;
							clearTimeout(timer);
							this.removeEventListener('message', receive);
							if (data.ok) resolve(data.result);
							else reject(new Error(data.error));
						};
						this.addEventListener('message', receive);
						this.postMessage({ id, type, ...payload });
					});
			}
		};
	});
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(String(error)));
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await expect(page.getByTestId('device-shell')).toHaveClass(/pi-3000/);
	const canvas = page.getByTestId('pi-tablet-lcd').locator('canvas');
	await expect(canvas).toHaveAttribute('width', '239');
	await expect(canvas).toHaveAttribute('height', '168');
	await rpc(page, 'step', { instructions: 400_000 });
	const snapshots: any[] = [];
	for (const [x, y] of [
		[32, 32],
		[205, 32],
		[32, 135],
		[205, 135],
	]) {
		await canvas.scrollIntoViewIfNeeded();
		const box = (await canvas.boundingBox())!;
		await page.mouse.click(box.x + ((x + 0.5) / 239) * box.width, box.y + ((y + 0.5) / 168) * box.height);
		await expect.poll(() => rpc(page, 'pi_state').then((s) => s.tablet[2])).toBe(1);
		await rpc(page, 'step', { instructions: 300_000 });
		snapshots.push(await rpc(page, 'pi_state'));
	}
	const state = snapshots.at(-1);
	expect(state.fault).toBeNull();
	expect(state.tablet.slice(0, 3)).toEqual([700, 700, 0]);
	expect(state.tablet[5]).toBe(971);
	expect(state.native_instructions).toBe(1053134);
	const observed = await rpc(page, 'eval_js', {
		source:
			'return { capture: await e.lcd.capture(1), completed: await Promise.all(Array.from({length:4}, (_,i) => e.memory.read(0xBB294 + 0x21E + i))), result: await e.memory.read(0xBB294 + 0x222) };',
	});
	expect(observed.error).toBeNull();
	const result = JSON.parse(observed.resultJson);
	expect(result.completed).toEqual([1, 1, 1, 1]);
	expect(result.result).toBe(2);
	const header = Buffer.from('P4\n240 336\n');
	const pbm = Buffer.alloc(header.length + 30 * 336);
	header.copy(pbm);
	for (let i = 0; i < result.capture.pixels.length; i++)
		if (result.capture.pixels[i] === 0) pbm[header.length + Math.floor(i / 8)] |= 1 << (7 - (i % 8));
	const digest = createHash('sha256').update(pbm).digest('hex');
	expect(digest).toBe('b45a13065798fa5f454fabf1cd073041eab62b8b663a54f0390ac22006cceabe');
	expect(errors).toEqual([]);
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(
			resolve(output, 'browser-calibration.json'),
			JSON.stringify({ snapshots, state, digest, completed: result.completed, result: result.result }),
		);
		await canvas.screenshot({ path: resolve(output, 'browser-calibration.png') });
		await rpc(page, 'step', { instructions: 500_000 });
		await canvas.screenshot({ path: resolve(output, 'browser-completion.png') });
	}
});

test('PI no-Worker browser fallback shares panel and assisted contact behavior', async ({ page }) => {
	test.setTimeout(180_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pi-3000', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pi-3000');
		(window as any).Worker = undefined;
	});
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(String(error)));
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('advanced-panel').locator('> summary').click();
	const step = page.getByRole('button', { name: 'Step 20k', exact: true });
	const canvas = page.getByTestId('pi-tablet-lcd').locator('canvas');
	async function advance(count: number) {
		for (let i = 0; i < count; i++) {
			await step.click();
			await expect(step).toBeEnabled();
		}
	}
	await advance(20);
	for (const [x, y] of [
		[32, 32],
		[205, 32],
		[32, 135],
		[205, 135],
	]) {
		await canvas.scrollIntoViewIfNeeded();
		const box = (await canvas.boundingBox())!;
		await page.mouse.click(box.x + ((x + 0.5) / 239) * box.width, box.y + ((y + 0.5) / 168) * box.height);
		await advance(15);
	}
	await expect(page.getByTestId('emu-status')).toContainText('1053134');
	const pixels = await canvas.evaluate((node: HTMLCanvasElement) => {
		const rgba = node.getContext('2d')!.getImageData(0, 0, node.width, node.height).data;
		return Array.from({ length: node.width * node.height }, (_, i) => (rgba[i * 4] === 0 ? 1 : 0));
	});
	expect(createHash('sha256').update(Buffer.from(pixels)).digest('hex')).toBe(
		'c150e9de6c0c9b27791666071fa061524f9d313aafac6b558122c8f6706417d8',
	);
	expect(errors).toEqual([]);
	if (output) {
		await mkdir(output, { recursive: true });
		await canvas.screenshot({ path: resolve(output, 'browser-fallback.png') });
		await writeFile(
			resolve(output, 'browser-fallback.json'),
			JSON.stringify({
				boundaries: 1_600_000,
				native_instructions: 1053134,
				panel_sha256: 'c150e9de6c0c9b27791666071fa061524f9d313aafac6b558122c8f6706417d8',
				errors,
			}),
		);
	}
});
