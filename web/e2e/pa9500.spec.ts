import { test, expect, type Page } from '@playwright/test';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const bundlePath = process.env.PA9500_ROM_PATH;
const output = process.env.PA9500_PROOF_DIR;
test.skip(!bundlePath, 'Supply the verified private PA bundle with PA9500_ROM_PATH');

async function request(page: Page, type: string, payload: any = {}) {
	return page.evaluate(({ type, payload }) => (window as any).__paRequest(type, payload), { type, payload });
}

async function openPa(page: Page) {
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pa-9500', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pa-9500');
		const Original = window.Worker;
		window.Worker = class extends Original {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				if (!String(url).includes('pce500.worker')) return;
				let sequence = 1_000_000;
				(window as any).__paRequest = (type: string, payload: any) =>
					new Promise((resolve, reject) => {
						const id = sequence++;
						const timer = setTimeout(() => reject(new Error('PA worker RPC timeout')), 120_000);
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
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await expect(page.getByTestId('device-shell')).toHaveClass(/pa-9500/);
	await expect(page.getByTestId('device-status')).toContainText('Paused');
}

test('PA fresh browser load and normal Y reach Calendar', async ({ page }) => {
	test.setTimeout(180_000);
	await openPa(page);
	await request(page, 'step', { instructions: 5_200_000 - 64 });
	await page.getByTestId('keyboard-target').locator('canvas').click();
	await page.keyboard.down('y');
	const down = await request(page, 'input_state');
	expect(down.rust.matrix).toEqual([38]);
	await request(page, 'step', { instructions: 200_000 });
	await page.keyboard.up('y');
	const up = await request(page, 'input_state');
	await request(page, 'step', { instructions: 3_400_000 });
	const observed = await request(page, 'eval_js', {
		source: 'return { status: e.pa9500.status(), capture: await e.lcd.capture() };',
	});
	expect(observed.error).toBeNull();
	const checkpoint = JSON.parse(observed.resultJson);
	expect(checkpoint.status.profile).toBe('bootstrap-v3');
	expect([checkpoint.capture.cols, checkpoint.capture.rows]).toEqual([195, 145]);
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(resolve(output, 'browser-calendar.json'), JSON.stringify(checkpoint));
		await writeFile(resolve(output, 'browser-input.json'), JSON.stringify({ down, up }));
		await page.getByTestId('keyboard-target').screenshot({ path: resolve(output, 'browser-calendar.png') });
	}
	expect(checkpoint.capture.pixels.filter((v: number) => v === 0)).toHaveLength(2907);
});

const replayPath = process.env.PA9500_BROWSER_CASE;
const expectedPath = process.env.PA9500_BROWSER_EXPECTED;
const cardRom = process.env.PA9500_CARD_ROM;
const cardSram = process.env.PA9500_CARD_SRAM;
const recoveryPath = process.env.PA9500_RECOVERY_CASE;

async function observation(page: Page) {
	const generation = (await request(page, 'input_state')).generation;
	const observed = await request(page, 'pa_observe', { generation });
	observed.capture.pixels = Array.from(observed.capture.pixels);
	return observed;
}

async function canvasMatches(page: Page, pixels: number[]) {
	await expect
		.poll(async () =>
			page
				.getByTestId('keyboard-target')
				.locator('canvas')
				.evaluate((canvas: HTMLCanvasElement, pixels) => {
					const rgba = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
					return (
						rgba.length === pixels.length * 4 &&
						pixels.every(
							(v: number, i: number) =>
								rgba[4 * i] === v && rgba[4 * i + 1] === v && rgba[4 * i + 2] === v && rgba[4 * i + 3] === 255,
						)
					);
				}, pixels),
		)
		.toBe(true);
}

async function replay(page: Page, spec: any, basic = false, label = 'browser') {
	const aliases: Record<number, string> = {
		1: 'F1',
		28: 'F2',
		44: 'F3',
		36: 'F4',
		52: 'F5',
		67: 'F6',
		68: 'F7',
		21: 'F8',
		23: 'F9',
		22: 'F10',
		92: 'F11',
		69: 'Enter',
		43: 'Backspace',
		18: 'Space',
	};
	const maps = JSON.parse(await readFile(resolve('src/lib/pa9500_keys.json'), 'utf8'));
	const letters: Record<number, string> = {};
	const nativeLetters: Record<number, string> = {};
	for (const [name, code] of Object.entries(maps.native)) nativeLetters[code as number] = name.toLowerCase();
	for (const [name, code] of Object.entries(basic ? maps.basic : maps.native)) letters[code] = name.toLowerCase();
	for (const [name, code] of Object.entries(maps.common))
		if (/^[0-9+\-*/=.%]$/.test(name) && letters[code] === undefined) letters[code] = name;
	const points = [
		...new Set<number>([
			...spec.checkpoints.filter((n: number) => n >= 64),
			...spec.contacts.map((row: number[]) => row[0]),
		]),
	].sort((a, b) => a - b);
	const snapshots = [],
		deliveries = [];
	let previous = 64,
		basicSelected = false;
	for (const boundary of points) {
		await request(page, 'step', { instructions: boundary - previous });
		if (basic && boundary >= 11_000_000 && !basicSelected) {
			await page.getByTestId('pa-input').selectOption('basic');
			basicSelected = true;
		}
		for (const [at, code, down] of spec.contacts) {
			if (at !== boundary) continue;
			const host = aliases[code] ?? (basic && !basicSelected ? nativeLetters[code] : letters[code]);
			if (!host) throw new Error('No DOM host binding for contact ' + code);
			if (down) {
				await page.getByTestId('keyboard-target').locator('canvas').click();
				await page.keyboard.down(host);
			} else await page.keyboard.up(host);
			const state = await request(page, 'input_state');
			expect(state.rust.matrix).toEqual(down ? [code] : []);
			deliveries.push({ boundary, code, down, rust: state.rust });
		}
		if (spec.checkpoints.includes(boundary)) {
			const observed = await observation(page);
			snapshots.push({ boundary, ...observed });
			await canvasMatches(page, observed.capture.pixels);
			if (output)
				await page.getByTestId('keyboard-target').screenshot({ path: resolve(output, `${label}-${boundary}.png`) });
		}
		previous = boundary;
	}
	return { snapshots, deliveries };
}

test('PA native app editing and switching through host DOM keys', async ({ page }) => {
	test.skip(!replayPath, 'Supply a frozen PA9500_BROWSER_CASE');
	test.setTimeout(240_000);
	await openPa(page);
	const spec = JSON.parse(await readFile(replayPath!, 'utf8'));
	const result = await replay(page, spec);
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(resolve(output, 'browser-replay.json'), JSON.stringify(result));
	}
	if (expectedPath) {
		const expected = JSON.parse(await readFile(expectedPath, 'utf8')).checkpoints;
		for (const actual of result.snapshots) {
			const target = expected.find((item: any) => item.boundary === actual.boundary);
			expect(actual.status).toEqual(target.status);
			expect(actual.capture.pixels).toEqual(target.capture.pixels);
			expect(actual.card_sram).toEqual(target.card_sram);
		}
	}
});

test('PA card DOM import and ordinary BASIC edit/list/run', async ({ page }) => {
	test.skip(!cardRom || !cardSram || !replayPath, 'Supply paired private card images and a frozen BASIC replay');
	test.setTimeout(240_000);
	await openPa(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('pa-card-rom').setInputFiles(cardRom!);
	await page.getByTestId('pa-card-sram').setInputFiles(cardSram!);
	await page.getByTestId('pa-card-load').click();
	await expect(page.getByTestId('pa-card-load')).toBeEnabled();
	await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	const spec = JSON.parse(await readFile(replayPath!, 'utf8'));
	const result = await replay(page, spec, true, 'card');
	const before = await observation(page);
	expect(before.card_sram).toHaveLength(32768);
	const download = page.waitForEvent('download');
	await page.getByTestId('pa-card-export').click();
	const exported = await readFile((await (await download).path())!);
	expect([...exported]).toEqual(before.card_sram);
	await page.getByRole('button', { name: 'More options' }).click();
	await page.getByTestId('reset-session').click();
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	const reset = await observation(page);
	expect(reset.card_sram).toEqual(before.card_sram);
	expect(reset.status.profile).toBe('bootstrap-v3');
	expect(reset.status.native_instructions).toBe(64);
	let recovery = null;
	if (recoveryPath) {
		const recoverySpec = JSON.parse(await readFile(recoveryPath, 'utf8'));
		recovery = await replay(page, recoverySpec, true, 'recovery');
	}
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(resolve(output, 'browser-card.json'), JSON.stringify({ result, before, reset, recovery }));
		await writeFile(resolve(output, 'exported-card-sram.bin'), exported);
	}
});

test('PA no-Worker fallback boots and edits through the same DOM controls', async ({ page }) => {
	test.skip(!replayPath, 'Supply a frozen native PA replay');
	test.setTimeout(300_000);
	const bundle = await readFile(bundlePath!);
	await page.route('**/api/rom?model=pa-9500', (route) => route.fulfill({ body: bundle }));
	await page.addInitScript(() => {
		localStorage.setItem('sc62015:rom-model', 'pa-9500');
		(window as any).Worker = undefined;
	});
	await page.goto('/');
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	await page.getByTestId('advanced-panel').locator('> summary').click();
	const spec = JSON.parse(await readFile(replayPath!, 'utf8'));
	const host: Record<number, string> = {
		38: 'y',
		28: 'F2',
		21: 'F8',
		32: 'm',
		39: 'a',
		37: 'r',
		43: 'Backspace',
		34: 'e',
		69: 'Enter',
		1: 'F1',
	};
	const points = [
		...new Set<number>([...spec.checkpoints.filter((n: number) => n > 0), ...spec.contacts.map((r: number[]) => r[0])]),
	].sort((a, b) => a - b);
	const snapshots = [];
	let previous = 0;
	const step = page.getByRole('button', { name: 'Step 20k', exact: true });
	for (const boundary of points) {
		const count = (boundary - previous) / 20_000;
		if (!Number.isInteger(count)) throw new Error('Fallback DOM fixture needs20k-aligned budgets');
		for (let i = 0; i < count; i++) {
			await step.click();
			await expect(step).toBeEnabled();
		}
		for (const [at, code, down] of spec.contacts) {
			if (at !== boundary) continue;
			if (!host[code]) throw new Error('Unsupported fallback host fixture contact');
			if (down) {
				await page.getByTestId('keyboard-target').locator('canvas').click();
				await page.keyboard.down(host[code]);
			} else await page.keyboard.up(host[code]);
		}
		if (spec.checkpoints.includes(boundary)) {
			const download = page.waitForEvent('download');
			await page.getByTestId('pa-observe').click();
			const observed = JSON.parse(await readFile((await (await download).path())!, 'utf8'));
			observed.capture.pixels = Array.from(observed.capture.pixels);
			expect(observed.capture.pixels).toHaveLength(195 * 145);
			await canvasMatches(page, observed.capture.pixels);
			snapshots.push({ boundary: boundary + 64, ...observed });
			if (output)
				await page
					.getByTestId('keyboard-target')
					.screenshot({ path: resolve(output, `fallback-${boundary + 64}.png`) });
		}
		previous = boundary;
	}
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(
			resolve(output, 'browser-fallback.json'),
			JSON.stringify({ snapshots, initial_loader_boundaries: 64 }),
		);
	}
});

test('PA invalid card import preserves the live machine and usable input generation', async ({ page }) => {
	test.skip(!cardSram, 'Supply a valid-size private SRAM image');
	test.setTimeout(90_000);
	await openPa(page);
	const before = await observation(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page
		.getByTestId('pa-card-rom')
		.setInputFiles({ name: 'invalid-rom.bin', mimeType: 'application/octet-stream', buffer: Buffer.alloc(0x20000) });
	await page.getByTestId('pa-card-sram').setInputFiles(cardSram!);
	await page.getByTestId('pa-card-load').click();
	await expect(page.getByTestId('pa-card-error')).toContainText('verified');
	const after = await observation(page);
	expect(after).toEqual(before);
	await page.getByTestId('keyboard-target').locator('canvas').click();
	await page.keyboard.down('y');
	expect((await request(page, 'input_state')).rust.matrix).toEqual([38]);
	await page.keyboard.up('y');
});
