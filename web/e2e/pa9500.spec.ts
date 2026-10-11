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
const recoveryExpected = process.env.PA9500_RECOVERY_EXPECTED;
const editedPath = process.env.PA9500_EDITED_CASE;
const editedExpected = process.env.PA9500_EDITED_EXPECTED;
const retainedInput = process.env.PA9500_RETAINED_INPUT;

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
		7: 'Pause',
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
		65: 'Tab',
		13: 'PageUp',
		9: 'PageDown',
		108: 'ContextMenu',
		43: 'Backspace',
		18: 'Space',
		8: 'ArrowLeft',
		14: 'ArrowRight',
		0: 'ArrowUp',
		15: 'ArrowDown',
		5: 'Escape',
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
			...(spec.on ?? []).map((row: number[]) => row[0]),
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
		for (const [at, down] of spec.on ?? []) {
			if (at !== boundary) continue;
			if (down) {
				await page.getByTestId('keyboard-target').locator('canvas').click();
				await page.keyboard.down('F12');
			} else await page.keyboard.up('F12');
			const state = await request(page, 'input_state');
			expect(state.rust.on).toBe(down);
			deliveries.push({ boundary, code: 'on', down, rust: state.rust });
		}
		for (const [at, action] of spec.host_actions ?? []) {
			if (at !== boundary) continue;
			await page.getByTestId('keyboard-target').locator('canvas').click();
			await page.keyboard.press(action === 'delete' ? 'Delete' : 'Insert');
		}
		for (const [at, code, down] of spec.contacts) {
			if (at !== boundary) continue;
			if (spec.macro_events?.some((e: any[]) => e[0] === at && e[1] === code && e[2] === down)) {
				const state = await request(page, 'input_state');
				expect(state.rust.matrix).toEqual(down ? [code] : []);
				deliveries.push({ boundary, code, down, rust: state.rust, shortcut: true });
				continue;
			}
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
			const serial = spec.capture_serial ? (await request(page, 'serial_status')).uart : undefined;
			snapshots.push({ boundary, ...observed, ...(spec.capture_serial ? { uart: serial } : {}) });
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
	if (spec.input_bindings === 'basic' || retainedInput || spec.profile === 'bootstrap-v4')
		await page.getByTestId('advanced-panel').locator('> summary').click();
	if (spec.profile === 'bootstrap-v4') {
		await page.getByTestId('pa-profile').selectOption('bootstrap-v4');
		await page.getByTestId('pa-card-remove').click();
		await expect(page.getByTestId('pa-card-remove')).toBeEnabled();
		await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	}
	if (cardRom && cardSram) {
		await page.getByTestId('pa-card-rom').setInputFiles(cardRom);
		await page.getByTestId('pa-card-sram').setInputFiles(cardSram);
		await page.getByTestId('pa-card-load').click();
		await expect(page.getByTestId('pa-card-load')).toBeEnabled();
		await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	}
	if (retainedInput) {
		await page.getByTestId('pa-retained-file').setInputFiles(retainedInput);
		await expect(page.getByTestId('pa-retained-file')).toBeEnabled();
		await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	}
	const result = await replay(page, spec, spec.input_bindings === 'basic');
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
	if (process.env.PA9500_CAPTURE_RETAINED && output && expectedPath) {
		const pending = page.waitForEvent('download');
		await page.getByTestId('pa-retained-export').click();
		const file = await pending;
		const bytes = await readFile((await file.path())!);
		const final = JSON.parse(await readFile(expectedPath, 'utf8')).checkpoints.at(-1);
		const record = final.backing.find((b: any) => b.kind === 'retained');
		expect(bytes).toEqual(await readFile(resolve(expectedPath, '..', record.file)));
		await writeFile(resolve(output, 'browser-retained.bin'), bytes);
	}
});

test('PA native record backing export and fresh CPU import through file controls', async ({ page }) => {
	test.skip(
		!replayPath || !expectedPath || !recoveryPath || !recoveryExpected || !output,
		'Supply frozen native save/recovery cases and proof output',
	);
	test.setTimeout(360_000);
	await openPa(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await mkdir(output!, { recursive: true });
	const save = await replay(page, JSON.parse(await readFile(replayPath!, 'utf8')), true, 'record-save');
	const expected = JSON.parse(await readFile(expectedPath!, 'utf8')).checkpoints;
	for (const actual of save.snapshots) {
		const target = expected.find((c: any) => c.boundary === actual.boundary);
		expect(actual.status).toEqual(target.status);
		expect(actual.capture.pixels).toEqual(target.capture.pixels);
	}
	const download = page.waitForEvent('download');
	await page.getByTestId('pa-retained-export').click();
	const file = await download;
	expect(file.suggestedFilename()).toBe('pa9500-ram-rtc.bin');
	const bytes = await readFile((await file.path())!);
	const nativeRecord = expected.at(-1).backing.find((b: any) => b.kind === 'retained');
	expect(bytes).toEqual(await readFile(resolve(expectedPath!, '..', nativeRecord.file)));
	const saved = resolve(output!, 'browser-retained.bin');
	await writeFile(saved, bytes);
	const corrupt = Buffer.from(bytes);
	corrupt[104] ^= 1;
	const bad = resolve(output!, 'corrupt-retained.bin');
	await writeFile(bad, corrupt);
	const before = await observation(page);
	await page.getByTestId('pa-retained-file').setInputFiles(bad);
	await expect(page.getByTestId('pa-card-error')).toContainText('checksum');
	expect(await observation(page)).toEqual(before);
	await page.getByTestId('pa-retained-file').setInputFiles(saved);
	await expect(page.getByTestId('pa-retained-file')).toBeEnabled();
	await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	const recovery = await replay(page, JSON.parse(await readFile(recoveryPath!, 'utf8')), false, 'record-recovery');
	const expectedRecovery = JSON.parse(await readFile(recoveryExpected!, 'utf8')).checkpoints;
	for (const actual of recovery.snapshots) {
		const target = expectedRecovery.find((c: any) => c.boundary === actual.boundary);
		expect(actual.status).toEqual(target.status);
		expect(actual.capture.pixels).toEqual(target.capture.pixels);
	}
	let edited;
	if (editedPath && editedExpected) {
		const secondDownload = page.waitForEvent('download');
		await page.getByTestId('pa-retained-export').click();
		const secondFile = await secondDownload;
		const modifiedBytes = await readFile((await secondFile.path())!);
		const target = expectedRecovery.at(-1).backing.find((b: any) => b.kind === 'retained');
		expect(modifiedBytes).toEqual(await readFile(resolve(recoveryExpected!, '..', target.file)));
		const modified = resolve(output!, 'browser-modified-retained.bin');
		await writeFile(modified, modifiedBytes);
		await page.getByTestId('pa-retained-file').setInputFiles(modified);
		await expect(page.getByTestId('pa-retained-file')).toBeEnabled();
		await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
		edited = await replay(page, JSON.parse(await readFile(editedPath, 'utf8')), false, 'record-edited');
		const expectedEdited = JSON.parse(await readFile(editedExpected, 'utf8')).checkpoints;
		for (const actual of edited.snapshots) {
			const checkpoint = expectedEdited.find((c: any) => c.boundary === actual.boundary);
			expect(actual.status).toEqual(checkpoint.status);
			expect(actual.capture.pixels).toEqual(checkpoint.capture.pixels);
		}
	}
	await writeFile(
		resolve(output!, 'browser-record.json'),
		JSON.stringify({ save, recovery, ...(edited ? { edited } : {}), corrupt_import_preserved_machine: true }),
	);
});

test('PA ordinary BASIC navigation and ON BREAK through host DOM keys', async ({ page }) => {
	test.skip(!cardRom || !cardSram || !replayPath, 'Supply paired card images and a frozen BASIC replay');
	test.setTimeout(240_000);
	await openPa(page);
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('pa-card-rom').setInputFiles(cardRom!);
	await page.getByTestId('pa-card-sram').setInputFiles(cardSram!);
	await page.getByTestId('pa-card-load').click();
	await expect(page.getByTestId('pa-card-load')).toBeEnabled();
	await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	const spec = JSON.parse(await readFile(replayPath!, 'utf8'));
	const result = await replay(page, spec, true);
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

test('PA serial TEST through ordinary BASIC and browser transport', async ({ page }) => {
	test.skip(
		!cardRom || !cardSram || !replayPath || !expectedPath,
		'Supply paired serial TEST media and native control',
	);
	test.setTimeout(240_000);
	await openPa(page);
	const spec = JSON.parse(await readFile(replayPath!, 'utf8'));
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.getByTestId('pa-uart').setChecked(spec.uart_candidate);
	await page.getByTestId('pa-card-rom').setInputFiles(cardRom!);
	await page.getByTestId('pa-card-sram').setInputFiles(cardSram!);
	await page.getByTestId('pa-card-load').click();
	await expect(page.getByTestId('pa-card-load')).toBeEnabled();
	await expect(page.getByTestId('pa-card-error')).toHaveCount(0);
	const result = await replay(page, spec, true);
	const expected = JSON.parse(await readFile(expectedPath!, 'utf8')).checkpoints;
	for (const actual of result.snapshots) {
		const target = expected.find((item: any) => item.boundary === actual.boundary);
		expect(actual.status).toEqual(target.status);
		expect(actual.capture.pixels).toEqual(target.capture.pixels);
		expect(actual.card_sram).toEqual(target.card_sram);
		expect(actual.uart).toEqual(target.serial.uart);
	}
	const bytes = expected.flatMap((cp: any) => cp.serial.bytes);
	await page.getByTestId('serial-panel').locator('> summary').click();
	await expect(page.getByTestId('serial-count')).toContainText(`${bytes.length} bytes received`);
	let raw = Buffer.alloc(0);
	if (bytes.length) {
		const download = page.waitForEvent('download');
		await page.getByTestId('serial-download').click();
		const file = await download;
		expect(file.suggestedFilename()).toBe('pa-9500-serial.raw');
		raw = await readFile((await file.path())!);
		expect([...raw]).toEqual(bytes);
	} else await expect(page.getByTestId('serial-download')).toBeDisabled();
	const before = await observation(page);
	await page.getByRole('button', { name: 'More options' }).click();
	await page.getByTestId('reset-session').click();
	await expect(page.getByTestId('pause-resume')).toBeEnabled();
	const reset = await observation(page);
	expect(reset.card_sram).toEqual(before.card_sram);
	expect(reset.status.native_instructions).toBe(64);
	const uartReset = (await request(page, 'serial_status')).uart;
	expect(Boolean(uartReset)).toBe(spec.uart_candidate);
	await expect(page.getByTestId('serial-count')).toContainText('0 bytes received');
	const recovered = await replay(page, spec, true, 'serial-recovered');
	await expect(page.getByTestId('serial-count')).toContainText(`${bytes.length} bytes received`);
	let recoveredRaw = Buffer.alloc(0);
	if (bytes.length) {
		const download = page.waitForEvent('download');
		await page.getByTestId('serial-download').click();
		recoveredRaw = await readFile((await (await download).path())!);
		expect([...recoveredRaw]).toEqual(bytes);
	}
	if (output) {
		await mkdir(output, { recursive: true });
		await writeFile(
			resolve(output, 'browser-serial.json'),
			JSON.stringify({ result, before, reset, uartReset, recovered }),
		);
		await writeFile(resolve(output, 'browser-serial.raw'), raw);
		await writeFile(resolve(output, 'browser-serial-recovered.raw'), recoveredRaw);
		await writeFile(resolve(output, 'browser-retained-card.bin'), Uint8Array.from(before.card_sram));
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
		for (const [at, action] of spec.host_actions ?? []) {
			if (at !== boundary) continue;
			await page.getByTestId('keyboard-target').locator('canvas').click();
			await page.keyboard.press(action === 'delete' ? 'Delete' : 'Insert');
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
