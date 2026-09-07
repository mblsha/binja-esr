import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function open(page: Page, model: string, realRom = false) {
	if (!realRom) {
		const rom = await readFile(resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom'));
		await page.route(`**/api/rom?model=${model}`, (route) => route.fulfill({ status: 200, body: rom }));
	}
	await page.addInitScript((model) => {
		localStorage.setItem('sc62015:rom-model', model);
		const NativeWorker = window.Worker;
		window.Worker = class extends NativeWorker {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				let nextId = 1_000_000;
				// Inspect the actual compiled machine worker, never a fake emulator.
				(window as any).__inputRequest = (type: string, payload: any = {}) =>
					new Promise((resolve, reject) => {
						const id = nextId++;
						const timer = setTimeout(() => {
							this.removeEventListener('message', receive);
							reject(new Error('Input test RPC timeout'));
						}, 5000);
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
	}, model);
	await page.goto('/');
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await expect(page.getByRole('button', { name: 'Step 20k' })).toBeEnabled();
}

const request = (page: Page, type: string, payload: any = {}) =>
	page.evaluate(({ type, payload }) => (window as any).__inputRequest(type, payload), { type, payload });
const contacts = (page: Page) => request(page, 'input_state').then((state) => state.rust);

async function previewPaste(page: Page, text: string) {
	await page.getByRole('button', { name: 'More options' }).click();
	await page.getByTestId('open-paste').click();
	await page.getByTestId('paste-editor').fill(text);
}

test('paste preview rejects unsupported text and streams a bounded, cancellable queue while paused', async ({
	page,
}) => {
	await open(page, 'iq-7000');
	await previewPaste(page, 'ABC:DEF');
	const generation = (await request(page, 'input_state')).generation;
	await expect(request(page, 'paste_text', { text: 'A', generation: generation - 1 })).rejects.toThrow(
		'Stale paste generation',
	);
	await expect(page.getByTestId('paste-unsupported')).toContainText('nothing will be typed');
	await expect(page.getByTestId('submit-paste')).toBeDisabled();
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await page.getByTestId('paste-editor').fill('A'.repeat(300));
	await page.getByTestId('submit-paste').click();
	await expect(page.getByTestId('paste-preview')).toHaveCount(0);
	let state = await request(page, 'input_state');
	expect(state.paste).toEqual({ pending: 300, total: 300 });
	expect(state.typing.pending).toBe(128);
	await page.waitForTimeout(50);
	expect((await request(page, 'input_state')).paste.pending).toBe(300);
	await request(page, 'step', { instructions: 80_000 });
	expect((await request(page, 'input_state')).paste.pending).toBe(299);
	await page.getByRole('button', { name: 'Cancel queued keys', exact: true }).click();
	state = await request(page, 'input_state');
	expect(state.paste.pending).toBe(0);
	expect(state.rust).toEqual({ matrix: [], on: false });
	await request(page, 'step', { instructions: 80_000 });
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
});

test('clipboard paste opens a preview without injecting text or stealing host editor paste', async ({ page }) => {
	await open(page, 'iq-7000');
	await page.getByTestId('keyboard-focus').click();
	const intercepted = await page.getByTestId('keyboard-target').evaluate((target) => {
		const data = new DataTransfer();
		data.setData('text/plain', 'ABC,123');
		return !target.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
	});
	expect(intercepted).toBe(true);
	await expect(page.getByTestId('paste-editor')).toHaveValue('ABC,123');
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	const editorIntercepted = await page
		.getByTestId('paste-editor')
		.evaluate((target) => !target.dispatchEvent(new ClipboardEvent('paste', { bubbles: true, cancelable: true })));
	expect(editorIntercepted).toBe(false);
});

test('clicking the device enables focus and buffered feedback follows delivered contacts', async ({ page }) => {
	await open(page, 'iq-7000');
	await page.locator('.iq-brand').click();
	await expect(page.getByTestId('physical-keyboard-toggle')).toBeChecked();
	await expect(page.getByTestId('keyboard-target')).toBeFocused();
	await page.keyboard.type('ab', { delay: 0 });
	await expect(page.getByTestId('vk-a')).toHaveClass(/delivered/);
	await expect(page.getByTestId('vk-a')).not.toHaveClass(/host-held/);
	await expect(page.getByTestId('vk-b')).not.toHaveClass(/delivered/);
	await request(page, 'step', { instructions: 80_000 });
	await expect(page.getByTestId('vk-a')).not.toHaveClass(/delivered/);
	await expect(page.getByTestId('vk-b')).toHaveClass(/delivered/);
	await page.getByTestId('clear-typing').click();
	await expect(page.getByTestId('vk-b')).not.toHaveClass(/delivered/);
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await page.getByRole('button', { name: 'More options' }).click();
	await page.getByRole('button', { name: 'Shortcuts', exact: true }).click();
	await expect(page.getByTestId('keyboard-shortcuts')).toContainText('Enter stores');
	await page.getByRole('button', { name: 'Close shortcuts' }).click();
	await expect(page.getByTestId('keyboard-shortcuts')).toHaveCount(0);
});

async function pointDown(page: Page, id: string) {
	const button = page.getByTestId(id);
	await button.scrollIntoViewIfNeeded();
	const box = (await button.boundingBox())!;
	await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
	await page.mouse.down();
}

async function script(page: Page, source: string, wait = true) {
	await page.getByTestId('fnr-panel').evaluate((panel: HTMLDetailsElement) => {
		panel.open = true;
	});
	await page.getByTestId('fnr-editor').fill(source);
	await page.getByTestId('fnr-run').click();
	if (wait) {
		await expect(page.getByTestId('fnr-run')).toBeEnabled();
		await expect(page.getByTestId('fnr-error')).toHaveCount(0);
	}
}

for (const model of ['pc-e500', 'iq-7000']) {
	const code = model === 'pc-e500' ? 0x56 : 0x18;
	const virtual = model === 'pc-e500' ? 'vk-pf1' : 'vk-calendar';
	const shift = model === 'pc-e500' ? 0x06 : 0x02;

	test(`${model}: physical, pointer and modifier owners do not release each other`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('assisted-taps-toggle').uncheck();
		await page.getByTestId('physical-keyboard-mode').selectOption('keycaps');
		await page.getByTestId('physical-keyboard-toggle').check();
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: false });
		await page.getByTestId(virtual).click();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: false });
		await page.keyboard.up('F1');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('ShiftLeft');
		await page.keyboard.down('ShiftRight');
		await page.keyboard.up('ShiftLeft');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [shift], on: false });
		await page.keyboard.up('ShiftRight');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
	});

	test(`${model}: assisted release is bounded, repress-safe, and cancelled on blur while paused`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId(virtual).click();
		let state = await request(page, 'input_state');
		expect(state.rust.matrix).toEqual([code]);
		expect(state.pendingVirtualRelease).toEqual([[code, 40_000]]);
		await request(page, 'step', { instructions: 10_000 });
		state = await request(page, 'input_state');
		expect(state.pendingVirtualRelease).toEqual([[code, 30_000]]);
		await pointDown(page, virtual);
		await request(page, 'step', { instructions: 40_000 });
		expect(await contacts(page)).toEqual({ matrix: [code], on: false });
		await page.mouse.up();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId(virtual).click();
		await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		expect((await request(page, 'input_state')).pendingVirtualRelease).toEqual([]);
	});

	test(`${model}: focus loss and text entry release host keys without stealing editor input`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('physical-keyboard-toggle').check();
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F1');
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F1');
		await page.getByTestId('fnr-panel').evaluate((panel: HTMLDetailsElement) => {
			panel.open = true;
		});
		await page.getByTestId('fnr-editor').focus();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F1');
		await page.keyboard.press('ShiftLeft');
		await page.keyboard.press('Enter');
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId('emu-status').click();
		await page.keyboard.down('F12');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: true });
		await page.getByTestId('physical-keyboard-toggle').uncheck();
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F12');
	});

	test(`${model}: isolated script cleanup releases its raw/ON keys and preserves another owner`, async ({ page }) => {
		await open(page, model);
		// A distinct raw host owner persists while the script owns the same matrix contact.
		await request(page, 'physical_key', { code, down: true, owner: 'test-host' });
		await script(
			page,
			`await e.keys.phys.press(${code}); await e.onKey.press(); e.print('contacts held'); while (true) {}`,
			false,
		);
		await expect.poll(() => contacts(page)).toEqual({ matrix: [code], on: true });
		await page.getByRole('button', { name: 'Stop', exact: true }).click();
		await expect(page.getByTestId('fnr-run')).toBeEnabled();
		expect(await contacts(page)).toEqual({ matrix: [code], on: false });
		await request(page, 'physical_key', { code, down: false, owner: 'test-host' });
		await script(page, `await e.keys.phys.press(${code}); await e.onKey.press();`);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await script(page, `await e.keys.event.press(${code});`);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});

	test(`${model}: stale input cannot affect a replacement machine`, async ({ page }) => {
		await open(page, model);
		const { generation } = await request(page, 'input_state');
		await expect(request(page, 'physical_key', { code, down: true, generation: generation - 1 })).rejects.toThrow(
			'Stale input',
		);
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});
}

for (const model of ['pc-e500', 'iq-7000']) {
	const plus = model === 'pc-e500' ? 0x47 : 0x38;
	const shift = model === 'pc-e500' ? 0x06 : 0x02;
	test(`${model}: focus action enables layout-aware laptop operators and highlights contacts`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('keyboard-focus').click();
		await expect(page.getByTestId('physical-keyboard-toggle')).toBeChecked();
		await expect(page.getByTestId('keyboard-target')).toBeFocused();
		await page.keyboard.down('ShiftLeft');
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.down('Equal');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [plus], on: false });
		await expect(page.getByTestId('vk-plus')).toHaveClass(/host-held/);
		await page.keyboard.up('ShiftLeft'); // Key-up now says "=", but must release "+".
		await page.keyboard.up('Equal');
		await request(page, 'step', { instructions: 80_000 }); // Drain assisted hold and release gap while paused.
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await expect(page.getByTestId('vk-plus')).not.toHaveClass(/host-held/);
		await page.keyboard.down('F9');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [shift], on: false });
		await page.getByTestId('physical-keyboard-mode').selectOption('keycaps');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('F9');
	});

	test(`${model}: shortcuts, IME, repeated keys and unsupported symbols are safe`, async ({ page }) => {
		await open(page, model);
		await page.getByTestId('keyboard-focus').click();
		await page.keyboard.down('F9');
		await page.keyboard.down('ControlLeft');
		await page.keyboard.down('KeyA');
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('KeyA');
		await page.keyboard.up('ControlLeft');
		await page.keyboard.up('F9');
		await page.getByTestId('keyboard-focus').click();
		await page.keyboard.press('!');
		await expect(page.getByTestId('keyboard-notice')).toContainText('No qualified');
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.down('ArrowDown');
		const prevented = await page.getByTestId('keyboard-target').evaluate((target) => {
			const repeat = new KeyboardEvent('keydown', {
				key: 'ArrowDown',
				code: 'ArrowDown',
				repeat: true,
				bubbles: true,
				cancelable: true,
			});
			target.dispatchEvent(repeat);
			return repeat.defaultPrevented;
		});
		expect(prevented).toBe(true);
		await page.getByTestId('keyboard-target').dispatchEvent('compositionstart');
		await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
		await page.keyboard.up('ArrowDown');
		await page.getByTestId('keyboard-target').dispatchEvent('keydown', { key: 'a', code: 'KeyA', isComposing: true });
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});
}

test('IQ direct comma reports the required sequence without sending a bogus contact', async ({ page }) => {
	await open(page, 'iq-7000');
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.down('F9');
	await page.keyboard.down('Comma');
	await expect(page.getByTestId('keyboard-notice')).toContainText('F9, release it, then press K');
	await expect.poll(() => contacts(page)).toEqual({ matrix: [0x02], on: false });
	await page.keyboard.up('Comma');
	await expect.poll(() => contacts(page)).toEqual({ matrix: [0x02], on: false });
	await page.keyboard.up('F9');
	await request(page, 'step', { instructions: 80_000 });
	await expect.poll(() => contacts(page)).toEqual({ matrix: [], on: false });
});

for (const model of ['pc-e500', 'iq-7000']) {
	test(`${model}: buffered rollover stays ordered, pauses, and cancels after all host keys are up`, async ({
		page,
	}) => {
		await open(page, model);
		await page.getByTestId('keyboard-focus').click();
		await page.keyboard.down('KeyA');
		await page.keyboard.down('KeyB');
		await page.keyboard.up('KeyB');
		await page.keyboard.up('KeyA');
		const a = model === 'iq-7000' ? 0x1c : 0x03;
		const b = model === 'iq-7000' ? 0x04 : 0x15;
		await expect(page.getByTestId('typing-status')).toContainText('2/128');
		expect(await contacts(page)).toEqual({ matrix: [a], on: false });
		await request(page, 'step', { instructions: 40_000 });
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await request(page, 'step', { instructions: 40_000 });
		expect(await contacts(page)).toEqual({ matrix: [b], on: false });
		await page.evaluate(() => window.dispatchEvent(new Event('blur')));
		await expect(page.getByTestId('typing-status')).toContainText('0/128');
		await request(page, 'step', { instructions: 200_000 });
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
		await page.getByTestId('keyboard-focus').click();
		await page.keyboard.type('ABBA', { delay: 0 });
		expect((await request(page, 'input_state')).typing.pending).toBe(4);
		await page.keyboard.down('F12'); // ON bypasses and cancels the backlog.
		expect(await contacts(page)).toEqual({ matrix: [], on: true });
		expect((await request(page, 'input_state')).typing.pending).toBe(0);
		await page.keyboard.up('F12');
	});
}

test('typing overflow pauses and blocks the remaining burst until clear, without resetting the machine', async ({
	page,
}) => {
	await open(page, 'iq-7000');
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.type('A'.repeat(130), { delay: 0 });
	await expect(page.getByRole('alert')).toContainText('Typing buffer overflow');
	expect((await request(page, 'input_state')).typing).toEqual({ pending: 0, blocked: true, capacity: 128 });
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await page.getByTestId('clear-typing').click();
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.press('KeyB');
	expect((await request(page, 'input_state')).typing.pending).toBe(1);
});

test('real IQ ROM consumes the browser physical MEMO and CAPS controls', async ({ page }, info) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await request(page, 'step', { instructions: 500_000 });
	await page.getByTestId('vk-memo').click();
	await request(page, 'step', { instructions: 200_000 });
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await script(page, `e.assert((await e.lcd.capture()).annunciators.caps, 'reset starts with CAPS');`);
	await page.getByTestId('vk-caps').click();
	await request(page, 'step', { instructions: 200_000 });
	await script(page, `e.assert(!(await e.lcd.capture()).annunciators.caps, 'physical CAPS must change ROM state');`);
	await page.locator('.lcd-display').screenshot({ path: info.outputPath('real-iq-physical-memo-caps.png') });
});

// Drive actual DOM keyboard/pointer events and advance the compiled Rust worker.
// No keyboard.injectEvent, RAM/FIFO writes, direct app calls or display patches.
async function hostTap(page: Page, key: string) {
	await page.getByTestId('emu-status').click();
	await page.keyboard.down(key);
	await request(page, 'step', { instructions: 40_000 });
	await page.keyboard.up(key);
	await request(page, 'step', { instructions: 40_000 });
}
async function virtualTap(page: Page, id: string, settle = 40_000) {
	await page.getByTestId('vk-' + id).click();
	await request(page, 'step', { instructions: 40_000 + settle });
}

async function shiftedHostTap(page: Page, key: string) {
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.down('ShiftLeft');
	await page.keyboard.down(key);
	await request(page, 'step', { instructions: 40_000 });
	await page.keyboard.up(key);
	await page.keyboard.up('ShiftLeft');
	await request(page, 'step', { instructions: 40_000 });
}

test('real IQ ROM: browser letter/digit/editor input stores and reopens edited MEMO', async ({ page }, info) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await page.getByTestId('physical-keyboard-toggle').check();
	await request(page, 'step', { instructions: 500_000 });
	await hostTap(page, 'F4');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	for (const key of ['KeyA', 'KeyB', 'KeyC']) await hostTap(page, key);
	await virtualTap(page, '2');
	await hostTap(page, 'Backspace');
	await hostTap(page, 'KeyD');
	await expect(page.getByTestId('lcd-text')).toContainText('ABCD');
	await hostTap(page, 'Enter');
	await hostTap(page, 'F4');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await hostTap(page, 'PageDown');
	await expect(page.getByTestId('lcd-text')).toHaveText('ABCD');
	await hostTap(page, 'F9'); // device SHIFT latch, not host uppercase composition
	await hostTap(page, 'KeyA'); // EDIT
	await virtualTap(page, 'x');
	await hostTap(page, 'Enter');
	await hostTap(page, 'F4');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await hostTap(page, 'PageDown');
	await expect(page.getByTestId('lcd-text')).toHaveText('XBCD');
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await page.locator('.lcd-display').screenshot({ path: info.outputPath('real-iq-edited-memo.png') });
	await page.getByTestId('device-shell').screenshot({ path: info.outputPath('real-iq-device-edited-memo.png') });
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.screenshot({ path: info.outputPath('real-iq-device-first-page.png'), fullPage: true });
});

test('real ROM: PC-E500 browser calculator consumes physical and virtual expression input', async ({ page }, info) => {
	test.skip(process.env.PCE500_E2E_REAL_ROM !== '1', 'Requires private PC-E500 ROM');
	await open(page, 'pc-e500', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await page.getByTestId('physical-keyboard-toggle').check();
	await request(page, 'step', { instructions: 500_000 });
	await virtualTap(page, 'pf1', 800_000);
	await expect(page.getByTestId('lcd-text')).toContainText('S1(MAIN):NEW CARD');
	await virtualTap(page, 'pf1', 800_000);
	await expect(page.getByTestId('lcd-text')).toContainText('MAIN MENU');
	await virtualTap(page, 'pf2', 800_000);
	await expect(page.getByTestId('lcd-text')).toContainText('0.');
	await hostTap(page, 'Digit2');
	await expect(page.getByTestId('lcd-text')).toContainText('2.');
	await shiftedHostTap(page, 'Equal');
	await hostTap(page, 'Digit2');
	await hostTap(page, 'Enter');
	await expect(page.getByTestId('lcd-text')).toContainText('4.');
	await hostTap(page, 'Escape');
	await hostTap(page, 'Digit4');
	await shiftedHostTap(page, 'Digit8');
	await hostTap(page, 'Digit3');
	await hostTap(page, 'Enter');
	await expect(page.getByTestId('lcd-text')).toContainText('12.');
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await page.locator('.lcd-display').screenshot({ path: info.outputPath('real-pc-calculator-12.png') });
	await page.getByTestId('device-shell').screenshot({ path: info.outputPath('real-pc-device-calculator-12.png') });
	await page.getByTestId('advanced-panel').locator('> summary').click();
	await page.screenshot({ path: info.outputPath('real-pc-device-first-page.png'), fullPage: true });
});

test('real ROM: PC-E500 zero-delay repeated digits reach a fresh calculator through physical contacts', async ({
	page,
}) => {
	test.skip(process.env.PCE500_E2E_REAL_ROM !== '1', 'Requires private PC-E500 ROM');
	await open(page, 'pc-e500', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await page.getByTestId('keyboard-focus').click();
	await request(page, 'step', { instructions: 500_000 });
	await virtualTap(page, 'pf1', 800_000);
	await virtualTap(page, 'pf1', 800_000);
	await virtualTap(page, 'pf2', 800_000);
	await expect(page.getByTestId('lcd-text')).toContainText('0.');
	await page.getByRole('button', { name: 'Run', exact: true }).click();
	await page.getByTestId('keyboard-focus').click();
	await page.keyboard.type('11+22', { delay: 0 });
	await page.keyboard.press('Enter');
	await expect(page.getByTestId('lcd-text')).toContainText('33.');
	await expect.poll(async () => (await request(page, 'input_state')).typing.pending).toBe(0);
	await page.getByRole('button', { name: 'Stop', exact: true }).click();
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
});

test('real IQ ROM: previewed paste enters comma and newline through actual MEMO keys', async ({ page }) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await request(page, 'step', { instructions: 500_000 });
	await virtualTap(page, 'memo');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await previewPaste(page, 'PASTE ONE,2\nSECOND');
	await page.getByTestId('submit-paste').click();
	await page.getByTestId('pause-resume').click();
	await expect(page.getByTestId('lcd-text')).toContainText('PASTE ONE,2');
	await expect(page.getByTestId('lcd-text')).toContainText('SECOND');
	await expect.poll(async () => (await request(page, 'input_state')).paste.pending).toBe(0);
	await page.getByTestId('pause-resume').click();
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
});

test('real ROM: previewed PC paste executes an explicitly confirmed calculator expression', async ({ page }) => {
	test.skip(process.env.PCE500_E2E_REAL_ROM !== '1', 'Requires private PC-E500 ROM');
	await open(page, 'pc-e500', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await request(page, 'step', { instructions: 500_000 });
	await virtualTap(page, 'pf1', 800_000);
	await virtualTap(page, 'pf1', 800_000);
	await virtualTap(page, 'pf2', 800_000);
	await expect(page.getByTestId('lcd-text')).toContainText('0.');
	await previewPaste(page, '11+22\n');
	await expect(page.getByTestId('paste-preview')).toContainText('may execute');
	await page.getByTestId('submit-paste').click();
	await page.getByTestId('pause-resume').click();
	await expect(page.getByTestId('lcd-text')).toContainText('33.');
	await expect.poll(async () => (await request(page, 'input_state')).paste.pending).toBe(0);
	await page.getByTestId('pause-resume').click();
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
});

for (const catchUp of [false, true])
	test(`real IQ ROM: zero-delay typing preserves order and repeated letters in live MEMO (catch-up ${catchUp})`, async ({
		page,
	}) => {
		test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
		await open(page, 'iq-7000', true);
		await page.getByText('LCD (decoded text)', { exact: true }).click();
		await page.getByTestId('keyboard-focus').click();
		await request(page, 'step', { instructions: 500_000 });
		await hostTap(page, 'F4');
		await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
		await page.getByTestId('typing-catch-up').setChecked(catchUp);
		await page.getByRole('button', { name: 'Run', exact: true }).click();
		await page.getByTestId('keyboard-focus').click();
		const started = Date.now();
		await page.keyboard.type('AABBCCDDEE1122', { delay: 0 });
		await expect(page.getByTestId('lcd-text')).toContainText('AABBCCDDEE1122');
		await expect.poll(async () => (await request(page, 'input_state')).typing.pending).toBe(0);
		console.log(JSON.stringify({ model: 'iq-7000', catchUp, burstToLcdMs: Date.now() - started }));
		await page.getByRole('button', { name: 'Stop', exact: true }).click();
		expect(await contacts(page)).toEqual({ matrix: [], on: false });
	});

test('real IQ ROM: live keyboard typing, symbols and laptop newline reach MEMO through ROM', async ({ page }, info) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await request(page, 'step', { instructions: 500_000 });
	await page.getByRole('button', { name: 'Run', exact: true }).click();
	await page.getByTestId('keyboard-focus').click();
	// Actual paced Run, not step RPCs between characters. Human-scale contacts
	// intentionally leave a release gap for the firmware scanner/debounce.
	const liveTap = async (key: string) => {
		await page.keyboard.press(key, { delay: 150 });
		await page.waitForTimeout(100);
	};
	await liveTap('F4');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	for (const key of ['KeyA', 'KeyB', 'KeyC', 'Shift+Equal', 'Digit2', 'F9', 'KeyK', 'Digit3']) await liveTap(key);
	await expect(page.getByTestId('lcd-text')).toContainText('ABC+2,3');
	await liveTap('Shift+Enter');
	for (const key of ['KeyD', 'KeyE', 'KeyF']) await liveTap(key);
	await expect(page.getByTestId('lcd-text')).toContainText('DEF');
	await liveTap('Enter');
	await liveTap('F4');
	await expect(page.getByTestId('lcd-text')).toContainText('MEMO ?');
	await liveTap('PageDown');
	await expect(page.getByTestId('lcd-text')).toContainText('ABC+2,3');
	await expect(page.getByTestId('lcd-text')).toContainText('DEF');
	await page.getByRole('button', { name: 'Stop', exact: true }).click();
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
	await expect(page.getByRole('alert')).toHaveCount(0);
	await page.getByTestId('keyboard-target').screenshot({ path: info.outputPath('real-iq-live-keyboard-memo.png') });
});

test('real IQ ROM: cursor directions and Insert/Delete have distinct editor effects', async ({ page }) => {
	test.skip(process.env.IQ7000_E2E_REAL_ROM !== '1', 'Requires private IQ-7000 ROM');
	await open(page, 'iq-7000', true);
	await page.getByText('LCD (decoded text)', { exact: true }).click();
	await page.getByTestId('physical-keyboard-toggle').check();
	await request(page, 'step', { instructions: 500_000 });
	for (const key of [
		'F4',
		'KeyA',
		'KeyB',
		'KeyC',
		'KeyD',
		'F11',
		'KeyE',
		'KeyF',
		'KeyG',
		'KeyH',
		'ArrowLeft',
		'ArrowLeft',
		'Insert',
		'KeyX',
	])
		await hostTap(page, key);
	await expect(page.getByTestId('lcd-text')).toContainText('EFXGH');
	await hostTap(page, 'Delete');
	await expect(page.getByTestId('lcd-text')).toContainText('EFXH');
	for (const key of ['ArrowUp', 'ArrowRight', 'KeyX']) await hostTap(page, key);
	await expect(page.getByTestId('lcd-text')).toContainText('ABCDX');
	for (const key of ['ArrowDown', 'KeyX']) await hostTap(page, key);
	await expect(page.getByTestId('lcd-text')).toContainText('EFXHX');
	expect(await contacts(page)).toEqual({ matrix: [], on: false });
});
