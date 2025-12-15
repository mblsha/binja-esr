import { cleanup, fireEvent, render } from '@testing-library/svelte';
import { waitFor } from '@testing-library/dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import Page from '../routes/+page.svelte';

function asFetchUrl(input: any): string {
	if (typeof input === 'string') return input;
	if (input instanceof URL) return input.href;
	if (input && typeof input.url === 'string') return input.url;
	return String(input);
}

async function loadTestRom(): Promise<Uint8Array> {
	const romPath = resolve(process.cwd(), 'emulator-wasm/testdata/pf1_demo_rom_window.rom');
	const bytes = await readFile(romPath);
	return new Uint8Array(bytes);
}

async function loadWasmBytes(url: string): Promise<Uint8Array> {
	if (url.startsWith('file:')) {
		const wasmPath = fileURLToPath(url);
		const bytes = await readFile(wasmPath);
		return new Uint8Array(bytes);
	}
	const fallbackPath = resolve(process.cwd(), 'src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm');
	const bytes = await readFile(fallbackPath);
	return new Uint8Array(bytes);
}

describe('PC-E500 web emulator', () => {
	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('boots, shows decoded LCD text, PF1 changes it', async () => {
		const romBytes = await loadTestRom();

		vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
			createImageData: (width: number, height: number) => ({
				data: new Uint8ClampedArray(width * height * 4)
			}),
			putImageData: () => {}
		} as any);

		vi.stubGlobal('fetch', async (input: any) => {
			const url = asFetchUrl(input);

			if (url === '/api/rom' || url.endsWith('/api/rom')) {
				return new Response(romBytes as any, {
					status: 200,
					headers: {
						'content-type': 'application/octet-stream',
						'cache-control': 'no-store',
						'x-rom-source': 'test:pf1_demo_rom_window.rom'
					}
				});
			}

			if (url.endsWith('pce500_wasm_bg.wasm')) {
				const wasmBytes = await loadWasmBytes(url);
				return new Response(wasmBytes as any, {
					status: 200,
					headers: {
						'content-type': 'application/wasm',
						'cache-control': 'no-store'
					}
				});
			}

			throw new Error(`Unexpected fetch: ${url}`);
		});

		const { getByTestId, getByText } = render(Page);

		const step20k = getByText('Step 20k') as HTMLButtonElement;
		await waitFor(() => expect(step20k.disabled).toBe(false));

		await fireEvent.click(step20k);
		await waitFor(() => {
			const text = (getByTestId('lcd-text').textContent ?? '').trim();
			expect(text).toContain('BOOT');
		});
		await waitFor(() => {
			const info = (getByTestId('build-info').textContent ?? '').trim();
			expect(info).toMatch(/WASM:\s*v\d+\./i);
		});
		expect((getByTestId('emu-status').textContent ?? '').toUpperCase()).toContain('STOPPED');
		expect(getByTestId('call-stack-empty').textContent ?? '').toContain('No frames');

		const pf1 = getByTestId('vk-pf1') as HTMLButtonElement;
		// Tap PF1 (press+release) and ensure the firmware sees it after stepping.
		await fireEvent.pointerDown(pf1);
		await fireEvent.pointerUp(pf1);
		await fireEvent.click(step20k);
		await fireEvent.click(step20k);
		await fireEvent.click(step20k);

		await waitFor(() => {
			const text = (getByTestId('lcd-text').textContent ?? '').trim();
			expect(text).toContain('MENU');
		});
		await waitFor(() => {
			const status = getByTestId('emu-status').textContent ?? '';
			expect(status).toMatch(/PC: 0x/i);
		});
		expect(getByTestId('regs-table')).toBeTruthy();
	});

	it('function runner can call with trace enabled (no wasm-bindgen aliasing)', async () => {
		const romBytes = await loadTestRom();

		vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
			createImageData: (width: number, height: number) => ({
				data: new Uint8ClampedArray(width * height * 4)
			}),
			putImageData: () => {}
		} as any);

		vi.stubGlobal('fetch', async (input: any) => {
			const url = asFetchUrl(input);

			if (url === '/api/rom' || url.endsWith('/api/rom')) {
				return new Response(romBytes as any, {
					status: 200,
					headers: {
						'content-type': 'application/octet-stream',
						'cache-control': 'no-store',
						'x-rom-source': 'test:pf1_demo_rom_window.rom'
					}
				});
			}

			if (url.endsWith('pce500_wasm_bg.wasm')) {
				const wasmBytes = await loadWasmBytes(url);
				return new Response(wasmBytes as any, {
					status: 200,
					headers: {
						'content-type': 'application/wasm',
						'cache-control': 'no-store'
					}
				});
			}

			throw new Error(`Unexpected fetch: ${url}`);
		});

		const { getByTestId, getByText, queryByTestId, getAllByTestId } = render(Page);

		const step20k = getByText('Step 20k') as HTMLButtonElement;
		await waitFor(() => expect(step20k.disabled).toBe(false));

		// Ensure ROM is loaded (the function runner uses the same emulator instance).
		await fireEvent.click(step20k);
		await waitFor(() => {
			const text = (getByTestId('lcd-text').textContent ?? '').trim();
			expect(text).toContain('BOOT');
		});

		// Open function runner panel.
		const panel = getByTestId('fnr-panel') as HTMLDetailsElement;
		panel.open = true;

		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		const script = `
await e.reset({ fresh: true, warmupTicks: 1_000 });
await e.keyboard.tap(0x56, 100); // PF1
const pc = e.reg(Reg.PC);
await e.call(pc, undefined, { maxInstructions: 2_000, trace: true });
`;
		await fireEvent.input(editor, { target: { value: script } });

		const run = getByTestId('fnr-run') as HTMLButtonElement;
		await fireEvent.click(run);

		await waitFor(() => {
			expect(queryByTestId('fnr-error')).toBeNull();
			expect(getAllByTestId('fnr-call').length).toBeGreaterThanOrEqual(1);
		});
	});
});
