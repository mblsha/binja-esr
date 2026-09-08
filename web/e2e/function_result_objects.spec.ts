import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

test('structured WASM call results match JSON through browser exports and release ownership', async ({ page }) => {
	for (const [url, file, contentType] of [
		['**/__result_test.js', 'pce500_wasm.js', 'text/javascript'],
		['**/__result_test.wasm', 'pce500_wasm_bg.wasm', 'application/wasm'],
	]) {
		const body = await readFile(resolve('src/lib/wasm/pce500_wasm', file));
		await page.route(url, (route) => route.fulfill({ body, contentType }));
	}
	await page.goto('/');
	const result = await page.evaluate(async () => {
		const url = '/__result_test.js';
		const api = await import(/* @vite-ignore */ url);
		await api.default({ module_or_path: '/__result_test.wasm' });
		const results = [];
		for (const model of ['pc-e500', 'iq-7000']) {
			const make = () => {
				const e = new api.Sc62015Emulator();
				e.load_rom_with_model(new Uint8Array(0x40000), model);
				e.set_reg('PC', 0xb8000);
				e.set_reg('S', 0xb9003);
				return e;
			};
			const legacy = make(),
				typed = make();
			try {
				for (const cancel of [false, true]) {
					const a = legacy.call_function_begin(0xb8000, 4, {});
					const b = typed.call_function_begin(0xb8000, 4, {});
					legacy.call_function_slice(a, 4, 16);
					typed.call_function_slice(b, 4, 16);
					const expected = JSON.parse(cancel ? legacy.call_function_cancel(a) : legacy.call_function_finish(a));
					const actual = cancel ? typed.call_function_cancel_object(b) : typed.call_function_finish_object(b);
					results.push({
						model,
						expected,
						actual,
						plainMap: Object.getPrototypeOf(actual.before_regs) === Object.prototype,
					});
					// This enters Rust again after serialization through the actual JS wrapper.
					typed.get_reg('PC');
				}
				const id = typed.call_function_begin(0xb8000, 4, { stubs: [{ id: 7, pc: 0xb8000 }] });
				const slice = typed.call_function_slice(id, 4, 16);
				if (slice.state !== 'stub') throw new Error('Expected a stub handoff');
				// Host callbacks may re-enter read APIs after a slice returns.
				const callback = () => ({ regs: [{ name: 'A', value: typed.get_reg('A') }], ret: { kind: 'stay' } });
				typed.call_function_apply_stub(id, slice.stub_request.sequence, callback());
				typed.call_function_cancel_object(id);
			} finally {
				legacy.free();
				typed.free();
			}
		}
		return results;
	});
	for (const item of result) {
		expect(item.actual).toEqual(item.expected);
		expect(item.plainMap).toBe(true);
	}
});
