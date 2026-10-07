/** Reproduce a supplied native proof through the normal shared WASM facade.
 * Usage: vite-node --script scripts/oz9600_public_wasm_parity.ts NATIVE_PROOF NEW_OUTPUT
 * Inputs are external verified fixtures; no ROMs or captured RAM ship here.
 */
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import init, { Sc62015Emulator } from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';
import { ozPbm, runOzPhysicalReplay } from '../src/lib/emulator/oz9600_replay';

const [nativeArg, outputArg] = process.argv.slice(2);
if (!nativeArg || !outputArg) throw new Error('Expected native proof directory and NEW output directory');
const native = resolve(nativeArg),
	output = resolve(outputArg);
await mkdir(output); // Never overwrite another proof.
const wasmPath = fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url));
const wasmBytes = await readFile(wasmPath);
await init({ module_or_path: wasmBytes });
await copyFile(wasmPath, resolve(output, 'pce500_wasm_bg.wasm'));
await copyFile(
	fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm.js', import.meta.url)),
	resolve(output, 'pce500_wasm.js'),
);
const bundle = await readFile(resolve(native, 'rom.ozrom'));
let store: Uint8Array, done: Uint8Array;
const summary: object[] = [];
const sha = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
for (const name of ['store', 'restart-view', 'done-store', 'done-restart']) {
	const retained =
		name === 'store'
			? await readFile(resolve(native, 'initial-retained.bin'))
			: name === 'done-restart'
				? done!
				: store!;
	const document = await readFile(resolve(native, name, 'input.json'), 'utf8');
	const expected = JSON.parse(await readFile(resolve(native, name, 'replay-report.json'), 'utf8'));
	const emulator = new Sc62015Emulator();
	emulator.load_oz9600(bundle, retained, 'experimental');
	assert.equal(emulator.oz9600_profile(), 'experimental');
	const reports: object[] = [];
	const boundaries = await runOzPhysicalReplay(emulator, document, {
		yieldHost: () => new Promise<void>((r) => setImmediate(r)),
		onStep: async (index, state) => {
			const pbm = ozPbm(emulator.lcd_capture());
			const report = { step: index, ...state, pbm_sha256: sha(pbm) };
			assert.deepEqual(report, expected[index], `${name} step ${index}`);
			reports.push(report);
		},
	});
	const bytes = emulator.export_oz9600_retained();
	const pbm = ozPbm(emulator.lcd_capture());
	assert.deepEqual(Buffer.from(bytes), await readFile(resolve(native, name, 'retained-state.bin')));
	assert.deepEqual(Buffer.from(pbm), await readFile(resolve(native, name, 'final.pbm')));
	const dir = resolve(output, name);
	await mkdir(dir);
	await writeFile(resolve(dir, 'input.json'), document);
	await writeFile(resolve(dir, 'replay-report.json'), JSON.stringify(reports, null, 2));
	await writeFile(resolve(dir, 'retained-state.bin'), bytes);
	await writeFile(resolve(dir, 'final.pbm'), pbm);
	if (name === 'store') store = bytes;
	if (name === 'done-store') done = bytes;
	// Replacement failures must leave the running model, CPU and backing intact.
	const state = emulator.oz9600_state();
	const bad = bytes.slice();
	bad[bad.length - 1] ^= 1;
	for (const action of [
		() => emulator.load_rom_with_model(new Uint8Array([0]), 'oz-9600'),
		() => emulator.restore_oz9600_retained(bad),
		() => emulator.load_oz9600(bundle, retained, 'implicit'),
		() => emulator.validate_oz9600_replay('{"steps":[{"boundaries":1,"rtc_causes":{"a":4}}]}'),
		() => emulator.lcd_chip_pixels(),
		() => emulator.lcd_trace(),
		() => emulator.call_function_begin(0xf0000, 1, {}),
	]) {
		assert.throws(action);
		assert.deepEqual(emulator.oz9600_state(), state);
		assert.deepEqual(emulator.export_oz9600_retained(), bytes);
	}
	emulator.reset();
	assert.equal(emulator.oz9600_profile(), 'experimental');
	assert.deepEqual(emulator.export_oz9600_retained(), bytes);
	assert.equal(emulator.instruction_count(), 0n);
	assert.equal(emulator.read_u8(0x1000ec), 0);
	emulator.free();
	const record = { name, steps: reports.length, boundaries, pbm_sha256: sha(pbm), retained_sha256: sha(bytes) };
	summary.push(record);
	console.log(JSON.stringify(record));
}
const strict = new Sc62015Emulator();
strict.load_rom_with_model(bundle, 'oz-9600');
assert.equal(strict.oz9600_profile(), 'strict');
assert.equal(strict.read_u8(0x1000ec), 0);
assert.throws(() => strict.step_scheduler_boundaries(20_000), /unqualified zero code target/);
strict.free();
await writeFile(
	resolve(output, 'summary.json'),
	JSON.stringify(
		{
			native,
			wasm_sha256: sha(wasmBytes),
			cases: summary,
			strict_cold_boot: 'explicit unresolved-zero-target error',
			hardware_goal_complete: false,
		},
		null,
		2,
	),
);
