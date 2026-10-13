/** Real WASM host-deadline contract; no guest writes or forced execution.
 * Usage: vite-node --script scripts/host_deadline_probe.ts BUNDLE RETAINED
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import init, { Sc62015Emulator } from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';
import { automaticHostSlice } from '../src/lib/emulator/host_pacing';
import { runHostSlice } from '../src/lib/emulator/bounded_step';

const [bundle, retained, ...extra] = process.argv.slice(2);
assert(bundle && retained && !extra.length, 'BUNDLE RETAINED');
await init({
	module_or_path: await readFile(
		fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url)),
	),
});
const emulator = new Sc62015Emulator();
emulator.load_oz9600(await readFile(bundle), await readFile(retained), 'provisional-v1');
const original = Object.getOwnPropertyDescriptor(globalThis, 'performance');
const ticks = [0, 1, 6]; // Pacing plan at 1 ms, host preempted until 6 ms.
const inputs = { limitBudget: (n: number) => n, advance: (n: number) => assert.equal(n, 0) };
try {
	Object.defineProperty(globalThis, 'performance', {
		configurable: true,
		value: {
			now: () => {
				assert(ticks.length, 'Unexpected clock read');
				return ticks.shift()!;
			},
		},
	});
	assert(automaticHostSlice(emulator, 10, inputs) <= 4);
	const state = emulator.oz9600_state(),
		ram = emulator.export_oz9600_retained();
	let result;
	const wrapper = {
		automatic_slice: (n: number) => {
			result = emulator.automatic_slice(n);
			console.log(JSON.stringify(result));
			return result;
		},
	};
	assert.equal(automaticHostSlice(wrapper, 10, inputs), 0);
	assert.equal(result!.plan.action, 'run');
	assert.equal(result!.slice.reason, 'host_yield');
	assert.equal(result!.slice.progress.boundary_budget_used, 0);
	assert.deepEqual(emulator.oz9600_state(), state);
	assert.deepEqual(emulator.export_oz9600_retained(), ram);
	assert.equal(ticks.length, 0);
} finally {
	if (original) Object.defineProperty(globalThis, 'performance', original);
	else Reflect.deleteProperty(globalThis, 'performance');
}
assert(runHostSlice(emulator, 100) > 0, 'Healthy execution resumes on the next host turn');
emulator.free();
console.log('Real WASM immediate host yield preserves guest state and input deadlines; next slice resumes.');
