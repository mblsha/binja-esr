/** Normal ROM workflow probe through the shared WASM model and physical inputs.
 * Usage: vite-node --script scripts/oz9600_workflow_probe.ts BUNDLE RETAINED|- INPUT NEW_OUTPUT PROFILE [BUILD_RECEIPT]
 * Captures actual controller pixels after every step. No guest RAM/PC patches.
 */
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import init, { Sc62015Emulator } from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';
import { ozPbm, runOzPhysicalReplay } from '../src/lib/emulator/oz9600_replay';

const [bundleArg, retainedArg, inputArg, outputArg, profile = 'strict', receiptArg] = process.argv.slice(2);
if (
	!bundleArg ||
	!retainedArg ||
	!inputArg ||
	!outputArg ||
	![
		'strict',
		'experimental',
		'experimental-isr-clear-only',
		'experimental-isr-mti-writable',
		'experimental-on-edge',
		'experimental-irq-imr',
		'experimental-rtc',
		'experimental-rtc-irq-imr',
		'provisional-v1',
	].includes(profile)
)
	throw new Error('Expected BUNDLE RETAINED|- INPUT NEW_OUTPUT and a known OZ profile');
const output = resolve(outputArg);
await mkdir(output);
const bytes = await readFile(
	fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url)),
);
await init({ module_or_path: bytes });
const bundle = await readFile(resolve(bundleArg));
const retained = retainedArg === '-' ? new Uint8Array() : await readFile(resolve(retainedArg));
const document = await readFile(resolve(inputArg), 'utf8');
const steps = JSON.parse(document).steps;
const emulator = new Sc62015Emulator();
emulator.load_oz9600(bundle, retained, profile);
emulator.validate_oz9600_replay(document);
// Read-only evidence before the first CPU boundary. Exporting backing and
// debugger peeks do not perform guest bus reads or mutate application memory.
const initialBattery = emulator.export_oz9600_retained();
const initialState = emulator.oz9600_state();
await writeFile(resolve(output, 'initial-battery.bin'), initialBattery);
await writeFile(
	resolve(output, 'initial-state.json'),
	JSON.stringify(
		{
			state: initialState,
			bp: emulator.read_u8(0x1000ec),
			factory_empty_sram: retained.length === 0,
			all_256k_sram_bytes_zero: initialBattery.slice(104, 104 + 0x40000).every((b) => b === 0),
		},
		null,
		2,
	),
);
await writeFile(resolve(output, 'input.json'), document);
await writeFile(resolve(output, 'initial-retained.bin'), retained);
const sha = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex');
let receiptSha: string | null = null;
if (receiptArg) {
	const receiptBytes = await readFile(resolve(receiptArg));
	const receipt = JSON.parse(receiptBytes.toString());
	const wrapper = await readFile(fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm.js', import.meta.url)));
	if (
		receipt.schema !== 'oz9600-build-receipt-1' ||
		receipt.artifacts.wasm.sha256 !== sha(bytes) ||
		receipt.artifacts.wasm_js.sha256 !== sha(wrapper) ||
		receipt.artifacts.bundle.sha256 !== sha(bundle)
	)
		throw new Error('Build receipt does not identify this WASM/ROM build');
	receiptSha = sha(receiptBytes);
} else {
	await writeFile(resolve(output, 'pce500_wasm_bg.wasm'), bytes);
	await copyFile(
		fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm.js', import.meta.url)),
		resolve(output, 'pce500_wasm.js'),
	);
}
const reports: object[] = [];
let failure: string | null = null;
let boundaries: number | null = null;
try {
	boundaries = await runOzPhysicalReplay(emulator, document, {
		yieldHost: () => new Promise<void>((r) => setImmediate(r)),
		onStep: async (index, state) => {
			const pbm = ozPbm(emulator.lcd_capture());
			await writeFile(resolve(output, `step-${index.toString().padStart(3, '0')}.pbm`), pbm);
			reports.push({ step: index, ...state, pbm_sha256: sha(pbm) });
			console.log(JSON.stringify({ step: index, label: steps[index].label, state, pbm_sha256: sha(pbm) }));
		},
	});
} catch (error) {
	failure = String(error);
	console.error(failure);
}
const final = ozPbm(emulator.lcd_capture());
const result = emulator.export_oz9600_retained();
await writeFile(resolve(output, 'final.pbm'), final);
await writeFile(resolve(output, 'retained-state.bin'), result);
await writeFile(resolve(output, 'replay-report.json'), JSON.stringify(reports, null, 2));
await writeFile(
	resolve(output, 'summary.json'),
	JSON.stringify(
		{
			schema: 'oz9600-shared-workflow-probe-1',
			profile,
			bundle_sha256: sha(bundle),
			initial_retained_sha256: sha(retained),
			input_sha256: sha(document),
			wasm_sha256: sha(bytes),
			build_receipt_sha256: receiptSha,
			completed_steps: reports.length,
			total_steps: steps.length,
			boundaries,
			failure,
			final_state: emulator.oz9600_state(),
			final_pbm_sha256: sha(final),
			final_retained_sha256: sha(result),
			qualification:
				'Verified ROM, fresh CPU with explicit retained backing and execution profile; physical hardware/reset/timing remain unqualified',
			hardware_goal_complete: false,
		},
		null,
		2,
	),
);
emulator.free();
if (failure) process.exitCode = 1;
