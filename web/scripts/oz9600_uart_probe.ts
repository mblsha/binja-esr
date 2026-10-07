/** Normal physical ROM replay plus a separate external UART byte plan.
 * Usage: vite-node --script scripts/oz9600_uart_probe.ts BUNDLE RETAINED INPUT PEER NEW_OUTPUT
 * Software rings/CPU registers are never written by this probe.
 */
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import init, { Sc62015Emulator } from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';
import { ozPbm, runOzPhysicalReplay } from '../src/lib/emulator/oz9600_replay';

const [bundleArg, retainedArg, inputArg, peerArg, outputArg] = process.argv.slice(2);
if (!bundleArg || !retainedArg || !inputArg || !peerArg || !outputArg)
	throw new Error('Expected BUNDLE RETAINED INPUT PEER NEW_OUTPUT');
const output = resolve(outputArg);
const bytes = await readFile(
	fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url)),
);
await init({ module_or_path: bytes });
const bundle = await readFile(resolve(bundleArg));
const retained = await readFile(resolve(retainedArg));
const document = await readFile(resolve(inputArg), 'utf8');
const peerBytes = await readFile(resolve(peerArg), 'utf8');
const steps = JSON.parse(document).steps;
const peer = JSON.parse(peerBytes) as { step: number; value: number }[];
if (
	!Array.isArray(peer) ||
	peer.some(
		(e) =>
			Object.keys(e).sort().join(',') !== 'step,value' ||
			!Number.isInteger(e.step) ||
			e.step < 0 ||
			e.step >= steps.length ||
			!Number.isInteger(e.value) ||
			e.value < 0 ||
			e.value > 255,
	)
)
	throw new Error('Invalid peer step/byte');
const emulator = new Sc62015Emulator();
emulator.load_oz9600(bundle, retained, 'experimental-isr-mti-writable');
emulator.validate_oz9600_replay(document);
await mkdir(output);
await writeFile(resolve(output, 'input.json'), document);
await writeFile(resolve(output, 'peer.json'), peerBytes);
await writeFile(resolve(output, 'initial-retained.bin'), retained);
await writeFile(resolve(output, 'pce500_wasm_bg.wasm'), bytes);
await copyFile(
	fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm.js', import.meta.url)),
	resolve(output, 'pce500_wasm.js'),
);
const sha = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex');
const reports: object[] = [];
const transmitted: number[] = [];
let failure: string | null = null;
let boundaries: number | null = null;
const queuePeer = (step: number) => {
	for (const event of peer.filter((e) => e.step === step)) emulator.sio_queue_rx_byte(event.value);
};
queuePeer(0);
try {
	boundaries = await runOzPhysicalReplay(emulator, document, {
		yieldHost: () => new Promise<void>((r) => setImmediate(r)),
		onStep: async (index, state) => {
			const pbm = ozPbm(emulator.lcd_capture());
			const tx = [...emulator.sio_drain_tx_bytes()];
			transmitted.push(...tx);
			await writeFile(resolve(output, `step-${index.toString().padStart(3, '0')}.pbm`), pbm);
			reports.push({
				step: index,
				label: steps[index].label,
				state,
				tx_bytes: tx,
				uart_device: emulator.sio_uart_report(),
				pbm_sha256: sha(pbm),
			});
			console.log(JSON.stringify({ step: index, label: steps[index].label, tx_bytes: tx, pbm_sha256: sha(pbm) }));
			queuePeer(index + 1);
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
			schema: 'oz9600-shared-uart-probe-1',
			profile: 'experimental-isr-mti-writable',
			bundle_sha256: sha(bundle),
			initial_retained_sha256: sha(retained),
			input_sha256: sha(document),
			peer_sha256: sha(peerBytes),
			wasm_sha256: sha(bytes),
			completed_steps: reports.length,
			total_steps: steps.length,
			boundaries,
			failure,
			transmitted,
			final_state: emulator.oz9600_state(),
			final_pbm_sha256: sha(final),
			final_retained_sha256: sha(result),
			hardware_goal_complete: false,
			qualification:
				'External register-UART peer and normal ROM input; physical clock/divisor/pin timing and PC Link protocol remain unqualified',
		},
		null,
		2,
	),
);
emulator.free();
if (failure) process.exitCode = 1;
