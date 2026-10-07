/** Real-ROM native/WASM PCM comparison. Inputs remain in the private proof.
 * Usage: vite-node --script scripts/oz9600_audio_parity.ts PROOF NEW_OUTPUT
 */
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import init, { Sc62015Emulator } from '../src/lib/wasm/pce500_wasm/pce500_wasm.js';
import { ozPbm, runOzPhysicalReplay } from '../src/lib/emulator/oz9600_replay';

const [baseArg, outArg] = process.argv.slice(2);
if (!baseArg || !outArg) throw new Error('Expected PROOF and NEW_OUTPUT');
const base = resolve(baseArg),
	out = resolve(outArg);
await mkdir(out);
const wasmPath = fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm_bg.wasm', import.meta.url));
const wasmBytes = await readFile(wasmPath);
await init({ module_or_path: wasmBytes });
await copyFile(wasmPath, resolve(out, 'pce500_wasm_bg.wasm'));
await copyFile(
	fileURLToPath(new URL('../src/lib/wasm/pce500_wasm/pce500_wasm.js', import.meta.url)),
	resolve(out, 'pce500_wasm.js'),
);
const sha = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
const commands = JSON.parse(await readFile(resolve(base, 'commands.json'), 'utf8'));
const results: object[] = [];
for (const { name, argv } of commands) {
	const emu = new Sc62015Emulator();
	const bundle = await readFile(argv[1]),
		retained = await readFile(argv[2]);
	const document = await readFile(argv[3], 'utf8');
	const expected = JSON.parse(await readFile(resolve(base, name, 'report.json'), 'utf8'));
	emu.load_oz9600(bundle, retained, 'experimental-rtc');
	assert.equal(emu.oz9600_audio_status().enabled, false);
	emu.set_oz9600_audio_enabled(true);
	const chunks: Buffer[] = [],
		steps: object[] = [];
	let count = 0,
		stepFirst = 0,
		stepChunks = 0;
	const drain = () => {
		const chunk = emu.take_oz9600_audio();
		assert.ok(chunk.samples instanceof Int16Array);
		assert.equal(chunk.sample_rate, 48000);
		assert.equal(chunk.first_sample, count);
		assert.equal(chunk.dropped_samples, 0);
		const bytes = Buffer.alloc(chunk.samples.length * 2);
		chunk.samples.forEach((sample: number, i: number) => bytes.writeInt16LE(sample, i * 2));
		chunks.push(bytes);
		count += chunk.samples.length;
		assert.equal(chunk.total_samples, count);
	};
	await runOzPhysicalReplay(emu, document, {
		yieldHost: async () => {
			drain();
			await new Promise<void>((r) => setImmediate(r));
		},
		onStep: async (index, rawState) => {
			drain();
			const state = rawState as any,
				previous = expected.endpoints[index];
			for (const field of ['pc', 'cycles', 'instructions', 'cpu_halted', 'selector', 'irq_total'])
				assert.equal(state[field], previous[field], `${name} step ${index} ${field}`);
			assert.deepEqual(state.rtc_registers, previous.rtc);
			const frame = ozPbm(emu.lcd_capture());
			assert.equal(sha(frame), previous.pbm_sha256);
			const bytes = Buffer.concat(chunks.slice(stepChunks));
			const record = {
				first_sample: stepFirst,
				samples: count - stepFirst,
				nonzero_samples: Array.from({ length: bytes.length / 2 }, (_, i) => bytes.readInt16LE(i * 2)).filter(
					(s) => s !== 0,
				).length,
				pcm_sha256: sha(bytes),
			};
			assert.deepEqual(record, previous.audio, `${name} step ${index} PCM`);
			steps.push({ step: index, ...record, pbm_sha256: sha(frame) });
			stepFirst = count;
			stepChunks = chunks.length;
		},
	});
	const pcm = Buffer.concat(chunks),
		status = emu.oz9600_audio_status();
	assert.equal(count, expected.audio.samples);
	assert.equal(sha(pcm), expected.audio.pcm_sha256);
	for (const [key, value] of Object.entries(status)) assert.deepEqual(value, expected.audio[key], key);
	assert.deepEqual(
		Buffer.from(emu.export_oz9600_retained()),
		await readFile(resolve(base, name, 'retained-state.bin')),
	);
	const saved = emu.oz9600_audio_status(),
		bad = retained.slice();
	bad[bad.length - 1] ^= 1;
	assert.throws(() => emu.restore_oz9600_retained(bad));
	assert.deepEqual(emu.oz9600_audio_status(), saved);
	emu.reset();
	assert.equal(emu.oz9600_audio_status().enabled, true);
	assert.equal(emu.take_oz9600_audio().total_samples, 0);
	await writeFile(resolve(out, `${name}.pcm`), pcm);
	const result = {
		name,
		steps,
		audio: status,
		pcm_sha256: sha(pcm),
		retained_sha256: sha(emu.export_oz9600_retained()),
	};
	await writeFile(resolve(out, `${name}.json`), JSON.stringify(result, null, 2) + '\n');
	results.push(result);
	console.log(JSON.stringify({ name, steps: steps.length, samples: count, pcm_sha256: sha(pcm) }));
	emu.free();
}
await writeFile(
	resolve(out, 'summary.json'),
	JSON.stringify(
		{
			schema: 'oz9600-audio-wasm-parity-1',
			wasm_sha256: sha(wasmBytes),
			cases: results,
			physical_audio_qualified: false,
			hardware_goal_complete: false,
		},
		null,
		2,
	) + '\n',
);
