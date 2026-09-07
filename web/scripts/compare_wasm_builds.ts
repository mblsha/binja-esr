/** Differential ROM smoke: physical contacts, full memory hashes, registers and LCD. */
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { deepStrictEqual } from 'node:assert';
import { physicalKey } from '../src/lib/keymap';

async function main() {
	const [baseDir, candidateDir, romPath, model, outputDir] = process.argv.slice(2);
	if (!baseDir || !candidateDir || !romPath || !outputDir || !['pc-e500', 'iq-7000'].includes(model))
		throw new Error('Usage: compare_wasm_builds.ts BASE_WASM_DIR CANDIDATE_WASM_DIR ROM pc-e500|iq-7000 OUTPUT_DIR');
	const rom = new Uint8Array(await readFile(resolve(romPath)));
	const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
	const load = async (dir: string) => {
		const api = await import(/* @vite-ignore */ pathToFileURL(resolve(dir, 'pce500_wasm.js')).href);
		const bytes = await readFile(resolve(dir, 'pce500_wasm_bg.wasm'));
		const wasm = api.initSync({ module: bytes });
		const e = new api.Sc62015Emulator();
		e.load_rom_with_model(rom, model);
		if (model === 'iq-7000') e.set_iq7000_rtc_yyyymmddhhmm('199201101050');
		return { e, wasm, wasmHash: hash(bytes) };
	};
	const base = await load(baseDir);
	const candidate = await load(candidateDir);
	const observe = ({ e, wasm }: typeof base) => ({
		registers: Object.fromEntries(['BA', 'I', 'X', 'Y', 'U', 'S', 'PC', 'F', 'IMR'].map((r) => [r, e.get_reg(r)])),
		retired: e.instruction_count().toString(),
		timing: e.cycle_count().toString(),
		power: e.power_state(),
		contacts: e.input_contacts(),
		text: e.lcd_text(),
		lcd: hash(e.lcd_capture().pixels),
		external: hash(new Uint8Array(wasm.memory.buffer, e.memory_external_ptr(), e.memory_external_len())),
		internal: hash(new Uint8Array(wasm.memory.buffer, e.memory_internal_ptr(), e.memory_internal_len())),
		rtc: model === 'iq-7000' ? e.iq7000_rtc_state() : null,
	});
	const stages: unknown[] = [];
	const compare = (stage: string, budget: number) => {
		base.e.step(budget);
		candidate.e.step(budget);
		const a = observe(base),
			b = observe(candidate);
		deepStrictEqual(b, a, `${model}: ${stage}`);
		stages.push({ stage, ...a });
	};
	try {
		compare('boot', model === 'pc-e500' ? 20_000 : 500_000);
		const keys =
			model === 'pc-e500'
				? ['PF1', '1', '2', '3', 'BS']
				: ['MEMO', 'CAPS', 'A', 'A', 'SPACE', 'SHIFT', 'K', 'ENTER', 'MEMO', 'SEARCH_DOWN'];
		for (const name of keys) {
			const code = physicalKey(model as 'pc-e500' | 'iq-7000', name);
			if (code === null) throw new Error(`No physical key ${name}`);
			base.e.press_matrix_code(code);
			candidate.e.press_matrix_code(code);
			compare(`${name} down`, name === 'PF1' ? 800_000 : 40_000);
			base.e.release_matrix_code(code);
			candidate.e.release_matrix_code(code);
			compare(`${name} up`, name === 'PF1' ? 800_000 : 40_000);
		}
		await mkdir(outputDir, { recursive: true });
		await writeFile(
			resolve(outputDir, `${model}-comparison.json`),
			JSON.stringify(
				{
					verified: true,
					model,
					romHash: hash(rom),
					baseHash: base.wasmHash,
					candidateHash: candidate.wasmHash,
					stages,
				},
				null,
				2,
			),
		);
		console.log(`${model}: ${stages.length} complete state comparisons passed`);
	} finally {
		base.e.free();
		candidate.e.free();
	}
}
main().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
