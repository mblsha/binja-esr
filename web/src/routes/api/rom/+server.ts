import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { DEFAULT_ROM_MODEL, normalizeRomModel, romBasename, type RomModel } from '$lib/rom_model';
import type { RequestHandler } from './$types';

type Candidate = {
	path: string;
	source: string;
};

function romCandidates(model: RomModel): Candidate[] {
	const env =
		model === 'pc-e500'
			? process.env.PCE500_ROM_PATH
			: process.env.IQ7000_ROM_PATH ?? process.env.IQ_7000_ROM_PATH;
	const candidates: Candidate[] = [];
	if (env) {
		candidates.push({
			path: env,
			source: model === 'pc-e500' ? 'env:PCE500_ROM_PATH' : 'env:IQ7000_ROM_PATH',
		});
	}

	const basename = romBasename(model);
	// When running `web/` inside the repo, the ROM symlink is usually at `../data/<rom>.bin`.
	candidates.push({ path: resolve(process.cwd(), `../data/${basename}`), source: `../data/${basename}` });
	candidates.push({ path: resolve(process.cwd(), `../../data/${basename}`), source: `../../data/${basename}` });
	candidates.push({ path: resolve(process.cwd(), `data/${basename}`), source: `data/${basename}` });
	return candidates;
}

export const GET: RequestHandler = async ({ url }) => {
	const model = normalizeRomModel(url.searchParams.get('model')) ?? DEFAULT_ROM_MODEL;

	for (const candidate of romCandidates(model)) {
		try {
			const bytes = await readFile(candidate.path);
			if (bytes.length === 0) continue;
			return new Response(bytes, {
				status: 200,
				headers: {
					'content-type': 'application/octet-stream',
					'cache-control': 'no-store',
					'x-rom-source': candidate.source,
					'x-rom-model': model,
				},
			});
		} catch {
			// Continue probing.
		}
	}

	return new Response(`ROM not found for model '${model}'`, {
		status: 404,
		headers: {
			'content-type': 'text/plain; charset=utf-8',
			'cache-control': 'no-store',
		},
	});
};
