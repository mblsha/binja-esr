import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { RequestHandler } from './$types';

type Candidate = {
	path: string;
	source: string;
};

function romCandidates(): Candidate[] {
	const env = process.env.PCE500_ROM_PATH;
	const candidates: Candidate[] = [];
	if (env) {
		candidates.push({ path: env, source: 'env:PCE500_ROM_PATH' });
	}

	// When running `web/` inside the repo, the ROM symlink is usually at `../data/pc-e500.bin`.
	candidates.push({ path: resolve(process.cwd(), '../data/pc-e500.bin'), source: '../data/pc-e500.bin' });
	candidates.push({ path: resolve(process.cwd(), '../../data/pc-e500.bin'), source: '../../data/pc-e500.bin' });
	candidates.push({ path: resolve(process.cwd(), 'data/pc-e500.bin'), source: 'data/pc-e500.bin' });
	return candidates;
}

export const GET: RequestHandler = async () => {
	for (const candidate of romCandidates()) {
		try {
			const bytes = await readFile(candidate.path);
			if (bytes.length === 0) continue;
			return new Response(bytes, {
				status: 200,
				headers: {
					'content-type': 'application/octet-stream',
					'cache-control': 'no-store',
					'x-rom-source': candidate.source,
				},
			});
		} catch {
			// Continue probing.
		}
	}

	return new Response('ROM not found', {
		status: 404,
		headers: {
			'content-type': 'text/plain; charset=utf-8',
			'cache-control': 'no-store',
		},
	});
};
