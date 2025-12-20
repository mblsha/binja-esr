import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import type { RequestHandler } from './$types';

type Candidate = {
	path: string;
	source: string;
};

type RomModel = 'iq-7000' | 'pc-e500';

const DEFAULT_MODEL: RomModel = 'iq-7000';

function normalizeModel(raw: string | null): RomModel | null {
	const trimmed = raw?.trim().toLowerCase();
	if (!trimmed) return null;
	if (trimmed === 'iq-7000' || trimmed === 'iq7000' || trimmed === 'iq_7000') return 'iq-7000';
	if (trimmed === 'pc-e500' || trimmed === 'pce500' || trimmed === 'pc_e500') return 'pc-e500';
	return null;
}

function reportRelativePath(model: RomModel): string {
	switch (model) {
		case 'iq-7000':
			return 'rom-analysis/iq-7000/bnida.json';
		case 'pc-e500':
			return 'rom-analysis/pc-e500/s3-en/bnida.json';
	}
}

function stripLeadingLineComments(raw: string): string {
	const lines = raw.split(/\r?\n/);
	let start = 0;
	while (start < lines.length && lines[start]?.trimStart().startsWith('//')) start += 1;
	return lines.slice(start).join('\n');
}

function walkParents(start: string, maxDepth = 6): string[] {
	const out: string[] = [];
	let current = start;
	for (let i = 0; i < maxDepth; i++) {
		out.push(current);
		const next = dirname(current);
		if (next === current) break;
		current = next;
	}
	return out;
}

function symbolCandidates(model: RomModel): Candidate[] {
	const env =
		model === 'pc-e500'
			? process.env.PCE500_BNIDA_ADDRESS_REPORT_PATH
			: process.env.IQ7000_BNIDA_ADDRESS_REPORT_PATH;
	const candidates: Candidate[] = [];
	if (env) {
		candidates.push({
			path: env,
			source: model === 'pc-e500' ? 'env:PCE500_BNIDA_ADDRESS_REPORT_PATH' : 'env:IQ7000_BNIDA_ADDRESS_REPORT_PATH',
		});
	}

	const reportPath = reportRelativePath(model);
	for (const root of walkParents(process.cwd())) {
		candidates.push({
			path: resolve(root, reportPath),
			source: `${root}/${reportPath}`,
		});
		candidates.push({
			path: resolve(root, 'binja-esr-tests', reportPath),
			source: `${root}/binja-esr-tests/${reportPath}`,
		});
	}
	return candidates;
}

function parseAddress(key: string): number | null {
	const trimmed = key.trim();
	if (!trimmed) return null;
	const value = Number.parseInt(trimmed, 10);
	if (!Number.isFinite(value)) return null;
	return value >>> 0;
}

export const GET: RequestHandler = async ({ url }) => {
	const model = normalizeModel(url.searchParams.get('model')) ?? DEFAULT_MODEL;

	for (const candidate of symbolCandidates(model)) {
		try {
			const raw = await readFile(candidate.path, 'utf8');
			const jsonText = stripLeadingLineComments(raw);
			const bnida = JSON.parse(jsonText) as { names?: Record<string, string> };
			const symbols = Object.entries(bnida.names ?? {})
				.map(([addr, name]) => ({ addr: parseAddress(addr), name: String(name ?? '').trim() }))
				.filter((entry) => typeof entry.addr === 'number' && entry.addr !== null && entry.name.length > 0)
				.map((entry) => ({ addr: (entry.addr as number) & 0x000f_ffff, name: entry.name }));
			return new Response(JSON.stringify({ source: candidate.source, symbols }), {
				status: 200,
				headers: {
					'content-type': 'application/json; charset=utf-8',
					'cache-control': 'no-store',
					'x-rom-model': model,
				},
			});
		} catch {
			// Continue probing.
		}
	}

	return new Response(`Symbol report not found for model '${model}'`, {
		status: 404,
		headers: {
			'content-type': 'text/plain; charset=utf-8',
			'cache-control': 'no-store',
		},
	});
};
