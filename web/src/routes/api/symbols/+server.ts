import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import type { RequestHandler } from './$types';

type Candidate = {
	path: string;
	source: string;
};

const REPORT_RELATIVE_PATH = 'rom-analysis/pc-e500/s3-en/bnida_address_report.json';

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

function symbolCandidates(): Candidate[] {
	const env = process.env.PCE500_BNIDA_ADDRESS_REPORT_PATH;
	const candidates: Candidate[] = [];
	if (env) {
		candidates.push({ path: env, source: 'env:PCE500_BNIDA_ADDRESS_REPORT_PATH' });
	}

	for (const root of walkParents(process.cwd())) {
		candidates.push({
			path: resolve(root, REPORT_RELATIVE_PATH),
			source: `${root}/${REPORT_RELATIVE_PATH}`,
		});
		candidates.push({
			path: resolve(root, 'binja-esr-tests', REPORT_RELATIVE_PATH),
			source: `${root}/binja-esr-tests/${REPORT_RELATIVE_PATH}`,
		});
	}
	return candidates;
}

function parseAddress(key: string): number | null {
	const trimmed = key.trim();
	if (!trimmed) return null;
	const hex = trimmed.toLowerCase().startsWith('0x') ? trimmed.slice(2) : trimmed;
	const value = Number.parseInt(hex, 16);
	if (!Number.isFinite(value)) return null;
	return value >>> 0;
}

export const GET: RequestHandler = async () => {
	for (const candidate of symbolCandidates()) {
		try {
			const raw = await readFile(candidate.path, 'utf8');
			const jsonText = stripLeadingLineComments(raw);
			const report = JSON.parse(jsonText) as Record<string, string>;
			const symbols = Object.entries(report)
				.map(([addr, name]) => ({ addr: parseAddress(addr), name: String(name ?? '').trim() }))
				.filter((entry) => typeof entry.addr === 'number' && entry.addr !== null && entry.name.length > 0)
				.map((entry) => ({ addr: (entry.addr as number) & 0x000f_ffff, name: entry.name }));
			return new Response(JSON.stringify({ source: candidate.source, symbols }), {
				status: 200,
				headers: {
					'content-type': 'application/json; charset=utf-8',
					'cache-control': 'no-store',
				},
			});
		} catch {
			// Continue probing.
		}
	}

	return new Response('Symbol report not found', {
		status: 404,
		headers: {
			'content-type': 'text/plain; charset=utf-8',
			'cache-control': 'no-store',
		},
	});
};
