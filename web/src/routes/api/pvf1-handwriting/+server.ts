import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	const path = process.env.PVF1_HANDWRITING_MODEL_PATH;
	if (!path) return new Response('No configured PV-F1 handwriting model', { status: 404 });
	try {
		const bytes = await readFile(path);
		if (!bytes.length) return new Response('Configured handwriting model is empty', { status: 422 });
		return new Response(bytes, {
			headers: {
				'content-type': 'application/octet-stream',
				'cache-control': 'no-store',
				'x-model-name': encodeURIComponent(basename(path)),
			},
		});
	} catch {
		return new Response('Configured handwriting model is unavailable', { status: 500 });
	}
};
