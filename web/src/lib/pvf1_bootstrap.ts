export type PvHandwritingAsset = { bytes: Uint8Array<ArrayBuffer>; name: string };

/** Optional server configuration; Rust owns model validation at ROM installation. */
export async function configuredPvHandwriting(fetcher: typeof fetch): Promise<PvHandwritingAsset | null> {
	const response = await fetcher('/api/pvf1-handwriting');
	if (response.status === 404) return null;
	if (!response.ok) throw new Error(`Configured handwriting model fetch failed (${response.status})`);
	const bytes = new Uint8Array(await response.arrayBuffer());
	if (!bytes.length) throw new Error('Configured handwriting model is empty');
	const encoded = response.headers.get('x-model-name');
	return { bytes, name: encoded ? decodeURIComponent(encoded) : 'configured handwriting model' };
}
