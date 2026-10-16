/** Encode only exact WHATWG Shift_JIS/Windows-31J round trips. Never substitute '?'. */
let reverse: Map<string, number[]> | undefined;
function table(): Map<string, number[]> {
	if (reverse) return reverse;
	const result = new Map<string, number[]>();
	let decoder = new TextDecoder('shift_jis', { fatal: true });
	const add = (bytes: number[]) => {
		try {
			const text = decoder.decode(Uint8Array.from(bytes));
			if ([...text].length === 1 && !result.has(text)) result.set(text, bytes);
		} catch {
			// WebKit can retain the failed pair after a fatal decode. A new decoder
			// keeps the next valid pair from inheriting those bytes.
			decoder = new TextDecoder('shift_jis', { fatal: true });
		}
	};
	for (let b = 0; b <= 255; b++) add([b]);
	for (const [first, last] of [
		[0x81, 0x9f],
		[0xe0, 0xfc],
	])
		for (let lead = first; lead <= last; lead++)
			for (let trail = 0x40; trail <= 0xfc; trail++) if (trail !== 0x7f) add([lead, trail]);
	reverse = result;
	return result;
}
export function encodeCp932(text: string): Uint8Array {
	const map = table(),
		bytes: number[] = [],
		missing = new Set<string>();
	for (const char of text) {
		const encoded = map.get(char);
		if (encoded) bytes.push(...encoded);
		else missing.add(char);
	}
	if (missing.size)
		throw new Error(
			`Unmappable CP932 characters: ${[...missing].map((c) => `${c} (U+${c.codePointAt(0)!.toString(16).toUpperCase()})`).join(', ')}`,
		);
	return Uint8Array.from(bytes);
}
