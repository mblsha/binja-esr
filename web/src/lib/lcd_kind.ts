export type LcdKind = 'hd61202' | 'iq7000-vram' | 'unknown';

export function normalizeLcdKind(raw: unknown): LcdKind | null {
	if (typeof raw !== 'string') return null;
	const trimmed = raw.trim().toLowerCase();
	if (!trimmed) return null;
	if (trimmed === 'hd61202') return 'hd61202';
	if (trimmed === 'iq7000-vram') return 'iq7000-vram';
	if (trimmed === 'unknown') return 'unknown';
	return null;
}
