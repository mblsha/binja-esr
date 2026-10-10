export type RomModel = 'iq-7000' | 'pc-e500' | 'oz-9600' | 'pa-9500';

export function normalizeRomModel(raw: string | null | undefined): RomModel | null {
	const trimmed = raw?.trim().toLowerCase();
	if (!trimmed) return null;
	if (trimmed === 'iq-7000' || trimmed === 'iq7000' || trimmed === 'iq_7000') return 'iq-7000';
	if (trimmed === 'pc-e500' || trimmed === 'pce500' || trimmed === 'pc_e500') return 'pc-e500';
	if (['oz-9600', 'oz9600', 'oz_9600'].includes(trimmed)) return 'oz-9600';
	if (['pa-9500', 'pa9500', 'db-z', 'dbz'].includes(trimmed)) return 'pa-9500';
	return null;
}

export function romBasename(model: RomModel): string {
	switch (model) {
		case 'iq-7000':
			return 'iq-7000.bin';
		case 'oz-9600':
			return 'oz-9600.ozrom';
		case 'pa-9500':
			return 'pa-9500.parom';
		case 'pc-e500':
			return 'pc-e500-en.bin';
	}
}
