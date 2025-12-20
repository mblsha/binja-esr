export type RomModel = 'iq-7000' | 'pc-e500';

export const DEFAULT_ROM_MODEL: RomModel = 'iq-7000';

export function normalizeRomModel(raw: string | null | undefined): RomModel | null {
	const trimmed = raw?.trim().toLowerCase();
	if (!trimmed) return null;
	if (trimmed === 'iq-7000' || trimmed === 'iq7000' || trimmed === 'iq_7000') return 'iq-7000';
	if (trimmed === 'pc-e500' || trimmed === 'pce500' || trimmed === 'pc_e500') return 'pc-e500';
	return null;
}

export function romBasename(model: RomModel): string {
	switch (model) {
		case 'iq-7000':
			return 'iq-7000.bin';
		case 'pc-e500':
			return 'pc-e500.bin';
	}
}

