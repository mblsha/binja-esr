export type LcdTabletContact = { pixel_x: number; pixel_y: number; pressed: boolean };
// Acquired FE:D005F/D0000: two columns of pen regions beyond the main LCD.
// These interior raw samples traverse the normal ROM input route. The layout
// below is functional; physical printed legends/geometry remain unmeasured.
export const PV_APPLICATIONS = [
	{ label: 'Calendar', raw: [900, 120] },
	{ label: 'Schedule', raw: [980, 120] },
	{ label: 'Telephone', raw: [900, 250] },
	{ label: 'Business cards', raw: [980, 250] },
	{ label: 'Action List', raw: [900, 380] },
	{ label: 'Memo', raw: [980, 380] },
	{ label: 'World Clock', raw: [900, 510] },
	{ label: 'Drawing', raw: [980, 510] },
	{ label: 'Calculator', raw: [900, 640] },
	{ label: 'Optical communication', raw: [980, 640] },
	{ label: 'Add-in Software', raw: [900, 770] },
	{ label: 'Action Planner', raw: [900, 900] },
	{ label: 'Search', raw: [980, 900] },
] as const;
// Displayed controller storage is336x292; the firmware main viewport ends239.
// Conversion to ADC belongs to the worker's live native calibration, not UI.
export function pvLcdPixels(x: number, y: number): [number, number] | null {
	if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x >= 1 || y < 0 || y >= 240 / 292) return null;
	return [Math.floor(x * 336), Math.floor(y * 292)];
}
