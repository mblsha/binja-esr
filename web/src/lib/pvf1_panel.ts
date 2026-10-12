export type LcdTabletContact = { pixel_x: number; pixel_y: number; pressed: boolean };
// Displayed controller storage is336x292; the firmware main viewport ends239.
// Conversion to ADC belongs to the worker's live native calibration, not UI.
export function pvLcdPixels(x: number, y: number): [number, number] | null {
	if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x >= 1 || y < 0 || y >= 240 / 292) return null;
	return [Math.floor(x * 336), Math.floor(y * 292)];
}
