// Firmware default calibration, not measured digitizer coordinates.
export const OZ_PANEL = [
	['Calendar', 61, 141],
	['Schedule', 122, 141],
	['Menu', 182, 141],
	['To Do', 61, 288],
	['Anniversary', 122, 288],
	['Calculator', 182, 288],
	['Telephone', 61, 434],
	['User File', 122, 434],
	['Clock', 182, 434],
	['Notebook', 61, 581],
	['Outline', 122, 581],
	['Scrapbook', 182, 581],
	['Filer', 61, 727],
	['Card', 122, 727],
	['Search', 122, 874],
] as const;

export function ozLcdTablet(x: number, y: number): [number, number] {
	const px = Math.max(0, Math.min(335, Math.floor(x * 336)));
	const py = Math.max(0, Math.min(239, Math.floor(y * 240)));
	return [Math.floor(((4 * px + 419) * 548) / 1008), Math.floor(((4 * py + 75) * 630) / 688)];
}
