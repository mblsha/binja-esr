// Panel presentation candidate: IOCS uses controller X=0..167/Y=0..238.
// The native controller and its hashes remain unchanged.
export function piPanelCapture<T extends { cols: number; rows: number; pixel_scale: number; pixels: Uint8Array }>(
	capture: T,
): T {
	const scale = capture.pixel_scale;
	if (
		!Number.isInteger(scale) ||
		scale < 1 ||
		capture.cols !== 240 * scale ||
		capture.rows !== 336 * scale ||
		capture.pixels.length !== capture.cols * capture.rows
	)
		throw new Error('Unexpected PI controller frame geometry');
	const cols = 239 * scale,
		rows = 168 * scale;
	const pixels = new Uint8Array(cols * rows);
	for (let y = 0; y < rows; y++)
		for (let x = 0; x < cols; x++) pixels[y * cols + x] = capture.pixels[x * capture.cols + y];
	return { ...capture, cols, rows, pixels };
}

// Virtual digitizer calibration, not a measured physical ADC transfer function.
// These four raw endpoints pass the ROM's ten-sample calibration consumer.
export function piLcdTablet(x: number, y: number): [number, number] {
	const px = Math.max(0, Math.min(238, Math.floor(x * 239)));
	const py = Math.max(0, Math.min(167, Math.floor(y * 168)));
	const adc = (value: number) => Math.max(0, Math.min(1023, Math.round(value)));
	return [adc(340 + ((px - 32) * 360) / 173), adc(240 + ((py - 32) * 460) / 103)];
}
