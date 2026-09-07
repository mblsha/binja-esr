import { grayscaleToRgba } from './lcd';

/** Export the observed Rust capture, never a screenshot of the styled case. */
export async function lcdPng(pixels: Uint8Array, cols: number, rows: number): Promise<Blob> {
	if (!Number.isSafeInteger(cols) || !Number.isSafeInteger(rows) || cols <= 0 || rows <= 0 || cols * rows > 4_000_000)
		throw new Error('Invalid LCD export dimensions');
	const rgba = grayscaleToRgba(pixels, cols, rows);
	const canvas = document.createElement('canvas');
	canvas.width = cols;
	canvas.height = rows;
	const context = canvas.getContext('2d');
	if (!context) throw new Error('Canvas unavailable for LCD export');
	const image = context.createImageData(cols, rows);
	image.data.set(rgba);
	context.putImageData(image, 0, 0);
	return new Promise((resolve, reject) =>
		canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed'))), 'image/png'),
	);
}

export function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = filename;
	document.body.append(link);
	link.click();
	link.remove();
	// Allow asynchronous browser download handling before releasing the URL.
	setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
