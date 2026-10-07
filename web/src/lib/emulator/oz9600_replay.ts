import { stepBounded, ExecutionCancelled, type SlicedEmulator } from './bounded_step';

export type OzProfile =
	| 'strict'
	| 'experimental'
	| 'experimental-isr-clear-only'
	| 'experimental-isr-mti-writable'
	| 'experimental-on-edge'
	| 'experimental-irq-imr'
	| 'experimental-rtc'
	| 'experimental-rtc-irq-imr'
	| 'provisional-v1';
export type TabletContact = { raw_x: number; raw_y: number; pressed: boolean };
type PhysicalStep = {
	boundaries: number;
	contact?: { column: number; row: number; pressed: boolean };
	tablet?: TabletContact;
	on_key?: boolean;
	label?: string;
};
type OzEmulator = SlicedEmulator & {
	validate_oz9600_replay(document: string): void;
	press_matrix_code(code: number): void;
	release_matrix_code(code: number): void;
	press_on_key(): void;
	release_on_key(): void;
	set_oz9600_tablet_contact(x: number, y: number, pressed: boolean): void;
	oz9600_state(): object;
};

/** Rust validates the whole document before any input or CPU boundary. */
export async function runOzPhysicalReplay(
	emulator: OzEmulator,
	document: string,
	options: {
		signal?: AbortSignal;
		onStep?: (index: number, state: object) => Promise<void>;
		onProgress?: (used: number) => void;
		yieldHost?: () => Promise<void>;
	} = {},
) {
	emulator.validate_oz9600_replay(document);
	const { steps } = JSON.parse(document) as { steps: PhysicalStep[] };
	const held = new Set<number>();
	let tablet: TabletContact | undefined;
	let onKey = false;
	let completed = 0;
	try {
		for (const [index, step] of steps.entries()) {
			if (options.signal?.aborted) throw new ExecutionCancelled(completed);
			if (step.on_key !== undefined && step.on_key !== null) {
				onKey = step.on_key;
				if (onKey) emulator.press_on_key();
				else emulator.release_on_key();
			}
			if (step.contact) {
				const { column, row, pressed } = step.contact;
				const code = column * 8 + row;
				if (pressed) {
					emulator.press_matrix_code(code);
					held.add(code);
				} else {
					emulator.release_matrix_code(code);
					held.delete(code);
				}
			}
			if (step.tablet) {
				tablet = step.tablet;
				emulator.set_oz9600_tablet_contact(tablet.raw_x, tablet.raw_y, tablet.pressed);
			}
			await stepBounded(emulator, step.boundaries, {
				signal: options.signal,
				yieldHost: options.yieldHost,
				onProgress: options.onProgress,
			});
			completed += step.boundaries;
			await options.onStep?.(index, emulator.oz9600_state());
		}
		return completed;
	} catch (error) {
		// Stop acknowledges the partial machine; release host-owned contacts without
		// advancing it or pretending to roll back guest RAM/peripheral effects.
		for (const code of held) emulator.release_matrix_code(code);
		if (onKey) emulator.release_on_key();
		if (tablet?.pressed) emulator.set_oz9600_tablet_contact(tablet.raw_x, tablet.raw_y, false);
		throw error;
	}
}

/** PBM bytes describe the full controller frame, independent of bezel artwork. */
export function ozPbm(capture: {
	cols: number;
	rows: number;
	pixels: Uint8Array;
	pixel_format: string;
	pixel_scale: number;
}) {
	if (
		capture.cols !== 336 ||
		capture.rows !== 240 ||
		capture.pixel_format !== 'gray8' ||
		capture.pixel_scale !== 1 ||
		capture.pixels.length !== 336 * 240
	)
		throw new Error('Expected full native OZ 336×240 gray8 capture');
	const header = new TextEncoder().encode('P4\n336 240\n');
	const bytes = new Uint8Array(header.length + 42 * 240);
	bytes.set(header);
	for (let i = 0; i < capture.pixels.length; i++)
		if (capture.pixels[i] === 0) bytes[header.length + (i >> 3)] |= 0x80 >> (i & 7);
	return bytes;
}
