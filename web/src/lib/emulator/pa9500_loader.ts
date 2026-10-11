export type PaProfile = 'bootstrap-v1' | 'bootstrap-v2' | 'bootstrap-v3';
export type PaInputMode = 'native' | 'basic';

export type PaEditorAction = 'delete' | 'insert';

/** ROM F76DA: modifier plus LEFT emits7F, plus RIGHT emits0100. */
export function paEditorSequence(action: PaEditorAction): number[] {
	if (action === 'delete') return [23, 8];
	if (action === 'insert') return [23, 14];
	throw new Error('Unknown PA editor action');
}
/** Use the same fresh constructor and normal ON release as the ROM replays. */
export function loadPa9500(
	emulator: any,
	bundle: Uint8Array,
	profile: PaProfile,
	cardRom: Uint8Array = new Uint8Array(),
	cardSram: Uint8Array = new Uint8Array(),
	uartCandidate = false,
	retained: Uint8Array = new Uint8Array(),
) {
	if (retained.length) emulator.load_pa9500_with_retained(bundle, cardRom, cardSram, profile, retained, uartCandidate);
	else if (uartCandidate) emulator.load_pa9500_with_uart(bundle, cardRom, cardSram, profile);
	else emulator.load_pa9500(bundle, cardRom, cardSram, profile);
	emulator.step(64);
	emulator.release_on_key();
}

export function resetPa9500(emulator: any) {
	emulator.reset();
	emulator.step(64);
	emulator.release_on_key();
}
