export type PaProfile = 'bootstrap-v1' | 'bootstrap-v2' | 'bootstrap-v3';
export type PaInputMode = 'native' | 'basic';

/** Use the same fresh constructor and normal ON release as the ROM replays. */
export function loadPa9500(
	emulator: any,
	bundle: Uint8Array,
	profile: PaProfile,
	cardRom: Uint8Array = new Uint8Array(),
	cardSram: Uint8Array = new Uint8Array(),
) {
	emulator.load_pa9500(bundle, cardRom, cardSram, profile);
	emulator.step(64);
	emulator.release_on_key();
}

export function resetPa9500(emulator: any) {
	emulator.reset();
	emulator.step(64);
	emulator.release_on_key();
}
