import { physicalKey } from '../keymap';
import type { RomModel } from '../rom_model';

export const MAX_PASTE_CHARACTERS = 4096;

/** Keycap composition only. CAPS/app state determines the resulting text. */
export function planPaste(text: string, model: RomModel) {
	const contacts: number[] = [];
	const unsupported: { character: string; position: number }[] = [];
	if (text.length > MAX_PASTE_CHARACTERS)
		return { contacts, unsupported, error: `Paste is limited to ${MAX_PASTE_CHARACTERS} characters.` };
	const characters = Array.from(text.replace(/\r\n?/g, '\n'));
	characters.forEach((character, index) => {
		let names: string[] = [];
		if (character === '\n') names = [model === 'iq-7000' ? 'RETURN' : 'ENTER'];
		else if (character === ' ') names = ['SPACE'];
		else if (model === 'iq-7000' && character === ',') names = ['SHIFT', 'K'];
		else if (/^[A-Za-z0-9+\-*/=.,;()]$/.test(character)) names = [character.toUpperCase()];
		const mapped = names.map((name) => physicalKey(model, name));
		if (!mapped.length || mapped.some((contact) => contact === null))
			unsupported.push({ character, position: index + 1 });
		else contacts.push(...(mapped as number[]));
	});
	return { contacts: unsupported.length ? [] : contacts, unsupported, error: null };
}
