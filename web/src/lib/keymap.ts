import type { RomModel } from './rom_model';
import type { InputContact } from './emulator/host_inputs';
import physicalMaps from '../../../sc62015/core/data/physical_keys.json';

export type VirtualKey = { label: string; code: InputContact; testId: string };
export type HostKeyboardMode = 'symbols' | 'keycaps';
const maps: Record<string, Record<string, number>> = physicalMaps;
export function physicalKey(model: RomModel, name: string): number | null {
	return maps[model][name] ?? null;
}

// Keycap mapping, not a desktop text composer: device CAPS controls case and
// host SHIFT operates the device's printed shifted legends. No translated FIFO
// events. Native Rust consumes this same checked-in physical contact table.
const commonHost: Record<string, string> = {
	ArrowUp: 'UP',
	ArrowDown: 'DOWN',
	ArrowLeft: 'LEFT',
	ArrowRight: 'RIGHT',
	ShiftLeft: 'SHIFT',
	ShiftRight: 'SHIFT',
	CapsLock: 'CAPS',
	Enter: 'ENTER',
	Backspace: 'BS',
	Delete: 'DEL',
	Insert: 'INS',
	Space: 'SPACE',
	Escape: 'CLEAR',
	Period: '.',
	Slash: '/',
	Minus: '-',
	Equal: '=',
	Comma: ',',
	Semicolon: ';',
	NumpadAdd: '+',
	NumpadSubtract: '-',
	NumpadMultiply: '*',
	NumpadDivide: '/',
	NumpadDecimal: '.',
	NumpadEnter: '=',
};
const pcHost: Record<string, string> = {
	F1: 'PF1',
	F2: 'PF2',
	F3: 'PF3',
	F4: 'PF4',
	F5: 'PF5',
	F6: 'BASIC',
	F7: 'MENU',
	F8: 'CLEAR',
	F9: 'SHIFT',
	F10: 'CAPS',
	F11: 'CTRL', // Keeps host Ctrl/Cmd shortcuts available to the browser.
	ControlLeft: 'CTRL',
	ControlRight: 'CTRL',
};
const iqHost: Record<string, string> = {
	F1: 'CALENDAR',
	F2: 'SCHEDULE',
	F3: 'TEL',
	F4: 'MEMO',
	F5: 'CALC',
	F6: 'CARD',
	F7: 'WORLD',
	F8: 'HOME',
	F9: 'SHIFT',
	F10: 'CAPS',
	F11: 'RETURN',
	PageUp: 'SEARCH_UP',
	PageDown: 'SEARCH_DOWN',
};
const ozHost: Record<string, string> = {
	F1: 'NEW ENTRY',
	F2: 'EDIT',
	F9: 'SHIFT',
	F10: 'CAPS',
	F11: '2ND',
	Escape: 'CANCEL',
	PageUp: 'PREV',
	PageDown: 'NEXT',
	NumpadEnter: 'ENTER',
};
const modelHost = (model: RomModel) => (model === 'iq-7000' ? iqHost : model === 'oz-9600' ? ozHost : pcHost);
function hostName(code: string, model: RomModel): string | null {
	if (/^Key[A-Z]$/.test(code)) return code.slice(3);
	if (/^(Digit|Numpad)[0-9]$/.test(code)) return code.slice(-1);
	return modelHost(model)[code] ?? commonHost[code] ?? null;
}

export function matrixCodeForKeyEvent(event: KeyboardEvent, model: RomModel = 'pc-e500'): InputContact | null {
	if (event.code === 'F12') return 'on';
	const name = hostName(event.code, model);
	return name ? physicalKey(model, name) : null;
}

// The UI's default maps the character selected by the host layout to an actual
// device key. Host Shift selects symbols; F9 remains the device SHIFT key.
// This is not text injection: device CAPS/ROM state still determines letter case.
// Keycap mode retains positional code mapping and literal host Shift contacts.
export function contactsForKeyEvent(event: KeyboardEvent, model: RomModel, mode: HostKeyboardMode): InputContact[] {
	if (event.isComposing || event.key === 'Dead' || event.key === 'Process' || event.key === 'Unidentified') return [];
	if (mode === 'keycaps') {
		const contact = matrixCodeForKeyEvent(event, model);
		return contact === null ? [] : [contact];
	}
	if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') return [];
	if (event.code === 'ControlLeft' || event.code === 'ControlRight') return [];
	if (event.code === 'Enter' && event.shiftKey && model === 'iq-7000') return [physicalKey(model, 'RETURN')!];
	// Num Lock off must retain the host's navigation semantics, not type digits.
	if (
		event.code.startsWith('Numpad') &&
		[
			'Home',
			'End',
			'PageUp',
			'PageDown',
			'Insert',
			'Delete',
			...['Up', 'Down', 'Left', 'Right'].map((d) => 'Arrow' + d),
		].includes(event.key)
	) {
		const contact = matrixCodeForKeyEvent({ code: event.key } as KeyboardEvent, model);
		return contact === null ? [] : [contact];
	}
	if (event.key?.length === 1) {
		const name = event.key === ' ' ? 'SPACE' : event.key.toUpperCase();
		const contact = physicalKey(model, name);
		if (contact !== null) return [contact];
		// IQ comma needs the ROM to observe SHIFT before K. Simultaneous raw
		// contacts typed K in the live browser test; use F9, release, then K.
		return []; // Never turn unsupported punctuation into the unshifted key.
	}
	const contact = matrixCodeForKeyEvent(event, model);
	return contact === null ? [] : [contact];
}

export function hostKeyHints(contact: InputContact, model: RomModel, mode: HostKeyboardMode): string {
	const hosts = { ...commonHost, ...modelHost(model) };
	const names = Object.entries(hosts)
		.filter(
			([host, name]) =>
				!host.startsWith('Control') &&
				(!host.startsWith('Shift') || mode === 'keycaps') &&
				(mode === 'keycaps' || !['Period', 'Slash', 'Minus', 'Equal', 'Comma', 'Semicolon'].includes(host)) &&
				physicalKey(model, name) === contact,
		)
		.map(([host]) => host.replace(/^Shift(Left|Right)$/, 'Shift').replace('Numpad', 'Keypad '));
	if (contact === 'on') names.push('F12');
	for (const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') if (physicalKey(model, c) === contact) names.push(c);
	if (mode === 'symbols') {
		for (const c of '%+-*/=.,;()') if (physicalKey(model, c) === contact) names.push(c);
		if (model === 'iq-7000' && contact === physicalKey(model, 'RETURN')) names.push('Shift+Enter');
	}
	return [...new Set(names)].join(' / ');
}

// Compatibility export for existing callers, derived from the shared table.
export const KEY_TO_MATRIX_CODE: Record<string, InputContact> = Object.fromEntries(
	[
		...Object.keys(commonHost),
		...Object.keys(pcHost),
		...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map((c) => 'Key' + c),
		...'0123456789'.split('').map((c) => 'Digit' + c),
		'F12',
	]
		.map((code) => [code, matrixCodeForKeyEvent({ code } as KeyboardEvent)])
		.filter((entry): entry is [string, InputContact] => entry[1] !== null),
);

export function virtualKeysForModel(model: RomModel): VirtualKey[] {
	const keys: VirtualKey[] = [];
	const add = (name: string, label = name, id = name.toLowerCase().replaceAll('_', '-')) => {
		const code = physicalKey(model, name);
		if (code !== null) keys.push({ label, code, testId: 'vk-' + id });
	};
	const modeKeys =
		model === 'iq-7000'
			? ['CALENDAR', 'SCHEDULE', 'TEL', 'MEMO', 'CALC', 'CARD', 'WORLD', 'HOME']
			: model === 'oz-9600'
				? ['NEW ENTRY', 'EDIT', 'PREV', 'NEXT', 'CANCEL', 'MENU', '2ND', 'WORD', 'SYMBOL', 'OFF']
				: ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'BASIC', 'MENU'];
	for (const name of modeKeys) add(name);
	if (model === 'oz-9600') {
		add('M+', 'M+', 'm-plus');
		add('M-', 'M−', 'm-minus');
	}
	for (const [name, label] of [
		['UP', '↑'],
		['LEFT', '←'],
		['RIGHT', '→'],
		['DOWN', '↓'],
		['SEARCH_UP', 'Search ↑'],
		['SEARCH_DOWN', 'Search ↓'],
		['SHIFT', 'SHIFT'],
		['CAPS', 'CAPS'],
		['INS', 'INS'],
		['DEL', 'DEL'],
		['BS', 'BS'],
		['CLEAR', 'C·CE'],
		['RETURN', 'Return'],
		['ENTER', 'ENTER'],
	])
		add(name, label);
	keys.push({ label: 'ON', code: 'on', testId: 'vk-on' });
	for (const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') add(c);
	add('SPACE');
	for (const [name, id] of [
		['%', 'percent'],
		['+', 'plus'],
		['-', 'minus'],
		['*', 'multiply'],
		['/', 'divide'],
		['=', 'equals'],
		['.', 'period'],
		[',', 'comma'],
		[';', 'semicolon'],
		['(', 'lparen'],
		[')', 'rparen'],
	])
		add(name, name, id);
	return keys;
}
