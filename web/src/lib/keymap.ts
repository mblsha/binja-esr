import type { RomModel } from './rom_model';
import type { InputContact } from './emulator/host_inputs';
import physicalMaps from '../../../sc62015/core/data/physical_keys.json';

export type VirtualKey = { label: string; code: InputContact; testId: string };
const maps: Record<string, Record<string, number>> = physicalMaps;
export function physicalKey(model: RomModel, name: string): number | null {
	return maps[model === 'iq-7000' ? 'iq-7000' : 'pc-e500'][name] ?? null;
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
function hostName(code: string, model: RomModel): string | null {
	if (/^Key[A-Z]$/.test(code)) return code.slice(3);
	if (/^(Digit|Numpad)[0-9]$/.test(code)) return code.slice(-1);
	return (model === 'iq-7000' ? iqHost : pcHost)[code] ?? commonHost[code] ?? null;
}

export function matrixCodeForKeyEvent(event: KeyboardEvent, model: RomModel = 'pc-e500'): InputContact | null {
	if (event.code === 'F12') return 'on';
	const name = hostName(event.code, model);
	return name ? physicalKey(model, name) : null;
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
			: ['PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'BASIC', 'MENU'];
	for (const name of modeKeys) add(name);
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
