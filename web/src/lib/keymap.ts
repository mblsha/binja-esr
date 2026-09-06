import type { RomModel } from './rom_model';
import type { InputContact } from './emulator/host_inputs';

export type VirtualKey = { label: string; code: InputContact; testId: string };
const key = (label: string, code: InputContact, id = label.toLowerCase()): VirtualKey => ({
	label,
	code,
	testId: `vk-${id}`,
});

// Physical contacts, not translated FIFO events. PC-E500 KEY_NAMES has LEFT
// at col 3,row 7 (0x1f); 0x27 is ENTER, not LEFT.
export const KEY_TO_MATRIX_CODE: Record<string, InputContact> = {
	F1: 0x56,
	F2: 0x55,
	F3: 0x54,
	F4: 0x53,
	F5: 0x52,
	ArrowUp: 0x1e,
	ArrowDown: 0x17,
	ArrowLeft: 0x1f,
	ArrowRight: 0x26,
	ShiftLeft: 0x06,
	ShiftRight: 0x06,
	CapsLock: 0x0e,
	Enter: 0x27,
	Backspace: 0x4d,
	Delete: 0x4c,
	Insert: 0x4e,
	Space: 0x16,
	F12: 'on',
};

// IQ-7000 raw contacts from ROM-driven app/scanner workflows. SHIFT 02 -> ROM
// keycode 01, CAPS 24 -> 09. Physical 09 selects HOME, NOT CAPS.
// This is an initial control subset; full physical character mapping is pending.
const IQ_KEYS = [
	key('CALENDAR', 0x18),
	key('SCHEDULE', 0x19),
	key('TEL', 0x10),
	key('MEMO', 0x08),
	key('CALC', 0x00),
	key('CARD', 0x1a),
	key('WORLD', 0x11),
	key('HOME', 0x09),
	key('SHIFT', 0x02),
	key('CAPS', 0x24),
	key('Search ↑', 0x03, 'search-up'),
	key('Search ↓', 0x0b, 'search-down'),
	key('Return', 0x3d),
	key('ENTER', 0x45),
	key('ON', 'on'),
];
const IQ_HOST_KEYS: Record<string, InputContact> = {
	F1: 0x18,
	F2: 0x19,
	F3: 0x10,
	F4: 0x08,
	F5: 0x00,
	F6: 0x1a,
	F7: 0x11,
	F8: 0x09,
	F12: 'on',
	ShiftLeft: 0x02,
	ShiftRight: 0x02,
	CapsLock: 0x24,
	PageUp: 0x03,
	PageDown: 0x0b,
	Enter: 0x45,
};

export function virtualKeysForModel(model: RomModel): VirtualKey[] {
	return model === 'iq-7000'
		? IQ_KEYS
		: [
				key('PF1', 0x56),
				key('PF2', 0x55),
				key('PF3', 0x54),
				key('PF4', 0x53),
				key('PF5', 0x52),
				key('↑', 0x1e, 'up'),
				key('←', 0x1f, 'left'),
				key('→', 0x26, 'right'),
				key('↓', 0x17, 'down'),
				key('SHIFT', 0x06),
				key('CAPS', 0x0e),
				key('ENTER', 0x27),
				key('ON', 'on'),
			];
}

export function matrixCodeForKeyEvent(event: KeyboardEvent, model: RomModel = 'pc-e500'): InputContact | null {
	return (model === 'iq-7000' ? IQ_HOST_KEYS : KEY_TO_MATRIX_CODE)[event.code] ?? null;
}
