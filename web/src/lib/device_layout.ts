import { physicalKey, virtualKeysForModel } from './keymap';
import type { InputContact } from './emulator/host_inputs';
import type { RomModel } from './rom_model';

// Reference-photo proportions, NOT measured dimensions or scan coordinates.
// Stable IDs join case geometry to input contacts; future scan meshes can use
// these same IDs without owning or changing the emulated keyboard protocol.
export type DeviceKey = {
	id: string;
	label: string;
	face: string;
	legend?: string;
	code: InputContact | null;
	testId: string;
	x: number;
	y: number;
	w: number;
	h: number;
	tone: 'normal' | 'mode' | 'shift' | 'clear';
};
export type DeviceLayout = {
	version: 1;
	model: RomModel;
	width: number;
	height: number;
	lcd: { x: number; y: number; w: number; h: number };
	keys: DeviceKey[];
	provenance: 'photo-reference-approximation';
};

export function deviceLayout(model: RomModel): DeviceLayout {
	const keys: DeviceKey[] = [];
	const existing = virtualKeysForModel(model);
	const add = (
		id: string,
		x: number,
		y: number,
		w: number,
		h: number,
		face = id,
		legend = '',
		tone: DeviceKey['tone'] = 'normal',
	) => {
		const code = id === 'ON' ? 'on' : physicalKey(model, id);
		const known = existing.find((key) => key.code === code);
		keys.push({
			id,
			label: known?.label ?? id,
			face,
			legend,
			code,
			testId:
				known?.testId ??
				(code !== null
					? `vk-${id.toLowerCase()}`
					: `vk-unmapped-${Array.from(id)
							.map((c) => c.codePointAt(0)!.toString(16))
							.join('-')}`),
			x,
			y,
			w,
			h,
			tone,
		});
	};
	if (model === 'iq-7000') {
		const modeRows = [
			['CALENDAR', 'SCHEDULE', 'TEL', 'MEMO', 'CALC'],
			['CARD', 'WORLD', 'HOME', 'OFF', 'ON'],
		];
		modeRows.forEach((row, r) => row.forEach((id, c) => add(id, 492 + c * 72, 47 + r * 51, 62, 32, id, '', 'mode')));
		add('LEFT', 492, 160, 39, 66, '◀', '♪');
		add('UP', 536, 155, 66, 32, '▲', 'MARK *');
		add('DOWN', 536, 197, 66, 32, '▼', 'SECRET');
		add('RIGHT', 607, 160, 39, 66, '▶', 'ALARM');
		['INS', 'DEL', 'SHIFT'].forEach((id, c) =>
			add(id, 668 + c * 59, 150, 51, 31, id, '', id === 'SHIFT' ? 'shift' : 'normal'),
		);
		add('SEARCH_UP', 668, 199, 79, 31, '⌃', 'SEARCH');
		add('SEARCH_DOWN', 755, 199, 79, 31, '⌄', 'SEARCH');
		const legends = ['EDIT', 'FUNCTION', 'OPTION', 'CALC DATA', 'ANN', '4↔8LINES', '¨', '^', '`', '´', ',', ''];
		'ABCDEFGHIJKLMNOPQRSTUVWX'
			.split('')
			.forEach((id, i) => add(id, 492 + (i % 6) * 59, 270 + Math.floor(i / 6) * 43, 48, 28, id, legends[i] ?? ''));
		['Y', 'Z', 'SPACE', 'RETURN'].forEach((id, c) =>
			add(id, 492 + c * 59, 442, 48, 28, id === 'SPACE' ? 'SPC' : id === 'RETURN' ? '↵' : id),
		);
		add('ENTER', 728, 442, 107, 28);
		['CAPS', 'SMBL', 'USER DIC', 'BS', 'CLEAR'].forEach((id, c) =>
			add(id, 492 + c * 71, 499, 58, 31, id === 'CLEAR' ? 'C·CE' : id, '', id === 'CLEAR' ? 'clear' : 'normal'),
		);
		const numeric = [
			['7', '8', '9', '/', 'R·CM'],
			['4', '5', '6', '*', 'M−'],
			['1', '2', '3', '-', 'M+'],
			['0', '.', '%', '+', '='],
		];
		numeric.forEach((row, r) =>
			row.forEach((id, c) =>
				add(
					id,
					492 + c * 71,
					552 + r * 46,
					58,
					30,
					id === '/' ? '÷' : id === '*' ? '×' : id,
					r === 3 ? ['', ':', 'AM', 'PM', 'ALARM'][c] : '',
				),
			),
		);
		return {
			version: 1,
			model,
			width: 900,
			height: 750,
			lcd: { x: 52, y: 101, w: 349, h: 196 },
			keys,
			provenance: 'photo-reference-approximation',
		};
	}
	const scientific = [
		['hyp', 'sin', 'cos', 'tan', 'FSE', 'CLEAR'],
		['→HEX', '→DEG', 'ln', 'log', '1/x', 'TITLE'],
		['EXP', 'yˣ', '√', 'x²', '(', ')'],
	];
	const scientificLegends = [
		['archyp', 'sin⁻¹', 'cos⁻¹', 'tan⁻¹', 'TAB', 'CA'],
		['→DEC', '→D·MS', 'eˣ', '10ˣ', '→r·θ', ''],
		['π', 'x√y', '³√', '%', '→x·y', 'n!'],
	];
	scientific.forEach((row, r) =>
		row.forEach((id, c) =>
			add(
				id,
				670 + c * 50,
				101 + r * 47,
				40,
				27,
				id === 'CLEAR' ? 'C·CE' : id,
				scientificLegends[r][c],
				id === 'CLEAR' ? 'clear' : 'normal',
			),
		),
	);
	['PF1', 'PF2', 'PF3', 'PF4', 'PF5'].forEach((id, c) =>
		add(id, 91 + c * 101, 256, 86, 25, id, ['BASIC', 'CAL', 'MATRIX', 'STAT', 'ENG'][c], 'mode'),
	);
	add('2ndF', 595, 256, 40, 25, '2ndF', '', 'shift');
	add('PF SCROLL', 30, 256, 41, 25, '↕');
	add('MENU', 30, 298, 41, 27, 'MENU', 'CAL', 'mode');
	add('BASIC', 30, 343, 41, 27, 'BASIC', 'AER', 'mode');
	add('OFF', 30, 389, 41, 27);
	add('ON', 82, 389, 41, 27, 'ON', 'BREAK');
	'QWERTYUIOP'
		.split('')
		.forEach((id, i) => add(id, 91 + i * 50, 298, 40, 27, id, ['!', '"', '#', '$', '%', '&', "'", '<', '>', '@'][i]));
	'ASDFGHJKL'
		.split('')
		.forEach((id, i) => add(id, 104 + i * 50, 343, 40, 27, id, ['[', ']', '{', '}', '\\', '|', '~', '_', '^'][i]));
	'ZXCVBNM,;'
		.split('')
		.forEach((id, i) => add(id, 132 + i * 50, 389, 40, 27, id, id === ',' ? '?' : id === ';' ? ':' : ''));
	add('STO', 595, 298, 40, 27);
	add('RCL', 595, 343, 40, 27);
	add('ENTER', 595, 389, 40, 70, '↵', 'P↔NP');
	add('SHIFT', 30, 438, 49, 25, 'SHIFT', '', 'shift');
	add('CTRL', 113, 438, 40, 25);
	add('CAPS', 162, 438, 40, 25);
	add('ANS', 211, 438, 40, 25);
	add('SPACE', 260, 438, 88, 25);
	['DOWN', 'UP', 'LEFT', 'RIGHT'].forEach((id, i) => add(id, 362 + i * 53, 438, 42, 25, ['↓', '↑', '◀', '▶'][i]));
	const numeric = [
		['7', '8', '9', '/', 'DEL'],
		['4', '5', '6', '*', 'BS'],
		['1', '2', '3', '-', 'INS'],
		['0', '+/−', '.', '+', '='],
	];
	numeric.forEach((row, r) =>
		row.forEach((id, c) => add(id, 673 + c * 59, 275 + r * 53, 47, 34, id === '/' ? '÷' : id === '*' ? '×' : id)),
	);
	return {
		version: 1,
		model,
		width: 1000,
		height: 500,
		lcd: { x: 64, y: 101, w: 547, h: 111 },
		keys,
		provenance: 'photo-reference-approximation',
	};
}

export function rectStyle(
	rect: { x: number; y: number; w: number; h: number },
	layout: { width: number; height: number },
) {
	return `left:${(rect.x / layout.width) * 100}%;top:${(rect.y / layout.height) * 100}%;width:${(rect.w / layout.width) * 100}%;height:${(rect.h / layout.height) * 100}%;`;
}
