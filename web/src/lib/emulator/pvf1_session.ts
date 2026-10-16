/** Browser backing, not a CPU snapshot. Restore boots the unmodified ROM fresh. */
export type PvSession = {
	retained: Uint8Array;
	cardRom: Uint8Array;
	cardSram: Uint8Array;
	ramCard: Uint8Array;
	ramCardWritable: boolean;
	handwritingModel: Uint8Array;
};
export type PvAssets = Pick<PvSession, 'cardRom' | 'ramCardWritable' | 'handwritingModel'>;
export interface PvBackingMachine {
	export_pvf1_retained(): Uint8Array;
	pvf1_card_sram(): Uint8Array;
	pvf1_ram_card_image(): Uint8Array;
}
export interface PvIdentityMachine {
	load_pvf1(
		bundle: Uint8Array,
		retained: Uint8Array,
		model: Uint8Array,
		cardRom: Uint8Array,
		cardSram: Uint8Array,
	): void;
	export_pvf1_retained(): Uint8Array;
	free(): void;
}
export interface PvSessionMachine extends Pick<PvIdentityMachine, 'load_pvf1'> {
	load_pvf1_with_ram_card(
		bundle: Uint8Array,
		retained: Uint8Array,
		model: Uint8Array,
		card: Uint8Array,
		writable: boolean,
	): void;
}

/** Native loaders validate a candidate before replacing the active machine. */
export function loadPvSession(machine: PvSessionMachine, bundle: Uint8Array, session: PvSession): PvAssets {
	validate(session);
	if (session.ramCard.length)
		machine.load_pvf1_with_ram_card(
			bundle,
			session.retained,
			session.handwritingModel,
			session.ramCard,
			session.ramCardWritable,
		);
	else machine.load_pvf1(bundle, session.retained, session.handwritingModel, session.cardRom, session.cardSram);
	return {
		cardRom: session.cardRom.slice(),
		handwritingModel: session.handwritingModel.slice(),
		ramCardWritable: session.ramCardWritable,
	};
}
const MAGIC = new TextEncoder().encode('PVWEB01\0');
const PARTS = ['retained', 'cardRom', 'cardSram', 'ramCard', 'handwritingModel'] as const;
const LIMIT = 64 * 1024 * 1024;
const PREFIX = 44;

export function pvBatteryIdentity(create: () => PvIdentityMachine, bundle: Uint8Array): string {
	const candidate = create();
	try {
		const empty = new Uint8Array();
		candidate.load_pvf1(bundle, empty, empty, empty, empty);
		return pvRetainedIdentity(candidate.export_pvf1_retained());
	} finally {
		candidate.free();
	}
}
export function pvRetainedIdentity(image: Uint8Array): string {
	const magic = new TextDecoder().decode(image.subarray(0, 8));
	if (image.length < 104 || !['PVRAM01\0', 'PVRAM02\0'].includes(magic))
		throw new Error('Invalid PV-F1 retained identity');
	// Exclude PVRAM01/02: adding learned templates must not fork a firmware's saves.
	return 'pvf1:' + Array.from(image.subarray(8, 72), (b) => b.toString(16).padStart(2, '0')).join('');
}
export function capturePvSession(machine: PvBackingMachine, assets: PvAssets): PvSession {
	// These are synchronous, passive exports: all parts refer to one CPU boundary.
	return {
		retained: Uint8Array.from(machine.export_pvf1_retained()),
		cardRom: assets.cardRom.slice(),
		cardSram: Uint8Array.from(machine.pvf1_card_sram()),
		ramCard: Uint8Array.from(machine.pvf1_ram_card_image()),
		ramCardWritable: assets.ramCardWritable,
		handwritingModel: assets.handwritingModel.slice(),
	};
}
function validate(session: PvSession) {
	pvRetainedIdentity(session.retained);
	for (const key of PARTS) if (!(session[key] instanceof Uint8Array)) throw new Error('Invalid PV-F1 session bytes');
	if (typeof session.ramCardWritable !== 'boolean') throw new Error('Invalid memory-card write protection');
	if (session.cardRom.length || session.cardSram.length) {
		if (session.cardRom.length !== 0x20000 || session.cardSram.length !== 0x8000 || session.ramCard.length)
			throw new Error('Invalid PV-F1 program-card backing');
	}
	if (session.ramCard.length && session.ramCard.length !== 0x20000)
		throw new Error('Invalid PV-F1 memory-card backing');
	if (PARTS.reduce((n, key) => n + session[key].length, 0) > LIMIT) throw new Error('PV-F1 session exceeds size limit');
}
export async function encodePvSession(session: PvSession): Promise<Uint8Array> {
	validate(session);
	const meta = new TextEncoder().encode(
		JSON.stringify({ lengths: PARTS.map((key) => session[key].length), writable: session.ramCardWritable }),
	);
	const owned = new Uint8Array(PREFIX + meta.length + PARTS.reduce((n, key) => n + session[key].length, 0));
	owned.set(MAGIC);
	new DataView(owned.buffer).setUint32(40, meta.length, true);
	owned.set(meta, PREFIX);
	let offset = PREFIX + meta.length;
	for (const key of PARTS) {
		owned.set(session[key], offset);
		offset += session[key].length;
	}
	owned.set(new Uint8Array(await crypto.subtle.digest('SHA-256', owned.subarray(40))), 8);
	return owned;
}
export async function decodePvSession(image: Uint8Array, identity?: string): Promise<PvSession> {
	// Own the input across the digest await; a caller cannot change checked bytes.
	const owned = new Uint8Array(image);
	if (owned.length < PREFIX || owned.length > LIMIT + 4096 || !MAGIC.every((b, i) => owned[i] === b))
		throw new Error('Invalid PV-F1 browser-save format');
	const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', owned.subarray(40)));
	if (!digest.every((b, i) => b === owned[8 + i])) throw new Error('PV-F1 browser-save checksum mismatch');
	const size = new DataView(owned.buffer).getUint32(40, true);
	if (size > 4096 || PREFIX + size > owned.length) throw new Error('Invalid PV-F1 browser-save header');
	const meta = JSON.parse(new TextDecoder().decode(owned.subarray(PREFIX, PREFIX + size)));
	if (
		!Array.isArray(meta.lengths) ||
		meta.lengths.length !== PARTS.length ||
		meta.lengths.some((n: number) => !Number.isSafeInteger(n) || n < 0) ||
		typeof meta.writable !== 'boolean'
	)
		throw new Error('Invalid PV-F1 browser-save parts');
	let offset = PREFIX + size;
	const session = { ramCardWritable: meta.writable } as PvSession;
	for (const [i, key] of PARTS.entries()) {
		const end = offset + meta.lengths[i];
		if (end > owned.length) throw new Error('Truncated PV-F1 browser save');
		session[key] = owned.slice(offset, end);
		offset = end;
	}
	if (offset !== owned.length) throw new Error('Trailing PV-F1 browser-save bytes');
	validate(session);
	if (identity && pvRetainedIdentity(session.retained) !== identity)
		throw new Error('PV-F1 browser save belongs to another firmware');
	return session;
}
