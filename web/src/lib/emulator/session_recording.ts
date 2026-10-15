import { stepBounded, ExecutionCancelled, type SlicedEmulator } from './bounded_step';
import type { OzProfile } from './oz9600_replay';

export type SessionModel = 'pv-f1' | 'oz-9600';
export type SessionContext = {
	model: SessionModel;
	profile: 'firmware' | OzProfile;
	firmwareSha256: string;
	coreIdentity: string;
	settings: { executionMode: string; typingCatchUp: boolean; targetFps: number; sound: boolean };
};
export type DeliveredInput =
	| { kind: 'matrix'; code: number; pressed: boolean }
	| { kind: 'on'; pressed: boolean }
	| { kind: 'tablet'; raw_x: number; raw_y: number; pressed: boolean };
export type SessionEvent = (DeliveredInput | { kind: 'frame'; sha256: string }) & { at: number };
export type RecordedSession = {
	context: SessionContext;
	initial: Uint8Array;
	events: SessionEvent[];
	duration: number;
};
const MAGIC = new TextEncoder().encode('ORGREC01');
const PREFIX = 44;
const MAX_METADATA = 8 * 1024 * 1024;
const MAX_INITIAL = 64 * 1024 * 1024 + 4096;
const MAX_EVENTS = 100_000;
const OZ_PROFILES = [
	'strict',
	'experimental',
	'experimental-isr-clear-only',
	'experimental-isr-mti-writable',
	'experimental-on-edge',
	'experimental-irq-imr',
	'experimental-rtc',
	'experimental-rtc-irq-imr',
	'provisional-v1',
];
const PV_KEYS = [0, 1, 5, 6, 7, 8, 9, 13, 14, 15];
const hex = (bytes: Uint8Array) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
export async function sessionHash(bytes: Uint8Array): Promise<string> {
	return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes))));
}
function natural(n: unknown): n is number {
	return Number.isSafeInteger(n) && (n as number) >= 0;
}
function fields(value: any, allowed: string[]) {
	if (
		!value ||
		typeof value !== 'object' ||
		Array.isArray(value) ||
		Object.keys(value).some((key) => !allowed.includes(key))
	)
		throw new Error('Invalid session fields');
}
function sha(value: unknown) {
	if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) throw new Error('Invalid session hash');
}
export function validateRecordedSession(session: RecordedSession) {
	fields(session, ['context', 'initial', 'events', 'duration']);
	const c = session.context;
	fields(c, ['model', 'profile', 'firmwareSha256', 'coreIdentity', 'settings']);
	if (
		!['pv-f1', 'oz-9600'].includes(c.model) ||
		(c.model === 'pv-f1' ? c.profile !== 'firmware' : !OZ_PROFILES.includes(c.profile))
	)
		throw new Error('Invalid session model/profile');
	sha(c.firmwareSha256);
	if (typeof c.coreIdentity !== 'string' || !c.coreIdentity.length || c.coreIdentity.length > 256)
		throw new Error('Invalid core identity');
	fields(c.settings, ['executionMode', 'typingCatchUp', 'targetFps', 'sound']);
	if (
		!['interactive', 'turbo', 'deterministic'].includes(c.settings.executionMode) ||
		typeof c.settings.typingCatchUp !== 'boolean' ||
		typeof c.settings.sound !== 'boolean' ||
		!natural(c.settings.targetFps) ||
		c.settings.targetFps < 1 ||
		c.settings.targetFps > 240
	)
		throw new Error('Invalid session settings');
	if (
		!(session.initial instanceof Uint8Array) ||
		!session.initial.length ||
		session.initial.length > MAX_INITIAL ||
		!natural(session.duration) ||
		!Array.isArray(session.events) ||
		session.events.length > MAX_EVENTS
	)
		throw new Error('Invalid session size/duration');
	let previous = 0;
	for (const e of session.events) {
		fields(
			e,
			e?.kind === 'frame'
				? ['at', 'kind', 'sha256']
				: e?.kind === 'matrix'
					? ['at', 'kind', 'code', 'pressed']
					: e?.kind === 'tablet'
						? ['at', 'kind', 'raw_x', 'raw_y', 'pressed']
						: ['at', 'kind', 'pressed'],
		);
		if (!natural(e.at) || e.at < previous || e.at > session.duration) throw new Error('Invalid session event order');
		previous = e.at;
		if (e.kind === 'frame') sha(e.sha256);
		else {
			if (typeof e.pressed !== 'boolean') throw new Error('Invalid session contact');
			if (e.kind === 'matrix') {
				if (!natural(e.code) || (c.model === 'pv-f1' ? !PV_KEYS.includes(e.code) : e.code >= 88))
					throw new Error('Invalid matrix contact');
			} else if (e.kind === 'tablet') {
				if (!natural(e.raw_x) || !natural(e.raw_y) || e.raw_x > 1023 || e.raw_y > 1023)
					throw new Error('Invalid tablet contact');
			} else if (e.kind !== 'on') throw new Error('Invalid session event kind');
		}
	}
	if (
		!session.events.length ||
		session.events.at(-1)?.kind !== 'frame' ||
		session.events.at(-1)?.at !== session.duration
	)
		throw new Error('Session requires a final screen checkpoint');
}
export async function encodeRecordedSession(session: RecordedSession): Promise<Uint8Array> {
	validateRecordedSession(session);
	const metadata = new TextEncoder().encode(
		JSON.stringify({ context: session.context, events: session.events, duration: session.duration }),
	);
	if (metadata.length > MAX_METADATA) throw new Error('Session metadata exceeds size limit');
	const bytes = new Uint8Array(PREFIX + metadata.length + session.initial.length);
	bytes.set(MAGIC);
	new DataView(bytes.buffer).setUint32(40, metadata.length, true);
	bytes.set(metadata, PREFIX);
	bytes.set(session.initial, PREFIX + metadata.length);
	bytes.set(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes.subarray(40))), 8);
	return bytes;
}
export async function decodeRecordedSession(input: Uint8Array): Promise<RecordedSession> {
	const bytes = new Uint8Array(input);
	if (
		bytes.length < PREFIX ||
		bytes.length > PREFIX + MAX_METADATA + MAX_INITIAL ||
		!MAGIC.every((b, i) => bytes[i] === b)
	)
		throw new Error('Invalid recording format');
	const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes.subarray(40)));
	if (!digest.every((b, i) => b === bytes[8 + i])) throw new Error('Recording checksum mismatch');
	const length = new DataView(bytes.buffer).getUint32(40, true);
	if (length > MAX_METADATA || PREFIX + length >= bytes.length) throw new Error('Invalid recording header');
	const metadata = JSON.parse(
		new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(PREFIX, PREFIX + length)),
	);
	fields(metadata, ['context', 'events', 'duration']);
	const session = { ...metadata, initial: bytes.slice(PREFIX + length) } as RecordedSession;
	validateRecordedSession(session);
	return session;
}

/** Observes delivered electrical contacts. Browser events and requested budgets
 * do not advance this timeline; only Rust's completed scheduler boundaries do. */
export class SessionRecorder {
	private position = 0;
	private events: SessionEvent[] = [];
	private pending: Promise<void>[] = [];
	private failure: string | null = null;
	private initial: Uint8Array;
	private context: SessionContext;
	constructor(context: SessionContext, initial: Uint8Array) {
		this.context = JSON.parse(JSON.stringify(context));
		this.initial = new Uint8Array(initial);
	}
	advance(used: number) {
		if (!natural(used) || !natural(this.position + used)) {
			this.failure = 'Invalid recording boundary progress';
			return;
		}
		this.position += used;
	}
	private append(event: SessionEvent) {
		if (this.events.length >= MAX_EVENTS) this.failure = 'Recording event limit reached; restart recording';
		if (!this.failure) this.events.push(event);
	}
	input(event: DeliveredInput) {
		this.append({ ...event, at: this.position });
	}
	checkpoint(pbm: Uint8Array) {
		const event: SessionEvent = { kind: 'frame', at: this.position, sha256: '' };
		this.append(event);
		// Reserve the checkpoint's timeline position before asynchronous hashing.
		const job = sessionHash(pbm).then((hash) => {
			event.sha256 = hash;
		});
		this.pending.push(job);
		return job;
	}
	status() {
		return { boundaries: this.position, events: this.events.length, error: this.failure };
	}
	async finish(): Promise<RecordedSession> {
		await Promise.all(this.pending);
		if (this.failure) throw new Error(this.failure);
		const result = {
			context: JSON.parse(JSON.stringify(this.context)),
			initial: this.initial.slice(),
			duration: this.position,
			events: this.events.map((e) => ({ ...e })),
		};
		validateRecordedSession(result);
		return result;
	}
}

export function sessionPbm(model: SessionModel, pixels: Uint8Array): Uint8Array {
	const rows = model === 'pv-f1' ? 292 : 240;
	if (pixels.length !== 336 * rows) throw new Error('Recording requires a full native LCD frame');
	const header = new TextEncoder().encode(`P4\n336 ${rows}\n`);
	const bytes = new Uint8Array(header.length + 42 * rows);
	bytes.set(header);
	for (let i = 0; i < pixels.length; i++) if (pixels[i]) bytes[header.length + (i >> 3)] |= 0x80 >> (i & 7);
	return bytes;
}
export type SessionReplayTarget = SlicedEmulator & {
	input(event: DeliveredInput): void;
	frame(): Uint8Array;
};
/** The caller validates firmware/backing and installs a fresh machine first.
 * This adapter never writes guest memory, PC, IRQ state or application queues. */
export async function replayRecordedSession(
	target: SessionReplayTarget,
	session: RecordedSession,
	options: { signal?: AbortSignal; yieldHost?: () => Promise<void>; onProgress?: (used: number) => void } = {},
) {
	validateRecordedSession(session);
	let completed = 0;
	let checkpoints = 0;
	const held = new Map<string, DeliveredInput>();
	const advance = async (end: number) => {
		while (completed < end)
			await stepBounded(target, Math.min(end - completed, 1_000_000), {
				signal: options.signal,
				yieldHost: options.yieldHost,
				onProgress: (used) => {
					completed += used;
					options.onProgress?.(used);
				},
			});
	};
	try {
		for (const event of session.events) {
			if (options.signal?.aborted) throw new ExecutionCancelled(completed);
			await advance(event.at);
			if (event.kind === 'frame') {
				const actual = await sessionHash(target.frame());
				if (actual !== event.sha256)
					throw new Error(
						`Screen checkpoint ${checkpoints} mismatch at boundary ${completed}: expected ${event.sha256}, got ${actual}`,
					);
				checkpoints++;
			} else {
				target.input(event);
				const key = event.kind === 'matrix' ? `matrix:${event.code}` : event.kind;
				if (event.pressed) held.set(key, event);
				else held.delete(key);
			}
		}
		return { boundaries: completed, checkpoints };
	} finally {
		for (const event of held.values()) target.input({ ...event, pressed: false });
	}
}
