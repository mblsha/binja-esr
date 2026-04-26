import { IOCS, IOCS_PUBLIC_ENTRY_ADDR } from './iocs';
import { buildMemoryWriteBlocks, type MemoryWriteBlock, type MemoryWriteEvent } from './memory_write_blocks';
import type { StubHandler, StubRegistration } from './sc62015_stub_types';

export type RegisterName =
	| 'A'
	| 'B'
	| 'BA'
	| 'IL'
	| 'IH'
	| 'I'
	| 'X'
	| 'Y'
	| 'U'
	| 'S'
	| 'PC'
	| 'F'
	| 'IMR'
	| 'FC'
	| 'FZ';

export type StatusFlag = 'C' | 'Z';

export type ProbeRegisters = Record<string, number>;
export type ProbeSample = { pc: number; count: number; regs: ProbeRegisters };
export type ProbeHandler = (sample: ProbeSample) => void;

export type CallArtifacts = {
	address: number;
	before_pc: number;
	after_pc: number;
	before_sp: number;
	after_sp: number;
	before_regs: Record<string, number>;
	after_regs: Record<string, number>;
	memory_writes: Array<{ addr: number; value: number }>;
	lcd_writes: Array<{
		page: number;
		col: number;
		value: number;
		trace: { pc: number; call_stack: { len: number; frames: number[] } };
	}>;
	probe_samples: ProbeSample[];
	perfetto_trace_b64?: string | null;
	report: {
		reason: string;
		steps: number;
		pc: number;
		sp: number;
		halted: boolean;
		fault: { kind: string; message: string } | null;
	};
};

export type CallHandle = {
	index: number;
	address: number;
	name: string | null;
	artifacts: {
		before: Record<string, number>;
		after: Record<string, number>;
		changed: string[];
		memoryBlocks: MemoryWriteBlock[];
		lcdWrites: CallArtifacts['lcd_writes'];
		probeSamples: ProbeSample[];
		perfettoTraceB64: string | null;
		result: CallArtifacts['report'];
		infoLog: string[];
	};
};

export type PerfettoTraceHandle = {
	index: number;
	name: string;
	byteLength: number;
	perfettoTraceB64: string;
};

export type PrintEntry = { index: number; value: unknown };

export type EvalEvent =
	| { kind: 'call'; sequence: number; handle: CallHandle }
	| { kind: 'perfetto_trace'; sequence: number; trace: PerfettoTraceHandle }
	| { kind: 'print'; sequence: number; entry: PrintEntry }
	| { kind: 'error'; sequence: number; message: string }
	| { kind: 'reset'; sequence: number; fresh: boolean; warmupTicks: number };

export type EvalCallOptions = {
	maxInstructions?: number;
	trace?: boolean;
	zeroMissing?: boolean;
};

// IOCS calls use a mix of registers (A/I) and "byte registers" stored in IMEM.
//
// TRM (PC‑E500): (bl)/(bh)/(cl)/(ch) are 1-byte "register halves" mapped to IMEM:
// - (bl) = IMEM[0xD4]
// - (bh) = IMEM[0xD5]
// - (cl) = IMEM[0xD6]
// - (ch) = IMEM[0xD7]
//
// Example: IOCS "one character output to arbitrary position" (0x0041):
// - (cx)=0x0000  => (cl)=0, (ch)=0
// - (bl)=x, (bh)=y
// - A=byte
// - I=0x0041
// - CALLF 0xFFFE8 (in our runner: e.call(0x00FFFE8, ...))
//
// Note: Some ROM-specific IOCS entry points are selected via `IL` rather than `I`
// (e.g. IOCS.LCD_PUTC). These still consume IMEM bytes for parameters in many cases.
export type EvalIocsImemArgs = {
	bl?: number;
	bh?: number;
	cl?: number;
	ch?: number;
};

export type EvalIocsCallOptions = EvalCallOptions & EvalIocsImemArgs;

export type EvalIocsDisplayPutcOptions = EvalCallOptions &
	EvalIocsImemArgs & {
		bl: number;
		bh: number;
	};

export type EvalResetOptions = {
	fresh?: boolean;
	warmupTicks?: number;
};

export type EvalWaitOptions = {
	chunkInstructions?: number;
	maxInstructions?: number;
	quietSamples?: number;
	requireNonBlank?: boolean;
};

export type EvalCalendarMonthOptions = {
	year: number;
	month: number;
	day?: number;
	width?: number;
	height?: number;
};

export type EvalCalendarMonthAssertion = {
	ok: true;
	title: string;
	year: number;
	month: number;
	day: number | null;
	weeks: number[][];
	checkedDays: number[];
	dayOfYear: number | null;
	daysRemaining: number | null;
	weekOfYear: number | null;
};

export type EvalProofMetadataInput = {
	label?: string;
	keySequence?: string[] | string;
	assertions?: unknown;
	artifacts?: Record<string, string>;
	includeGeneratedAt?: boolean;
	context?: Record<string, unknown>;
};

export type EvalApiOptions = {};

export interface EmulatorAdapter {
	callFunction(
		address: number,
		maxInstructions: number,
		options?: {
			trace?: boolean;
			probe?: { pc: number; maxSamples?: number };
			stubs?: Array<{ id: number; pc: number }>;
		} | null,
	): Promise<CallArtifacts>;
	startPerfettoTrace?(name: string): Promise<void> | void;
	stopPerfettoTrace?(): Promise<string> | string;
	reset(fresh?: boolean): Promise<void> | void;
	step(instructions: number): Promise<void> | void;
	getReg(name: string): number;
	setReg(name: string, value: number): void;
	read8(addr: number): number;
	write8(addr: number, value: number): void;
	lcdText?(): string[] | null;
	lcdPixels?(): Uint8Array | number[] | null;
	pressMatrixCode?(code: number): void;
	releaseMatrixCode?(code: number): void;
	injectMatrixEvent?(code: number, release: boolean): void;
	pressOnKey?(): void;
	releaseOnKey?(): void;
	registerStub?(stub: StubRegistration): void;
	clearStubs?(): void;
}

export interface EvalApi {
	readonly calls: CallHandle[];
	readonly perfettoTraces: PerfettoTraceHandle[];
	readonly prints: PrintEntry[];
	readonly events: EvalEvent[];
	last(): CallHandle | null;
	reset(options?: EvalResetOptions): Promise<void>;
	step(instructions: number): Promise<void>;
	call(
		reference: string | number,
		registers?: Partial<Record<RegisterName, number>>,
		options?: EvalCallOptions,
	): Promise<CallHandle>;
	reg(name: RegisterName): number;
	flag(flag: StatusFlag): boolean;
	assert(condition: unknown, message?: string): void;
	print(...items: unknown[]): void;
	withProbe<T>(pc: number, handler: ProbeHandler, body: () => Promise<T> | T): Promise<T>;
	memory: {
		read(address: number, size?: 1 | 2 | 3): Promise<number>;
		write(address: number, size: 1 | 2 | 3, value: number): Promise<void>;
	};
	lcd: {
		text(): Promise<string[]>;
		textString(): Promise<string>;
		pixels(): Promise<number[]>;
		assertCalendarMonth(options: EvalCalendarMonthOptions): Promise<EvalCalendarMonthAssertion>;
	};
	keyboard: {
		press(code: number): Promise<void>;
		release(code: number): Promise<void>;
		tap(code: number, holdInstructions?: number): Promise<void>;
		injectEvent(code: number, release: boolean): Promise<void>;
	};
	keys: {
		tap(spec: string | number, holdInstructions?: number): Promise<void>;
		event: {
			press(code: number): Promise<void>;
			release(code: number): Promise<void>;
			tap(code: number, holdInstructions?: number): Promise<void>;
		};
		phys: {
			press(code: number): Promise<void>;
			release(code: number): Promise<void>;
			tap(code: number, holdInstructions?: number): Promise<void>;
		};
		app: {
			tap(name: string, holdInstructions?: number): Promise<void>;
		};
	};
	onKey: {
		press(): Promise<void>;
		release(): Promise<void>;
		tap(holdInstructions?: number): Promise<void>;
	};
	wait: {
		lcdStable(options?: EvalWaitOptions): Promise<{ instructions: number; signature: string }>;
		screenChange(
			options?: EvalWaitOptions & { baseline?: number[] },
		): Promise<{ instructions: number; signature: string }>;
		textIncludes(text: string, options?: EvalWaitOptions): Promise<{ instructions: number; text: string }>;
	};
	proof: {
		metadata(input?: EvalProofMetadataInput): Promise<Record<string, unknown>>;
		metadataYaml(input?: EvalProofMetadataInput): Promise<string>;
	};
	perfetto: {
		trace<T>(name: string, body: () => Promise<T> | T): Promise<T>;
	};
	iocs: {
		putc(ch: string | number, options?: EvalIocsCallOptions): Promise<CallHandle>;
		text(text: string, options?: EvalIocsCallOptions): Promise<CallHandle[]>;
		putcXY(ch: string | number, options: EvalIocsDisplayPutcOptions): Promise<CallHandle>;
	};
	stub(pc: number, name: string | null, handler: StubHandler): StubRegistration;
	clearStubs(): void;
}

const DEFAULT_MAX_INSTRUCTIONS = 200_000;
const DEFAULT_WARMUP_TICKS = 0;
const DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS = 40_000;
const DEFAULT_PROBE_MAX_SAMPLES = 256;
const DEFAULT_WAIT_CHUNK_INSTRUCTIONS = 20_000;
const DEFAULT_WAIT_MAX_INSTRUCTIONS = 1_000_000;
const DEFAULT_WAIT_QUIET_SAMPLES = 3;
const IMEM_BASE_ADDR = 0x0010_0000;
const IQ7000_LCD_WIDTH = 96;
const IQ7000_LCD_HEIGHT = 64;

const IQ7000_APP_EVENT_CODES = Object.freeze({
	calc: 0x00,
	shift: 0x01,
	memo: 0x08,
	home: 0x09,
	tel: 0x10,
	telephone: 0x10,
	calendar: 0x18,
	schedule: 0x19,
	card: 0x1a,
	'card-samples': 0x1a,
	samples: 0x1a,
	world: 0x1b,
} satisfies Record<string, number>);

const IQ7000_CALENDAR_DAY_ONES_X = [5, 19, 33, 47, 61, 75, 89];
const IQ7000_CALENDAR_DAY_Y = [17, 25, 33, 41, 49, 57];
const IQ7000_COMPACT_DIGITS = Object.freeze({
	'0': ['####', '#..#', '#..#', '#..#', '#..#', '####'],
	'1': ['...#', '...#', '...#', '...#', '...#', '...#'],
	'2': ['####', '...#', '####', '#...', '#...', '####'],
	'3': ['####', '...#', '####', '...#', '...#', '####'],
	'4': ['#..#', '#..#', '####', '...#', '...#', '...#'],
	'5': ['####', '#...', '####', '...#', '...#', '####'],
	'6': ['####', '#...', '####', '#..#', '#..#', '####'],
	'7': ['####', '#..#', '#..#', '...#', '...#', '...#'],
	'8': ['####', '#..#', '####', '#..#', '#..#', '####'],
	'9': ['####', '#..#', '####', '...#', '...#', '####'],
} satisfies Record<string, string[]>);

const RESULT_REGISTER_ORDER: RegisterName[] = ['A', 'B', 'BA', 'IL', 'IH', 'I', 'X', 'Y', 'U', 'S', 'F', 'IMR', 'PC'];

function normalizeAddress(addr: number): number {
	return (addr >>> 0) & 0x00ff_ffff;
}

function base64ByteLength(b64: string): number {
	const len = b64.length;
	if (len === 0) return 0;
	const padding = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
	return Math.max(0, Math.floor((len * 3) / 4) - padding);
}

function resolveReference(reference: string | number): { address: number; name: string | null } {
	if (typeof reference === 'number') {
		return { address: normalizeAddress(reference), name: null };
	}
	const trimmed = reference.trim().toLowerCase();
	if (trimmed.startsWith('0x')) {
		const value = Number.parseInt(trimmed.slice(2), 16);
		if (Number.isFinite(value)) return { address: normalizeAddress(value), name: null };
	}
	const asDec = Number.parseInt(trimmed, 10);
	if (Number.isFinite(asDec)) return { address: normalizeAddress(asDec), name: null };
	throw new Error(`Unsupported function reference '${reference}' (use an address like 0x00F29B8).`);
}

function normalizeRegValue(name: RegisterName, value: number): number {
	if (name === 'A' || name === 'B' || name === 'IL' || name === 'IH' || name === 'F' || name === 'IMR')
		return value & 0xff;
	if (name === 'BA' || name === 'I') return value & 0xffff;
	if (name === 'PC') return value & 0x0f_ffff;
	return value >>> 0;
}

function buildAssignments(
	provided: Partial<Record<RegisterName, number>> | undefined,
	zeroMissing: boolean,
): Map<RegisterName, number> {
	const out = new Map<RegisterName, number>();
	if (provided) {
		for (const [rawName, value] of Object.entries(provided)) {
			if (typeof value !== 'number') continue;
			const upper = rawName.trim().toUpperCase() as RegisterName;
			out.set(upper, normalizeRegValue(upper, value));
		}
	}
	if (zeroMissing) {
		for (const name of ['A', 'B', 'BA', 'I', 'X', 'Y', 'U'] as RegisterName[]) {
			if (!out.has(name)) out.set(name, 0);
		}
	}
	return out;
}

function diffRegisters(before: Record<string, number>, after: Record<string, number>): string[] {
	const changed: string[] = [];
	for (const key of Object.keys(after)) {
		if (before[key] !== after[key]) changed.push(key);
	}
	changed.sort((a, b) => a.localeCompare(b));
	return changed;
}

function parseByteSpec(raw: string): number | null {
	const trimmed = raw.trim().toLowerCase();
	if (!trimmed) return null;
	const value = trimmed.startsWith('0x') ? Number.parseInt(trimmed.slice(2), 16) : Number.parseInt(trimmed, 10);
	if (!Number.isFinite(value) || value < 0 || value > 0xff) return null;
	return value & 0xff;
}

function resolveAppEventCode(name: string): number {
	const key = name.trim().toLowerCase();
	const code = IQ7000_APP_EVENT_CODES[key as keyof typeof IQ7000_APP_EVENT_CODES];
	if (code === undefined) {
		throw new Error(`Unknown IQ-7000 app key '${name}'`);
	}
	return code;
}

function pixelSignature(pixels: number[]): string {
	let hash = 0x811c9dc5;
	let lit = 0;
	for (const value of pixels) {
		if (value) lit += 1;
		hash ^= value ? 1 : 0;
		hash = Math.imul(hash, 0x01000193) >>> 0;
	}
	return `${hash.toString(16).padStart(8, '0')}:${lit}`;
}

function isBlankPixels(pixels: number[]): boolean {
	return !pixels.some(Boolean);
}

function compactDigitMatches(pixels: number[], width: number, x: number, y: number, digit: string): boolean {
	const pattern = IQ7000_COMPACT_DIGITS[digit as keyof typeof IQ7000_COMPACT_DIGITS];
	if (!pattern) return false;
	if (x < 0 || y < 0 || y + pattern.length > Math.floor(pixels.length / width)) return false;
	for (let row = 0; row < pattern.length; row++) {
		for (let col = 0; col < pattern[row].length; col++) {
			const expected = pattern[row][col] === '#';
			const actual = Boolean(pixels[(y + row) * width + x + col]);
			if (actual !== expected) return false;
		}
	}
	return true;
}

function assertCompactNumber(pixels: number[], width: number, onesX: number, y: number, value: number): void {
	const digits = String(value);
	if (digits.length === 1) {
		if (!compactDigitMatches(pixels, width, onesX, y, digits)) {
			throw new Error(`Calendar day ${value} did not match compact digit pixels at (${onesX}, ${y})`);
		}
		return;
	}
	if (digits.length !== 2) {
		throw new Error(`Calendar compact assertion only supports 1-2 digit day numbers, got ${value}`);
	}
	const tensX = onesX - 5;
	if (!compactDigitMatches(pixels, width, tensX, y, digits[0])) {
		throw new Error(`Calendar day ${value} tens digit did not match compact pixels at (${tensX}, ${y})`);
	}
	if (!compactDigitMatches(pixels, width, onesX, y, digits[1])) {
		throw new Error(`Calendar day ${value} ones digit did not match compact pixels at (${onesX}, ${y})`);
	}
}

function daysInMonth(year: number, month: number): number {
	return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

function monthWeeksSundayFirst(year: number, month: number): number[][] {
	const days = daysInMonth(year, month);
	const firstDay = new Date(Date.UTC(year, month - 1, 1)).getUTCDay();
	const weeks: number[][] = [];
	let day = 1;
	while (day <= days) {
		const week = Array(7).fill(0) as number[];
		for (let dow = weeks.length === 0 ? firstDay : 0; dow < 7 && day <= days; dow++) {
			week[dow] = day++;
		}
		weeks.push(week);
	}
	return weeks;
}

function dayOfYearUtc(year: number, month: number, day: number): number {
	const start = Date.UTC(year, 0, 1);
	const current = Date.UTC(year, month - 1, day);
	return Math.floor((current - start) / 86_400_000) + 1;
}

function daysInYearUtc(year: number): number {
	return dayOfYearUtc(year, 12, 31);
}

function isoWeekUtc(year: number, month: number, day: number): number {
	const date = new Date(Date.UTC(year, month - 1, day));
	const dow = date.getUTCDay() || 7;
	date.setUTCDate(date.getUTCDate() + 4 - dow);
	const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
	return Math.ceil(((date.getTime() - yearStart.getTime()) / 86_400_000 + 1) / 7);
}

function monthTitle(year: number, month: number): string {
	const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
	return `*** ${months[month - 1] ?? '???'} ${year} ***`;
}

function yamlScalar(value: string): string {
	if (/^[A-Za-z0-9_./:-]+$/.test(value) && value !== 'null' && value !== 'true' && value !== 'false') {
		return value;
	}
	return JSON.stringify(value);
}

function toYaml(value: unknown, indent = 0): string {
	const pad = ' '.repeat(indent);
	if (value === null || value === undefined) return 'null';
	if (typeof value === 'string') return yamlScalar(value);
	if (typeof value === 'number' || typeof value === 'boolean') return String(value);
	if (Array.isArray(value)) {
		if (!value.length) return '[]';
		return value
			.map((entry) => {
				if (entry && typeof entry === 'object') {
					const nested = toYaml(entry, indent + 2);
					return `${pad}- ${nested.trimStart()}`;
				}
				return `${pad}- ${toYaml(entry, indent + 2).trimStart()}`;
			})
			.join('\n');
	}
	if (typeof value === 'object') {
		const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined);
		if (!entries.length) return '{}';
		return entries
			.map(([key, entry]) => {
				if (entry && typeof entry === 'object') {
					const nested = toYaml(entry, indent + 2);
					if (!nested.includes('\n')) return `${pad}${key}: ${nested.trimStart()}`;
					return `${pad}${key}:\n${nested}`;
				}
				return `${pad}${key}: ${toYaml(entry, indent + 2)}`;
			})
			.join('\n');
	}
	return yamlScalar(String(value));
}

export const Reg = Object.freeze(
	RESULT_REGISTER_ORDER.reduce(
		(acc, name) => {
			(acc as any)[name] = name;
			return acc;
		},
		{} as Record<RegisterName, RegisterName>,
	),
);

export const Flag = Object.freeze({
	C: 'C',
	Z: 'Z',
} satisfies Record<StatusFlag, StatusFlag>);

export function createEvalApi(adapter: EmulatorAdapter, _options?: EvalApiOptions): EvalApi {
	const calls: CallHandle[] = [];
	const perfettoTraces: PerfettoTraceHandle[] = [];
	const prints: PrintEntry[] = [];
	const events: EvalEvent[] = [];
	let sequence = 0;
	let callIndex = 0;
	let traceIndex = 0;
	let perfettoActive = false;
	const probeStack: Array<{ pc: number; handler: ProbeHandler; maxSamples: number }> = [];
	const stubs: StubRegistration[] = [];
	let stubId = 1;
	let perfettoActiveName: string | null = null;

	function isLikelyWasmTrap(message: string): boolean {
		const lower = message.toLowerCase();
		return (
			/\bunreachable\b/.test(lower) ||
			lower.includes('out of memory') ||
			lower.includes('memory allocation') ||
			lower.includes('wasm trap')
		);
	}

	function attachPerfettoHint(err: unknown): unknown {
		const message = err instanceof Error ? err.message : String(err);
		if (!message || !isLikelyWasmTrap(message)) return err;
		if (message.includes('Perfetto tracing may have')) return err;

		const name = perfettoActiveName ? `'${perfettoActiveName}' ` : '';
		const hint =
			`Perfetto tracing ${name}may have exhausted WASM memory (traces are buffered in memory). ` +
			`Try tracing fewer instructions or splitting into multiple traces; you may need to reload the emulator after a trap.`;
		const wrapped = new Error(`${message}\n${hint}`);
		(wrapped as any).cause = err;
		return wrapped;
	}

	async function writeIocsImemArgs(args: EvalIocsImemArgs | undefined) {
		if (!args) return;
		if (args.bl !== undefined) await api.memory.write(IMEM_BASE_ADDR + 0xd4, 1, args.bl);
		if (args.bh !== undefined) await api.memory.write(IMEM_BASE_ADDR + 0xd5, 1, args.bh);
		if (args.cl !== undefined) await api.memory.write(IMEM_BASE_ADDR + 0xd6, 1, args.cl);
		if (args.ch !== undefined) await api.memory.write(IMEM_BASE_ADDR + 0xd7, 1, args.ch);
	}

	function resolvePutcByte(ch: string | number): number {
		if (typeof ch === 'number') return ch & 0xff;
		if (typeof ch === 'string') {
			if (!ch.length) throw new Error('iocs.putc requires a character');
			const byte = ch.codePointAt(0) ?? 0;
			if (byte > 0xff) throw new Error('iocs.putc only supports single-byte characters');
			return byte;
		}
		throw new Error('iocs.putc requires a string or number');
	}

	async function tapEvent(code: number, holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) {
		adapter.injectMatrixEvent?.(code & 0xff, false);
		if (holdInstructions > 0) await Promise.resolve(adapter.step(holdInstructions));
		adapter.injectMatrixEvent?.(code & 0xff, true);
	}

	async function tapPhysical(code: number, holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) {
		adapter.pressMatrixCode?.(code & 0xff);
		if (holdInstructions > 0) await Promise.resolve(adapter.step(holdInstructions));
		adapter.releaseMatrixCode?.(code & 0xff);
	}

	function normalizeWaitOptions(options?: EvalWaitOptions) {
		return {
			chunkInstructions: options?.chunkInstructions ?? DEFAULT_WAIT_CHUNK_INSTRUCTIONS,
			maxInstructions: options?.maxInstructions ?? DEFAULT_WAIT_MAX_INSTRUCTIONS,
			quietSamples: options?.quietSamples ?? DEFAULT_WAIT_QUIET_SAMPLES,
			requireNonBlank: options?.requireNonBlank ?? true,
		};
	}

	const api: EvalApi = {
		calls,
		perfettoTraces,
		prints,
		events,
		last: () => (calls.length ? calls[calls.length - 1] : null),
		reset: async (options?: EvalResetOptions) => {
			const fresh = options?.fresh ?? true;
			const warmupTicks = options?.warmupTicks ?? DEFAULT_WARMUP_TICKS;
			if (fresh) {
				calls.length = 0;
				prints.length = 0;
				events.length = 0;
				sequence = 0;
				callIndex = 0;
			}
			await Promise.resolve(adapter.reset(fresh));
			if (warmupTicks > 0) {
				await Promise.resolve(adapter.step(warmupTicks));
			}
			events.push({ kind: 'reset', sequence: sequence++, fresh, warmupTicks });
		},
		step: async (instructions: number) => {
			if (typeof adapter.step !== 'function') throw new Error('EmulatorAdapter.step is not available.');
			await Promise.resolve(adapter.step(instructions));
		},
		call: async (reference, registers, options) => {
			const { address, name } = resolveReference(reference);
			const maxInstructions = options?.maxInstructions ?? DEFAULT_MAX_INSTRUCTIONS;
			const zeroMissing = options?.zeroMissing ?? false;
			const trace = options?.trace ?? false;
			if (trace && perfettoActive) {
				throw new Error(
					'Nested tracing is unsupported: disable per-call trace when using e.perfetto.trace(name, ...).',
				);
			}
			const assignments = buildAssignments(registers, zeroMissing);
			for (const [regName, value] of assignments.entries()) {
				adapter.setReg(regName, value);
			}

			const activeProbe = probeStack.length ? probeStack[probeStack.length - 1] : null;
			const stubSpecs = stubs.map((stub) => ({ id: stub.id, pc: stub.pc }));
			const artifacts = await adapter.callFunction(
				address,
				maxInstructions,
				activeProbe
					? {
							trace,
							probe: { pc: activeProbe.pc, maxSamples: activeProbe.maxSamples },
							stubs: stubSpecs,
						}
					: { trace, stubs: stubSpecs },
			);

			if (activeProbe && artifacts.probe_samples?.length) {
				for (const sample of artifacts.probe_samples) {
					try {
						activeProbe.handler(sample);
					} catch {
						/* ignore probe handler errors */
					}
				}
			}

			const memoryEvents: MemoryWriteEvent[] = artifacts.memory_writes.map((e) => ({
				addr: e.addr >>> 0,
				value: e.value & 0xff,
				size: 1,
			}));
			const memoryBlocks = buildMemoryWriteBlocks(memoryEvents);
			const before = artifacts.before_regs;
			const after = artifacts.after_regs;
			const changed = diffRegisters(before, after);

			const fault = artifacts.report.fault;
			const infoLog: string[] = [
				`Execution reason: ${artifacts.report.reason}`,
				fault ? `Fault: ${fault.kind}: ${fault.message}` : '',
				memoryEvents.length
					? `Captured ${memoryEvents.length} memory write byte(s) (${memoryBlocks.length} block(s)).`
					: 'No memory writes captured.',
				artifacts.lcd_writes.length
					? `Captured ${artifacts.lcd_writes.length} LCD addressing-unit write(s).`
					: 'No LCD writes captured.',
				artifacts.perfetto_trace_b64
					? `Perfetto trace captured (${artifacts.perfetto_trace_b64.length} b64 chars).`
					: '',
			].filter(Boolean);

			const handle: CallHandle = {
				index: callIndex++,
				address,
				name,
				artifacts: {
					before,
					after,
					changed,
					memoryBlocks,
					lcdWrites: artifacts.lcd_writes,
					probeSamples: artifacts.probe_samples ?? [],
					perfettoTraceB64: artifacts.perfetto_trace_b64 ?? null,
					result: artifacts.report,
					infoLog,
				},
			};
			calls.push(handle);
			events.push({ kind: 'call', sequence: sequence++, handle });
			return handle;
		},
		reg: (name) => adapter.getReg(name),
		flag: (flag) => {
			if (flag === 'C') return Boolean(adapter.getReg('FC') & 1);
			if (flag === 'Z') return Boolean(adapter.getReg('FZ') & 1);
			return false;
		},
		assert: (condition, message) => {
			if (!condition) throw new Error(message ?? 'Assertion failed');
		},
		print: (...items) => {
			for (const value of items) {
				const entry: PrintEntry = { index: prints.length, value };
				prints.push(entry);
				events.push({ kind: 'print', sequence: sequence++, entry });
			}
		},
		withProbe: async <T>(pc: number, handler: ProbeHandler, body: () => Promise<T> | T) => {
			if (typeof handler !== 'function') throw new Error('withProbe requires a handler function');
			if (typeof body !== 'function') throw new Error('withProbe requires a callback');
			const normalizedPc = normalizeAddress(pc);
			probeStack.push({
				pc: normalizedPc,
				handler,
				maxSamples: DEFAULT_PROBE_MAX_SAMPLES,
			});
			try {
				return await body();
			} finally {
				probeStack.pop();
			}
		},
		stub: (pc, name, handler) => {
			if (typeof handler !== 'function') throw new Error('stub requires a handler function');
			if (typeof adapter.registerStub !== 'function') {
				throw new Error('stub support is not available in this runtime');
			}
			const normalizedPc = normalizeAddress(pc);
			const stub: StubRegistration = {
				id: stubId++,
				pc: normalizedPc,
				name: name ?? null,
				handler,
			};
			stubs.push(stub);
			adapter.registerStub(stub);
			return stub;
		},
		clearStubs: () => {
			stubs.length = 0;
			adapter.clearStubs?.();
		},
		memory: {
			read: async (address, size = 1) => {
				const addr = normalizeAddress(address);
				if (size === 1) return adapter.read8(addr);
				if (size === 2) return (adapter.read8(addr) | (adapter.read8(addr + 1) << 8)) >>> 0;
				if (size === 3)
					return (adapter.read8(addr) | (adapter.read8(addr + 1) << 8) | (adapter.read8(addr + 2) << 16)) >>> 0;
				throw new Error(`Unsupported read size ${size}`);
			},
			write: async (address, size, value) => {
				const addr = normalizeAddress(address);
				if (size === 1) {
					adapter.write8(addr, value & 0xff);
					return;
				}
				if (size === 2) {
					adapter.write8(addr, value & 0xff);
					adapter.write8(addr + 1, (value >> 8) & 0xff);
					return;
				}
				if (size === 3) {
					adapter.write8(addr, value & 0xff);
					adapter.write8(addr + 1, (value >> 8) & 0xff);
					adapter.write8(addr + 2, (value >> 16) & 0xff);
					return;
				}
				throw new Error(`Unsupported write size ${size}`);
			},
		},
		lcd: {
			text: async () => adapter.lcdText?.() ?? [],
			textString: async () => (adapter.lcdText?.() ?? []).join('\n'),
			pixels: async () => Array.from(adapter.lcdPixels?.() ?? []),
			assertCalendarMonth: async (options) => {
				const { year, month } = options;
				if (!Number.isInteger(year) || year < 1) throw new Error(`Invalid calendar year: ${year}`);
				if (!Number.isInteger(month) || month < 1 || month > 12) throw new Error(`Invalid calendar month: ${month}`);
				const width = options.width ?? IQ7000_LCD_WIDTH;
				const height = options.height ?? IQ7000_LCD_HEIGHT;
				const pixels = Array.from(adapter.lcdPixels?.() ?? []);
				if (pixels.length !== width * height) {
					throw new Error(`LCD pixel payload length mismatch: expected ${width * height}, got ${pixels.length}`);
				}
				const weeks = monthWeeksSundayFirst(year, month);
				if (weeks.length > IQ7000_CALENDAR_DAY_Y.length) {
					throw new Error(
						`Calendar month needs ${weeks.length} week rows; assertion supports ${IQ7000_CALENDAR_DAY_Y.length}`,
					);
				}
				const checkedDays: number[] = [];
				for (let row = 0; row < weeks.length; row++) {
					for (let col = 0; col < 7; col++) {
						const day = weeks[row][col];
						if (!day) continue;
						assertCompactNumber(pixels, width, IQ7000_CALENDAR_DAY_ONES_X[col], IQ7000_CALENDAR_DAY_Y[row], day);
						checkedDays.push(day);
					}
				}
				const day = options.day ?? null;
				const dayOfYear = day === null ? null : dayOfYearUtc(year, month, day);
				return {
					ok: true,
					title: monthTitle(year, month),
					year,
					month,
					day,
					weeks,
					checkedDays,
					dayOfYear,
					daysRemaining: day === null ? null : daysInYearUtc(year) - dayOfYearUtc(year, month, day),
					weekOfYear: day === null ? null : isoWeekUtc(year, month, day),
				};
			},
		},
		keyboard: {
			press: async (code: number) => {
				adapter.pressMatrixCode?.(code & 0xff);
			},
			release: async (code: number) => {
				adapter.releaseMatrixCode?.(code & 0xff);
			},
			injectEvent: async (code: number, release: boolean) => {
				adapter.injectMatrixEvent?.(code & 0xff, Boolean(release));
			},
			tap: async (code: number, holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) => {
				await tapEvent(code, holdInstructions);
			},
		},
		keys: {
			tap: async (spec, holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) => {
				if (typeof spec === 'number') {
					await tapEvent(spec, holdInstructions);
					return;
				}
				const [kindRaw, valueRaw] = spec.includes(':') ? spec.split(/:(.*)/s, 2) : ['event', spec];
				const kind = kindRaw.trim().toLowerCase();
				const value = valueRaw.trim();
				if (kind === 'event') {
					const code = parseByteSpec(value);
					if (code === null) throw new Error(`Invalid event key spec '${spec}'`);
					await tapEvent(code, holdInstructions);
					return;
				}
				if (kind === 'phys' || kind === 'physical' || kind === 'matrix') {
					const code = parseByteSpec(value);
					if (code === null) throw new Error(`Invalid physical key spec '${spec}'`);
					await tapPhysical(code, holdInstructions);
					return;
				}
				if (kind === 'app') {
					await tapEvent(resolveAppEventCode(value), holdInstructions);
					return;
				}
				throw new Error(`Unknown key namespace '${kindRaw}' in '${spec}'`);
			},
			event: {
				press: async (code) => adapter.injectMatrixEvent?.(code & 0xff, false),
				release: async (code) => adapter.injectMatrixEvent?.(code & 0xff, true),
				tap: tapEvent,
			},
			phys: {
				press: async (code) => adapter.pressMatrixCode?.(code & 0xff),
				release: async (code) => adapter.releaseMatrixCode?.(code & 0xff),
				tap: tapPhysical,
			},
			app: {
				tap: async (name, holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) => {
					await tapEvent(resolveAppEventCode(name), holdInstructions);
				},
			},
		},
		onKey: {
			press: async () => {
				adapter.pressOnKey?.();
			},
			release: async () => {
				adapter.releaseOnKey?.();
			},
			tap: async (holdInstructions = DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS) => {
				adapter.pressOnKey?.();
				if (holdInstructions > 0) await Promise.resolve(adapter.step(holdInstructions));
				adapter.releaseOnKey?.();
			},
		},
		wait: {
			lcdStable: async (options) => {
				const opts = normalizeWaitOptions(options);
				let instructions = 0;
				let last: string | null = null;
				let stable = 0;
				while (instructions <= opts.maxInstructions) {
					await Promise.resolve(adapter.step(opts.chunkInstructions));
					instructions += opts.chunkInstructions;
					const pixels = Array.from(adapter.lcdPixels?.() ?? []);
					if (opts.requireNonBlank && isBlankPixels(pixels)) {
						last = null;
						stable = 0;
						continue;
					}
					const signature = pixelSignature(pixels);
					if (signature === last) {
						stable += 1;
						if (stable >= opts.quietSamples) return { instructions, signature };
					} else {
						last = signature;
						stable = 1;
					}
				}
				throw new Error(`LCD did not become stable within ${opts.maxInstructions} instructions`);
			},
			screenChange: async (options) => {
				const opts = normalizeWaitOptions({ ...options, requireNonBlank: options?.requireNonBlank ?? false });
				const baseline = options?.baseline
					? pixelSignature(options.baseline)
					: pixelSignature(Array.from(adapter.lcdPixels?.() ?? []));
				let instructions = 0;
				while (instructions <= opts.maxInstructions) {
					await Promise.resolve(adapter.step(opts.chunkInstructions));
					instructions += opts.chunkInstructions;
					const pixels = Array.from(adapter.lcdPixels?.() ?? []);
					if (opts.requireNonBlank && isBlankPixels(pixels)) continue;
					const signature = pixelSignature(pixels);
					if (signature !== baseline) return { instructions, signature };
				}
				throw new Error(`LCD did not change within ${opts.maxInstructions} instructions`);
			},
			textIncludes: async (text, options) => {
				const opts = normalizeWaitOptions({ ...options, requireNonBlank: false });
				let instructions = 0;
				while (instructions <= opts.maxInstructions) {
					const current = (adapter.lcdText?.() ?? []).join('\n');
					if (current.includes(text)) return { instructions, text: current };
					await Promise.resolve(adapter.step(opts.chunkInstructions));
					instructions += opts.chunkInstructions;
				}
				throw new Error(`LCD text did not include '${text}' within ${opts.maxInstructions} instructions`);
			},
		},
		proof: {
			metadata: async (input = {}) => {
				const pixels = Array.from(adapter.lcdPixels?.() ?? []);
				const lines = adapter.lcdText?.() ?? [];
				const keySequence =
					typeof input.keySequence === 'string'
						? input.keySequence
								.split(/[;,]/)
								.map((part) => part.trim())
								.filter(Boolean)
						: input.keySequence;
				return {
					schema: 'iq7000-screen-proof/v1',
					label: input.label,
					generated_at: input.includeGeneratedAt === false ? undefined : new Date().toISOString(),
					context: input.context,
					key_sequence: keySequence,
					assertions: input.assertions,
					emulator: {
						final_pc: `0x${adapter.getReg('PC').toString(16).toUpperCase().padStart(5, '0')}`,
					},
					lcd: {
						text: lines,
						pixels: {
							width: IQ7000_LCD_WIDTH,
							height: IQ7000_LCD_HEIGHT,
							count: pixels.length,
							signature: pixelSignature(pixels),
						},
					},
					artifacts: input.artifacts,
				};
			},
			metadataYaml: async (input = {}) => {
				const metadata = await api.proof.metadata(input);
				return `${toYaml(metadata)}\n`;
			},
		},
		perfetto: {
			trace: async <T>(name: string, body: () => Promise<T> | T): Promise<T> => {
				const trimmed = typeof name === 'string' ? name.trim() : '';
				if (!trimmed) throw new Error('perfetto.trace(name, ...) requires a non-empty name');
				if (perfettoActive) {
					throw new Error('Nested perfetto.trace(...) calls are unsupported');
				}
				if (typeof adapter.startPerfettoTrace !== 'function' || typeof adapter.stopPerfettoTrace !== 'function') {
					throw new Error('Perfetto tracing is not available in this runtime.');
				}

				perfettoActive = true;
				perfettoActiveName = trimmed;
				let started = false;
				let stopAttempted = false;

				try {
					await Promise.resolve(adapter.startPerfettoTrace(trimmed));
					started = true;
					const result = await body();
					stopAttempted = true;
					const perfettoTraceB64 = await Promise.resolve(adapter.stopPerfettoTrace());
					const trace: PerfettoTraceHandle = {
						index: traceIndex++,
						name: trimmed,
						byteLength: base64ByteLength(perfettoTraceB64),
						perfettoTraceB64,
					};
					perfettoTraces.push(trace);
					events.push({ kind: 'perfetto_trace', sequence: sequence++, trace });
					return result;
				} catch (err) {
					if (started && !stopAttempted) {
						try {
							stopAttempted = true;
							const perfettoTraceB64 = await Promise.resolve(adapter.stopPerfettoTrace());
							const trace: PerfettoTraceHandle = {
								index: traceIndex++,
								name: trimmed,
								byteLength: base64ByteLength(perfettoTraceB64),
								perfettoTraceB64,
							};
							perfettoTraces.push(trace);
							events.push({ kind: 'perfetto_trace', sequence: sequence++, trace });
						} catch {
							// ignore secondary stop errors
						}
					}
					throw attachPerfettoHint(err);
				} finally {
					perfettoActive = false;
					perfettoActiveName = null;
				}
			},
		},
		iocs: {
			putc: async (ch: string | number, options?: EvalIocsCallOptions) => {
				const byte = resolvePutcByte(ch);
				await writeIocsImemArgs(options);
				return await api.call(
					IOCS_PUBLIC_ENTRY_ADDR,
					{ IL: IOCS.LCD_PUTC, A: byte },
					options ? { ...options, zeroMissing: false } : { zeroMissing: false },
				);
			},
			text: async (text: string, options?: EvalIocsCallOptions) => {
				const out: CallHandle[] = [];
				for (const ch of text) out.push(await api.iocs.putc(ch, options));
				return out;
			},
			putcXY: async (ch: string | number, options: EvalIocsDisplayPutcOptions) => {
				const byte = resolvePutcByte(ch);
				await writeIocsImemArgs(options);
				return await api.call(
					IOCS_PUBLIC_ENTRY_ADDR,
					{ I: IOCS.DISPLAY_PUTCHAR_XY, A: byte },
					{ ...options, zeroMissing: false },
				);
			},
		},
	};

	return api;
}
