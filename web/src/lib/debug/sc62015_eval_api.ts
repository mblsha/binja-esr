import { IOCS, IOCS_PUBLIC_ENTRY_ADDR } from './iocs';
import { buildMemoryWriteBlocks, type MemoryWriteBlock, type MemoryWriteEvent } from './memory_write_blocks';

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

export type EvalApiOptions = {};

export interface EmulatorAdapter {
	callFunction(
		address: number,
		maxInstructions: number,
		options?: {
			trace?: boolean;
			probe?: { pc: number; maxSamples?: number };
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
	pressMatrixCode?(code: number): void;
	releaseMatrixCode?(code: number): void;
	injectMatrixEvent?(code: number, release: boolean): void;
	pressOnKey?(): void;
	releaseOnKey?(): void;
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
	};
	keyboard: {
		press(code: number): Promise<void>;
		release(code: number): Promise<void>;
		tap(code: number, holdInstructions?: number): Promise<void>;
		injectEvent(code: number, release: boolean): Promise<void>;
	};
	onKey: {
		press(): Promise<void>;
		release(): Promise<void>;
		tap(holdInstructions?: number): Promise<void>;
	};
	perfetto: {
		trace<T>(name: string, body: () => Promise<T> | T): Promise<T>;
	};
	iocs: {
		putc(ch: string | number, options?: EvalIocsCallOptions): Promise<CallHandle>;
		text(text: string, options?: EvalIocsCallOptions): Promise<CallHandle[]>;
		putcXY(ch: string | number, options: EvalIocsDisplayPutcOptions): Promise<CallHandle>;
	};
}

const DEFAULT_MAX_INSTRUCTIONS = 200_000;
const DEFAULT_WARMUP_TICKS = 0;
const DEFAULT_VIRTUAL_HOLD_INSTRUCTIONS = 40_000;
const DEFAULT_PROBE_MAX_SAMPLES = 256;
const IMEM_BASE_ADDR = 0x0010_0000;

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
			const artifacts = await adapter.callFunction(
				address,
				maxInstructions,
				activeProbe
					? {
							trace,
							probe: { pc: activeProbe.pc, maxSamples: activeProbe.maxSamples },
						}
					: { trace },
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
				adapter.injectMatrixEvent?.(code & 0xff, false);
				if (holdInstructions > 0) await Promise.resolve(adapter.step(holdInstructions));
				adapter.injectMatrixEvent?.(code & 0xff, true);
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
					throw err;
				} finally {
					perfettoActive = false;
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
