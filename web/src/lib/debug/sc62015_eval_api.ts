import { buildMemoryWriteBlocks, type MemoryWriteBlock, type MemoryWriteEvent } from './memory_write_blocks';

export type RegisterName =
	| 'A'
	| 'B'
	| 'BA'
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
	report: { reason: string; steps: number; pc: number; sp: number; halted: boolean };
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
		result: CallArtifacts['report'];
		infoLog: string[];
	};
};

export type PrintEntry = { index: number; value: unknown };

export type EvalEvent =
	| { kind: 'call'; sequence: number; handle: CallHandle }
	| { kind: 'print'; sequence: number; entry: PrintEntry }
	| { kind: 'error'; sequence: number; message: string };

export type EvalCallOptions = {
	maxInstructions?: number;
	zeroMissing?: boolean;
};

export type EvalApiOptions = {};

export interface EmulatorAdapter {
	callFunction(address: number, maxInstructions: number): Promise<CallArtifacts>;
	getReg(name: string): number;
	setReg(name: string, value: number): void;
	read8(addr: number): number;
	write8(addr: number, value: number): void;
}

export interface EvalApi {
	readonly calls: CallHandle[];
	readonly prints: PrintEntry[];
	readonly events: EvalEvent[];
	last(): CallHandle | null;
	call(
		reference: string | number,
		registers?: Partial<Record<RegisterName, number>>,
		options?: EvalCallOptions
	): Promise<CallHandle>;
	reg(name: RegisterName): number;
	flag(flag: StatusFlag): boolean;
	assert(condition: unknown, message?: string): void;
	print(...items: unknown[]): void;
	memory: {
		read(address: number, size?: 1 | 2 | 3): Promise<number>;
		write(address: number, size: 1 | 2 | 3, value: number): Promise<void>;
	};
}

const DEFAULT_MAX_INSTRUCTIONS = 200_000;

const RESULT_REGISTER_ORDER: RegisterName[] = [
	'A',
	'B',
	'BA',
	'I',
	'X',
	'Y',
	'U',
	'S',
	'F',
	'IMR',
	'PC'
];

function normalizeAddress(addr: number): number {
	return (addr >>> 0) & 0x00ff_ffff;
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
	if (name === 'A' || name === 'B' || name === 'F' || name === 'IMR') return value & 0xff;
	if (name === 'BA' || name === 'I') return value & 0xffff;
	if (name === 'PC') return value & 0x0f_ffff;
	return value >>> 0;
}

function buildAssignments(
	provided: Partial<Record<RegisterName, number>> | undefined,
	zeroMissing: boolean
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
	RESULT_REGISTER_ORDER.reduce((acc, name) => {
		(acc as any)[name] = name;
		return acc;
	}, {} as Record<RegisterName, RegisterName>)
);

export const Flag = Object.freeze({
	C: 'C',
	Z: 'Z'
} satisfies Record<StatusFlag, StatusFlag>);

export function createEvalApi(adapter: EmulatorAdapter, _options?: EvalApiOptions): EvalApi {
	const calls: CallHandle[] = [];
	const prints: PrintEntry[] = [];
	const events: EvalEvent[] = [];
	let sequence = 0;
	let callIndex = 0;

	const api: EvalApi = {
		calls,
		prints,
		events,
		last: () => (calls.length ? calls[calls.length - 1] : null),
		call: async (reference, registers, options) => {
			const { address, name } = resolveReference(reference);
			const maxInstructions = options?.maxInstructions ?? DEFAULT_MAX_INSTRUCTIONS;
			const zeroMissing = options?.zeroMissing ?? true;
			const assignments = buildAssignments(registers, zeroMissing);
			for (const [regName, value] of assignments.entries()) {
				adapter.setReg(regName, value);
			}

			const artifacts = await adapter.callFunction(address, maxInstructions);

			const memoryEvents: MemoryWriteEvent[] = artifacts.memory_writes.map((e) => ({
				addr: e.addr >>> 0,
				value: e.value & 0xff,
				size: 1
			}));
			const memoryBlocks = buildMemoryWriteBlocks(memoryEvents);
			const before = artifacts.before_regs;
			const after = artifacts.after_regs;
			const changed = diffRegisters(before, after);

			const infoLog: string[] = [
				`Execution reason: ${artifacts.report.reason}`,
				memoryEvents.length
					? `Captured ${memoryEvents.length} memory write byte(s) (${memoryBlocks.length} block(s)).`
					: 'No memory writes captured.',
				artifacts.lcd_writes.length
					? `Captured ${artifacts.lcd_writes.length} LCD addressing-unit write(s).`
					: 'No LCD writes captured.'
			];

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
					result: artifacts.report,
					infoLog
				}
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
		memory: {
			read: async (address, size = 1) => {
				const addr = normalizeAddress(address);
				if (size === 1) return adapter.read8(addr);
				if (size === 2) return (adapter.read8(addr) | (adapter.read8(addr + 1) << 8)) >>> 0;
				if (size === 3)
					return (
						adapter.read8(addr) |
						(adapter.read8(addr + 1) << 8) |
						(adapter.read8(addr + 2) << 16)
					) >>> 0;
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
			}
		}
	};

	return api;
}

