import type { StubPatch, StubRegistration, StubReturn, StubMemory } from './sc62015_stub_types';

type StubRegEntry = { name: string; value: number };
type StubWrite = { addr: number; value: number; size: number };

type StubDispatcherOptions = {
	wasmMemory: WebAssembly.Memory;
	externalPtr: number;
	externalLen: number;
	internalPtr: number;
	internalLen: number;
};

export type StubDispatcher = {
	registerStub(stub: StubRegistration): void;
	clearStubs(): void;
};

const ADDRESS_MASK = 0x00ff_ffff;
const INTERNAL_BASE = 0x100000;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeAddr(value: unknown): number | null {
	if (typeof value === 'number' && Number.isFinite(value)) return value >>> 0;
	if (typeof value === 'string') {
		const trimmed = value.trim();
		if (!trimmed) return null;
		const parsed = trimmed.startsWith('0x') ? Number.parseInt(trimmed.slice(2), 16) : Number.parseInt(trimmed, 10);
		return Number.isFinite(parsed) ? parsed >>> 0 : null;
	}
	return null;
}

function normalizeEntryList(value: unknown): StubRegEntry[] {
	if (!value) return [];
	if (Array.isArray(value)) {
		return value
			.map((entry) => {
				if (isRecord(entry) && typeof entry.name === 'string' && typeof entry.value === 'number') {
					return { name: entry.name, value: entry.value };
				}
				return null;
			})
			.filter((entry): entry is StubRegEntry => Boolean(entry));
	}
	if (isRecord(value)) {
		return Object.entries(value)
			.map(([name, rawValue]) =>
				typeof rawValue === 'number' && Number.isFinite(rawValue) ? { name, value: rawValue } : null,
			)
			.filter((entry): entry is StubRegEntry => Boolean(entry));
	}
	return [];
}

function normalizeWrites(value: unknown): StubWrite[] {
	if (!value) return [];
	if (Array.isArray(value)) {
		return value
			.map((entry) => {
				if (!isRecord(entry)) return null;
				const addr = normalizeAddr(entry.addr);
				const rawValue = typeof entry.value === 'number' && Number.isFinite(entry.value) ? entry.value : null;
				if (addr === null || rawValue === null) return null;
				const sizeRaw = typeof entry.size === 'number' ? Math.trunc(entry.size) : 1;
				const size = sizeRaw === 2 || sizeRaw === 3 ? sizeRaw : 1;
				return { addr, value: rawValue, size };
			})
			.filter((entry): entry is StubWrite => Boolean(entry));
	}
	if (isRecord(value)) {
		return Object.entries(value)
			.map(([addrKey, rawValue]) => {
				const addr = normalizeAddr(addrKey);
				const value = typeof rawValue === 'number' && Number.isFinite(rawValue) ? rawValue : null;
				if (addr === null || value === null) return null;
				return { addr, value, size: 1 };
			})
			.filter((entry): entry is StubWrite => Boolean(entry));
	}
	return [];
}

function makeMemReader(getViews: () => { external: Uint8Array; internal: Uint8Array }): StubMemory {
	const read8 = (addr: number): number => {
		const { external, internal } = getViews();
		const masked = (addr >>> 0) & ADDRESS_MASK;
		if (masked >= INTERNAL_BASE) {
			const idx = masked - INTERNAL_BASE;
			if (idx >= internal.length) {
				throw new Error(`stub memory read out of range: 0x${masked.toString(16)}`);
			}
			return internal[idx] ?? 0;
		}
		if (!external.length) return 0;
		const idx = masked % external.length;
		return external[idx] ?? 0;
	};
	return {
		read8,
		read16: (addr) => read8(addr) | (read8(addr + 1) << 8),
		read24: (addr) => read8(addr) | (read8(addr + 1) << 8) | (read8(addr + 2) << 16),
	};
}

export function createStubDispatcher(options: StubDispatcherOptions): StubDispatcher {
	const registry = new Map<number, StubRegistration>();
	let cachedBuffer: ArrayBuffer | null = null;
	let externalView: Uint8Array | null = null;
	let internalView: Uint8Array | null = null;

	const getViews = () => {
		const buffer = options.wasmMemory.buffer;
		if (!buffer || buffer.byteLength === 0) {
			throw new Error('stub memory is unavailable (missing wasm buffer)');
		}
		if (buffer !== cachedBuffer) {
			cachedBuffer = buffer;
			externalView = new Uint8Array(buffer, options.externalPtr, options.externalLen);
			internalView = new Uint8Array(buffer, options.internalPtr, options.internalLen);
		}
		if (!externalView || !internalView) {
			throw new Error('stub memory views are unavailable');
		}
		return { external: externalView, internal: internalView };
	};

	const mem = makeMemReader(getViews);

	const dispatch = (stubId: number, regsEntries: unknown, flagEntries: unknown) => {
		const stub = registry.get(stubId);
		if (!stub) throw new Error(`stub ${stubId} is not registered`);
		const regs = Object.fromEntries(normalizeEntryList(regsEntries).map((entry) => [entry.name, entry.value]));
		const flags = Object.fromEntries(normalizeEntryList(flagEntries).map((entry) => [entry.name, entry.value])) as {
			C: number;
			Z: number;
		};
		let patch: StubPatch | void;
		try {
			patch = stub.handler(mem, regs, flags);
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			throw new Error(`stub ${stub.name ?? stub.id} failed: ${msg}`);
		}
		const normalizedRegs = normalizeEntryList(patch && isRecord(patch) ? patch.regs : null);
		const normalizedFlags = normalizeEntryList(patch && isRecord(patch) ? patch.flags : null);
		const normalizedWrites = normalizeWrites(patch && isRecord(patch) ? patch.mem_writes : null);
		const ret = patch && isRecord(patch) && 'ret' in patch ? (patch.ret as StubReturn | undefined) : undefined;
		return {
			mem_writes: normalizedWrites,
			regs: normalizedRegs,
			flags: normalizedFlags,
			ret,
		};
	};

	(globalThis as any).__sc62015_stub_dispatch = dispatch;

	return {
		registerStub: (stub) => {
			registry.set(stub.id, stub);
		},
		clearStubs: () => {
			registry.clear();
		},
	};
}
