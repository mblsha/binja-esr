export interface MemoryWriteEvent {
	addr: number;
	value: number;
	size?: 1 | 2 | 3 | 4;
}

export interface MemoryWriteBlock {
	start: number;
	lines: string[];
	byteCount: number;
}

function normalizeAddress(addr: number): number {
	return (addr & 0x00ff_ffff) >>> 0;
}

function eventBytes(event: MemoryWriteEvent): Array<{ addr: number; value: number }> {
	const size = (event.size ?? 1) as 1 | 2 | 3 | 4;
	const baseAddr = normalizeAddress(event.addr);
	const mask = size === 1 ? 0xff : size === 2 ? 0xffff : size === 3 ? 0xffffff : 0xffffffff;
	const value = event.value & mask;
	const bytes: Array<{ addr: number; value: number }> = [];
	for (let i = 0; i < size; i++) {
		const shift = (size - 1 - i) * 8;
		const byte = (value >>> shift) & 0xff;
		bytes.push({
			addr: normalizeAddress(baseAddr + i),
			value: byte
		});
	}
	return bytes;
}

type BlockFormatOptions = {
	groupSize: number;
	lineFormatter: (lineAddr: number, bytes: number[]) => string;
};

function defaultLineFormatter(_addr: number, bytes: number[]): string {
	return bytes
		.map((b) => b.toString(16).padStart(2, '0').toUpperCase())
		.join(' ');
}

function makeBlock(start: number, bytes: number[], options: BlockFormatOptions): MemoryWriteBlock {
	const lines: string[] = [];
	for (let i = 0; i < bytes.length; i += options.groupSize) {
		const slice = bytes.slice(i, i + options.groupSize);
		const lineAddr = normalizeAddress(start + i);
		lines.push(options.lineFormatter(lineAddr, slice));
	}
	return {
		start,
		lines,
		byteCount: bytes.length
	};
}

export type MemoryWriteBlockOptions = {
	ignoreRanges?: Array<{ start: number; end: number }>;
	groupSize?: number;
	lineFormatter?: (lineAddr: number, bytes: number[]) => string;
};

function normalizeRange(range: { start: number; end: number }): { start: number; end: number } | null {
	const start = normalizeAddress(range.start);
	const end = normalizeAddress(range.end);
	if (end < start) return null;
	return { start, end };
}

function shouldIgnore(
	addr: number,
	ignoreRanges: Array<{ start: number; end: number }> | undefined
): boolean {
	if (!ignoreRanges || ignoreRanges.length === 0) return false;
	for (const range of ignoreRanges) {
		if (addr >= range.start && addr <= range.end) {
			return true;
		}
	}
	return false;
}

// Reduce a write stream down to "last value per byte address", then render contiguous spans.
export function buildMemoryWriteBlocks(
	events: MemoryWriteEvent[],
	options?: MemoryWriteBlockOptions
): MemoryWriteBlock[] {
	if (!events.length) return [];

	const normalizedRanges =
		options?.ignoreRanges
			?.map((range) => normalizeRange(range))
			.filter((range): range is { start: number; end: number } => range !== null) ?? [];

	const formatOptions: BlockFormatOptions = {
		groupSize: Math.max(1, options?.groupSize ?? 16),
		lineFormatter: options?.lineFormatter ?? defaultLineFormatter
	};

	const finalBytes = new Map<number, number>();
	for (const event of events) {
		for (const { addr, value } of eventBytes(event)) {
			if (shouldIgnore(addr, normalizedRanges)) continue;
			finalBytes.set(addr, value);
		}
	}

	if (finalBytes.size === 0) return [];

	const sorted = Array.from(finalBytes.entries()).sort(([a], [b]) => a - b);

	const blocks: MemoryWriteBlock[] = [];
	let currentStart = sorted[0][0];
	let currentBytes: number[] = [];
	let prevAddr: number | null = null;

	for (const [addr, value] of sorted) {
		if (prevAddr === null || normalizeAddress(prevAddr + 1) !== addr) {
			if (currentBytes.length) blocks.push(makeBlock(currentStart, currentBytes, formatOptions));
			currentStart = addr;
			currentBytes = [value];
		} else {
			currentBytes.push(value);
		}
		prevAddr = addr;
	}

	if (currentBytes.length) blocks.push(makeBlock(currentStart, currentBytes, formatOptions));
	return blocks;
}

export function formatAddress(value: number): string {
	return '0x' + (value >>> 0).toString(16).padStart(6, '0').toUpperCase();
}

