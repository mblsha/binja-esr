export const PCE500_IOCS_WORKSPACE_PTR_ADDR = 0x00bfd17;
export const PCE500_KEY_FIFO_BASE_OFFSET = 0x02;
export const PCE500_KEY_FIFO_TAIL_OFFSET = 0x04;
export const PCE500_KEY_FIFO_HEAD_OFFSET = 0x05;
export const PCE500_KEY_FIFO_CAPACITY = 0x10;

export type Pce500KeyboardFifoAddresses = {
	workspaceBase: number;
	fifoBase: number;
	fifoTail: number;
	fifoHead: number;
};

type ReadU8 = (address: number) => number | null | undefined;

function readByte(readU8: ReadU8, address: number): number | null {
	const value = readU8(address);
	return typeof value === 'number' ? value & 0xff : null;
}

/** Resolve the relocatable PC-E500 keyboard FIFO through [BFD17]/(E6). */
export function resolvePce500KeyboardFifo(readU8: ReadU8): Pce500KeyboardFifoAddresses | null {
	const base0 = readByte(readU8, PCE500_IOCS_WORKSPACE_PTR_ADDR);
	const base1 = readByte(readU8, PCE500_IOCS_WORKSPACE_PTR_ADDR + 1);
	const base2 = readByte(readU8, PCE500_IOCS_WORKSPACE_PTR_ADDR + 2);
	if (base0 === null || base1 === null || base2 === null) return null;

	const workspaceBase = base0 | (base1 << 8) | (base2 << 16);
	if (workspaceBase === 0) return null;
	const offset0 = readByte(readU8, workspaceBase + PCE500_KEY_FIFO_BASE_OFFSET);
	const offset1 = readByte(readU8, workspaceBase + PCE500_KEY_FIFO_BASE_OFFSET + 1);
	if (offset0 === null || offset1 === null) return null;

	const fifoOffset = offset0 | (offset1 << 8);
	return {
		workspaceBase,
		fifoBase: workspaceBase + fifoOffset,
		fifoTail: workspaceBase + PCE500_KEY_FIFO_TAIL_OFFSET,
		fifoHead: workspaceBase + PCE500_KEY_FIFO_HEAD_OFFSET,
	};
}
