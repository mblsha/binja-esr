import { describe, expect, it } from 'vitest';
import { resolvePce500KeyboardFifo } from './pce500_iocs_workspace';

function reader(entries: Array<[number, number]>) {
	const memory = new Map(entries);
	return (address: number) => memory.get(address);
}

describe('resolvePce500KeyboardFifo', () => {
	it('resolves the English-ROM cold-start mapping', () => {
		const resolved = resolvePce500KeyboardFifo(
			reader([
				[0x00bfd17, 0xb4],
				[0x00bfd18, 0xf9],
				[0x00bfd19, 0x0b],
				[0x00bf9b6, 0x50],
				[0x00bf9b7, 0x00],
			]),
		);

		expect(resolved).toEqual({
			workspaceBase: 0x00bf9b4,
			fifoBase: 0x00bfa04,
			fifoTail: 0x00bf9b8,
			fifoHead: 0x00bf9b9,
		});
	});

	it('follows a relocated workspace and its stored FIFO offset', () => {
		const resolved = resolvePce500KeyboardFifo(
			reader([
				[0x00bfd17, 0x00],
				[0x00bfd18, 0xfa],
				[0x00bfd19, 0x0b],
				[0x00bfa02, 0x70],
				[0x00bfa03, 0x01],
			]),
		);

		expect(resolved).toEqual({
			workspaceBase: 0x00bfa00,
			fifoBase: 0x00bfb70,
			fifoTail: 0x00bfa04,
			fifoHead: 0x00bfa05,
		});
	});

	it('returns null when the workspace pointer is unavailable', () => {
		expect(resolvePce500KeyboardFifo(() => undefined)).toBeNull();
	});

	it('rejects an all-zero uninitialised workspace pointer', () => {
		expect(resolvePce500KeyboardFifo(() => 0)).toBeNull();
	});
});
