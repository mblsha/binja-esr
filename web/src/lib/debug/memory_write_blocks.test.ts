import { describe, expect, it } from 'vitest';

import { buildMemoryWriteBlocks } from './memory_write_blocks';

describe('buildMemoryWriteBlocks', () => {
	it('reduces to last value per address and merges contiguous bytes', () => {
		const blocks = buildMemoryWriteBlocks([
			{ addr: 0x0010, value: 0x11 },
			{ addr: 0x0010, value: 0x22 },
			{ addr: 0x0011, value: 0x33 },
		]);

		expect(blocks).toHaveLength(1);
		expect(blocks[0].start).toBe(0x0010);
		expect(blocks[0].byteCount).toBe(2);
		expect(blocks[0].lines.join('\n')).toContain('22 33');
	});

	it('expands multi-byte writes into per-byte entries', () => {
		const blocks = buildMemoryWriteBlocks([{ addr: 0x0100, value: 0x1122, size: 2 }], {
			groupSize: 16,
		});
		expect(blocks).toHaveLength(1);
		expect(blocks[0].start).toBe(0x0100);
		expect(blocks[0].byteCount).toBe(2);
		expect(blocks[0].lines.join('\n')).toContain('11 22');
	});
});
