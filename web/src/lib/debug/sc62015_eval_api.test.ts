import { describe, expect, it } from 'vitest';

import { createEvalApi } from './sc62015_eval_api';

describe('createEvalApi', () => {
	it('calls adapter.callFunction and builds last-value memory blocks', async () => {
		const regWrites: Array<{ name: string; value: number }> = [];
		const adapter = {
			callFunction: async (address: number, maxInstructions: number) => {
				expect(address).toBe(0x00012345);
				expect(maxInstructions).toBe(7);
				return {
					address,
					before_pc: 0,
					after_pc: 0,
					before_sp: 0,
					after_sp: 0,
					before_regs: { A: 0, B: 0, PC: address },
					after_regs: { A: 1, B: 0, PC: address },
					memory_writes: [
						{ addr: 0x0010, value: 0xaa },
						{ addr: 0x0010, value: 0xbb },
						{ addr: 0x0011, value: 0xcc }
					],
					lcd_writes: [],
					report: { reason: 'returned', steps: 3, pc: 0, sp: 0, halted: false }
				};
			},
			getReg: (_name: string) => 0,
			setReg: (name: string, value: number) => regWrites.push({ name, value }),
			read8: (_addr: number) => 0,
			write8: (_addr: number, _value: number) => {}
		};

		const api = createEvalApi(adapter as any);
		const handle = await api.call('0x00012345', { A: 0x12 }, { maxInstructions: 7, zeroMissing: false });

		expect(regWrites).toEqual([{ name: 'A', value: 0x12 }]);
		expect(handle.address).toBe(0x00012345);
		expect(handle.artifacts.changed).toContain('A');
		expect(handle.artifacts.memoryBlocks).toHaveLength(1);
		expect(handle.artifacts.memoryBlocks[0].start).toBe(0x0010);
		expect(handle.artifacts.memoryBlocks[0].lines.join('\n')).toContain('BB CC');
		expect(api.events.find((ev) => ev.kind === 'call')).toBeTruthy();
	});
});

