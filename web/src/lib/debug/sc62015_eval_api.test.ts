import { describe, expect, it } from 'vitest';

import { createEvalApi } from './sc62015_eval_api';

describe('createEvalApi', () => {
	it('calls adapter.callFunction and builds last-value memory blocks', async () => {
		const regWrites: Array<{ name: string; value: number }> = [];
		const calls: Array<{ address: number; maxInstructions: number; options: any }> = [];
		const adapter = {
			callFunction: async (address: number, maxInstructions: number, options?: any) => {
				calls.push({ address, maxInstructions, options });
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
					probe_samples: [],
					trace_events: [],
					trace_truncated: false,
					report: { reason: 'returned', steps: 3, pc: 0, sp: 0, halted: false, fault: null }
				};
			},
			reset: () => {},
			step: () => {},
			getReg: (_name: string) => 0,
			setReg: (name: string, value: number) => regWrites.push({ name, value }),
			read8: (_addr: number) => 0,
			write8: (_addr: number, _value: number) => {}
		};

		const api = createEvalApi(adapter as any);
		const handle = await api.call(
			'0x00012345',
			{ A: 0x12 },
			{ maxInstructions: 7, zeroMissing: false, trace: true }
		);

		expect(regWrites).toEqual([{ name: 'A', value: 0x12 }]);
		expect(calls).toEqual([
			{
				address: 0x00012345,
				maxInstructions: 7,
				options: { trace: true }
			}
		]);
		expect(handle.address).toBe(0x00012345);
		expect(handle.artifacts.changed).toContain('A');
		expect(handle.artifacts.memoryBlocks).toHaveLength(1);
		expect(handle.artifacts.memoryBlocks[0].start).toBe(0x0010);
		expect(handle.artifacts.memoryBlocks[0].lines.join('\n')).toContain('BB CC');
		expect(api.events.find((ev) => ev.kind === 'call')).toBeTruthy();
	});

	it('reset calls adapter.reset and warms up via step', async () => {
		const ops: string[] = [];
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: async () => ops.push('reset'),
			step: async (n: number) => ops.push(`step:${n}`),
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {}
		};
		const api = createEvalApi(adapter as any);
		await api.reset({ fresh: true, warmupTicks: 123 });
		expect(ops).toEqual(['reset', 'step:123']);
		expect(api.events[0]?.kind).toBe('reset');
	});

	it('withProbe forwards probe to callFunction and invokes handler for returned samples', async () => {
		const probeHits: number[] = [];
		const capturedOptions: any[] = [];
		const adapter = {
			callFunction: async (_address: number, _max: number, options?: any) => {
				capturedOptions.push(options);
				return {
					address: 0x10,
					before_pc: 0,
					after_pc: 0,
					before_sp: 0,
					after_sp: 0,
					before_regs: {},
					after_regs: {},
					memory_writes: [],
					lcd_writes: [],
					probe_samples: [
						{ pc: 0x123, count: 1, regs: { A: 1 } },
						{ pc: 0x123, count: 2, regs: { A: 2 } }
					],
					trace_events: [],
					trace_truncated: false,
					report: { reason: 'returned', steps: 1, pc: 0, sp: 0, halted: false, fault: null }
				};
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {}
		};
		const api = createEvalApi(adapter as any);
		await api.withProbe(0x123, (s) => probeHits.push(s.count), async () => {
			await api.call(0x10);
		});
		expect(probeHits).toEqual([1, 2]);
		expect(capturedOptions[0]?.probe?.pc).toBe(0x123);
	});

	it('keyboard helpers forward to adapter', async () => {
		const ops: string[] = [];
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: () => {},
			step: async (n: number) => ops.push(`step:${n}`),
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
			injectMatrixEvent: (code: number, release: boolean) => ops.push(`inject:${code}:${release ? 1 : 0}`)
		};
		const api = createEvalApi(adapter as any);
		await api.keyboard.tap(0x56, 5);
		expect(ops).toEqual(['inject:86:0', 'step:5', 'inject:86:1']);
	});
});
