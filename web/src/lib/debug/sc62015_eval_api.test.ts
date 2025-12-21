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
						{ addr: 0x0011, value: 0xcc },
					],
					lcd_writes: [],
					probe_samples: [],
					perfetto_trace_b64: 'ZHVtbXk=',
					report: { reason: 'returned', steps: 3, pc: 0, sp: 0, halted: false, fault: null },
				};
			},
			reset: () => {},
			step: () => {},
			getReg: (_name: string) => 0,
			setReg: (name: string, value: number) => regWrites.push({ name, value }),
			read8: (_addr: number) => 0,
			write8: (_addr: number, _value: number) => {},
		};

		const api = createEvalApi(adapter as any);
		const handle = await api.call('0x00012345', { A: 0x12 }, { maxInstructions: 7, zeroMissing: false, trace: true });

		expect(regWrites).toEqual([{ name: 'A', value: 0x12 }]);
		expect(calls).toEqual([
			{
				address: 0x00012345,
				maxInstructions: 7,
				options: { trace: true },
			},
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
			write8: () => {},
		};
		const api = createEvalApi(adapter as any);
		await api.reset({ fresh: true, warmupTicks: 123 });
		expect(ops).toEqual(['reset', 'step:123']);
		expect(api.events[0]?.kind).toBe('reset');
	});

	it('step forwards to adapter.step', async () => {
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
			write8: () => {},
		};
		const api = createEvalApi(adapter as any);
		await api.step(42);
		expect(ops).toEqual(['step:42']);
	});

	it('lcd.textString returns joined decoded lines', async () => {
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: async () => {},
			step: async () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
			lcdText: () => ['A', 'B'],
		};
		const api = createEvalApi(adapter as any);
		expect(await api.lcd.text()).toEqual(['A', 'B']);
		expect(await api.lcd.textString()).toEqual('A\nB');
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
						{ pc: 0x123, count: 2, regs: { A: 2 } },
					],
					perfetto_trace_b64: null,
					report: { reason: 'returned', steps: 1, pc: 0, sp: 0, halted: false, fault: null },
				};
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
		};
		const api = createEvalApi(adapter as any);
		await api.withProbe(
			0x123,
			(s) => probeHits.push(s.count),
			async () => {
				await api.call(0x10);
			},
		);
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
			injectMatrixEvent: (code: number, release: boolean) => ops.push(`inject:${code}:${release ? 1 : 0}`),
		};
		const api = createEvalApi(adapter as any);
		await api.keyboard.tap(0x56, 5);
		expect(ops).toEqual(['inject:86:0', 'step:5', 'inject:86:1']);
	});

	it('iocs.putc writes IMEM byte registers when provided', async () => {
		const writes: Array<{ addr: number; value: number }> = [];
		const regWrites: Array<{ name: string; value: number }> = [];
		const calls: Array<{ address: number }> = [];
		const adapter = {
			callFunction: async (address: number, _max: number, _options?: any) => {
				calls.push({ address });
				return {
					address,
					before_pc: 0,
					after_pc: 0,
					before_sp: 0,
					after_sp: 0,
					before_regs: {},
					after_regs: {},
					memory_writes: [],
					lcd_writes: [],
					probe_samples: [],
					perfetto_trace_b64: null,
					report: { reason: 'returned', steps: 1, pc: 0, sp: 0, halted: false, fault: null },
				};
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: (name: string, value: number) => regWrites.push({ name, value }),
			read8: () => 0,
			write8: (addr: number, value: number) => writes.push({ addr, value }),
		};
		const api = createEvalApi(adapter as any);
		await api.iocs.putc('A', { bl: 1, bh: 2, cl: 3, ch: 4 });
		expect(writes).toEqual([
			{ addr: 0x00100000 + 0xd4, value: 1 },
			{ addr: 0x00100000 + 0xd5, value: 2 },
			{ addr: 0x00100000 + 0xd6, value: 3 },
			{ addr: 0x00100000 + 0xd7, value: 4 },
		]);
		expect(regWrites).toContainEqual({ name: 'IL', value: 0x0d });
		expect(regWrites).toContainEqual({ name: 'A', value: 0x41 });
		expect(calls[0]?.address).toBe(0x00fffe8);
	});

	it('iocs.putcXY calls display IOCS 0x0041 and writes IMEM bl/bh', async () => {
		const writes: Array<{ addr: number; value: number }> = [];
		const regWrites: Array<{ name: string; value: number }> = [];
		const calls: Array<{ address: number }> = [];
		const adapter = {
			callFunction: async (address: number, _max: number, _options?: any) => {
				calls.push({ address });
				return {
					address,
					before_pc: 0,
					after_pc: 0,
					before_sp: 0,
					after_sp: 0,
					before_regs: {},
					after_regs: {},
					memory_writes: [],
					lcd_writes: [],
					probe_samples: [],
					perfetto_trace_b64: null,
					report: { reason: 'returned', steps: 1, pc: 0, sp: 0, halted: false, fault: null },
				};
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: (name: string, value: number) => regWrites.push({ name, value }),
			read8: () => 0,
			write8: (addr: number, value: number) => writes.push({ addr, value }),
		};
		const api = createEvalApi(adapter as any);
		await api.iocs.putcXY('Z', { bl: 9, bh: 8, cl: 0, ch: 0 });
		expect(writes).toEqual([
			{ addr: 0x00100000 + 0xd4, value: 9 },
			{ addr: 0x00100000 + 0xd5, value: 8 },
			{ addr: 0x00100000 + 0xd6, value: 0 },
			{ addr: 0x00100000 + 0xd7, value: 0 },
		]);
		expect(regWrites).toContainEqual({ name: 'I', value: 0x0041 });
		expect(regWrites).toContainEqual({ name: 'A', value: 0x5a });
		expect(calls[0]?.address).toBe(0x00fffe8);
	});

	it('perfetto.trace starts/stops capture and records an event', async () => {
		const ops: string[] = [];
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			startPerfettoTrace: (name: string) => ops.push(`start:${name}`),
			stopPerfettoTrace: () => {
				ops.push('stop');
				return 'ZHVtbXk=';
			},
			reset: () => {},
			step: (n: number) => ops.push(`step:${n}`),
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
		};

		const api = createEvalApi(adapter as any);
		const result = await api.perfetto.trace('boot', async () => {
			await api.step(10);
			return 123;
		});

		expect(result).toBe(123);
		expect(ops).toEqual(['start:boot', 'step:10', 'stop']);
		expect(api.events.some((e) => e.kind === 'perfetto_trace')).toBe(true);
	});

	it('perfetto.trace rejects nested capture', async () => {
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			startPerfettoTrace: (_name: string) => {},
			stopPerfettoTrace: () => 'ZHVtbXk=',
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
		};

		const api = createEvalApi(adapter as any);
		await expect(
			api.perfetto.trace('outer', async () => {
				await api.perfetto.trace('inner', async () => {});
			}),
		).rejects.toThrow(/nested/i);
	});

	it('perfetto.trace rejects per-call trace inside an active capture', async () => {
		const adapter = {
			callFunction: async () => {
				throw new Error('call should be blocked before reaching adapter');
			},
			startPerfettoTrace: (_name: string) => {},
			stopPerfettoTrace: () => 'ZHVtbXk=',
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
		};

		const api = createEvalApi(adapter as any);
		await expect(
			api.perfetto.trace('outer', async () => {
				await api.call(0x10, undefined, { trace: true });
			}),
		).rejects.toThrow(/nested tracing/i);
	});
});
