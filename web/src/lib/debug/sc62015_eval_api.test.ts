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
				options: { trace: true, stubs: [] },
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

	it('lcd.pixels returns a plain array for screenshots', async () => {
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
			lcdPixels: () => new Uint8Array([0, 1, 1, 0]),
		};
		const api = createEvalApi(adapter as any);
		expect(await api.lcd.pixels()).toEqual([0, 1, 1, 0]);
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

	it('stub registers with adapter and passes stubs into calls', async () => {
		const registered: any[] = [];
		const cleared: string[] = [];
		const callOptions: any[] = [];
		const adapter = {
			callFunction: async (_address: number, _max: number, options?: any) => {
				callOptions.push(options);
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
					probe_samples: [],
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
			registerStub: (stub: any) => registered.push(stub),
			clearStubs: () => cleared.push('ok'),
		};
		const api = createEvalApi(adapter as any);
		const handler = () => ({ regs: { A: 1 } });
		const stub = api.stub(0x1234, 'stub', handler as any);
		await api.call(0x10);
		api.clearStubs();

		expect(registered).toHaveLength(1);
		expect(registered[0].pc).toBe(0x1234);
		expect(callOptions[0].stubs).toEqual([{ id: stub.id, pc: 0x1234 }]);
		expect(cleared).toEqual(['ok']);
	});

	it('stub throws when adapter lacks stub support', () => {
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
		};
		const api = createEvalApi(adapter as any);
		expect(() => api.stub(0x1234, 'stub', () => ({}))).toThrow(/stub support/);
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

	it('explicit key namespaces distinguish event and physical input', async () => {
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
			injectMatrixEvent: (code: number, release: boolean) => ops.push(`event:${code}:${release ? 1 : 0}`),
			pressMatrixCode: (code: number) => ops.push(`phys-press:${code}`),
			releaseMatrixCode: (code: number) => ops.push(`phys-release:${code}`),
		};
		const api = createEvalApi(adapter as any);
		await api.keys.tap('app:calendar', 1);
		await api.keys.tap('event:0x0B', 2);
		await api.keys.tap('phys:0x56', 3);
		expect(ops).toEqual([
			'event:24:0',
			'step:1',
			'event:24:1',
			'event:11:0',
			'step:2',
			'event:11:1',
			'phys-press:86',
			'step:3',
			'phys-release:86',
		]);
	});

	it('wait.lcdStable waits for repeated nonblank LCD signatures', async () => {
		const frames = [
			[0, 0, 0, 0],
			[1, 0, 0, 0],
			[1, 1, 0, 0],
			[1, 1, 0, 0],
		];
		let frame = 0;
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: () => {},
			step: async () => {
				frame = Math.min(frame + 1, frames.length - 1);
			},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
			lcdPixels: () => frames[frame],
		};
		const api = createEvalApi(adapter as any);
		const result = await api.wait.lcdStable({ chunkInstructions: 10, quietSamples: 2, maxInstructions: 100 });
		expect(result.instructions).toBe(30);
	});

	it('calendar pixel assertion checks compact day-number grid', async () => {
		const width = 96;
		const height = 64;
		const pixels = new Array(width * height).fill(0);
		const digits: Record<string, string[]> = {
			'0': ['####', '#..#', '#..#', '#..#', '#..#', '####'],
			'1': ['...#', '...#', '...#', '...#', '...#', '...#'],
			'2': ['####', '...#', '####', '#...', '#...', '####'],
			'3': ['####', '...#', '####', '...#', '...#', '####'],
			'4': ['#..#', '#..#', '####', '...#', '...#', '...#'],
			'5': ['####', '#...', '####', '...#', '...#', '####'],
			'6': ['####', '#...', '####', '#..#', '#..#', '####'],
			'7': ['####', '#..#', '#..#', '...#', '...#', '...#'],
			'8': ['####', '#..#', '####', '#..#', '#..#', '####'],
			'9': ['####', '#..#', '####', '...#', '...#', '####'],
		};
		const onesX = [5, 19, 33, 47, 61, 75, 89];
		const rowY = [17, 25, 33, 41, 49, 57];
		function drawDigit(x: number, y: number, digit: string) {
			for (let row = 0; row < digits[digit].length; row++) {
				for (let col = 0; col < digits[digit][row].length; col++) {
					if (digits[digit][row][col] === '#') pixels[(y + row) * width + x + col] = 1;
				}
			}
		}
		function drawDay(row: number, col: number, day: number) {
			const text = String(day);
			if (text.length === 1) {
				drawDigit(onesX[col], rowY[row], text);
			} else {
				drawDigit(onesX[col] - 5, rowY[row], text[0]);
				drawDigit(onesX[col], rowY[row], text[1]);
			}
		}
		const weeks = [
			[0, 0, 0, 1, 2, 3, 4],
			[5, 6, 7, 8, 9, 10, 11],
			[12, 13, 14, 15, 16, 17, 18],
			[19, 20, 21, 22, 23, 24, 25],
			[26, 27, 28, 29, 30, 0, 0],
		];
		for (let row = 0; row < weeks.length; row++) {
			for (let col = 0; col < weeks[row].length; col++) {
				if (weeks[row][col]) drawDay(row, col, weeks[row][col]);
			}
		}
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: () => {},
			step: () => {},
			getReg: () => 0,
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
			lcdPixels: () => pixels,
		};
		const api = createEvalApi(adapter as any);
		const assertion = await api.lcd.assertCalendarMonth({ year: 2026, month: 4, day: 26 });
		expect(assertion.title).toBe('*** APR 2026 ***');
		expect(assertion.checkedDays).toHaveLength(30);
		expect(assertion.dayOfYear).toBe(116);
		expect(assertion.daysRemaining).toBe(249);
		expect(assertion.weekOfYear).toBe(17);
	});

	it('proof metadata renders readable yaml', async () => {
		const adapter = {
			callFunction: async () => {
				throw new Error('not used');
			},
			reset: () => {},
			step: () => {},
			getReg: (name: string) => (name === 'PC' ? 0xf1234 : 0),
			setReg: () => {},
			read8: () => 0,
			write8: () => {},
			lcdText: () => ['*** APR 2026 ***'],
			lcdPixels: () => [1, 0, 1, 0],
		};
		const api = createEvalApi(adapter as any);
		const yaml = await api.proof.metadataYaml({
			label: 'calendar-apr-2026',
			keySequence: 'app:calendar',
			assertions: { calendar: 'ok' },
			includeGeneratedAt: false,
		});
		expect(yaml).toContain('schema: iq7000-screen-proof/v1');
		expect(yaml).toContain('label: calendar-apr-2026');
		expect(yaml).toContain('- app:calendar');
		expect(yaml).toContain('final_pc: 0xF1234');
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
