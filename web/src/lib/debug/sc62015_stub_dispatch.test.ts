import { describe, expect, it } from 'vitest';

import { createStubDispatcher } from './sc62015_stub_dispatch';

function makeMemory() {
	const wasmMemory = new WebAssembly.Memory({ initial: 1 });
	const buffer = new Uint8Array(wasmMemory.buffer);
	const externalPtr = 0;
	const externalLen = 0x40;
	const internalPtr = 0x40;
	const internalLen = 0x40;
	buffer[externalPtr] = 0x11;
	buffer[internalPtr] = 0x22;
	return { wasmMemory, externalPtr, externalLen, internalPtr, internalLen };
}

describe('createStubDispatcher', () => {
	it('dispatches handler and normalizes patch output', () => {
		const memory = makeMemory();
		const dispatcher = createStubDispatcher(memory);
		let seenExternal = 0;
		let seenInternal = 0;
		dispatcher.registerStub({
			id: 1,
			pc: 0x1234,
			name: 'stub',
			handler: (mem) => {
				seenExternal = mem.read8(0);
				seenInternal = mem.read8(0x100000);
				return {
					mem_writes: { 0x10: 0xaa, '0x20': 0xbb },
					regs: { A: 0x12 },
					flags: { C: 1, Z: 0 },
					ret: { kind: 'ret' },
				};
			},
		});

		const dispatch = (globalThis as any).__sc62015_stub_dispatch as (id: number, regs: unknown, flags: unknown) => any;
		const patch = dispatch(
			1,
			[{ name: 'A', value: 0 }],
			[
				{ name: 'C', value: 0 },
				{ name: 'Z', value: 0 },
			],
		);

		expect(seenExternal).toBe(0x11);
		expect(seenInternal).toBe(0x22);
		expect(patch.mem_writes).toEqual([
			{ addr: 0x10, value: 0xaa, size: 1 },
			{ addr: 0x20, value: 0xbb, size: 1 },
		]);
		expect(patch.regs).toEqual([{ name: 'A', value: 0x12 }]);
		expect(patch.flags).toEqual([
			{ name: 'C', value: 1 },
			{ name: 'Z', value: 0 },
		]);
		expect(patch.ret).toEqual({ kind: 'ret' });
	});

	it('throws when dispatching an unknown stub id', () => {
		const memory = makeMemory();
		createStubDispatcher(memory);
		const dispatch = (globalThis as any).__sc62015_stub_dispatch as (id: number, regs: unknown, flags: unknown) => any;
		expect(() => dispatch(99, [], [])).toThrow(/not registered/);
	});

	it('bubbles handler errors with the stub name', () => {
		const memory = makeMemory();
		const dispatcher = createStubDispatcher(memory);
		dispatcher.registerStub({
			id: 7,
			pc: 0x9999,
			name: 'boom',
			handler: () => {
				throw new Error('nope');
			},
		});
		const dispatch = (globalThis as any).__sc62015_stub_dispatch as (id: number, regs: unknown, flags: unknown) => any;
		expect(() => dispatch(7, [], [])).toThrow(/boom/);
	});

	it('throws when stub reads past internal memory window', () => {
		const memory = makeMemory();
		const dispatcher = createStubDispatcher(memory);
		dispatcher.registerStub({
			id: 8,
			pc: 0x1000,
			name: 'oor',
			handler: (mem) => {
				mem.read8(0x100000 + memory.internalLen);
				return {};
			},
		});
		const dispatch = (globalThis as any).__sc62015_stub_dispatch as (id: number, regs: unknown, flags: unknown) => any;
		expect(() => dispatch(8, [], [])).toThrow(/oor/);
		expect(() => dispatch(8, [], [])).toThrow(/out of range/);
	});
});
