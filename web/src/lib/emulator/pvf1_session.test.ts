import { describe, expect, it, vi } from 'vitest';
import {
	capturePvSession,
	decodePvSession,
	encodePvSession,
	pvBatteryIdentity,
	pvRetainedIdentity,
	type PvSession,
} from './pvf1_session';

function backing(): PvSession {
	const retained = new Uint8Array(120);
	retained.set(new TextEncoder().encode('PVRAM01\0'));
	retained[8] = 42;
	return {
		retained,
		cardRom: new Uint8Array(0x20000),
		cardSram: new Uint8Array(0x8000),
		ramCard: new Uint8Array(),
		ramCardWritable: true,
		handwritingModel: new Uint8Array([5, 6]),
	};
}
describe('PV-F1 browser session backing', () => {
	it('round trips organizer, program card and recognition configuration as one checked image', async () => {
		const input = backing();
		input.cardSram[9] = 71;
		const image = await encodePvSession(input);
		const read = await decodePvSession(image, pvRetainedIdentity(input.retained));
		expect(read).toEqual(input);
		read.cardSram[9] = 99;
		expect((await decodePvSession(image)).cardSram[9]).toBe(71);
	});
	it('preserves a write-protected RAM card and rejects conflicting media', async () => {
		const input = backing();
		input.cardRom = input.cardSram = new Uint8Array();
		input.ramCard = new Uint8Array(0x20000).fill(17);
		input.ramCardWritable = false;
		expect(await decodePvSession(await encodePvSession(input))).toEqual(input);
		input.cardRom = new Uint8Array(0x20000);
		input.cardSram = new Uint8Array(0x8000);
		await expect(encodePvSession(input)).rejects.toThrow('program-card');
	});
	it('rejects truncated, changed, appended and wrong-firmware saves before loading a machine', async () => {
		const input = backing();
		const image = await encodePvSession(input);
		for (const invalid of [image.slice(0, 20), image.slice(0, -1), new Uint8Array([...image, 0])])
			await expect(decodePvSession(invalid)).rejects.toThrow();
		const changed = image.slice();
		changed[changed.length - 1] ^= 1;
		await expect(decodePvSession(changed)).rejects.toThrow('checksum');
		input.retained[8] = 1;
		await expect(decodePvSession(image, pvRetainedIdentity(input.retained))).rejects.toThrow('another firmware');
	});
	it('owns bytes across asynchronous digest validation', async () => {
		const input = backing();
		const image = await encodePvSession(input);
		const pending = decodePvSession(image);
		image.fill(0);
		expect(await pending).toEqual(input);
	});
	it('uses ROM/layout identity across registered-template format upgrades and frees failed candidates', () => {
		const input = backing();
		const old = pvRetainedIdentity(input.retained);
		input.retained[6] = '2'.charCodeAt(0);
		expect(pvRetainedIdentity(input.retained)).toBe(old);
		const machine = { load_pvf1: vi.fn(), export_pvf1_retained: () => input.retained, free: vi.fn() };
		expect(pvBatteryIdentity(() => machine, new Uint8Array([1]))).toBe(old);
		machine.load_pvf1.mockImplementationOnce(() => {
			throw new Error('bad bundle');
		});
		expect(() => pvBatteryIdentity(() => machine, new Uint8Array())).toThrow('bad bundle');
		expect(machine.free).toHaveBeenCalledTimes(2);
	});
	it('captures active backing without advancing or borrowing mutable core buffers', () => {
		const input = backing();
		const machine = {
			export_pvf1_retained: () => input.retained,
			pvf1_card_sram: () => input.cardSram,
			pvf1_ram_card_image: () => input.ramCard,
		};
		const copy = capturePvSession(machine, input);
		input.retained[104] = 88;
		input.cardRom[5] = 99;
		expect(copy.retained[104]).toBe(0);
		expect(copy.cardRom[5]).toBe(0);
	});
});
