import { describe, expect, it, vi } from 'vitest';
import {
	BatterySession,
	ozBatteryIdentity,
	type BatteryStorage,
	type BatteryVersion,
	type SavedBattery,
	type BatteryLock,
} from './oz_battery_store';
function fixture() {
	const records = new Map<string, SavedBattery>();
	const storage: BatteryStorage = {
		read: vi.fn(async (key) => records.get(key) ?? null),
		write: vi.fn(async (key, record) => {
			records.set(key, {
				image: record.image.slice(),
				session: record.session?.slice(),
				savedAt: record.savedAt,
				previous: record.previous?.map((v: BatteryVersion) => ({
					image: v.image.slice(),
					session: v.session?.slice(),
					savedAt: v.savedAt,
				})),
			});
		}),
	};
	const locks = new Set<string>();
	const lock: BatteryLock = async (key) => {
		if (locks.has(key)) throw new Error('another owner');
		locks.add(key);
		return async () => {
			locks.delete(key);
		};
	};
	return { records, storage, lock, locks };
}
describe('OZ battery session ownership and persistence', () => {
	it('commits bounded previous versions with the image and preserves them after failed writes', async () => {
		const f = fixture();
		const session = await BatterySession.open('pvf1:rom', f.storage, f.lock);
		for (let n = 1; n <= 5; n++) await session.save(new Uint8Array([n]));
		expect(session.versions().map((v) => v.image[0])).toEqual([5, 4, 3, 2]);
		const before = f.records.get('pvf1:rom');
		vi.mocked(f.storage.write).mockRejectedValueOnce(new Error('quota'));
		await expect(session.save(new Uint8Array([6]))).rejects.toThrow('quota');
		expect(f.records.get('pvf1:rom')).toEqual(before);
		expect(session.versions().map((v) => v.image[0])).toEqual([5, 4, 3, 2]);
		const copies = session.versions();
		copies[1].image[0] = 99;
		await session.close();
		const recovered = await BatterySession.open('pvf1:rom', f.storage, f.lock);
		expect(recovered.versions().map((v) => v.image[0])).toEqual([5, 4, 3, 2]);
		// Choosing an old save creates a new commit; it does not destroy the current one.
		await recovered.save(recovered.versions()[2].image);
		expect(recovered.versions().map((v) => v.image[0])).toEqual([3, 5, 4, 3]);
		await recovered.close();
	});
	it('loads committed images, deduplicates and excludes another owner', async () => {
		const f = fixture();
		const first = await BatterySession.open('rom-a', f.storage, f.lock);
		expect(first.loaded).toBeNull();
		await expect(BatterySession.open('rom-a', f.storage, f.lock)).rejects.toThrow('another owner');
		const image = new Uint8Array([1, 2, 3]);
		await expect(first.save(image)).resolves.toMatchObject({ changed: true });
		image[0] = 99;
		await expect(first.save(new Uint8Array([1, 2, 3]))).resolves.toMatchObject({ changed: false });
		expect(f.storage.write).toHaveBeenCalledTimes(1);
		await first.close();
		const second = await BatterySession.open('rom-a', f.storage, f.lock);
		expect(second.loaded?.image).toEqual(new Uint8Array([1, 2, 3]));
		await second.close();
	});
	it('failed commit preserves old data and the same replacement can retry', async () => {
		const f = fixture();
		const session = await BatterySession.open('rom', f.storage, f.lock);
		await session.save(new Uint8Array([1]));
		vi.mocked(f.storage.write).mockRejectedValueOnce(new Error('quota'));
		await expect(session.save(new Uint8Array([2]))).rejects.toThrow('quota');
		expect(f.records.get('rom')?.image).toEqual(new Uint8Array([1]));
		await expect(session.save(new Uint8Array([2]))).resolves.toMatchObject({ changed: true });
		await session.close();
	});
	it('close waits for accepted writes, releases ownership and rejects late snapshots', async () => {
		const f = fixture();
		let finish!: () => void;
		vi.mocked(f.storage.write).mockImplementationOnce(
			() =>
				new Promise<void>((resolve) => {
					finish = resolve;
				}),
		);
		const session = await BatterySession.open('rom', f.storage, f.lock);
		const save = session.save(new Uint8Array([7]));
		await Promise.resolve();
		const close = session.close();
		expect(f.locks.has('rom')).toBe(true);
		await expect(session.save(new Uint8Array([8]))).rejects.toThrow('closed');
		finish();
		await save;
		await close;
		expect(f.locks.has('rom')).toBe(false);
	});
	it('read failure releases ownership without writing or silently loading empty memory', async () => {
		const f = fixture();
		vi.mocked(f.storage.read).mockRejectedValueOnce(new Error('damaged storage'));
		await expect(BatterySession.open('rom', f.storage, f.lock)).rejects.toThrow('damaged storage');
		expect(f.locks.size).toBe(0);
		expect(f.storage.write).not.toHaveBeenCalled();
	});
	it('firmware identities remain independent', async () => {
		const f = fixture();
		const a = await BatterySession.open('a', f.storage, f.lock);
		const b = await BatterySession.open('b', f.storage, f.lock);
		await a.save(new Uint8Array([1]));
		await b.save(new Uint8Array([2]));
		expect(f.records.get('a')?.image).toEqual(new Uint8Array([1]));
		expect(f.records.get('b')?.image).toEqual(new Uint8Array([2]));
		await a.close();
		await b.close();
	});
	it('identity validation frees the disposable machine on success and rejection', () => {
		const image = new Uint8Array(104);
		image[8] = 42;
		const machine = { load_oz9600: vi.fn(), export_oz9600_retained: () => image, free: vi.fn() };
		const key = ozBatteryIdentity(() => machine, new Uint8Array([1]));
		image[73] = 99;
		expect(ozBatteryIdentity(() => machine, new Uint8Array([2]))).toBe(key);
		machine.load_oz9600.mockImplementationOnce(() => {
			throw new Error('bad ROM');
		});
		expect(() => ozBatteryIdentity(() => machine, new Uint8Array())).toThrow('bad ROM');
		expect(machine.free).toHaveBeenCalledTimes(3);
	});
});

it('commits RAM and running state atomically; quota failures retain both and allow retry', async () => {
	const f = fixture();
	const a = await BatterySession.open('paired', f.storage, f.lock);
	const ram = new Uint8Array([1]),
		draft = new Uint8Array([2]);
	await a.save(ram, draft);
	ram[0] = 9;
	draft[0] = 9;
	expect(f.records.get('paired')?.session).toEqual(new Uint8Array([2]));
	await expect(a.save(new Uint8Array([1]), new Uint8Array([2]))).resolves.toMatchObject({ changed: false });
	vi.mocked(f.storage.write).mockRejectedValueOnce(new Error('quota'));
	await expect(a.save(new Uint8Array([3]), new Uint8Array([4]))).rejects.toThrow('quota');
	expect(f.records.get('paired')?.image).toEqual(new Uint8Array([1]));
	expect(f.records.get('paired')?.session).toEqual(new Uint8Array([2]));
	await a.save(new Uint8Array([3]), new Uint8Array([4]));
	await a.close();
	const b = await BatterySession.open('paired', f.storage, f.lock);
	expect(b.loaded?.session).toEqual(new Uint8Array([4]));
	const history = b.versions();
	expect(history.map(({ image, session }) => [Array.from(image), session && Array.from(session)])).toEqual([
		[[3], [4]],
		[[1], [2]],
	]);
	history[1].session![0] = 99;
	expect(b.versions()[1].session).toEqual(new Uint8Array([2]));
	await b.close();
});
