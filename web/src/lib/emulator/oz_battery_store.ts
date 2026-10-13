/** Host storage only. Core-validated battery images contain their ROM identity. */
export interface IdentityMachine {
	load_oz9600(bundle: Uint8Array, retained: Uint8Array, profile: string): void;
	export_oz9600_retained(): Uint8Array;
	free(): void;
}
/** Validate in a disposable zero-instruction machine without replacing the live
 * machine. The key excludes filename, captured RAM, clock and pixels. */
export function ozBatteryIdentity(create: () => IdentityMachine, bundle: Uint8Array): string {
	const candidate = create();
	try {
		candidate.load_oz9600(bundle, new Uint8Array(), 'strict');
		const image = candidate.export_oz9600_retained();
		if (image.length < 104) throw new Error('Incomplete battery-image identity');
		return 'oz9600:' + Array.from(image.subarray(0, 72), (b) => b.toString(16).padStart(2, '0')).join('');
	} finally {
		candidate.free();
	}
}
export type SavedBattery = { image: Uint8Array; session?: Uint8Array; savedAt: number };
export interface BatteryStorage {
	read(key: string): Promise<SavedBattery | null>;
	write(key: string, record: SavedBattery): Promise<void>;
}
export type BatteryLock = (key: string) => Promise<() => Promise<void>>;
const DB = 'sc62015-oz9600-battery';
const STORE = 'images';
async function database(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DB, 1);
		let blocked = false;
		request.onupgradeneeded = () => request.result.createObjectStore(STORE);
		request.onerror = () => reject(request.error ?? new Error('Cannot open saved records'));
		request.onblocked = () => {
			blocked = true;
			reject(new Error('Close other emulator tabs to update saved-record storage'));
		};
		request.onsuccess = () => {
			if (blocked) request.result.close();
			else resolve(request.result);
		};
	});
}
async function transaction<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
	const db = await database();
	try {
		return await new Promise((resolve, reject) => {
			const tx = db.transaction(STORE, mode);
			const request = action(tx.objectStore(STORE));
			tx.oncomplete = () => resolve(request.result);
			tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('Saved-record transaction failed'));
		});
	} finally {
		db.close();
	}
}
export const browserBatteryStorage: BatteryStorage = {
	async read(key) {
		const record = await transaction('readonly', (store) => store.get(key));
		if (record === undefined) return null;
		if (!(record?.image instanceof Uint8Array) || !Number.isFinite(record?.savedAt))
			throw new Error('Saved records have an invalid storage format; the original was preserved');
		if (record.session !== undefined && !(record.session instanceof Uint8Array))
			throw new Error('Saved session has an invalid storage format; the original was preserved');
		return {
			image: record.image.slice(),
			...(record.session !== undefined ? { session: record.session.slice() } : {}),
			savedAt: record.savedAt,
		};
	},
	async write(key, record) {
		await transaction('readwrite', (store) =>
			store.put(
				{
					image: record.image.slice(),
					...(record.session !== undefined ? { session: record.session.slice() } : {}),
					savedAt: record.savedAt,
				},
				key,
			),
		);
	},
};
/** A held Web Lock protects one firmware identity between tabs and releases on
 * tab termination. Close waits for accepted writes before releasing. */
export const browserBatteryLock: BatteryLock = async (key) => {
	if (typeof navigator === 'undefined' || !navigator.locks)
		throw new Error('Automatic saving needs browser profile locks; use backups or a supported browser');
	let acquired!: () => void;
	let failed!: (error: unknown) => void;
	let release!: () => void;
	const ownership = new Promise<void>((resolve, reject) => {
		acquired = resolve;
		failed = reject;
	});
	const released = new Promise<void>((resolve) => {
		release = resolve;
	});
	const holding = navigator.locks.request(`sc62015-battery:${key}`, { ifAvailable: true }, async (lock) => {
		if (!lock) {
			failed(new Error('Another tab is using these saved records; close it and retry'));
			return;
		}
		acquired();
		await released;
	});
	void holding.catch(failed);
	await ownership;
	return async () => {
		release();
		await holding;
	};
};
function equalImage(a: Uint8Array | null, b: Uint8Array): boolean {
	return a !== null && a.length === b.length && a.every((byte, index) => byte === b[index]);
}
export class BatterySession {
	private pending: Promise<void> = Promise.resolve();
	private closing: Promise<void> | null = null;
	private image: Uint8Array | null;
	private savedAt: number | null;
	private session: Uint8Array | null;
	private constructor(
		readonly key: string,
		readonly loaded: SavedBattery | null,
		private storage: BatteryStorage,
		private release: () => Promise<void>,
	) {
		this.image = loaded?.image.slice() ?? null;
		this.savedAt = loaded?.savedAt ?? null;
		this.session = loaded?.session?.slice() ?? null;
	}
	static async open(key: string, storage = browserBatteryStorage, lock = browserBatteryLock): Promise<BatterySession> {
		const release = await lock(key);
		try {
			return new BatterySession(key, await storage.read(key), storage, release);
		} catch (error) {
			await release();
			throw error;
		}
	}
	/** The caller supplies successfully exported core images. Failed writes retain
	 * the last committed image and allow the same replacement to retry. */
	save(image: Uint8Array, session?: Uint8Array): Promise<{ changed: boolean; savedAt: number }> {
		if (this.closing) return Promise.reject(new Error('Saved-record session is closed'));
		const owned = image.slice();
		const ownedSession = session?.slice() ?? null;
		const write = this.pending.then(async () => {
			if (
				equalImage(this.image, owned) &&
				(ownedSession === null ? this.session === null : equalImage(this.session, ownedSession))
			)
				return { changed: false, savedAt: this.savedAt! };
			const savedAt = Date.now();
			await this.storage.write(this.key, {
				image: owned,
				...(ownedSession !== null ? { session: ownedSession } : {}),
				savedAt,
			});
			this.image = owned;
			this.session = ownedSession;
			this.savedAt = savedAt;
			return { changed: true, savedAt };
		});
		this.pending = write.then(
			() => {},
			() => {},
		);
		return write;
	}
	close(): Promise<void> {
		return (this.closing ??= this.pending.then(() => this.release()));
	}
}
