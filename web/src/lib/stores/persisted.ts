import { writable, type Writable } from 'svelte/store';

type PersistCodec<T> = {
        serialize: (value: T) => string;
        deserialize: (raw: string) => T;
};

const jsonCodec = <T>(): PersistCodec<T> => ({
        serialize: (value) => JSON.stringify(value),
        deserialize: (raw) => JSON.parse(raw) as T
});

function hasStorage(): boolean {
	try {
		return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
	} catch {
		return false;
	}
}

export function createPersistedStore<T>(
        key: string,
        initialValue: T,
        codec: PersistCodec<T> = jsonCodec<T>()
): Writable<T> {
	let startValue = initialValue;
	if (hasStorage()) {
		try {
			const raw = window.localStorage.getItem(key);
			if (raw != null) startValue = codec.deserialize(raw);
		} catch {
			/* ignore persisted load failures */
		}
	}

	const store = writable<T>(startValue);
	if (hasStorage()) {
		store.subscribe((value) => {
			try {
				window.localStorage.setItem(key, codec.serialize(value));
			} catch {
				/* ignore persisted write failures */
			}
		});
	}
	return store;
}

