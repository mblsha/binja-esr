import { describe, expect, it, vi } from 'vitest';

import { get } from 'svelte/store';

import { createPersistedStore } from './persisted';

describe('createPersistedStore', () => {
	it('loads initial value from localStorage and writes updates back', () => {
		const backing = new Map<string, string>();
		const localStorageMock = {
			getItem: (key: string) => backing.get(key) ?? null,
			setItem: (key: string, value: string) => backing.set(key, value),
			removeItem: (key: string) => backing.delete(key),
			clear: () => backing.clear()
		};
                vi.stubGlobal(
                        'window',
                        { localStorage: localStorageMock } as unknown as Window & typeof globalThis
                );

                backing.set('k', JSON.stringify({ a: 1 }));
                const store = createPersistedStore('k', { a: 0 });
                expect(get(store)).toEqual({ a: 1 });

                store.set({ a: 2 });
		expect(backing.get('k')).toBe(JSON.stringify({ a: 2 }));
	});
});

