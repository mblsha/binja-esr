import { cleanup, fireEvent, render } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import FunctionRunnerPanel from './FunctionRunnerPanel.svelte';

describe('FunctionRunnerPanel', () => {
	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('persists editor text to localStorage', async () => {
		const backing = new Map<string, string>();
		const localStorageMock = {
			getItem: (key: string) => backing.get(key) ?? null,
			setItem: (key: string, value: string) => backing.set(key, value),
			removeItem: (key: string) => backing.delete(key),
			clear: () => backing.clear()
		};
		vi.stubGlobal('window', { localStorage: localStorageMock } as any);

		const { getByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun: async () => ({ events: [], calls: [], prints: [], resultJson: null, error: null })
		});
		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		await fireEvent.input(editor, { target: { value: 'hello' } });

		expect(backing.get('pce500:function-runner:source')).toContain('hello');
	});

	it('runs script and shows prints', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async (_source: string) => ({
			events: [],
			calls: [],
			prints: [{ index: 0, value: 'ok' }],
			resultJson: null,
			error: null
		}));
		const { getByTestId, findByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		await fireEvent.click(getByTestId('fnr-run'));
		expect(onRun).toHaveBeenCalled();
		const prints = await findByTestId('fnr-prints');
		expect(prints.textContent ?? '').toContain('ok');
	});
});

