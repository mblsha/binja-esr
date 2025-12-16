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

	it('shows perfetto download button when trace captured', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async (_source: string) => ({
			events: [],
			prints: [],
			resultJson: null,
			error: null,
			calls: [
				{
					index: 0,
					address: 0x10,
					name: null,
					artifacts: {
						before: {},
						after: {},
						changed: [],
						memoryBlocks: [],
						lcdWrites: [],
						probeSamples: [],
						perfettoTraceB64: 'ZHVtbXk=',
						result: { reason: 'returned', steps: 1, pc: 0, sp: 0, halted: false, fault: null },
						infoLog: []
					}
				}
			]
		}));
		const { getByTestId, findByText } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		await fireEvent.click(getByTestId('fnr-run'));
		expect(onRun).toHaveBeenCalled();
		expect(await findByText('Download')).toBeTruthy();
	});

	it('Cmd+Enter runs the script from the editor', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async (_source: string) => ({
			events: [],
			calls: [],
			prints: [],
			resultJson: null,
			error: null
		}));
		const { getByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		await fireEvent.keyDown(editor, { key: 'Enter', metaKey: true });
		await new Promise((r) => setTimeout(r, 0));
		expect(onRun).toHaveBeenCalled();
	});

	it('Cmd+/ toggles line comments for the selection', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async () => ({ events: [], calls: [], prints: [], resultJson: null, error: null }));
		const { getByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		await fireEvent.input(editor, { target: { value: 'line1\n  line2\n\nline3' } });
		editor.selectionStart = 0;
		editor.selectionEnd = editor.value.length;

		await fireEvent.keyDown(editor, { key: '/', metaKey: true });
		await new Promise((r) => setTimeout(r, 0));
		expect(editor.value).toBe('// line1\n  // line2\n\n// line3');

		await fireEvent.keyDown(editor, { key: '/', metaKey: true });
		await new Promise((r) => setTimeout(r, 0));
		expect(editor.value).toBe('line1\n  line2\n\nline3');
	});

	it('Tab indents selection by two spaces and Shift+Tab unindents', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async () => ({ events: [], calls: [], prints: [], resultJson: null, error: null }));
		const { getByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		await fireEvent.input(editor, { target: { value: 'a\n  b\nc' } });
		editor.selectionStart = 0;
		editor.selectionEnd = editor.value.length;

		await fireEvent.keyDown(editor, { key: 'Tab' });
		await new Promise((r) => setTimeout(r, 0));
		expect(editor.value).toBe('  a\n    b\n  c');

		await fireEvent.keyDown(editor, { key: 'Tab', shiftKey: true });
		await new Promise((r) => setTimeout(r, 0));
		expect(editor.value).toBe('a\n  b\nc');
	});

	it('Tab inserts two spaces at cursor when no selection', async () => {
		vi.stubGlobal('window', { localStorage: { getItem: () => null, setItem: () => {} } } as any);
		const onRun = vi.fn(async () => ({ events: [], calls: [], prints: [], resultJson: null, error: null }));
		const { getByTestId } = render(FunctionRunnerPanel, {
			disabled: false,
			busy: false,
			onRun
		});
		const editor = getByTestId('fnr-editor') as HTMLTextAreaElement;
		await fireEvent.input(editor, { target: { value: 'ab' } });
		editor.selectionStart = 1;
		editor.selectionEnd = 1;

		await fireEvent.keyDown(editor, { key: 'Tab' });
		await new Promise((r) => setTimeout(r, 0));
		expect(editor.value).toBe('a  b');
	});
});
