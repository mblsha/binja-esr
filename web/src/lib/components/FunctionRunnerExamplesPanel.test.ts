import { fireEvent, render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';

import FunctionRunnerExamplesPanel from './FunctionRunnerExamplesPanel.svelte';

describe('FunctionRunnerExamplesPanel', () => {
	it('copies example code to clipboard', async () => {
		const writeText = vi.fn(async (_text: string) => {});
		vi.stubGlobal('navigator', { clipboard: { writeText } } as any);

		const { getAllByRole } = render(FunctionRunnerExamplesPanel);
		const buttons = getAllByRole('button', { name: 'Copy' });
		expect(buttons.length).toBeGreaterThan(0);

		await fireEvent.click(buttons[0]);
		await new Promise((r) => setTimeout(r, 0));

		expect(writeText).toHaveBeenCalled();
		const copied = writeText.mock.calls[0]?.[0] ?? '';
		expect(String(copied)).toContain('await e.reset');

		expect((buttons[0]?.textContent ?? '').trim()).toBe('Copied');
	});
});
