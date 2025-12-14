import { cleanup, fireEvent, render } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import VirtualKeyboard from './VirtualKeyboard.svelte';

describe('VirtualKeyboard', () => {
	afterEach(() => cleanup());

	it('calls onPress/onRelease for PF1', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { disabled: false, onPress, onRelease });
		const pf1 = getByTestId('vk-pf1');

		await fireEvent.pointerDown(pf1);
		await fireEvent.pointerUp(pf1);

		expect(onPress).toHaveBeenCalledWith(0x56);
		expect(onRelease).toHaveBeenCalledWith(0x56);
	});

	it('does not call handlers when disabled', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { disabled: true, onPress, onRelease });
		const pf1 = getByTestId('vk-pf1');

		await fireEvent.pointerDown(pf1);
		await fireEvent.pointerUp(pf1);

		expect(onPress).not.toHaveBeenCalled();
		expect(onRelease).not.toHaveBeenCalled();
	});
});
