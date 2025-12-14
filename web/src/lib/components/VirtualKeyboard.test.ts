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

	it('uses the same matrix codes as keymap for arrows', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { disabled: false, onPress, onRelease });
		const up = getByTestId('vk-up');
		const down = getByTestId('vk-down');
		const left = getByTestId('vk-left');
		const right = getByTestId('vk-right');

		await fireEvent.pointerDown(up);
		await fireEvent.pointerUp(up);
		await fireEvent.pointerDown(down);
		await fireEvent.pointerUp(down);
		await fireEvent.pointerDown(left);
		await fireEvent.pointerUp(left);
		await fireEvent.pointerDown(right);
		await fireEvent.pointerUp(right);

		expect(onPress).toHaveBeenCalledWith(0x1e);
		expect(onRelease).toHaveBeenCalledWith(0x1e);
		expect(onPress).toHaveBeenCalledWith(0x17);
		expect(onRelease).toHaveBeenCalledWith(0x17);
		expect(onPress).toHaveBeenCalledWith(0x27);
		expect(onRelease).toHaveBeenCalledWith(0x27);
		expect(onPress).toHaveBeenCalledWith(0x26);
		expect(onRelease).toHaveBeenCalledWith(0x26);
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
