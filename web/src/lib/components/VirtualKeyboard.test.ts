import { cleanup, fireEvent, render } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import VirtualKeyboard from './VirtualKeyboard.svelte';
import { deviceLayout } from '../device_layout';
import { virtualKeysForModel } from '../keymap';

describe('VirtualKeyboard', () => {
	afterEach(() => cleanup());

	it('distinguishes host-held keys from contacts delivered to Rust', () => {
		const { getByTestId } = render(VirtualKeyboard, {
			model: 'iq-7000',
			onPress: vi.fn(),
			onRelease: vi.fn(),
			physicalHighlights: new Set([0x1c]),
			deliveredContacts: new Set([virtualKeysForModel('iq-7000').find((key) => key.testId === 'vk-b')!.code]),
		});
		expect(getByTestId('vk-a').classList.contains('host-held')).toBe(true);
		expect(getByTestId('vk-a').classList.contains('delivered')).toBe(false);
		expect(getByTestId('vk-b').classList.contains('delivered')).toBe(true);
		expect(getByTestId('vk-b').classList.contains('host-held')).toBe(false);
	});

	it('device geometry retains physical actions and disables unqualified keycaps', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const layout = deviceLayout('iq-7000');
		const { getByTestId, getByRole } = render(VirtualKeyboard, { model: 'iq-7000', layout, onPress, onRelease });
		await fireEvent.pointerDown(getByTestId('vk-a'));
		await fireEvent.pointerUp(window);
		expect(onPress).toHaveBeenCalledWith(0x1c, expect.any(String));
		expect(onRelease).toHaveBeenCalledWith(0x1c, expect.any(String), false);
		const off = getByRole('button', { name: 'OFF (unmapped)' }) as HTMLButtonElement;
		expect(off.disabled).toBe(true);
		await fireEvent.pointerDown(off);
		await fireEvent.click(off);
		expect(onPress).toHaveBeenCalledTimes(1);
	});

	it('calls onPress/onRelease for PF1', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { disabled: false, onPress, onRelease });
		const pf1 = getByTestId('vk-pf1');

		await fireEvent.pointerDown(pf1);
		await fireEvent.pointerUp(pf1);

		expect(onPress).toHaveBeenCalledWith(0x56, expect.any(String));
		expect(onRelease).toHaveBeenCalledWith(0x56, expect.any(String), false);
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

		for (const code of [0x1e, 0x17, 0x1f, 0x26]) {
			expect(onPress).toHaveBeenCalledWith(code, expect.any(String));
			expect(onRelease).toHaveBeenCalledWith(code, expect.any(String), false);
		}
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

	it('cancels held and already-released assisted keys on blur, disable and model change', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const onCancelAll = vi.fn();
		const { getByTestId, rerender, unmount } = render(VirtualKeyboard, { onPress, onRelease, onCancelAll });
		await fireEvent.pointerDown(getByTestId('vk-pf1'));
		await fireEvent.blur(window);
		expect(onRelease).toHaveBeenLastCalledWith(0x56, expect.any(String), true);
		await fireEvent.pointerDown(getByTestId('vk-pf1'));
		await rerender({ disabled: true });
		expect(onRelease).toHaveBeenCalledTimes(2);
		await rerender({ disabled: false });
		await fireEvent.pointerDown(getByTestId('vk-pf1'));
		await rerender({ model: 'iq-7000' });
		expect(onRelease).toHaveBeenCalledTimes(3);
		await fireEvent.pointerDown(getByTestId('vk-shift'));
		await fireEvent.pointerUp(getByTestId('vk-shift'));
		expect(onPress).toHaveBeenLastCalledWith(0x02, expect.any(String));
		const before = onCancelAll.mock.calls.length;
		await unmount();
		expect(onCancelAll.mock.calls.length).toBeGreaterThan(before);
	});

	it('releases on pointercancel and external pointerup exactly once', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { onPress, onRelease });
		await fireEvent.pointerDown(getByTestId('vk-pf1'));
		await fireEvent.pointerCancel(window);
		await fireEvent.pointerUp(window);
		expect(onRelease).toHaveBeenCalledTimes(1);
		expect(onRelease).toHaveBeenLastCalledWith(0x56, expect.any(String), true);
		await fireEvent.pointerDown(getByTestId('vk-pf1'));
		await fireEvent.pointerUp(window);
		expect(onRelease).toHaveBeenLastCalledWith(0x56, expect.any(String), false);
	});

	it('supports keyboard activation and ON without double-firing auto-repeat', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { onPress, onRelease });
		const on = getByTestId('vk-on');
		await fireEvent.keyDown(on, { key: ' ', code: 'Space' });
		await fireEvent.keyDown(on, { key: ' ', code: 'Space', repeat: true });
		await fireEvent.keyUp(on, { key: ' ', code: 'Space' });
		expect(onPress.mock.calls).toEqual([['on', 'key:Space:on']]);
		expect(onRelease.mock.calls).toEqual([['on', 'key:Space:on', false]]);
	});

	it('keeps simultaneous pointers distinct and uses distinct assisted owners when one pointer moves to another key', async () => {
		const onPress = vi.fn();
		const onRelease = vi.fn();
		const { getByTestId } = render(VirtualKeyboard, { onPress, onRelease });
		const pointer = async (target: Element | Window, type: string, pointerId: number) => {
			const event = new Event(type, { bubbles: true, cancelable: true });
			Object.assign(event, { pointerId, button: 0 });
			await fireEvent(target, event);
		};
		await pointer(getByTestId('vk-pf1'), 'pointerdown', 1);
		await pointer(getByTestId('vk-pf1'), 'pointerdown', 2);
		await pointer(window, 'pointerup', 1);
		expect(onRelease.mock.calls).toEqual([[0x56, 'pointer:1:86', false]]);
		await pointer(getByTestId('vk-pf2'), 'pointerdown', 1);
		await pointer(window, 'pointerup', 2);
		await pointer(window, 'pointerup', 1);
		expect(onPress.mock.calls).toEqual([
			[0x56, 'pointer:1:86'],
			[0x56, 'pointer:2:86'],
			[0x55, 'pointer:1:85'],
		]);
		expect(onRelease.mock.calls).toEqual([
			[0x56, 'pointer:1:86', false],
			[0x56, 'pointer:2:86', false],
			[0x55, 'pointer:1:85', false],
		]);
	});
});
