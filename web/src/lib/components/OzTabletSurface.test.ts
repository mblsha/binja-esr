import { render, cleanup } from '@testing-library/svelte';
import { expect, it, vi, afterEach } from 'vitest';
import OzTabletSurface from './OzTabletSurface.svelte';
afterEach(cleanup);
function setup() {
	const onContact = vi.fn();
	const view = render(OzTabletSurface, {
		label: 'LCD',
		trackMovement: true,
		coordinates: (x: number, y: number) =>
			x < 0 || y < 0 || x >= 1 || y >= 1 ? null : [Math.round(x * 100), Math.round(y * 100)],
		onContact,
	});
	const el = view.getByRole('button');
	vi.spyOn(el, 'getBoundingClientRect').mockReturnValue({
		left: 0,
		top: 0,
		width: 100,
		height: 100,
		right: 100,
		bottom: 100,
		x: 0,
		y: 0,
		toJSON() {},
	});
	Object.assign(el, { setPointerCapture: vi.fn(), hasPointerCapture: () => true, releasePointerCapture: vi.fn() });
	const send = (type: string, x: number, y: number, extra: Record<string, unknown> = {}) => {
		const e = new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, clientX: x, clientY: y });
		for (const [key, value] of Object.entries({ pointerId: 7, pointerType: 'pen', isPrimary: true, ...extra }))
			Object.defineProperty(e, key, { value });
		el.dispatchEvent(e);
	};
	return { view, onContact, send };
}
it('preserves coalesced samples, the dispatched endpoint and the pen-up endpoint in order', () => {
	const { onContact, send } = setup();
	send('pointerdown', 10, 10);
	send('pointermove', 40, 40, {
		getCoalescedEvents: () => [
			{ clientX: 20, clientY: 20 },
			{ clientX: 30, clientY: 30 },
			{ clientX: 40, clientY: 40 },
		],
	});
	send('pointerup', 50, 50);
	expect(onContact.mock.calls.map(([c]) => [c.raw_x, c.raw_y, c.pressed])).toEqual([
		[10, 10, true],
		[20, 20, true],
		[30, 30, true],
		[40, 40, true],
		[50, 50, true],
		[50, 50, false],
	]);
	expect(onContact.mock.calls.every(([, owner]) => owner === 'tablet:LCD:7')).toBe(true);
});
it('ignores secondary touch/pen owners, keeps the last valid sample, and cancels without inventing an endpoint', () => {
	const { onContact, send } = setup();
	send('pointerdown', 90, 90, { isPrimary: false, pointerType: 'touch' });
	expect(onContact).not.toHaveBeenCalled();
	send('pointerdown', 10, 10);
	send('pointermove', 20, 20, { pointerId: 8 });
	send('pointerup', 30, 30, { pointerId: 8 });
	send('pointermove', 120, 120);
	send('pointercancel', 50, 50);
	expect(onContact.mock.calls.map(([c, , cancel]) => [c.raw_x, c.raw_y, c.pressed, cancel])).toEqual([
		[10, 10, true, false],
		[10, 10, false, true],
	]);
	send('pointerdown', 40, 40);
	send('pointerup', 45, 45);
	expect(onContact.mock.calls.at(-1)?.[0]).toEqual({ raw_x: 45, raw_y: 45, pressed: false });
});
it('cancels on disable and permits a new stroke after re-enabling', async () => {
	const { view, onContact, send } = setup();
	send('pointerdown', 10, 10);
	await view.rerender({ disabled: true });
	expect(onContact.mock.calls.at(-1)?.[2]).toBe(true);
	await view.rerender({ disabled: false });
	send('pointerdown', 20, 20);
	send('pointerup', 20, 20);
	expect(onContact.mock.calls.at(-1)?.[0]).toEqual({ raw_x: 20, raw_y: 20, pressed: false });
});
