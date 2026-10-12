import { describe, expect, it, vi } from 'vitest';
import { render, fireEvent, cleanup } from '@testing-library/svelte';
import { afterEach } from 'vitest';
import { pvLcdPixels } from './pvf1_panel';
import OzTabletSurface from './components/OzTabletSurface.svelte';
import PvTabletControls from './components/PvTabletControls.svelte';
afterEach(cleanup);
describe('PV-F1 logical LCD pen', () => {
	it('delivers application pen contacts and releases on cancellation', async () => {
		const onContact = vi.fn();
		const { getByTestId } = render(PvTabletControls, { onContact });
		await fireEvent.pointerDown(getByTestId('pv-app-drawing'), { button: 0, pointerId: 1 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 980, raw_y: 510, pressed: true }, 'tablet:Drawing:1', false);
		await fireEvent.pointerUp(window, { pointerId: 1 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 980, raw_y: 510, pressed: false }, 'tablet:Drawing:1', false);
		await fireEvent.pointerDown(getByTestId('pv-app-calculator'), { button: 0, pointerId: 2 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 900, raw_y: 640, pressed: true }, 'tablet:Calculator:2', false);
		await fireEvent.pointerCancel(window, { pointerId: 2 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 900, raw_y: 640, pressed: false }, 'tablet:Calculator:2', true);
		await fireEvent.pointerDown(getByTestId('pv-app-search'), { button: 0, pointerId: 3 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 980, raw_y: 900, pressed: true }, 'tablet:Search:3', false);
		await fireEvent.pointerUp(window, { pointerId: 3 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 980, raw_y: 900, pressed: false }, 'tablet:Search:3', false);
	});
	it('keeps storage rows outside the main viewport and preserves native pixel locations', () => {
		expect(pvLcdPixels(222.5 / 336, 16.5 / 292)).toEqual([222, 16]);
		expect(pvLcdPixels(335.5 / 336, 239.5 / 292)).toEqual([335, 239]);
		for (const [x, y] of [
			[0, 240 / 292],
			[-1, 0],
			[1, 0],
			[0, 1],
			[NaN, 0],
			[0, Infinity],
		])
			expect(pvLcdPixels(x, y)).toBeNull();
	});
	it('tracks one pointer and releases its last position when canceled', async () => {
		const onContact = vi.fn();
		const { getByTestId } = render(OzTabletSurface, {
			label: 'LCD',
			testPrefix: 'pv-tablet',
			coordinates: pvLcdPixels,
			trackMovement: true,
			onContact,
		});
		const el = getByTestId('pv-tablet-lcd');
		vi.spyOn(el, 'getBoundingClientRect').mockReturnValue({
			left: 0,
			top: 0,
			width: 336,
			height: 292,
			right: 336,
			bottom: 292,
			x: 0,
			y: 0,
			toJSON: () => {},
		});
		await fireEvent.pointerDown(el, { button: 0, pointerId: 1, clientX: 222, clientY: 16 });
		await fireEvent.pointerMove(el, { pointerId: 1, clientX: 100, clientY: 90 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 100, raw_y: 90, pressed: true }, 'tablet:LCD:1', false);
		await fireEvent.pointerCancel(window, { pointerId: 1 });
		expect(onContact).toHaveBeenLastCalledWith({ raw_x: 100, raw_y: 90, pressed: false }, 'tablet:LCD:1', true);
		const count = onContact.mock.calls.length;
		await fireEvent.pointerDown(el, { button: 0, pointerId: 2, clientX: 100, clientY: 270 });
		expect(onContact).toHaveBeenCalledTimes(count);
	});
});
