import { describe, expect, it, vi } from 'vitest';
import { piLcdTablet, piPanelCapture } from './pi3000_panel';
import { TabletInputs } from './emulator/tablet_inputs';
import { normalizeRomModel, romBasename } from './rom_model';
import { deviceLayout } from './device_layout';
import { contactsForKeyEvent } from './keymap';

describe('PI panel and raw input', () => {
	it('keeps native frames immutable and transposes the visible rectangle without indicators', () => {
		const pixels = new Uint8Array(240 * 336).fill(255);
		pixels[238 * 240 + 167] = 17;
		pixels[239 * 240] = 42;
		pixels[238 * 240 + 168] = 43;
		pixels[0] = 9;
		const native = { cols: 240, rows: 336, pixel_scale: 1, pixels };
		const panel = piPanelCapture(native);
		expect([panel.cols, panel.rows]).toEqual([239, 168]);
		expect(panel.pixels[0]).toBe(9);
		expect(panel.pixels[panel.pixels.length - 1]).toBe(17);
		expect(panel.pixels).not.toContain(42);
		expect(panel.pixels).not.toContain(43);
		expect(native.pixels).toBe(pixels);
		expect(native.cols).toBe(240);
		expect(() => piPanelCapture({ ...native, rows: 240 })).toThrow('geometry');
	});
	it('maps the four screen targets to raw endpoints accepted by ROM calibration', () => {
		for (const [x, y, rx, ry] of [
			[32, 32, 340, 240],
			[205, 32, 700, 240],
			[32, 135, 340, 700],
			[205, 135, 700, 700],
		])
			expect(piLcdTablet((x + 0.5) / 239, (y + 0.5) / 168)).toEqual([rx, ry]);
		for (const point of [
			[-1, -1],
			[2, 2],
		])
			for (const adc of piLcdTablet(point[0], point[1])) expect(adc >= 0 && adc <= 1023).toBe(true);
	});
	it('retains a short PI tap through ten-sample acquisition while preserving pen ownership', () => {
		const apply = vi.fn();
		const pen = new TabletInputs(apply, undefined, undefined, () => 200_000);
		const contact = { raw_x: 340, raw_y: 240, pressed: true };
		pen.set('one', contact);
		pen.set('one', { ...contact, pressed: false });
		pen.set('two', { ...contact, raw_x: 700 });
		pen.advance(100_000);
		expect(apply).toHaveBeenCalledTimes(1);
		expect(pen.limitBudget(200_000)).toBe(100_000);
		pen.advance(100_000);
		expect(apply).toHaveBeenLastCalledWith({ ...contact, pressed: false });
		pen.set('two', contact);
		pen.set('two', { ...contact, pressed: false }, true);
		expect(apply).toHaveBeenLastCalledWith({ ...contact, pressed: false });
	});
	it('selects its bundle and exposes ON without inventing a keyboard matrix', () => {
		expect(normalizeRomModel('PI3000')).toBe('pi-3000');
		expect(romBasename('pi-3000')).toBe('pi-3000.pirom');
		expect(deviceLayout('pi-3000').keys.map((k) => k.code)).toEqual(['on', 1, 6, 9, 13, 14, 7]);
		expect(contactsForKeyEvent({ code: 'PageUp', key: 'PageUp' } as KeyboardEvent, 'pi-3000', 'symbols')).toEqual([9]);
		expect(contactsForKeyEvent({ code: 'Enter', key: 'Enter' } as KeyboardEvent, 'pi-3000', 'symbols')).toEqual([]);
		expect(contactsForKeyEvent({ code: 'F12', key: 'F12' } as KeyboardEvent, 'pi-3000', 'symbols')).toEqual(['on']);
	});
});
