import { describe, expect, it, vi } from 'vitest';
import { loadPa9500, resetPa9500 } from './pa9500_loader';

describe('PA explicit RAM/RTC restart', () => {
	it('validates candidate backing before releasing ON or stepping', () => {
		const emulator = {
			load_pa9500_with_retained: vi.fn(() => {
				throw new Error('checksum');
			}),
			step: vi.fn(),
			release_on_key: vi.fn(),
		};
		expect(() =>
			loadPa9500(
				emulator,
				new Uint8Array(),
				'bootstrap-v3',
				new Uint8Array(),
				new Uint8Array(),
				false,
				new Uint8Array([1]),
			),
		).toThrow('checksum');
		expect(emulator.step).not.toHaveBeenCalled();
		expect(emulator.release_on_key).not.toHaveBeenCalled();
	});
	it('keeps ordinary reset and UART-free fresh loading separate from retained imports', () => {
		const emulator = {
			load_pa9500: vi.fn(),
			load_pa9500_with_retained: vi.fn(),
			step: vi.fn(),
			release_on_key: vi.fn(),
			reset: vi.fn(),
		};
		loadPa9500(emulator, new Uint8Array(), 'bootstrap-v3');
		expect(emulator.load_pa9500).toHaveBeenCalledOnce();
		expect(emulator.load_pa9500_with_retained).not.toHaveBeenCalled();
		resetPa9500(emulator);
		expect(emulator.reset).toHaveBeenCalledOnce();
		expect(emulator.step.mock.calls).toEqual([[64], [64]]);
	});
});
