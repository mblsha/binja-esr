import { describe, expect, it } from 'vitest';
import { HostInputs } from './host_inputs';
import { TabletInputs } from './tablet_inputs';
import { PvReadiness } from './pvf1_readiness';
import { decodePvLiveState, encodePvLiveState, type PvLiveState } from './pvf1_live_state';
function fixture(): PvLiveState {
	return {
		context: { firmwareSha256: 'a'.repeat(64), coreIdentity: 'wasm-sha256:' + 'b'.repeat(64) },
		settings: { executionMode: 'interactive', typingCatchUp: false, targetFps: 30, sound: false },
		inputs: new HostInputs(() => {}).checkpoint(),
		tablet: new TabletInputs(() => {}).checkpoint(),
		readiness: new PvReadiness().checkpoint(),
		model: new Uint8Array([1, 2, 3]),
		native: new Uint8Array([4, 5, 6]),
	};
}
describe('portable live-state envelope', () => {
	it('owns pending queues and assets across checksum awaits', async () => {
		const value = fixture(),
			bytes = await encodePvLiveState(value),
			copy = bytes.slice();
		const pending = decodePvLiveState(bytes, value.context);
		bytes.fill(0);
		expect(await pending).toEqual(value);
		expect(await encodePvLiveState(await decodePvLiveState(copy, value.context))).toEqual(copy);
	});
	it('rejects corrupt extents, wrong builds/firmware and invalid ownership before native installation', async () => {
		const value = fixture(),
			bytes = await encodePvLiveState(value);
		for (const bad of [bytes.slice(0, 20), bytes.slice(0, -1), new Uint8Array([...bytes, 0])])
			await expect(decodePvLiveState(bad, value.context)).rejects.toThrow();
		for (const context of [
			{ ...value.context, firmwareSha256: 'c'.repeat(64) },
			{ ...value.context, coreIdentity: 'wasm-sha256:' + 'd'.repeat(64) },
		])
			await expect(decodePvLiveState(bytes, context)).rejects.toThrow('matching firmware and emulator build');
		value.tablet.physical = 9;
		await expect(encodePvLiveState(value)).rejects.toThrow('tablet-input');
	});
});
