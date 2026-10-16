import { HostInputs } from './host_inputs';
import { TabletInputs } from './tablet_inputs';
import { PvReadiness } from './pvf1_readiness';

export type PvLiveState = {
	context: { firmwareSha256: string; coreIdentity: string };
	settings: Omit<import('./session_recording').SessionContext['settings'], 'executionMode'> & {
		executionMode: import('./host_pacing').ExecutionMode;
	};
	inputs: ReturnType<HostInputs['checkpoint']>;
	tablet: ReturnType<TabletInputs['checkpoint']>;
	readiness: ReturnType<PvReadiness['checkpoint']>;
	model: Uint8Array;
	native: Uint8Array;
};
const MAGIC = new TextEncoder().encode('PVSTATE1'),
	LIMIT = 80 * 1024 * 1024;
export function validateLiveHost(value: PvLiveState) {
	if (
		!value ||
		!value.context ||
		!/^[0-9a-f]{64}$/.test(value.context.firmwareSha256) ||
		!/^wasm-sha256:[0-9a-f]{64}$/.test(value.context.coreIdentity) ||
		!(value.model instanceof Uint8Array) ||
		!(value.native instanceof Uint8Array) ||
		!value.native.length
	)
		throw new Error('Invalid live-state metadata/assets');
	const settings = value.settings;
	if (
		!settings ||
		!['interactive', 'turbo', 'deterministic'].includes(settings.executionMode) ||
		typeof settings.typingCatchUp !== 'boolean' ||
		typeof settings.sound !== 'boolean' ||
		!Number.isInteger(settings.targetFps) ||
		settings.targetFps < 1 ||
		settings.targetFps > 240
	)
		throw new Error('Invalid live-state execution settings');
	new HostInputs(() => {}).restoreCheckpoint(value.inputs);
	new TabletInputs(() => {}).restoreCheckpoint(value.tablet);
	new PvReadiness().restoreCheckpoint(value.readiness);
}
export async function encodePvLiveState(value: PvLiveState): Promise<Uint8Array> {
	validateLiveHost(value);
	const { model, native, ...host } = value;
	const meta = new TextEncoder().encode(JSON.stringify({ ...host, lengths: [model.length, native.length] }));
	const size = 44 + meta.length + model.length + native.length;
	if (size > LIMIT || meta.length > 2 * 1024 * 1024) throw new Error('Live state exceeds size limit');
	const bytes = new Uint8Array(size);
	bytes.set(MAGIC);
	new DataView(bytes.buffer).setUint32(40, meta.length, true);
	bytes.set(meta, 44);
	bytes.set(model, 44 + meta.length);
	bytes.set(native, 44 + meta.length + model.length);
	bytes.set(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes.subarray(40))), 8);
	return bytes;
}
export async function decodePvLiveState(input: Uint8Array, context: PvLiveState['context']): Promise<PvLiveState> {
	const bytes = new Uint8Array(input);
	if (bytes.length < 44 || bytes.length > LIMIT || !MAGIC.every((b, i) => bytes[i] === b))
		throw new Error('Invalid PV-F1 live-state format');
	const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes.subarray(40)));
	if (!digest.every((b, i) => bytes[8 + i] === b)) throw new Error('Live-state checksum mismatch');
	const size = new DataView(bytes.buffer).getUint32(40, true);
	if (size > 2 * 1024 * 1024 || 44 + size > bytes.length) throw new Error('Invalid live-state header length');
	const data = JSON.parse(new TextDecoder().decode(bytes.subarray(44, 44 + size)));
	if (
		!Array.isArray(data.lengths) ||
		data.lengths.length !== 2 ||
		data.lengths.some((n: number) => !Number.isSafeInteger(n) || n < 0) ||
		44 + size + data.lengths[0] + data.lengths[1] !== bytes.length
	)
		throw new Error('Invalid live-state asset extents');
	if (data.context?.firmwareSha256 !== context.firmwareSha256 || data.context?.coreIdentity !== context.coreIdentity)
		throw new Error('Live state requires its matching firmware and emulator build');
	const { lengths, ...host } = data;
	const value: PvLiveState = {
		...host,
		model: bytes.slice(44 + size, 44 + size + lengths[0]),
		native: bytes.slice(44 + size + lengths[0]),
	};
	validateLiveHost(value);
	return value;
}
