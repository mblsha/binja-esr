declare module '$lib/wasm/pce500_wasm/pce500_wasm.js' {
	export default function init(input?: unknown): Promise<void>;

	export function default_device_model(): string;

	export class Sc62015Emulator {
		constructor();
		[key: string]: unknown;
	}
}
