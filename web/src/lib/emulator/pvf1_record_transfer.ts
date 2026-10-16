import { stepBounded, type SlicedEmulator } from './bounded_step';
import { queueSerialInput, type SerialDevice } from './oz_serial';
import { parseNativeRecords } from './pvf1_records';
type RecordDevice = SlicedEmulator &
	Omit<SerialDevice, 'sio_uart_report'> & {
		sio_uart_report(): { baud: number; pending_rx: number; rejected_rx: number; rx_ready: boolean } | null;
	};
/** Host byte peer. Native UART timing, IRQs, parser and storage own the import. */
export async function transferNativeRecords(
	device: RecordDevice,
	input: Uint8Array,
	options: {
		signal?: AbortSignal;
		onWire?: (bytes: Uint8Array) => void;
		onProgress?: (sent: number, total: number, boundaries: number) => void;
	} = {},
) {
	const bytes = parseNativeRecords(input).bytes,
		initial = device.sio_uart_report();
	if (!initial?.baud || device.power_state() === 'off')
		throw new Error('Open Receive for this application on the organizer first');
	if (initial.pending_rx || initial.rx_ready) throw new Error('Wait for pending serial input before importing records');
	let sent = 0,
		boundaries = 0,
		flowPaused = false;
	const budget = Math.min(0xffff_ffff, 10_000_000 + bytes.length * 4000);
	const wire = () => {
		const output = device.sio_drain_tx_bytes();
		for (const byte of output) {
			if (byte === 0x13) flowPaused = true;
			if (byte === 0x11) flowPaused = false;
		}
		if (output.length) options.onWire?.(output);
	};
	while (boundaries < budget) {
		if (options.signal?.aborted) throw new Error('Record import cancelled');
		wire();
		const uart = device.sio_uart_report();
		if (!uart) throw new Error('Serial device disappeared during import');
		if (uart.rejected_rx !== initial.rejected_rx) throw new Error('Serial input was rejected');
		if (!uart.baud) {
			if (sent !== bytes.length) throw new Error('Organizer closed Receive before the complete file');
			return { bytes: sent, boundaries, nativeClosedReceive: true };
		}
		if (!flowPaused && sent < bytes.length) {
			const count = Math.min(8 - uart.pending_rx, bytes.length - sent);
			if (count > 0) {
				queueSerialInput(device, bytes.slice(sent, sent + count));
				sent += count;
			}
		}
		const n = Math.min(20_000, budget - boundaries);
		await stepBounded(device, n, { signal: options.signal, onProgress: () => wire() });
		boundaries += n;
		options.onProgress?.(sent, bytes.length, boundaries);
	}
	throw new Error('Organizer did not complete Receive within the transfer budget');
}
