import { expect, it } from 'vitest';
import { transferNativeRecords } from './pvf1_record_transfer';
const bytes = new TextEncoder().encode('"i","MEM1"\r\n"T","Body"\r\n"D","sample text"\r\n\x1a');
function peer() {
	const queue: number[] = [],
		received: number[] = [],
		wire: number[] = [];
	let baud = 9600,
		steps = 0,
		max = 0;
	return {
		received,
		get max() {
			return max;
		},
		sio_uart_report: () => ({ baud, pending_rx: queue.length, rejected_rx: 0, rx_ready: false }),
		power_state: () => 'running',
		sio_queue_rx_byte: (b: number) => {
			queue.push(b);
			max = Math.max(max, queue.length);
		},
		sio_drain_tx_bytes: () => Uint8Array.from(wire.splice(0)),
		run_slice: (budget: number) => {
			steps++;
			received.push(...queue.splice(0, 4));
			if (steps === 1) wire.push(0x13);
			if (steps === 2) wire.push(0x11);
			if (received.at(-1) === 0x1a) baud = 0;
			return {
				reason: 'boundary_budget' as const,
				progress: { boundary_budget_used: budget, instructions_retired: 0, timing_units_advanced: budget },
			};
		},
	};
}
it('streams every original byte with a bounded backlog and respects native XOFF/XON', async () => {
	const device = peer(),
		wire: number[] = [];
	const result = await transferNativeRecords(device, bytes, { onWire: (b) => wire.push(...b) });
	expect(device.received).toEqual([...bytes]);
	expect(device.max).toBeLessThanOrEqual(8);
	expect(wire).toEqual([0x13, 0x11]);
	expect(result.nativeClosedReceive).toBe(true);
});
it('rejects malformed files and disabled receivers before admitting bytes', async () => {
	const device = peer();
	await expect(transferNativeRecords(device, bytes.slice(0, -1))).rejects.toThrow('EOF');
	expect(device.received).toEqual([]);
	device.sio_uart_report = () => ({ baud: 0, pending_rx: 0, rejected_rx: 0, rx_ready: false });
	await expect(transferNativeRecords(device, bytes)).rejects.toThrow('Open Receive');
	expect(device.received).toEqual([]);
});
it('honors cancellation before advancing or admitting input', async () => {
	const device = peer(),
		controller = new AbortController();
	controller.abort();
	await expect(transferNativeRecords(device, bytes, { signal: controller.signal })).rejects.toThrow('cancelled');
	expect(device.received).toEqual([]);
	expect(device.max).toBe(0);
});
