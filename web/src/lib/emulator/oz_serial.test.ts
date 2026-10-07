import { describe, expect, it } from 'vitest';
import { SerialCapture, SerialDelivery, queueSerialInput, serialInput, type SerialPacket } from './oz_serial';

describe('serial byte admission', () => {
	it('preserves UTF-8, hex binary values and explicit line endings', () => {
		expect([...serialInput('UART OK', 'text', 'crlf')]).toEqual([...new TextEncoder().encode('UART OK\r\n')]);
		expect([...serialInput('00 0A ff 1b', 'hex', 'none')]).toEqual([0, 10, 255, 27]);
		expect([...serialInput('é', 'text', 'none')]).toEqual([195, 169]);
		for (const input of ['0', 'GG', '0102', '12,34']) expect(() => serialInput(input, 'hex', 'none')).toThrow();
		expect(() => serialInput('x'.repeat(4096), 'text', 'cr')).toThrow();
	});
	it('rejects disabled/full/off UARTs atomically and queues enabled input without software writes', () => {
		let state = { baud: 1200, pending_rx: 0, rejected_rx: 0 };
		const received: number[] = [];
		let power = 'on';
		const device = {
			sio_uart_report: () => state,
			power_state: () => power,
			sio_queue_rx_byte: (b: number) => received.push(b),
			sio_drain_tx_bytes: () => new Uint8Array(),
		};
		state.baud = 0;
		expect(() => queueSerialInput(device, new Uint8Array([1, 2]))).toThrow();
		state.baud = 1200;
		state.pending_rx = 4095;
		expect(() => queueSerialInput(device, new Uint8Array([1, 2]))).toThrow();
		state.pending_rx = 0;
		power = 'off';
		expect(() => queueSerialInput(device, new Uint8Array([1, 2]))).toThrow();
		expect(received).toEqual([]);
		power = 'on';
		expect(queueSerialInput(device, new Uint8Array([0, 255]))).toBe(2);
		expect(received).toEqual([0, 255]);
	});
});

describe('serial delivery and raw capture', () => {
	it('delivers independently, bounds stalled consumers and rejects stale reset credit', () => {
		const packets: SerialPacket[] = [];
		const d = new SerialDelivery((p) => packets.push(p));
		const old = d.reset(1);
		d.pump(() => new Uint8Array([1]));
		for (let i = 0; i < 5; i++) d.pump(() => new Uint8Array(4096).fill(i));
		expect(packets).toHaveLength(1);
		expect(d.snapshot()).toMatchObject({ pending_bytes: 16384, dropped_bytes: 4096, total_bytes: 20481 });
		d.consumed(2, old.epoch, packets[0].sequence);
		d.pump(() => new Uint8Array());
		expect(packets).toHaveLength(1);
		d.consumed(1, old.epoch, packets[0].sequence);
		d.pump(() => new Uint8Array());
		expect(packets[1].first_byte).toBe(4097);
		expect(new Uint8Array(packets[1].bytes)[0]).toBe(1);
		const fresh = d.reset(1);
		d.pump(() => new Uint8Array([9]));
		d.consumed(1, old.epoch, packets[2].sequence);
		d.pump(() => new Uint8Array([10]));
		expect(packets).toHaveLength(3);
		d.consumed(1, fresh.epoch, packets[2].sequence);
		d.pump(() => new Uint8Array());
		expect(packets).toHaveLength(4);
	});
	it('keeps raw zero/control/high bytes and chronological tails through wrap, clear and reset', () => {
		const packets: SerialPacket[] = [];
		const d = new SerialDelivery((p) => packets.push(p));
		const c = new SerialCapture(5);
		c.begin(d.reset(4));
		d.pump(() => new Uint8Array([0, 255, 13, 10]));
		expect(c.push(packets[0])).toBe(true);
		expect(c.push(packets[0])).toBe(false);
		expect([...c.bytes()]).toEqual([0, 255, 13, 10]);
		expect(c.preview()).toBe('\\x00\\xff\\r\n');
		d.consumed(4, packets[0].epoch, packets[0].sequence);
		d.pump(() => new Uint8Array([1, 2, 3]));
		c.push(packets[1]);
		expect([...c.bytes()]).toEqual([13, 10, 1, 2, 3]);
		expect(c.status()).toMatchObject({ received: 7, retained: 5, captureDropped: 2 });
		c.clear();
		expect(c.push(packets[1])).toBe(false);
		expect(c.bytes()).toHaveLength(0);
		c.begin(d.reset(5));
		expect(c.push(packets[1])).toBe(false);
		d.pump(() => new Uint8Array([88]));
		c.push(packets[2]);
		expect([...c.bytes()]).toEqual([88]);
	});
});
