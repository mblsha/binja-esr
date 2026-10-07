/** Host byte transport only. The Rust UART owns timing, flags and IRQs. */
export const SERIAL_INPUT_CAPACITY = 4096;
export const SERIAL_DELIVERY_CAPACITY = 16384;
export const SERIAL_PACKET_CAPACITY = 4096;
export const SERIAL_CAPTURE_CAPACITY = 4 * 1024 * 1024;
export type SerialFormat = 'text' | 'hex';
export type SerialEnding = 'none' | 'cr' | 'lf' | 'crlf';
export type SerialReset = { generation: number; epoch: number };
export type SerialPacket = SerialReset & {
	sequence: number;
	first_byte: number;
	total_bytes: number;
	dropped_bytes: number;
	bytes: ArrayBuffer;
};
type UartReport = { baud: number; pending_rx: number; rejected_rx: number };
export type SerialDevice = {
	sio_uart_report(): UartReport | null;
	sio_queue_rx_byte(value: number): void;
	sio_drain_tx_bytes(): Uint8Array;
	power_state(): string;
};

export function serialInput(text: string, format: SerialFormat, ending: SerialEnding): Uint8Array {
	if (text.length > SERIAL_INPUT_CAPACITY * 3) throw new Error('Send at most 4096 bytes at a time');
	let bytes: Uint8Array;
	if (format === 'text') bytes = new TextEncoder().encode(text);
	else if (format === 'hex') {
		const words = text.trim() ? text.trim().split(/\s+/) : [];
		if (words.some((word) => !/^[0-9a-fA-F]{2}$/.test(word)))
			throw new Error('Hex input needs byte pairs separated by spaces, for example 55 41 52 54');
		bytes = Uint8Array.from(words.map((word) => parseInt(word, 16)));
	} else throw new Error('Unknown serial input format');
	const tail = { none: [], cr: [13], lf: [10], crlf: [13, 10] }[ending];
	if (!tail) throw new Error('Unknown line ending');
	if (!bytes.length && !tail.length) throw new Error('Enter some text or hex bytes');
	if (bytes.length + tail.length > SERIAL_INPUT_CAPACITY) throw new Error('Send at most 4096 bytes at a time');
	return Uint8Array.from([...bytes, ...tail]);
}

/** Check the entire input before touching the UART; no partial invalid sends. */
export function queueSerialInput(device: SerialDevice, bytes: Uint8Array): number {
	if (!(bytes instanceof Uint8Array) || bytes.length === 0 || bytes.length > SERIAL_INPUT_CAPACITY)
		throw new Error('Send between 1 and 4096 bytes');
	const uart = device.sio_uart_report();
	if (!uart || !uart.baud || device.power_state() === 'off')
		throw new Error('The UART is disabled. Open Terminal and choose Connect first');
	if (
		!Number.isInteger(uart.pending_rx) ||
		uart.pending_rx < 0 ||
		uart.pending_rx + bytes.length > SERIAL_INPUT_CAPACITY
	)
		throw new Error('Serial input is full; let the device receive the pending bytes first');
	for (const byte of bytes) device.sio_queue_rx_byte(byte);
	return bytes.length;
}

/** One packet in flight, bounded backlog, explicit loss. No display credit. */
export class SerialDelivery {
	private generation = -1;
	private epoch = 0;
	private sequence = 0;
	private inFlight: number | null = null;
	private backlog: number[] = [];
	private total = 0;
	private dropped = 0;
	constructor(private send: (packet: SerialPacket) => void) {}
	reset(generation: number): SerialReset {
		this.generation = generation;
		this.epoch++;
		this.inFlight = null;
		this.backlog = [];
		this.total = this.dropped = 0;
		return { generation, epoch: this.epoch };
	}
	consumed(generation: number, epoch: number, sequence: number) {
		if (generation === this.generation && epoch === this.epoch && sequence === this.inFlight) this.inFlight = null;
	}
	pump(take: () => Uint8Array) {
		const bytes = take();
		if (!(bytes instanceof Uint8Array) || bytes.length > SERIAL_INPUT_CAPACITY)
			throw new Error('Invalid UART output block');
		this.total += bytes.length;
		this.backlog.push(...bytes);
		const skip = Math.max(0, this.backlog.length - SERIAL_DELIVERY_CAPACITY);
		this.dropped += skip;
		if (skip) this.backlog.splice(0, skip);
		if (this.inFlight !== null || !this.backlog.length) return;
		const first_byte = this.total - this.backlog.length;
		const packet = Uint8Array.from(this.backlog.splice(0, SERIAL_PACKET_CAPACITY));
		this.inFlight = ++this.sequence;
		try {
			this.send({
				generation: this.generation,
				epoch: this.epoch,
				sequence: this.sequence,
				first_byte,
				total_bytes: this.total,
				dropped_bytes: this.dropped,
				bytes: packet.buffer as ArrayBuffer,
			});
		} catch (error) {
			this.inFlight = null;
			this.dropped += packet.length;
			throw error;
		}
	}
	snapshot() {
		return {
			generation: this.generation,
			epoch: this.epoch,
			inFlight: this.inFlight,
			pending_bytes: this.backlog.length,
			total_bytes: this.total,
			dropped_bytes: this.dropped,
		};
	}
}

/** Bounded raw capture. Text preview never changes the bytes saved to disk. */
export class SerialCapture {
	private generation = -1;
	private epoch = -1;
	private sequence = -1;
	private buffer: Uint8Array | null = null;
	private head = 0;
	private size = 0;
	private received = 0;
	private transportDropped = 0;
	constructor(private capacity = SERIAL_CAPTURE_CAPACITY) {
		if (!Number.isSafeInteger(capacity) || capacity < 1) throw new Error('Invalid serial capture capacity');
	}
	begin(reset: SerialReset) {
		this.generation = reset.generation;
		this.epoch = reset.epoch;
		this.sequence = -1;
		this.clear();
		this.transportDropped = 0;
	}
	clear() {
		this.head = this.size = this.received = 0;
	}
	push(packet: SerialPacket): boolean {
		if (packet.generation !== this.generation || packet.epoch !== this.epoch || packet.sequence <= this.sequence)
			return false;
		const bytes = new Uint8Array(packet.bytes);
		if (
			!Number.isSafeInteger(packet.sequence) ||
			packet.sequence < 1 ||
			bytes.length > SERIAL_PACKET_CAPACITY ||
			!Number.isSafeInteger(packet.first_byte) ||
			packet.first_byte < 0 ||
			!Number.isSafeInteger(packet.total_bytes) ||
			packet.total_bytes < packet.first_byte + bytes.length ||
			!Number.isSafeInteger(packet.dropped_bytes) ||
			packet.dropped_bytes < 0
		)
			throw new Error('Invalid serial output packet');
		this.sequence = packet.sequence;
		this.transportDropped = packet.dropped_bytes;
		this.received += bytes.length;
		this.buffer ??= new Uint8Array(this.capacity);
		for (const byte of bytes) {
			if (this.size === this.capacity) this.head = (this.head + 1) % this.capacity;
			else this.size++;
			this.buffer[(this.head + this.size - 1) % this.capacity] = byte;
		}
		return true;
	}
	bytes(): Uint8Array {
		const out = new Uint8Array(this.size);
		if (this.buffer) {
			const first = Math.min(this.size, this.capacity - this.head);
			out.set(this.buffer.subarray(this.head, this.head + first));
			out.set(this.buffer.subarray(0, this.size - first), first);
		}
		return out;
	}
	preview(): string {
		const count = Math.min(this.size, 2048);
		const bytes = Array.from(
			{ length: count },
			(_, i) => this.buffer![(this.head + this.size - count + i) % this.capacity],
		);
		return Array.from(bytes, (b) =>
			b === 13
				? '\\r'
				: b === 10
					? '\n'
					: b >= 32 && b < 127
						? String.fromCharCode(b)
						: `\\x${b.toString(16).padStart(2, '0')}`,
		).join('');
	}
	status() {
		return {
			received: this.received,
			retained: this.size,
			captureDropped: this.received - this.size,
			transportDropped: this.transportDropped,
			capacity: this.capacity,
		};
	}
}
