/** PCM delivery is independent of coalesced display frames. No guest writes. */
export type PcmChunk = {
	sample_rate: number;
	first_sample: number;
	total_samples: number;
	dropped_samples: number;
	samples: Int16Array;
};
export type AudioPacket = Omit<PcmChunk, 'samples'> & {
	generation: number;
	epoch: number;
	sequence: number;
	samples: ArrayBuffer;
};
export const MAX_AUDIO_SAMPLES = 4800; // At most 100 ms transferred at once.

/** One block in flight. Rust keeps a bounded queue while the host lacks credit. */
export class AudioDelivery {
	private generation = -1;
	private epoch = 0;
	private enabled = false;
	private inFlight: number | null = null;
	private sequence = 0;
	private discarded = 0;
	constructor(private send: (packet: AudioPacket) => void) {}
	reset(generation: number, enabled: boolean) {
		this.generation = generation;
		this.enabled = enabled;
		this.inFlight = null;
		this.epoch++;
		return { generation, epoch: this.epoch, enabled };
	}
	consumed(generation: number, epoch: number, sequence: number) {
		if (generation === this.generation && epoch === this.epoch && sequence === this.inFlight) this.inFlight = null;
	}
	pump(take: () => PcmChunk) {
		if (!this.enabled || this.inFlight !== null) return;
		const chunk = take();
		if (chunk.samples.length === 0) return;
		const skip = Math.max(0, chunk.samples.length - MAX_AUDIO_SAMPLES);
		this.discarded += skip;
		const samples = chunk.samples.slice(skip);
		const sequence = ++this.sequence;
		this.inFlight = sequence;
		try {
			this.send({
				...chunk,
				generation: this.generation,
				epoch: this.epoch,
				sequence,
				first_sample: chunk.first_sample + skip,
				samples: samples.buffer as ArrayBuffer,
			});
		} catch (error) {
			this.inFlight = null;
			throw error;
		}
	}
	snapshot() {
		return {
			generation: this.generation,
			epoch: this.epoch,
			enabled: this.enabled,
			inFlight: this.inFlight,
			discarded_samples: this.discarded,
		};
	}
}

/** WebAudio only consumes already-emulated PCM; it never synthesizes a tone. */
export class AudioPlayback {
	private context: AudioContext | null = null;
	private filter: BiquadFilterNode | null = null;
	private wanted = false;
	private remoteEnabled = false;
	private generation = -1;
	private epoch = -1;
	private enableVersion = 0;
	private nextTime = 0;
	private nextSample: number | null = null;
	private lastSequence = -1;
	private sources = new Set<AudioBufferSourceNode>();
	private dropped = 0;
	private scheduled = 0;
	constructor(private createContext = () => new AudioContext({ sampleRate: 48000 })) {}
	async enable() {
		const version = ++this.enableVersion;
		this.wanted = true;
		try {
			if (!this.context) {
				this.context = this.createContext();
				this.filter = this.context.createBiquadFilter();
				this.filter.type = 'highpass';
				this.filter.frequency.value = 40; // Host DC removal; speaker response is unqualified.
				this.filter.connect(this.context.destination);
			}
			await this.context.resume();
			if (this.context.state !== 'running') throw new Error('Browser audio did not start');
			return this.wanted && version === this.enableVersion;
		} catch (error) {
			if (version === this.enableVersion) this.disable();
			throw error;
		}
	}
	disable() {
		this.enableVersion++;
		this.wanted = false;
		this.pause();
	}
	pause() {
		this.remoteEnabled = false;
		this.clear();
	}
	begin(generation: number, epoch: number, enabled: boolean) {
		if (generation < this.generation || (generation === this.generation && epoch < this.epoch)) return;
		this.clear();
		this.generation = generation;
		this.epoch = epoch;
		this.remoteEnabled = enabled;
		this.nextSample = null;
		this.lastSequence = -1;
	}
	push(packet: AudioPacket) {
		const ctx = this.context;
		if (
			!this.wanted ||
			!this.remoteEnabled ||
			!ctx ||
			ctx.state !== 'running' ||
			packet.generation !== this.generation ||
			packet.epoch !== this.epoch
		)
			return;
		if (packet.sequence <= this.lastSequence) return;
		if (
			packet.sample_rate !== 48000 ||
			!(packet.samples instanceof ArrayBuffer) ||
			packet.samples.byteLength % 2 !== 0 ||
			packet.samples.byteLength > MAX_AUDIO_SAMPLES * 2 ||
			!Number.isSafeInteger(packet.first_sample) ||
			packet.first_sample < 0
		)
			throw new Error('Invalid emulator audio block');
		const samples = new Int16Array(packet.samples);
		if (!samples.length) return;
		this.lastSequence = packet.sequence;
		if (this.nextSample !== null && this.nextSample !== packet.first_sample) this.clear();
		this.nextSample = packet.first_sample + samples.length;
		const duration = samples.length / packet.sample_rate;
		if (this.nextTime + duration > ctx.currentTime + 0.15 || this.sources.size >= 128) {
			this.clear(); // Catch up to current emulated sound, with explicit host loss.
		}
		const buffer = ctx.createBuffer(1, samples.length, packet.sample_rate);
		const data = buffer.getChannelData(0);
		for (let i = 0; i < samples.length; i++) data[i] = samples[i] / 32768;
		const source = ctx.createBufferSource();
		source.buffer = buffer;
		source.connect(this.filter!);
		source.onended = () => {
			this.sources.delete(source);
			source.disconnect();
		};
		this.sources.add(source);
		const start = Math.max(this.nextTime, ctx.currentTime + 0.02);
		source.start(start);
		this.nextTime = start + duration;
		this.scheduled += samples.length;
	}
	private clear() {
		for (const source of this.sources) {
			this.dropped += source.buffer?.length ?? 0;
			try {
				source.stop();
			} catch {
				/* Already ended. */
			}
			source.disconnect();
		}
		this.sources.clear();
		this.nextTime = 0;
	}
	snapshot() {
		return {
			wanted: this.wanted,
			enabled: this.remoteEnabled,
			context_state: this.context?.state ?? 'absent',
			active_sources: this.sources.size,
			scheduled_samples: this.scheduled,
			cancelled_buffer_samples: this.dropped,
			queued_seconds: Math.max(0, this.nextTime - (this.context?.currentTime ?? 0)),
		};
	}
	close() {
		this.disable();
		void this.context?.close();
	}
}
