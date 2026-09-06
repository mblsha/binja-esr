/** One transferred frame in flight, plus one lazy request for the newest state.
 * Capture only after the consumer returns credit; never build a queue of pixel
 * buffers or stale debug snapshots behind a blocked/background UI.
 */
export class LatestFrame<T> {
	private pending: (() => T) | null = null;
	private inFlight: number | null = null;
	private scheduled = false;
	private sequence = 0;
	private coalesced = 0;
	private failures = 0;

	constructor(
		private readonly send: (frame: T, sequence: number) => void,
		private readonly onError: (error: unknown) => void,
		private readonly schedule: (flush: () => void) => void = (flush) => {
			setTimeout(flush, 0);
		},
	) {}

	request(capture: () => T): void {
		if (this.pending) this.coalesced++;
		this.pending = capture;
		this.scheduleFlush();
	}

	consumed(sequence: number): void {
		if (sequence !== this.inFlight) return; // Stale/duplicate acknowledgements cannot create extra credit.
		this.inFlight = null;
		this.scheduleFlush();
	}

	/** Discard uncaptured work after a machine fault; already-transferred frames
	 * remain valid historical observations and still require their normal credit.
	 */
	discardPending(): void {
		this.pending = null;
	}

	private scheduleFlush() {
		if (this.scheduled || this.inFlight !== null || !this.pending) return;
		this.scheduled = true;
		this.schedule(() => {
			this.scheduled = false;
			if (this.inFlight !== null || !this.pending) return;
			const capture = this.pending;
			this.pending = null;
			try {
				const frame = capture();
				this.inFlight = ++this.sequence;
				this.send(frame, this.inFlight);
			} catch (error) {
				this.inFlight = null;
				this.failures++;
				this.onError(error);
			}
		});
	}

	snapshot() {
		return {
			inFlight: this.inFlight,
			pending: this.pending !== null,
			scheduled: this.scheduled,
			sequence: this.sequence,
			coalesced: this.coalesced,
			failures: this.failures,
		};
	}
}
