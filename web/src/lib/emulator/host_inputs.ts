import { checkBoundaryBudget } from './bounded_step';

export type InputContact = number | 'on';
export type InputSource = 'physical' | 'virtual' | 'script' | 'diagnostic';
export type ContactChange = {
	source: InputSource;
	owner: string;
	contact: InputContact;
	down: boolean;
	minimumHold?: number;
	/** Serialize ordinary typing through real contacts. Raw mode is unchanged. */
	buffered?: boolean;
	/** Cancellation bypasses assisted-tap timing, even when the machine is paused. */
	cancel?: boolean;
};
type HeldContact = {
	source: InputSource;
	owner: string;
	contact: InputContact;
	down: boolean;
	remaining: number;
};

export class InputBufferOverflow extends Error {
	constructor() {
		super('Typing buffer overflow (128 keys). Pending typing cancelled; clear queued keys before continuing.');
	}
}
type BufferedContact = HeldContact & { id: string };
// Compatibility policy shared by both browser models, not measured silicon
// timing. Native IQ typing uses this same hold/gap budget. No CPU runs on input.
export const TYPING_HOLD = 40_000;
export const TYPING_GAP = 40_000;
export const TYPING_CAPACITY = 128;

/** Host ownership only: the sink changes real matrix/ON contacts, never FIFO or
 * IRQ registers. Assisted holds count submitted scheduler boundaries, not CPU
 * cycles, retired instructions, or wall time. No background execution is added.
 */
export class HostInputs {
	private held = new Map<string, HeldContact>();
	private typing: BufferedContact[] = [];
	private typingDown = new Map<string, BufferedContact>();
	private typingActive: BufferedContact | null = null;
	private typingGap = 0;
	private typingSequence = 0;
	private typingBlocked = false;
	constructor(private readonly sink: (contact: InputContact, down: boolean) => void) {}

	private isHeld(contact: InputContact, except?: string): boolean {
		return [...this.held].some(([id, state]) => id !== except && state.contact === contact);
	}

	set(change: ContactChange): void {
		const { source, owner, contact, down, cancel = false, minimumHold = 0, buffered = false } = change;
		if (!['physical', 'virtual', 'script', 'diagnostic'].includes(source)) throw new Error('Invalid input source');
		if (typeof owner !== 'string' || !owner.length || owner.length > 128) throw new Error('Invalid input owner');
		if (contact !== 'on' && (!Number.isInteger(contact) || contact < 0 || contact >= 128))
			throw new Error('Physical matrix contact must be an integer in 0..127, or ON');
		if (typeof down !== 'boolean' || typeof cancel !== 'boolean') throw new Error('Invalid contact transition');
		checkBoundaryBudget(minimumHold);
		if (typeof buffered !== 'boolean' || (buffered && (source !== 'physical' || contact === 'on')))
			throw new Error('Buffered typing requires a physical matrix key');
		if (buffered) {
			this.setBuffered(owner, contact, down, cancel);
			return;
		}
		const id = `${source}:${owner}`;
		const previous = this.held.get(id);
		if (previous && previous.contact !== contact) throw new Error('Release the old contact before reusing its owner');
		if (cancel || !down) {
			if (!previous) return;
			if (!cancel && previous.remaining > 0) previous.down = false;
			else this.remove(id, previous);
			return;
		}
		if (previous?.down) return; // Browser auto-repeat is not a second contact edge.
		if (!previous && this.held.size >= 512) throw new Error('Too many input owners');
		if (!this.isHeld(contact)) this.sink(contact, true);
		// Repress cancels the old release deadline and starts a new minimum hold.
		// Without an actual up edge this is one continuous electrical contact.
		this.held.set(id, { source, owner, contact, down: true, remaining: minimumHold });
	}

	private setBuffered(owner: string, contact: InputContact, down: boolean, cancel: boolean) {
		if (this.typingBlocked) throw new InputBufferOverflow();
		const previous = this.typingDown.get(owner);
		if (previous && previous.contact !== contact)
			throw new Error('Release the old typing contact before reusing its owner');
		if (cancel) {
			this.clearTyping();
			return;
		}
		if (!down) {
			if (!previous) return;
			previous.down = false;
			this.typingDown.delete(owner);
			if (previous === this.typingActive && previous.remaining === 0) this.finishTyping();
			return;
		}
		if (previous) return; // Native repeat belongs to the ROM, not the browser.
		if (this.typing.length >= TYPING_CAPACITY) {
			this.clearTyping();
			this.typingBlocked = true; // Ignore the rest of the in-flight burst too.
			throw new InputBufferOverflow();
		}
		const entry: BufferedContact = {
			source: 'physical',
			owner,
			contact,
			down: true,
			remaining: TYPING_HOLD,
			id: `typing:${++this.typingSequence}`,
		};
		this.typing.push(entry);
		this.typingDown.set(owner, entry);
		this.startTyping();
	}

	private startTyping() {
		if (this.typingActive || this.typingGap || !this.typing.length) return;
		const next = this.typing[0];
		if (!this.isHeld(next.contact)) this.sink(next.contact, true);
		this.held.set(next.id, next); // Different namespace from raw owners.
		this.typingActive = next;
	}

	private finishTyping() {
		const active = this.typingActive!;
		this.remove(active.id, active);
		this.typing.shift();
		this.typingActive = null;
		this.typingGap = TYPING_GAP;
	}

	clearTyping() {
		if (this.typingActive) this.remove(this.typingActive.id, this.typingActive);
		this.typingActive = null;
		this.typing = [];
		this.typingDown.clear();
		this.typingGap = 0;
		this.typingBlocked = false;
	}

	typingStatus() {
		return { pending: this.typing.length, blocked: this.typingBlocked, capacity: TYPING_CAPACITY };
	}

	/** Accelerate only scan/debounce work, not a user's sustained key hold. */
	typingBoostBudget = (requested: number): number => {
		checkBoundaryBudget(requested);
		if (!this.typing.length) return 0;
		if (this.typingGap) return Math.min(requested, this.typingGap);
		return Math.min(requested, this.typingActive?.remaining ?? 0);
	};

	private remove(id: string, state: HeldContact) {
		if (!this.isHeld(state.contact, id)) this.sink(state.contact, false);
		this.held.delete(id);
	}

	releaseSource(source: InputSource): void {
		if (source === 'physical') this.clearTyping();
		for (const [id, state] of this.held) if (state.source === source) this.remove(id, state);
	}

	clear(): void {
		this.clearTyping();
		for (const [id, state] of this.held) this.remove(id, state);
	}

	/** Legacy diagnostic injection also writes raw contacts; restore the owned
	 * electrical level afterwards. It still deliberately perturbs debounce/FIFO.
	 */
	reapply(contact: InputContact): void {
		this.sink(contact, this.isHeld(contact));
	}

	/** End a Rust slice at the next pending release, independent of host speed. */
	limitBudget = (requested: number): number => {
		checkBoundaryBudget(requested);
		let limit = requested;
		if (this.typingGap) limit = Math.min(limit, this.typingGap);
		for (const state of this.held.values()) if (!state.down) limit = Math.min(limit, state.remaining);
		return limit;
	};

	advance = (used: number): void => {
		checkBoundaryBudget(used);
		if (this.limitBudget(used) < used) throw new Error('Execution crossed a pending input deadline');
		// Count the gap that existed before this slice, never a gap created by
		// releasing its active key at the end of the slice.
		this.typingGap = Math.max(0, this.typingGap - used);
		for (const [id, state] of this.held) {
			state.remaining = Math.max(0, state.remaining - used);
			if (!state.down && state.remaining === 0) {
				if (state === this.typingActive) this.finishTyping();
				else this.remove(id, state);
			}
		}
		this.startTyping();
	};

	snapshot() {
		const contacts = [...new Set([...this.held.values()].map((state) => state.contact))];
		return {
			typing: this.typingStatus(),
			pressedCodes: contacts.filter((c): c is number => typeof c === 'number').sort((a, b) => a - b),
			onHeld: contacts.includes('on'),
			owners: [...this.held.values()].map((state) => ({ ...state })),
			pendingVirtualRelease: [...this.held.values()]
				.filter((state) => state.source === 'virtual' && !state.down && typeof state.contact === 'number')
				.map((state) => [state.contact as number, state.remaining] as [number, number]),
		};
	}
}

/** The same contact sink is used by the machine worker and no-worker frontend. */
export function applyContact(emulator: any, contact: InputContact, down: boolean): void {
	const method =
		contact === 'on' ? (down ? 'press_on_key' : 'release_on_key') : down ? 'press_matrix_code' : 'release_matrix_code';
	if (typeof emulator?.[method] !== 'function') throw new Error(`Rust/WASM input export ${method} is unavailable`);
	if (contact === 'on') emulator[method]();
	else emulator[method](contact);
}
