import { checkBoundaryBudget } from './bounded_step';

export type InputContact = number | 'on';
export type InputSource = 'physical' | 'virtual' | 'script' | 'diagnostic';
export type ContactChange = {
	source: InputSource;
	owner: string;
	contact: InputContact;
	down: boolean;
	minimumHold?: number;
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

/** Host ownership only: the sink changes real matrix/ON contacts, never FIFO or
 * IRQ registers. Assisted holds count submitted scheduler boundaries, not CPU
 * cycles, retired instructions, or wall time. No background execution is added.
 */
export class HostInputs {
	private held = new Map<string, HeldContact>();
	constructor(private readonly sink: (contact: InputContact, down: boolean) => void) {}

	private isHeld(contact: InputContact, except?: string): boolean {
		return [...this.held].some(([id, state]) => id !== except && state.contact === contact);
	}

	set(change: ContactChange): void {
		const { source, owner, contact, down, cancel = false, minimumHold = 0 } = change;
		if (!['physical', 'virtual', 'script', 'diagnostic'].includes(source)) throw new Error('Invalid input source');
		if (typeof owner !== 'string' || !owner.length || owner.length > 128) throw new Error('Invalid input owner');
		if (contact !== 'on' && (!Number.isInteger(contact) || contact < 0 || contact >= 128))
			throw new Error('Physical matrix contact must be an integer in 0..127, or ON');
		if (typeof down !== 'boolean' || typeof cancel !== 'boolean') throw new Error('Invalid contact transition');
		checkBoundaryBudget(minimumHold);
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

	private remove(id: string, state: HeldContact) {
		if (!this.isHeld(state.contact, id)) this.sink(state.contact, false);
		this.held.delete(id);
	}

	releaseSource(source: InputSource): void {
		for (const [id, state] of this.held) if (state.source === source) this.remove(id, state);
	}

	clear(): void {
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
		for (const state of this.held.values()) if (!state.down) limit = Math.min(limit, state.remaining);
		return limit;
	};

	advance = (used: number): void => {
		checkBoundaryBudget(used);
		if (this.limitBudget(used) < used) throw new Error('Execution crossed a pending input deadline');
		for (const [id, state] of this.held) {
			state.remaining = Math.max(0, state.remaining - used);
			if (!state.down && state.remaining === 0) this.remove(id, state);
		}
	};

	snapshot() {
		const contacts = [...new Set([...this.held.values()].map((state) => state.contact))];
		return {
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
