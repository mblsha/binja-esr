import type { TabletContact } from './oz9600_replay';

type HeldPen = { owner: string; contact: TabletContact; remaining: number; grace: number; release: boolean };
type QueuedPen = { owner: string; contact: TabletContact; down: boolean };

/** Host click assistance. Raw ROM replay bypasses this adapter. No CPU runs on input. */
export class TabletInputs {
	private held: HeldPen | null = null;
	private queue: QueuedPen[] = [];
	private gap = 0;
	private buffered: BufferedTabletInputs;
	constructor(
		private apply: (contact: TabletContact) => void,
		private ready: () => boolean = () => true,
		private mailboxBusy: () => boolean = () => false,
		private minimumHold: () => number = () => 40_000,
		private minimumGap: () => number = () => 0,
	) {
		this.buffered = new BufferedTabletInputs(apply, minimumHold, minimumGap);
	}
	private canRelease() {
		return this.held?.remaining === 0 && (this.ready() || this.held.grace === 0);
	}
	private startQueued() {
		if (this.held || this.gap || !this.queue.length || this.mailboxBusy()) return;
		const next = this.queue.shift()!;
		this.held = {
			owner: next.owner,
			contact: next.contact,
			remaining: this.minimumHold(),
			grace: 400_000,
			release: !next.down,
		};
		this.apply(next.contact);
	}
	private finishHeld() {
		const held = this.held;
		this.held = null;
		if (held) {
			this.apply({ ...held.contact, pressed: false });
			this.gap = 40_000;
		}
	}
	set(owner: string, contact: TabletContact, cancel = false) {
		if (this.minimumGap() > 0) return this.buffered.set(owner, contact, cancel);
		if (
			!Number.isInteger(contact.raw_x) ||
			!Number.isInteger(contact.raw_y) ||
			contact.raw_x < 0 ||
			contact.raw_y < 0 ||
			contact.raw_x > 1023 ||
			contact.raw_y > 1023
		)
			throw new Error('Tablet contact outside ten-bit ADC range');
		if (cancel) {
			this.queue = this.queue.filter((q) => q.owner !== owner);
			if (this.held?.owner === owner) this.finishHeld();
			this.gap = 0;
			return;
		}
		if (contact.pressed && this.queue.some((q) => q.down && q.owner !== owner)) return;
		const pending = [...this.queue].reverse().find((q) => q.owner === owner && q.down);
		if (pending) {
			if (contact.pressed) pending.contact = contact;
			else pending.down = false;
			return;
		}
		if (!contact.pressed) {
			if (this.held?.owner === owner) {
				if (this.canRelease()) this.finishHeld();
				else this.held.release = true;
			}
			return;
		}
		if (this.held && !this.held.release) {
			if (this.held.owner !== owner) return;
			this.held.contact = contact;
			this.apply(contact);
			return; // Drag, not another down edge.
		}
		if (this.queue.length >= 16)
			throw new Error('Tap buffer full (16 taps); cancel pending contacts before continuing.');
		this.queue.push({ owner, contact, down: true });
		this.startQueued();
	}
	discard() {
		this.buffered.discard();
		this.held = null;
		this.queue = [];
		this.gap = 0;
	}
	clear() {
		this.buffered.clear();
		this.queue = [];
		this.finishHeld();
		this.gap = 0;
	}
	limitBudget = (requested: number) => {
		if (this.minimumGap() > 0) return this.buffered.limitBudget(requested);
		if (this.gap) return Math.min(requested, this.gap);
		if (this.held && this.held.remaining > 0) return Math.min(requested, this.held.remaining);
		if (this.held?.release && this.held.grace > 0) return Math.min(requested, this.held.grace, 256);
		return this.queue.length ? Math.min(requested, 4096) : requested;
	};
	advance = (used: number) => {
		if (this.minimumGap() > 0) return this.buffered.advance(used);
		this.gap = Math.max(0, this.gap - used);
		if (this.held) {
			if (this.held.release) this.held.grace = Math.max(0, this.held.grace - Math.max(0, used - this.held.remaining));
			this.held.remaining = Math.max(0, this.held.remaining - used);
			if (this.held.release && this.canRelease()) this.finishHeld();
		}
		this.startQueued();
	};
}


type Stroke = { owner: string; contact: TabletContact; remaining: number; release: boolean };

/** One physical pen owns the ADC contact; assisted release uses CPU boundaries. */
class BufferedTabletInputs {
	private held: Stroke | null = null;
	private physical: Stroke | null = null;
	private pending: Stroke[] = [];
	private gap = 0;
	private blocked = false;
	constructor(
		private apply: (contact: TabletContact) => void,
		private minimumHold: () => number = () => 40_000,
		/** Positive gaps serialize distinct strokes. Host assistance, not a
		 * measured controller or silicon timing contract. */
		private minimumGap: () => number = () => 0,
	) {}
	set(owner: string, contact: TabletContact, cancel = false) {
		if (
			!Number.isInteger(contact.raw_x) ||
			!Number.isInteger(contact.raw_y) ||
			contact.raw_x < 0 ||
			contact.raw_y < 0 ||
			contact.raw_x > 1023 ||
			contact.raw_y > 1023
		)
			throw new Error('Tablet contact outside ten-bit ADC range');
		if (this.minimumGap() > 0) {
			this.setBuffered(owner, contact, cancel);
			return;
		}
		if (contact.pressed) {
			if (this.held && this.held.owner !== owner) return;
			this.held = { owner, contact, remaining: this.held?.remaining ?? this.minimumHold(), release: false };
			this.apply(contact);
		} else if (this.held?.owner === owner) {
			if (cancel || this.held.remaining === 0) this.clear();
			else this.held.release = true;
		}
	}

	private setBuffered(owner: string, contact: TabletContact, cancel: boolean) {
		if (cancel) {
			if (this.held?.owner === owner || this.pending.some((stroke) => stroke.owner === owner)) this.clear();
			return;
		}
		if (this.blocked) {
			if (contact.pressed) throw new Error('Tablet stroke buffer overflow; clear input before continuing');
			return;
		}
		if (!contact.pressed) {
			if (this.physical?.owner !== owner) return;
			const stroke = this.physical;
			this.physical = null;
			stroke.release = true;
			if (stroke === this.held && stroke.remaining === 0) this.finishStroke();
			return;
		}
		if (this.physical) {
			if (this.physical.owner !== owner) return;
			this.physical.contact = { ...contact };
			if (this.physical === this.held) this.apply(contact);
			return;
		}
		if (this.pending.length + Number(this.held !== null) >= 128) {
			this.clear();
			this.blocked = true;
			throw new Error('Tablet stroke buffer overflow; clear input before continuing');
		}
		const stroke: Stroke = { owner, contact: { ...contact }, remaining: this.minimumHold(), release: false };
		this.physical = stroke;
		this.pending.push(stroke);
		this.startStroke();
	}

	private startStroke() {
		if (this.held || this.gap) return;
		this.held = this.pending.shift() ?? null;
		if (this.held) this.apply(this.held.contact);
	}

	private finishStroke() {
		const held = this.held!;
		this.held = null;
		this.apply({ ...held.contact, pressed: false });
		this.gap = this.minimumGap();
	}

	discard() {
		this.held = null;
		this.physical = null;
		this.pending = [];
		this.gap = 0;
		this.blocked = false;
	}
	clear() {
		const held = this.held;
		this.discard();
		if (held) this.apply({ ...held.contact, pressed: false });
	}
	limitBudget = (requested: number) => {
		if (this.gap > 0) return Math.min(requested, this.gap);
		return this.held && this.held.remaining > 0 ? Math.min(requested, this.held.remaining) : requested;
	};
	advance = (used: number) => {
		if (this.gap > 0) {
			this.gap = Math.max(0, this.gap - used);
			if (this.gap === 0) this.startStroke();
			return;
		}
		if (!this.held) return;
		this.held.remaining = Math.max(0, this.held.remaining - used);
		if (this.held.release && this.held.remaining === 0) {
			if (this.minimumGap() > 0) this.finishStroke();
			else this.clear();
		}
	};
}
