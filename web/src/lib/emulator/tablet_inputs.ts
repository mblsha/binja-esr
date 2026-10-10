import type { TabletContact } from './oz9600_replay';

type HeldPen = { owner: string; contact: TabletContact; remaining: number; grace: number; release: boolean };
type QueuedPen = { owner: string; contact: TabletContact; down: boolean };

/** Host click assistance. Raw ROM replay bypasses this adapter. No CPU runs on input. */
export class TabletInputs {
	private held: HeldPen | null = null;
	private queue: QueuedPen[] = [];
	private gap = 0;
	constructor(
		private apply: (contact: TabletContact) => void,
		private ready: () => boolean = () => true,
		private mailboxBusy: () => boolean = () => false,
	) {}
	private canRelease() {
		return this.held?.remaining === 0 && (this.ready() || this.held.grace === 0);
	}
	private startQueued() {
		if (this.held || this.gap || !this.queue.length || this.mailboxBusy()) return;
		const next = this.queue.shift()!;
		this.held = { owner: next.owner, contact: next.contact, remaining: 40_000, grace: 400_000, release: !next.down };
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
		this.held = null;
		this.queue = [];
		this.gap = 0;
	}
	clear() {
		this.queue = [];
		this.finishHeld();
		this.gap = 0;
	}
	limitBudget = (requested: number) => {
		if (this.gap) return Math.min(requested, this.gap);
		if (this.held && this.held.remaining > 0) return Math.min(requested, this.held.remaining);
		if (this.held?.release && this.held.grace > 0) return Math.min(requested, this.held.grace, 256);
		return this.queue.length ? Math.min(requested, 4096) : requested;
	};
	advance = (used: number) => {
		this.gap = Math.max(0, this.gap - used);
		if (this.held) {
			if (this.held.release) this.held.grace = Math.max(0, this.held.grace - Math.max(0, used - this.held.remaining));
			this.held.remaining = Math.max(0, this.held.remaining - used);
			if (this.held.release && this.canRelease()) this.finishHeld();
		}
		this.startQueued();
	};
}
