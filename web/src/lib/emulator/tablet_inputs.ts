import type { TabletContact } from './oz9600_replay';

/** One physical pen owns the ADC contact; assisted release uses CPU boundaries. */
export class TabletInputs {
	private held: { owner: string; contact: TabletContact; remaining: number; release: boolean } | null = null;
	constructor(private apply: (contact: TabletContact) => void) {}
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
		if (contact.pressed) {
			if (this.held && this.held.owner !== owner) return;
			this.held = { owner, contact, remaining: this.held?.remaining ?? 40_000, release: false };
			this.apply(contact);
		} else if (this.held?.owner === owner) {
			if (cancel || this.held.remaining === 0) this.clear();
			else this.held.release = true;
		}
	}
	discard() {
		this.held = null;
	}
	clear() {
		const held = this.held;
		this.held = null;
		if (held) this.apply({ ...held.contact, pressed: false });
	}
	limitBudget = (requested: number) =>
		this.held && this.held.remaining > 0 ? Math.min(requested, this.held.remaining) : requested;
	advance = (used: number) => {
		if (!this.held) return;
		this.held.remaining = Math.max(0, this.held.remaining - used);
		if (this.held.release && this.held.remaining === 0) this.clear();
	};
}
