import type { TabletContact } from './oz9600_replay';

type Stroke = {
	owner: string;
	contact: TabletContact;
	remaining: number;
	release: boolean;
	points: TabletContact[];
	pointRemaining: number;
	pointHold: number;
};

/** One physical pen owns the ADC contact; assisted release uses CPU boundaries. */
export class TabletInputs {
	private held: Stroke | null = null;
	private physical: Stroke | null = null;
	private pending: Stroke[] = [];
	private gap = 0;
	private blocked = false;
	private pendingPoints = 0;
	constructor(
		private apply: (contact: TabletContact) => void,
		private minimumHold: () => number = () => 40_000,
		/** Positive gaps serialize distinct strokes. Host assistance, not a
		 * measured controller or silicon timing contract. */
		private minimumGap: () => number = () => 0,
		/** Positive cadence preserves deferred motion using scheduler boundaries.
		 * Zero retains the existing immediate/coalesced contact policy. */
		private replayPointHold: () => number = () => 0,
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
			this.held = {
				owner,
				contact,
				remaining: this.held?.remaining ?? this.minimumHold(),
				release: false,
				points: [],
				pointRemaining: 0,
				pointHold: 0,
			};
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
			if (stroke === this.held) this.maybeFinishStroke();
			return;
		}
		if (this.physical) {
			if (this.physical.owner !== owner) return;
			if (this.physical.pointHold > 0) {
				const last = this.physical.points.at(-1) ?? this.physical.contact;
				if (last.raw_x === contact.raw_x && last.raw_y === contact.raw_y) return;
				if (this.pendingPoints >= 4096) {
					this.clear();
					this.blocked = true;
					throw new Error('Tablet movement buffer overflow; clear input before continuing');
				}
				this.physical.points.push({ ...contact });
				this.pendingPoints++;
				if (this.physical === this.held && this.physical.pointRemaining === 0) this.nextPoint();
				return;
			}
			this.physical.contact = { ...contact };
			if (this.physical === this.held) this.apply(contact);
			return;
		}
		if (this.pending.length + Number(this.held !== null) >= 128) {
			this.clear();
			this.blocked = true;
			throw new Error('Tablet stroke buffer overflow; clear input before continuing');
		}
		const stroke: Stroke = {
			owner,
			contact: { ...contact },
			remaining: this.minimumHold(),
			release: false,
			points: [],
			pointRemaining: 0,
			pointHold: this.replayPointHold(),
		};
		this.physical = stroke;
		this.pending.push(stroke);
		this.startStroke();
	}

	private startStroke() {
		if (this.held || this.gap) return;
		this.held = this.pending.shift() ?? null;
		if (this.held) {
			this.held.pointRemaining = this.held.pointHold;
			this.apply(this.held.contact);
		}
	}
	private nextPoint() {
		const held = this.held!;
		const next = held.points.shift();
		if (next) {
			this.pendingPoints--;
			held.contact = next;
			held.pointRemaining = held.pointHold;
			this.apply(next);
		}
	}
	private maybeFinishStroke() {
		const held = this.held;
		if (held?.release && held.remaining === 0 && held.pointRemaining === 0 && !held.points.length) this.finishStroke();
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
		this.pendingPoints = 0;
	}
	clear() {
		const held = this.held;
		this.discard();
		if (held) this.apply({ ...held.contact, pressed: false });
	}
	limitBudget = (requested: number) => {
		if (this.gap > 0) return Math.min(requested, this.gap);
		if (this.held?.pointRemaining) requested = Math.min(requested, this.held.pointRemaining);
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
		if (this.held.pointHold > 0) {
			this.held.pointRemaining = Math.max(0, this.held.pointRemaining - used);
			if (this.held.pointRemaining === 0) this.nextPoint();
			this.maybeFinishStroke();
			return;
		}
		if (this.held.release && this.held.remaining === 0) {
			if (this.minimumGap() > 0) this.finishStroke();
			else this.clear();
		}
	};
}
