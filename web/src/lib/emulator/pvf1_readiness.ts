export type PvInputPhase = 'starting' | 'waking' | 'off' | 'ready';
/** ROM-derived observation, not an elapsed-time or electrical timing estimate.
 * F0B6B HALT follows scanner KOL/KOH setup; F0B6C is its stopped PC. Readiness
 * latches across ordinary IRQs, and resets on OFF/load. Calibration still needs
 * live pen input before the first observed foreground wait.
 */
export class PvReadiness {
	private phase: PvInputPhase = 'starting';
	private waitingForPowerOn = false;
	checkpoint() {
		return { phase: this.phase, waitingForPowerOn: this.waitingForPowerOn };
	}
	restoreCheckpoint(input: ReturnType<PvReadiness['checkpoint']>) {
		if (
			!input ||
			!['starting', 'waking', 'off', 'ready'].includes(input.phase) ||
			typeof input.waitingForPowerOn !== 'boolean' ||
			(input.waitingForPowerOn && input.phase !== 'waking')
		)
			throw new Error('Invalid readiness checkpoint');
		this.phase = input.phase;
		this.waitingForPowerOn = input.waitingForPowerOn;
	}
	reset() {
		this.phase = 'starting';
		this.waitingForPowerOn = false;
	}
	noteOn(wasOff: boolean) {
		if (wasOff || this.phase === 'off') {
			this.phase = 'waking';
			this.waitingForPowerOn = true;
		}
	}
	observe(power: string, halted: boolean, pc: number) {
		if (power === 'off' && this.waitingForPowerOn) return this.phase;
		if (power !== 'off') this.waitingForPowerOn = false;
		if (power === 'off') this.phase = 'off';
		else if (this.phase === 'off') this.phase = 'waking';
		if (power !== 'off' && halted && pc === 0xf0b6c) this.phase = 'ready';
		return this.phase;
	}
	status() {
		return this.phase;
	}
	canDeliver() {
		return this.phase !== 'waking';
	}
}
