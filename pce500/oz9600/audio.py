"""Nominal digital capture of software and provisional clocked CO modes."""

from collections import deque

SAMPLE_RATE = 48_000
TIMEBASE_HZ = 1_024_000
QUEUE_CAPACITY = SAMPLE_RATE * 2
QUALIFICATION = (
    "Nominal digital CO: 000/001 low/high, provisional 010/011 2/4 kHz, "
    "100/101 low/high, 110/111 explicit CI input. CI wiring, physical mode table, "
    "pitch, amplitude and speaker remain unqualified."
)
U64_MASK = 2**64 - 1


class AudioCapture:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.phase = self.area = 0
        self.samples = deque()
        self.total_samples = self.dropped_samples = 0
        self.elapsed_units = self.unsupported_units = 0
        self.provisional_units = 0
        self.mode = None
        self.oscillator_phase = 0
        self.oscillator_high = False

    def set_enabled(self, enabled):
        self.__init__(enabled)

    def reset(self):
        self.set_enabled(self.enabled)

    def status(self):
        return dict(
            enabled=self.enabled,
            sample_rate=SAMPLE_RATE,
            timebase_hz=TIMEBASE_HZ,
            elapsed_units=self.elapsed_units,
            total_samples=self.total_samples,
            queued_samples=len(self.samples),
            dropped_samples=self.dropped_samples,
            unsupported_units=self.unsupported_units,
            provisional_units=self.provisional_units,
            qualification=QUALIFICATION,
        )

    def advance(self, units, scr, off=False, ci=False):
        if not self.enabled:
            return
        mode = (scr >> 4) & 7
        if not off and mode >= 6:
            self.unsupported_units = (self.unsupported_units + units) & U64_MASK
        if not off and mode > 1:
            self.provisional_units = (self.provisional_units + units) & U64_MASK
        self.elapsed_units = (self.elapsed_units + units) & U64_MASK
        active_mode = None if off else mode
        if active_mode != self.mode:
            self.mode = active_mode
            self.oscillator_phase = 0
            self.oscillator_high = True
        if not off and mode in (2, 3):
            half_period = TIMEBASE_HZ // (4000 if mode == 2 else 8000)
            while units:
                part = min(units, half_period - self.oscillator_phase)
                self._integrate(part, 8192 if self.oscillator_high else 0)
                self.oscillator_phase += part
                units -= part
                if self.oscillator_phase == half_period:
                    self.oscillator_phase = 0
                    self.oscillator_high = not self.oscillator_high
        else:
            high = not off and (mode in (1, 5) or (mode >= 6 and ci))
            self._integrate(units, 8192 if high else 0)

    def _integrate(self, units, level):
        remaining = units * SAMPLE_RATE
        while remaining:
            part = min(remaining, TIMEBASE_HZ - self.phase)
            self.area += level * part
            self.phase += part
            remaining -= part
            if self.phase == TIMEBASE_HZ:
                if len(self.samples) == QUEUE_CAPACITY:
                    self.samples.popleft()
                    self.dropped_samples = (self.dropped_samples + 1) & U64_MASK
                self.samples.append(self.area // TIMEBASE_HZ)
                self.total_samples = (self.total_samples + 1) & U64_MASK
                self.phase = self.area = 0

    def take(self):
        result = dict(
            sample_rate=SAMPLE_RATE,
            first_sample=(self.total_samples - len(self.samples)) & U64_MASK,
            total_samples=self.total_samples,
            dropped_samples=self.dropped_samples,
            samples=list(self.samples),
        )
        self.samples.clear()
        return result
