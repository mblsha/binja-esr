"""PV-F1 RTC A2 divider candidate; calendar and silicon timing are unqualified."""


class HalfSecondClock:
    def __init__(self):
        self.period = None
        self.phase = self.ticks = self.elapsed = 0

    def configure(self, period):
        if period is not None and (
            not isinstance(period, int) or not 0 < period < 2**64
        ):
            raise ValueError("PV-F1 RTC candidate requires a positive period")
        self.period, self.phase = period, 0

    def advance(self, elapsed):
        if not isinstance(elapsed, int) or not 0 <= elapsed < 2**64:
            raise ValueError("Invalid RTC elapsed timing units")
        if self.period is None:
            return False
        ticks, self.phase = divmod(self.phase + elapsed, self.period)
        self.ticks = (self.ticks + ticks) % 2**64
        self.elapsed = (self.elapsed + elapsed) % 2**64
        return ticks != 0

    def snapshot(self):
        return {
            "half_second_period_candidate": self.period,
            "phase_units": self.phase,
            "ticks": self.ticks,
            "elapsed_timing_units": self.elapsed,
            "cause_A": 4,
            "physical_verified": False,
            "calendar_progression": False,
        }
