"""PV-F1 RTC A2 divider candidate; calendar and silicon timing are unqualified."""

from pce500.oz9600.rtc import CONTROL, HOLD


class CalendarClock:
    """Nominal calendar candidate with a frozen HOLD phase and no host time."""

    def __init__(self):
        self.period = None
        self.phase = self.seconds = self.held = self.uninitialized = self.elapsed = 0

    def configure(self, period):
        if period is not None and (
            not isinstance(period, int) or not 0 < period < 2**64
        ):
            raise ValueError("PV-F1 calendar candidate requires a positive period")
        self.period, self.phase = period, 0

    def advance(self, rtc, elapsed):
        if not isinstance(elapsed, int) or not 0 <= elapsed < 2**64:
            raise ValueError("Invalid RTC elapsed timing units")
        if self.period is None:
            return
        if rtc.read(CONTROL) & HOLD:
            self.held = (self.held + elapsed) % 2**64
        else:
            try:
                rtc.clock()
            except ValueError:
                self.uninitialized = (self.uninitialized + elapsed) % 2**64
            else:
                seconds, phase = divmod(self.phase + elapsed, self.period)
                if seconds:
                    rtc.advance_experimental_seconds(seconds)
                self.phase = phase
                self.seconds = (self.seconds + seconds) % 2**64
        self.elapsed = (self.elapsed + elapsed) % 2**64

    def snapshot(self):
        return {
            "second_period_candidate": self.period,
            "calendar_progression": self.period is not None,
            "phase_units": self.phase,
            "elapsed_seconds": self.seconds,
            "held_units": self.held,
            "uninitialized_units": self.uninitialized,
            "elapsed_timing_units": self.elapsed,
            "physical_verified": False,
            "HOLD_freeze_candidate": True,
        }


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
        }


def alarm_wake_level(rtc):
    """Candidate OFF wake from enabled A0/A1; periodic sources never wake."""
    return bool(rtc.read(8) & rtc.read(10) & 3)
