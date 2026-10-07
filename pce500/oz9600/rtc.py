"""Packed binary calendar and mask/cause latches, matching ROM getter behavior."""

from dataclasses import dataclass, replace

MASK_A, MASK_B, CAUSE_A, CAUSE_B, CONTROL, STATUS, HOLD = 8, 9, 10, 11, 12, 13, 128


@dataclass(frozen=True)
class Clock:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    weekday: int

    def month_length(self):
        return (
            30
            if self.month in (4, 6, 9, 11)
            else (29 if self.year % 4 == 0 else 28)
            if self.month == 2
            else 31
        )

    def validate(self):
        if not (
            1900 <= self.year <= 2099
            and 1 <= self.month <= 12
            and 1 <= self.day <= self.month_length()
            and 0 <= self.hour < 24
            and 0 <= self.minute < 60
            and 0 <= self.second < 60
            and 0 <= self.weekday < 7
        ):
            raise ValueError("invalid ordinary RTC date/time")
        return self

    @classmethod
    def decode(cls, data):
        return cls(
            1900 + (data[3] >> 4 | (data[4] & 15) << 4),
            data[3] & 15,
            data[2] >> 3,
            data[1] >> 6 | (data[2] & 7) << 2,
            data[1] & 63,
            data[0] & 63,
            data[4] >> 4 & 7,
        ).validate()

    def encode(self, previous):
        self.validate()
        year = self.year - 1900
        return bytes(
            [
                (previous[0] & 192) | self.second,
                self.minute | (self.hour & 3) << 6,
                self.hour >> 2 | self.day << 3,
                self.month | (year & 15) << 4,
                previous[4] & 128 | self.weekday << 4 | year >> 4,
            ]
        )

    def advance_seconds(self, seconds):
        self.validate()
        total = self.hour * 3600 + self.minute * 60 + self.second + seconds
        if total > 2**64 - 1 or seconds < 0:
            raise ValueError("RTC elapsed time overflow")
        clock = replace(
            self,
            hour=total % 86400 // 3600,
            minute=total % 3600 // 60,
            second=total % 60,
        )
        for _ in range(total // 86400):
            day, month, year = clock.day + 1, clock.month, clock.year
            if day > clock.month_length():
                day, month = 1, month + 1
                if month > 12:
                    month, year = 1, year + 1
                    if year > 2099:
                        raise ValueError(
                            "RTC year rollover beyond 2099 is not qualified"
                        )
            clock = replace(
                clock, day=day, month=month, year=year, weekday=(clock.weekday + 1) % 7
            )
        return clock


class Rtc:
    def __init__(self):
        self.backing = bytearray(32)

    def read(self, offset):
        offset &= 31
        return (
            (self.backing[STATUS] & ~HOLD) | (self.backing[CONTROL] & HOLD)
            if offset == STATUS
            else self.backing[offset]
        )

    def write(self, offset, value):
        self.backing[offset & 31] = value

    def registers(self):
        return bytes(self.read(n) for n in range(32))

    def clock(self):
        return Clock.decode(self.backing[:5])

    def advance_seconds(self, seconds):
        if self.backing[CONTROL] & HOLD:
            raise ValueError("RTC time advancement during hold is not qualified")
        self.backing[:5] = (
            self.clock().advance_seconds(seconds).encode(self.backing[:5])
        )

    def advance_experimental_seconds(self, seconds):
        """Opt-in cyclic/deadline policy plus observed Schedule seconds-3F form.

        A crossed second inside the programmed minute matches. Other wildcard
        fields and exact silicon reassert timing remain unqualified.
        """
        if self.backing[CONTROL] & HOLD:
            raise ValueError("RTC time advancement during hold is not qualified")
        old = self.clock()
        new = old.advance_seconds(seconds)
        encoded = new.encode(self.backing[:5])
        causes = 0

        def key(c):
            return (c.year, c.month, c.day, c.hour, c.minute, c.second)

        if seconds:
            causes |= 8
            if seconds >= 60 - old.second:
                causes |= 16
            if seconds >= 3600 - (old.minute * 60 + old.second):
                causes |= 32
            if seconds >= 86400 - (old.hour * 3600 + old.minute * 60 + old.second):
                causes |= 64
            for channel, offset in enumerate((16, 24)):
                if self.backing[offset + 5] != 7:
                    continue
                raw = bytearray(self.backing[offset : offset + 5])
                any_second = raw[0] & 0x3F == 0x3F
                if any_second:
                    raw[0] &= 0xC0
                try:
                    alarm = Clock.decode(raw)
                except ValueError:
                    continue
                matched = (
                    key(old) < (*key(alarm)[:5], 59) and key(alarm) <= key(new)
                    if any_second
                    else key(old) < key(alarm) <= key(new)
                )
                if matched:
                    causes |= 1 << channel
        self.backing[:5] = encoded
        self.raise_causes(causes, 0)

    def raise_causes(self, a, b):
        self.backing[CAUSE_A] |= a
        self.backing[CAUSE_B] |= b

    def interrupt_asserted(self):
        return bool(
            (
                (self.backing[MASK_A] & self.backing[CAUSE_A])
                | (self.backing[MASK_B] & self.backing[CAUSE_B])
            )
            & 127
        )
