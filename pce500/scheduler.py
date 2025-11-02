"""Timer and interrupt scheduler for the PC-E500 emulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List


class TimerSource(Enum):
    """Supported hardware timer sources."""

    MTI = auto()
    STI = auto()


@dataclass
class TimerScheduler:
    """Deterministic timer scheduler used by the emulator."""

    mti_period: int
    sti_period: int
    enabled: bool = True

    def __post_init__(self) -> None:
        self.mti_period = int(self.mti_period)
        self.sti_period = int(self.sti_period)
        self._next_mti = self.mti_period
        self._next_sti = self.sti_period

    def reset(self, *, cycle_base: int = 0) -> None:
        """Reset timer state to default offsets."""

        self._next_mti = cycle_base + self.mti_period
        self._next_sti = cycle_base + self.sti_period

    def advance(self, cycle_count: int) -> Iterable[TimerSource]:
        """Advance timers to ``cycle_count`` and yield fired sources."""

        if not self.enabled:
            return []

        fired: List[TimerSource] = []

        if cycle_count >= self._next_mti and self.mti_period > 0:
            while cycle_count >= self._next_mti:
                self._next_mti += self.mti_period
            fired.append(TimerSource.MTI)

        if cycle_count >= self._next_sti and self.sti_period > 0:
            while cycle_count >= self._next_sti:
                self._next_sti += self.sti_period
            fired.append(TimerSource.STI)

        return fired

    @property
    def next_mti(self) -> int:
        return self._next_mti

    @next_mti.setter
    def next_mti(self, value: int) -> None:
        self._next_mti = int(value)

    @property
    def next_sti(self) -> int:
        return self._next_sti

    @next_sti.setter
    def next_sti(self, value: int) -> None:
        self._next_sti = int(value)


__all__ = ["TimerScheduler", "TimerSource"]
