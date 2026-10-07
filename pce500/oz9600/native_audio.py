"""Pure host PCM queue reference; no CPU/peripheral access or audio device IO.

Matches the native window's bounded queue, rational zero-order rate conversion
and 40 Hz DC blocker. It does not model the physical buzzer or calibrate clocks.
"""

from __future__ import annotations

import math
import struct
from collections import deque

SAMPLE_RATE = 48_000
CAPACITY = SAMPLE_RATE // 10


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


class Queue:
    def __init__(self, output_rate: int):
        if output_rate <= 0:
            raise ValueError("Output rate must be positive")
        self.output_rate = output_rate
        self.alpha = f32(math.exp(f32(f32(-f32(math.tau) * 40) / output_rate)))
        self.samples: deque[int] = deque()
        self.expected: int | None = None
        self.phase = 0
        self.current: float | None = None
        self.previous_input = self.previous_output = 0.0
        self.dropped = self.rendered = self.nonzero = 0

    def clear(self):
        self.samples.clear()
        self.expected = None
        self.phase = 0
        self.current = None
        self.previous_input = self.previous_output = 0.0

    def push(
        self, first_sample: int, samples: list[int], sample_rate: int = SAMPLE_RATE
    ):
        if sample_rate != SAMPLE_RATE:
            self.clear()
            return
        if self.expected is not None and self.expected != first_sample:
            self.clear()
        self.expected = (first_sample + len(samples)) % (1 << 64)
        drop = max(0, len(self.samples) + len(samples) - CAPACITY)
        if drop:
            for _ in range(min(drop, len(self.samples))):
                self.samples.popleft()
            self.dropped += drop
            self.phase = 0
            self.current = None
            self.previous_input = self.previous_output = 0.0
        self.samples.extend(samples[max(0, len(samples) - CAPACITY) :])

    def _take(self):
        return self.samples.popleft() / 32768 if self.samples else None

    def next(self):
        if self.current is None:
            self.current = self._take()
        if self.current is None:
            self.phase = 0
            self.previous_input = self.previous_output = 0.0
            self.rendered += 1
            return 0.0
        output = f32(
            self.alpha
            * f32(f32(self.previous_output + self.current) - self.previous_input)
        )
        self.previous_input, self.previous_output = self.current, output
        self.phase += SAMPLE_RATE
        while self.phase >= self.output_rate:
            self.phase -= self.output_rate
            self.current = self._take()
            if self.current is None:
                self.phase = 0
                break
        self.rendered += 1
        self.nonzero += abs(output) > 0.0001
        return max(-1.0, min(1.0, output))
