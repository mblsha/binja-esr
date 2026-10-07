"""Register-only UART reference, independent of IOCS and software receive rings.

The scheduler clock/divisor and two-bit holding-register load latency are explicit
functional hypotheses. This models byte transport, not electrical waveforms.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

BAUD = (0, 300, 600, 1200, 2400, 4800, 9600, 19200)
QUEUE_LIMIT = 4096


@dataclass
class RxByte:
    value: int
    parity_error: bool = False
    overrun_error: bool = False
    framing_error: bool = False


class Uart:
    def __init__(self, timebase_hz=1_024_000, baud_divisor=2):
        if timebase_hz <= 0 or baud_divisor <= 0:
            raise ValueError("UART clock and divisor must be positive")
        self.timebase_hz, self.baud_divisor = timebase_hz, baud_divisor
        self.control = self.last_tx = self.rx_data = self.rx_errors = 0
        self.rx_latch = None
        self.pending_rx = deque()
        self.rx_remaining = None
        self.tx_holding = self.tx_shift = None
        self.tx_load_remaining = self.tx_remaining = None
        self.shift_break = False
        self.completed_tx = deque()
        self.rejected_rx = self.rejected_tx = self.dropped_tx = 0
        self.suppressed_break_frames = 0

    def baud(self):
        return BAUD[(self.control >> 4) & 7] // self.baud_divisor

    def bit_units(self):
        baud = max(self.baud(), 1)
        return (self.timebase_hz + baud - 1) // baud

    def frame_units(self):
        data = 7 if self.control & 2 else 8
        parity = not self.control & 8
        stop = 2 if self.control & 1 else 1
        return (1 + data + parity + stop) * self.bit_units()

    def data_mask(self):
        return 127 if self.control & 2 else 255

    def status(self):
        return (
            0x20 * (self.rx_latch is not None)
            | 0x10 * (self.tx_shift is None)
            | 8 * (self.tx_holding is None)
            | self.rx_errors
        )

    def write_control(self, value):
        self.control = value & 255
        if not value & 0x70:
            self.rx_latch = None
            self.rx_data = self.rx_errors = 0
            self.pending_rx.clear()
            self.rx_remaining = None
            self.tx_holding = self.tx_shift = None
            self.tx_load_remaining = self.tx_remaining = None
            self.shift_break = False
        elif value & 0x80 and self.tx_shift is not None:
            self.shift_break = True

    def write_tx(self, value):
        self.last_tx = value & 255
        if not self.baud() or self.tx_holding is not None:
            self.rejected_tx += 1
            return False
        self.tx_holding = value & self.data_mask()
        if self.tx_shift is None:
            self.tx_load_remaining = 2 * self.bit_units()
        return True

    def queue_rx(self, entry):
        if not self.baud() or len(self.pending_rx) >= QUEUE_LIMIT:
            self.rejected_rx += 1
            return False
        self.pending_rx.append(deepcopy(entry))
        if self.rx_remaining is None:
            self.rx_remaining = self.frame_units()
        return True

    def consume_rx(self):
        entry, self.rx_latch = self.rx_latch, None
        return entry

    def read_rx(self):
        self.rx_latch = None
        return self.rx_data

    def take_tx(self):
        return self.completed_tx.popleft() if self.completed_tx else None

    def pending_tx(self):
        return [
            value for value in (self.tx_shift, self.tx_holding) if value is not None
        ]

    def load_shift(self, events):
        assert self.tx_holding is not None
        self.tx_shift, self.tx_holding = self.tx_holding, None
        self.tx_remaining = self.frame_units()
        self.tx_load_remaining = None
        self.shift_break = bool(self.control & 0x80)
        events.append(("TxReady", self.tx_shift))

    def advance(self, units):
        if units < 0:
            raise ValueError("UART time must not run backward")
        events = []
        while units:
            fields = ("rx_remaining", "tx_load_remaining", "tx_remaining")
            deadlines = [
                getattr(self, key) for key in fields if getattr(self, key) is not None
            ]
            if not deadlines:
                break
            step = min(units, min(deadlines))
            for key in fields:
                value = getattr(self, key)
                if value is not None:
                    setattr(self, key, value - step)
            units -= step
            if self.rx_remaining == 0:
                entry = self.pending_rx.popleft()
                entry.value &= self.data_mask()
                entry.overrun_error |= self.rx_latch is not None
                self.rx_errors = (
                    int(entry.parity_error)
                    | 2 * entry.overrun_error
                    | 4 * entry.framing_error
                )
                self.rx_data, self.rx_latch = entry.value, entry
                self.rx_remaining = self.frame_units() if self.pending_rx else None
                events.append(("RxReady", entry.value))
            if self.tx_load_remaining == 0:
                self.load_shift(events)
            if self.tx_remaining == 0:
                byte, self.tx_shift = self.tx_shift, None
                self.tx_remaining = None
                if self.shift_break:
                    self.suppressed_break_frames += 1
                else:
                    if len(self.completed_tx) == QUEUE_LIMIT:
                        self.completed_tx.popleft()
                        self.dropped_tx += 1
                    self.completed_tx.append(byte)
                    events.append(("TxComplete", byte))
                if self.tx_holding is not None:
                    self.load_shift(events)
        return events

    def report(self):
        return dict(
            control=self.control,
            status=self.status(),
            rx_data=self.rx_data,
            last_tx=self.last_tx,
            baud=self.baud(),
            timebase_hz=self.timebase_hz,
            baud_divisor=self.baud_divisor,
            bit_units=self.bit_units(),
            frame_units=self.frame_units(),
            tx_holding=self.tx_holding,
            tx_shift=self.tx_shift,
            rx_remaining=self.rx_remaining,
            tx_load_remaining=self.tx_load_remaining,
            tx_remaining=self.tx_remaining,
            pending_rx=len(self.pending_rx),
            rx_ready=self.rx_latch is not None,
            completed_tx=len(self.completed_tx),
            rejected_rx=self.rejected_rx,
            rejected_tx=self.rejected_tx,
            dropped_tx=self.dropped_tx,
            suppressed_break_frames=self.suppressed_break_frames,
        )
