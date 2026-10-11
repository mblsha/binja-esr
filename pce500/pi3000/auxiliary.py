"""Provisional startup command/acknowledgment protocol; no query data stubs."""


class Auxiliary:
    def __init__(self):
        self.written, self.ready_at, self.complete = 0, None, False
        self.response_bytes = bytes(32)
        self.accepted = []
        self.accepted_count = 0

    def mark_written(self, offset):
        self.written |= 1 << offset

    def start(self, request, cycle):
        if self.ready_at is not None or self.complete:
            raise ValueError(
                "PI auxiliary start before prior completion acknowledgment"
            )
        header = int.from_bytes(request[:2], "little")
        if header not in (0x00FE, 0x01FE, 0x30FB, 0x02FE):
            raise ValueError(
                f"unimplemented PI auxiliary command {header:04X}, written mask {self.written:08X}"
            )
        if self.written != 3:
            raise ValueError(
                "PI auxiliary startup control requires exactly two written bytes"
            )
        self.written = 0
        self.response_bytes = b"\x01" + bytes(31)
        self.ready_at = cycle + 32
        self.accepted_count += 1
        if len(self.accepted) < 256:
            self.accepted.append(header)

    def advance(self, cycle):
        if self.ready_at is not None and cycle >= self.ready_at:
            self.ready_at, self.complete = None, True

    def status(self):
        return 2 | (0x80 if self.ready_at is not None else 0x40 if self.complete else 0)

    def response(self, offset):
        return self.response_bytes[0] if self.complete and offset == 0 else None

    def acknowledge(self, value):
        if value != 0xBF:
            raise ValueError(f"unimplemented PI auxiliary acknowledgment {value:02X}")
        self.complete = False
