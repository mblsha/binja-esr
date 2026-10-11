"""Provisional status command/acknowledgment protocol; no query data stubs."""


class Auxiliary:
    def __init__(self):
        self.written, self.ready_at, self.complete = 0, None, False
        self.response_byte = None
        self.capture_active = False
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
        request_bytes = {
            0x00FE: 2,
            0x01FE: 2,
            0x30FB: 2,
            0x02FE: 2,
            0x00F9: 2,
            0x01F9: 2,
            0x02F9: 2,
            0x03FC: 2,
            0x05FE: 2,
            0x04FE: 3,
            0x03FE: 8,
        }.get(header)
        if request_bytes is None:
            raise ValueError(
                f"unimplemented PI auxiliary command {header:04X}, written mask {self.written:08X}"
            )
        if self.written != (1 << request_bytes) - 1:
            count = {2: "two", 3: "three", 8: "eight"}[request_bytes]
            raise ValueError(
                f"PI auxiliary control requires exactly {count} written bytes"
            )
        self.written = 0
        self.response_byte = None if header == 0x03FC else 1
        if header == 0x03FC:
            self.capture_active = True
        if header == 0x05FE:
            self.capture_active = False
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
        return self.response_byte if self.complete and offset == 0 else None

    def acknowledge(self, value):
        if value != 0xBF:
            raise ValueError(f"unimplemented PI auxiliary acknowledgment {value:02X}")
        self.complete = False
