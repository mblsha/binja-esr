"""PV-F1 confirmed status controls over the shared handshake engine."""

from pce500.pi3000.auxiliary import Auxiliary as ControlEngine


class Auxiliary:
    def __init__(self, control=False):
        self.control = control
        self.request, self.controls = bytearray(32), bytearray(3)
        self.enabled, self.starts = True, 0
        self.engine = ControlEngine() if control else None

    def peek(self, address):
        if address == 0x8A00:
            status = self.engine.status() if self.engine else 0
            return (status & ~2) | (2 if self.enabled else 0)
        if address in (0x8A04, 0x8A05, 0x8A09):
            return 0
        if address == 0x8A40 and self.engine:
            return self.engine.response(0)
        return None

    def advance(self, cycle):
        if self.engine:
            self.engine.advance(cycle)

    def write(self, address, value, cycle):
        if address == 0x8A01 and value in (2, 0xFD):
            self.__init__(self.control)
            self.enabled = value == 2
        elif 0x8A20 <= address <= 0x8A3F:
            offset = address - 0x8A20
            self.request[offset] = value
            if self.engine:
                self.engine.mark_written(offset)
        elif 0x8A06 <= address <= 0x8A08:
            self.controls[address - 0x8A06] = value
        elif address == 0x8A02 and value == 0x80:
            if self.engine:
                command = int.from_bytes(self.request[:2], "little")
                if command not in (
                    0x03FC,
                    0x05FE,
                    0x00FE,
                    0x01FE,
                    0x02FE,
                    0x30FB,
                    0x00F9,
                    0x01F9,
                    0x02F9,
                    0x04FE,
                    0x03FE,
                ):
                    raise ValueError(
                        f"unrecovered PV-F1 auxiliary command {command:04X}"
                    )
                self.engine.start(self.request, cycle)
            else:
                self.starts += 1
        elif address == 0x8A03 and value == 0xBF:
            if self.engine:
                self.engine.acknowledge(value)
        else:
            raise ValueError(
                f"unrecovered PV-F1 auxiliary write {address:05X}={value:02X}"
            )
