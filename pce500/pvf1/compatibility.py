"""PV-F1 split legacy fetch, scalar outputs and shared IRAM/C/Z candidates.

The shared Python legacy CPU owns instruction execution. Native scheduling
remains in CoreRuntime; neither legacy timing nor physical handoff is implied.
"""

from pce500.pa9500.compat_probe import StartupState, step


def entered_legacy(opcode, target, encoded_target, gate):
    return (
        opcode == 3 and target == encoded_target and target < 65536 and bool(gate & 2)
    )


class Coordinator:
    def __init__(self, hardware):
        self.hardware = hardware
        self.state = StartupState(0)
        self.active = False
        self.boundaries = 0

    def enter(self, internal, flags, target):
        self.state.pc = target
        self.state.internal[:] = internal[:128]
        self.state.carry, self.state.zero = bool(flags & 1), bool(flags & 2)
        self.active = True

    def step_legacy(self):
        target = step(self.state, LegacyBus(self.hardware))
        self.boundaries += 1
        if target is not None:
            if self.hardware.gate[15] & 2:
                raise ValueError("PV-F1 legacy return did not clear gate bit1")
            self.active = False
        return target


class LegacyBus:
    def __init__(self, hardware):
        self.hardware = hardware
        self.ram, self.gate, self.banks = hardware.ram, hardware.gate, hardware.banks
        self.outputs = []

    def fetch(self, address):
        if address < 0x2000:
            selector, offset = 0, 0x10000 + address
        elif address < 0x4000:
            selector, offset = 0, address
        elif address >= 0xC000 and self.gate[6] <= 0x7F:
            page = self.gate[6]
            selector, offset = page >> 3, (page & 7) * 0x4000 + (address & 0x3FFF)
        else:
            raise ValueError(f"unqualified PV-F1 fetch {address:04X}")
        return self.banks[selector][offset]

    def read(self, address):
        if address >= 0xC000:
            return self.fetch(address)
        if 0x4000 <= address <= 0x7FFF:
            return self.ram[address - 0x4000]
        if 0x8000 <= address <= 0xBFFF and self.gate[4] == 1:
            return 255
        raise ValueError(f"unqualified PV-F1 data {address:04X}")

    def write(self, address, value):
        if not 0x4000 <= address <= 0x7FFF:
            raise ValueError(f"unqualified PV-F1 write {address:04X}")
        self.ram[address - 0x4000] = value

    def fill(self, start, count, value):
        addresses = [(start + i) & 65535 for i in range(count)]
        if not count or any(not 0x4000 <= a <= 0x7FFF for a in addresses):
            raise ValueError("unqualified PV-F1 fill span")
        for address in addresses:
            self.write(address, value)

    def writ(self, dp, p, value):
        port = dp & 255
        allowed = (
            (port == 15 and p == 2 and value in (0, 2, 3))
            or (port in (4, 5) and p in (2, 12, 13) and value <= 3)
            or (port == 6 and p == 2 and value <= 0x7F)
            or (port == 0 and p == 2 and value <= 7)
            or (port == 7 and p == 2 and value == 1)
        )
        if not allowed:
            raise ValueError(f"unqualified WRIT {dp:04X}/{p:02X}/{value:02X}")
        self.gate[port] = value
        self.outputs.append({"port": port, "p": p, "value": value})

    def outf(self, value):
        if value:
            raise ValueError("unqualified OUTF")
