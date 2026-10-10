"""Mapping/hand-off reference for the explicit PA mixed-ISA exploration.

Native scheduling remains in CoreRuntime; this module mirrors the legacy bus
and the declared gate/far-transfer and shared C/Z/IRAM hypotheses.
"""


def entered_legacy(opcode, target, gate):
    return opcode == 3 and target < 65536 and bool(gate & 2)


def legacy_flags(native_f):
    return bool(native_f & 1), bool(native_f & 2)


def native_flags(carry, zero):
    return int(carry) | (int(zero) << 1)


class LegacyBus:
    def __init__(self, hardware, absent_legacy_media=False, pc=0, paged_rom_data=False):
        self.hardware, self.absent_legacy_media, self.pc = (
            hardware,
            absent_legacy_media,
            pc,
        )
        self.events = []
        self.paged_rom_data = paged_rom_data

    def fetch(self, address):
        hw = self.hardware
        if 0 <= address <= 8191:
            selector, offset = 0, 65536 + address
        elif 8192 <= address <= 16383:
            selector, offset = 0, address
        elif 0xC000 <= address <= 0xFFFF and hw.gate[6] <= 0x37:
            page = hw.gate[6]
            selector, offset = page >> 3, (page & 7) * 16384 + (address & 16383)
        else:
            raise ValueError(f"unqualified legacy fetch {address:04X}")
        if selector not in hw.banks:
            raise ValueError("legacy page selector missing")
        return hw.banks[selector][offset]

    def read(self, address):
        hw = self.hardware
        if 0xC000 <= address <= 0xFFFF and self.paged_rom_data:
            return self.fetch(address)
        if 0x4000 <= address <= 0x7FFF:
            return hw.ram[address - 0x4000]
        if 0x8000 <= address <= 0xBFFF and hw.gate[4] == 1 and self.absent_legacy_media:
            return 255
        raise ValueError(f"unqualified legacy data {address:04X}")

    def write(self, address, value):
        if not 0x4000 <= address <= 0x7FFF:
            raise ValueError("unqualified legacy data write")
        self.hardware.ram[address - 0x4000] = value
        self.events.append(
            {
                "kind": "legacy_data_write",
                "address": address,
                "native_alias": 0x80000 + address - 0x4000,
                "value": value,
                "pc": self.pc,
            }
        )

    def writ(self, dp, p, value):
        register = dp & 255
        allowed = (
            (register == 15 and p == 2 and value in (0, 2, 3))
            or (register in (4, 5) and p in (2, 12, 13) and value <= 3)
            or (register == 6 and p == 2 and value <= 0x37)
            or (register == 0 and p == 2 and value <= 7)
            or (register == 7 and p == 2 and value == 1)
        )
        if not allowed:
            raise ValueError("unqualified legacy WRIT")
        self.hardware.gate[register] = value
        self.events.append(
            {
                "kind": "legacy_WRIT",
                "dp": dp,
                "register": register,
                "p": p,
                "value": value,
                "pc": self.pc,
            }
        )

    def fill(self, start, count, value):
        if count <= 0 or any(
            not 0x4000 <= ((start + i) & 65535) <= 0x7FFF for i in range(count)
        ):
            raise ValueError("unqualified legacy fill")
        for i in range(count):
            self.hardware.ram[((start + i) & 65535) - 0x4000] = value
        self.events.append(
            {
                "kind": "legacy_fill",
                "start": start,
                "count": count,
                "value": value,
                "pc": self.pc,
            }
        )

    def outf(self, value):
        if value != 0:
            raise ValueError("unqualified legacy OUTF")
        self.events.append({"kind": "legacy_OUTF", "value": value, "pc": self.pc})
