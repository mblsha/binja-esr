"""PA-9C3 logical ROM/SRAM view; physical aliases/presence remain provisional."""

import hashlib

ROM_SHA256 = "55f3c327abce50a1d44b0076f3bb6384ef64e4d7af3edd4ad9b53dc84ae87332"


class Pa9c3Card:
    def __init__(self, rom, sram):
        if (
            len(rom) != 0x20000
            or len(sram) != 0x8000
            or hashlib.sha256(rom).hexdigest() != ROM_SHA256
        ):
            raise ValueError(
                "PA-9C3 requires the verified 128 KiB ROM and a 32 KiB SRAM image"
            )
        self.rom, self.sram = bytes(rom), bytearray(sram)

    def read(self, address):
        if 0x30000 <= address <= 0x3FFFF:
            return self.sram[(address - 0x30000) % 0x8000]
        if 0x40000 <= address <= 0x7FFFF:
            return self.rom[(address - 0x40000) % 0x20000]
        return None

    def write(self, address, value):
        if 0x38000 <= address <= 0x3FFFF:
            self.sram[address - 0x38000] = value
