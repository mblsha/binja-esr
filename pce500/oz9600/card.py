"""Explicit OZ-707 logical cartridge view, independently mirroring Rust.

Canonical upper SRAM, lower read-only mirror and SSR bit-1 presence are
ROM-qualified hypotheses, not a physical mapper/wiring claim.
"""

import hashlib

ROM_SIZE = 0x20000
SRAM_SIZE = 0x8000
ROM_SHA256 = "a8a1afb91bf39f07f528a9690a60d45c6cc61232a126fb0fe1ccff95ec112d9d"


class Oz707Card:
    def __init__(self, rom: bytes, sram: bytes):
        if len(rom) != ROM_SIZE or len(sram) != SRAM_SIZE:
            raise ValueError("OZ-707 requires a 128 KiB ROM and 32 KiB SRAM image")
        if hashlib.sha256(rom).hexdigest() != ROM_SHA256:
            raise ValueError("OZ-707 ROM identity does not match the verified capture")
        self.rom, self.sram = bytes(rom), bytearray(sram)

    def read(self, address: int) -> int | None:
        if 0x30000 <= address <= 0x3FFFF:
            return self.sram[(address - 0x30000) % SRAM_SIZE]
        if 0x40000 <= address <= 0x7FFFF:
            return self.rom[(address - 0x40000) % ROM_SIZE]
        return None

    def write(self, address: int, value: int) -> None:
        if 0x38000 <= address <= 0x3FFFF:
            self.sram[address - 0x38000] = value

    def presence_ssr(self, ssr: int) -> int:
        return ssr | 2
