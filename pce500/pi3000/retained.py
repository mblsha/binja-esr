"""PI restart backing: separate scratch/main RAM and raw RTC registers.

This emulator format keeps CPU, mapper, LCD and input state fresh; physical
battery-backed regions are not established by its logical layout.
"""

import hashlib

MAGIC = b"PIRAM01\0"
LAYOUT_TAG = b"pi3000-scratch16k-main256k-rtc32-1"
MAIN_RAM_SIZE, SCRATCH_SIZE, RTC_SIZE = 0x40000, 0x4000, 32
HEADER_SIZE = 104
PAYLOAD_SIZE = MAIN_RAM_SIZE + SCRATCH_SIZE + RTC_SIZE
IMAGE_SIZE = HEADER_SIZE + PAYLOAD_SIZE


def bank_identity(hw):
    identity = hashlib.sha256(LAYOUT_TAG)
    for selector, bank in sorted(hw.banks.items()):
        identity.update(bytes([selector]))
        identity.update(bank)
    return identity.digest()


def retained_state(external, hw):
    payload = bytes(hw.ram) + bytes(hw.scratch) + bytes(hw.rtc.backing)
    return (
        MAGIC
        + hashlib.sha256(external[0xE0000:0x100000]).digest()
        + bank_identity(hw)
        + hashlib.sha256(payload).digest()
        + payload
    )


def restore_retained_state(external, hw, image, *, instructions=0, timing_units=0):
    if instructions or timing_units:
        raise ValueError(
            "PI retained state must be restored before the first CPU boundary"
        )
    if len(image) != IMAGE_SIZE or image[:8] != MAGIC:
        raise ValueError("PI retained-state version/length mismatch")
    payload = image[HEADER_SIZE:]
    if (
        image[8:40] != hashlib.sha256(external[0xE0000:0x100000]).digest()
        or image[40:72] != bank_identity(hw)
        or image[72:104] != hashlib.sha256(payload).digest()
    ):
        raise ValueError(
            "PI retained-state ROM/bank/layout identity or payload checksum mismatch"
        )
    hw.ram[:] = payload[:MAIN_RAM_SIZE]
    hw.scratch[:] = payload[MAIN_RAM_SIZE : MAIN_RAM_SIZE + SCRATCH_SIZE]
    hw.rtc.backing[:] = payload[-RTC_SIZE:]
