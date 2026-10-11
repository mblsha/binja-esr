"""PA bootstrap high RAM and RTC backing; CPU/card/scratch remain separate."""

import hashlib

from .bootstrap import FIXED_ROM_HASH

MAGIC = b"PABAT01\0"
LAYOUT_TAG = b"pa9500-bootstrap-high-ram-80000-bffff-rtc-1"
MAIN_RAM_SIZE = 0x40000
IMAGE_SIZE = 104 + MAIN_RAM_SIZE + 32


def profile(hw):
    if (
        len(hw.ram) != MAIN_RAM_SIZE
        or hw.rtc is None
        or hw.native_zero_data != 0xE0
        or not hw.card_eil_presence_candidate
        or not hw.lcd_coordinate_restore_candidate
        or hw.scratch_candidate is None
        or len(hw.scratch_candidate) != 4096
        or hw.card_lower_write_alias
    ):
        raise ValueError("PA retained backing layout/profile mismatch")
    flags = (hw.observed_low_pair, hw.observed_low_triad, hw.observed_field_words)
    if hw.observed_power_pair is not None or hw.irq_imr_only_candidate:
        if (
            flags == (bytes([0xE0] * 2), bytes([0xE0] * 3), bytes([0xE0] * 4))
            and hw.observed_power_pair == bytes([0xE0] * 2)
            and hw.irq_imr_only_candidate
        ):
            return 4
        raise ValueError("PA retained backing has a noncanonical power profile")
    if flags == (None, None, None):
        return 1
    if flags == (bytes([0xE0] * 2), bytes([0xE0] * 3), None):
        return 2
    if flags == (bytes([0xE0] * 2), bytes([0xE0] * 3), bytes([0xE0] * 4)):
        return 3
    raise ValueError("PA retained backing has a noncanonical bootstrap profile")


def identity(hw):
    digest = hashlib.sha256(LAYOUT_TAG + bytes([profile(hw)]))
    for selector, bank in sorted(hw.banks.items()):
        digest.update(bytes([selector]))
        digest.update(bank)
    return digest.digest()


def fixed_identity(external):
    fixed = hashlib.sha256(external[0xE0000:0x100000]).digest()
    if fixed.hex() != FIXED_ROM_HASH:
        raise ValueError("PA retained backing requires the verified fixed ROM")
    return fixed


def retained_state(external, hw):
    """Read RAM/RTC backing only; the caller owns the bootstrap coordinator."""
    fixed, layout = fixed_identity(external), identity(hw)
    payload = bytes(hw.ram) + bytes(hw.rtc.backing)
    return MAGIC + fixed + layout + hashlib.sha256(payload).digest() + payload


def restore_retained_state(
    external, hw, image, *, instructions=0, timing_units=0, legacy_boundaries=0
):
    if instructions or timing_units or legacy_boundaries or hw.cycle:
        raise ValueError(
            "PA retained backing must be restored before the first CPU boundary"
        )
    if len(image) != IMAGE_SIZE or image[:8] != MAGIC:
        raise ValueError("PA retained backing version/length mismatch")
    payload = image[104:]
    if (
        image[8:40] != fixed_identity(external)
        or image[40:72] != identity(hw)
        or image[72:104] != hashlib.sha256(payload).digest()
    ):
        raise ValueError("PA retained backing ROM/bank/profile/checksum mismatch")
    hw.ram[:] = payload[:MAIN_RAM_SIZE]
    hw.rtc.backing[:] = payload[-32:]
