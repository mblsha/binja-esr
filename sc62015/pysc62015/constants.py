"""Shared architecture constants for the SC62015 emulator.

This module centralizes common, non-volatile constants used across the
assembler/emulator and tests.
"""

from enum import IntFlag

# Number of bytes in the internal RAM region. The hardware provides a
# contiguous 256-byte block of memory immediately following the external
# address space.
INTERNAL_MEMORY_LENGTH = 0x100  # 256 bytes

# Total addressable memory size: 1 MB of external space plus the internal RAM
# block, giving 0x100100 bytes in total.
ADDRESS_SPACE_SIZE = 0x100000 + INTERNAL_MEMORY_LENGTH

# Address of the first byte of internal RAM.
# The internal region occupies addresses [INTERNAL_MEMORY_START, ADDRESS_SPACE_SIZE - 1].
INTERNAL_MEMORY_START = ADDRESS_SPACE_SIZE - INTERNAL_MEMORY_LENGTH

# Mask for the program counter. Although stored in three bytes, the PC
# only uses the lower 20 bits.
PC_MASK = 0xFFFFF

# F is transferred through a byte-wide stack slot, but the available TRM, ROM,
# and emulator evidence establishes semantics only for carry and zero.  Do not
# silently invent, preserve, or discard bits 2-7: reject those images at
# execution/snapshot boundaries until a real-hardware trace defines them.
MODELED_F_MASK = 0x03


class HardwareEvidenceRequired(RuntimeError):
    """Raised when execution reaches behavior that needs a silicon trace."""


def validate_f_image(value: int) -> int:
    """Return a supported F image or quarantine unverified upper flag bits."""

    raw = int(value)
    image = raw & 0xFF
    if raw < 0 or raw & ~MODELED_F_MASK:
        raise HardwareEvidenceRequired(
            f"unsupported SC62015 F image 0x{image:02X}: "
            "bits 2-7 require real-hardware tracing"
        )
    return image


class IMRFlag(IntFlag):
    """Interrupt Mask Register (IMR, 0xFB) bit masks.

    Bit layout (see README.md):
        7   6    5    4     3     2    1    0
      +----+----+----+-----+-----+----+----+----+
      |IRM |EXM |RXRM|TXRM |ONKM |KEYM|STM |MTM |
      +----+----+----+-----+-----+----+----+----+

    Naming follows the datasheet/README. Aliases are provided for
    convenience (e.g., MTI→MTM, STI→STM, KEY→KEYM, ONK→ONKM).
    """

    MTM = 0x01  # Main timer mask
    STM = 0x02  # Sub timer mask
    KEYM = 0x04  # Keyboard matrix mask
    ONKM = 0x08  # ON-key mask
    TXRM = 0x10  # UART TX ready mask
    RXRM = 0x20  # UART RX ready mask
    EXM = 0x40  # External interrupt mask
    IRM = 0x80  # Master interrupt enable

    # Common aliases used in higher-level tests/code
    MTI = MTM
    STI = STM
    KEY = KEYM
    ONK = ONKM


class ISRFlag(IntFlag):
    """Interrupt Status Register (ISR, 0xFC) bit masks.

    Bit layout (see README.md):
        7    6     5     4      3      2     1     0
      +----+-----+-----+------+------+-----+-----+-----+
      |    | EXI | RXRI| TXRI | ONKI | KEYI| STI | MTI |
      +----+-----+-----+------+------+-----+-----+-----+
    """

    MTI = 0x01  # Main timer pending
    STI = 0x02  # Sub timer pending
    KEYI = 0x04  # Keyboard matrix pending
    ONKI = 0x08  # ON-key pending
    TXRI = 0x10  # UART TX ready pending
    RXRI = 0x20  # UART RX ready pending
    EXI = 0x40  # External interrupt pending
