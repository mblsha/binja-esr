"""Shared architecture constants for the SC62015 emulator."""

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
