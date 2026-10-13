"""Read-only host aid for the verified ROM's three tablet event records."""


def tablet_mailbox_busy(hardware):
    def byte(address):
        return hardware.ram[0x30000 + address - 0x10000]

    pointer = byte(0x1FF47) | byte(0x1FF48) << 8 | byte(0x1FF49) << 16
    if not 0x10000 <= pointer <= 0x1FFE4:
        return False
    return any(
        0xC8 <= byte(pointer + offset) | byte(pointer + offset + 1) << 8 <= 0xCA
        for offset in (0, 0x0B, 0x1A)
    )
