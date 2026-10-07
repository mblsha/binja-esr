"""One 256 KiB RAM backing plus RTC; low workspace aliases the last quarter."""

import hashlib
from .hardware import FIXED_ROM_HASH, WORKSPACE_RAM_OFFSET

WORKSPACE_START, WORKSPACE_SIZE, MAIN_RAM_SIZE = 0x10000, 0x10000, 0x40000
PAYLOAD_SIZE = MAIN_RAM_SIZE + 32
IMAGE_SIZE = 104 + PAYLOAD_SIZE
LAYOUT_TAG = b"oz9600-top-page-ram-rtc-2"
LEGACY_LAYOUT_TAG = b"oz9600-logical-ram-rtc-1"


def bank_identity(hw, layout=LAYOUT_TAG):
    digest = hashlib.sha256(layout)
    for selector, bank in sorted(hw.banks.items()):
        digest.update(bytes([selector]))
        digest.update(bank)
    return digest.digest()


def _image_from_payload(external, hw, payload):
    return (
        b"OZBAT02\0"
        + hashlib.sha256(external[0xE0000:0x100000]).digest()
        + bank_identity(hw)
        + hashlib.sha256(payload).digest()
        + payload
    )


def retained_state(external, hw):
    return _image_from_payload(external, hw, bytes(hw.ram) + bytes(hw.rtc.backing))


def _validated_payload(external, hw, image, magic, layout, size):
    if len(image) != 104 + size or image[:8] != magic:
        raise ValueError("retained-state version/length mismatch")
    fixed = hashlib.sha256(external[0xE0000:0x100000]).digest()
    if fixed.hex() != FIXED_ROM_HASH:
        raise ValueError("retained restore requires the verified fixed ROM")
    payload = image[104:]
    if (
        image[8:40] != fixed
        or image[40:72] != bank_identity(hw, layout)
        or image[72:104] != hashlib.sha256(payload).digest()
    ):
        raise ValueError(
            "retained-state ROM/bank identity or payload checksum mismatch"
        )
    return payload


def restore_retained_state(external, hw, image, *, instructions=0, cycles=0):
    if instructions or cycles:
        raise ValueError(
            "retained state must be restored before the first CPU boundary"
        )
    if image[:8] == b"OZBAT01\0":
        raise ValueError(
            "OZBAT01 used independent workspace; convert explicitly to OZBAT02"
        )
    payload = _validated_payload(
        external, hw, image, b"OZBAT02\0", LAYOUT_TAG, PAYLOAD_SIZE
    )
    hw.ram[:] = payload[:MAIN_RAM_SIZE]
    hw.rtc.backing[:] = payload[-32:]
    hw.retained_loaded = True


def _merged_legacy_payload(payload):
    workspace = payload[:WORKSPACE_SIZE]
    ram = bytearray(payload[WORKSPACE_SIZE : WORKSPACE_SIZE + MAIN_RAM_SIZE])
    if any(workspace) or any(ram):

        def word(a):
            return int.from_bytes(workspace[a : a + 3], "little")

        floor, start, end = word(0xFF38), word(0xFD00), word(0xFD03)
        if not (
            0x10000 <= floor <= 0x20000
            and end == floor + 0xA0000
            and 0x80000 <= start < end
        ):
            raise ValueError(
                "legacy image lacks consistent workspace/filesystem reservation"
            )
        offset = floor - WORKSPACE_START
        if any(
            low and low != high
            for low, high in zip(
                workspace[:offset],
                ram[WORKSPACE_RAM_OFFSET : WORKSPACE_RAM_OFFSET + offset],
            )
        ):
            raise ValueError("legacy workspace prefix conflicts with main RAM")
        ram[WORKSPACE_RAM_OFFSET + offset :] = workspace[offset:]
    return bytes(ram) + payload[WORKSPACE_SIZE + MAIN_RAM_SIZE :]


def convert_legacy_image(external, hw, image):
    """Validate and coalesce legacy regions offline, without runtime mutation."""
    payload = _validated_payload(
        external,
        hw,
        image,
        b"OZBAT01\0",
        LEGACY_LAYOUT_TAG,
        WORKSPACE_SIZE + PAYLOAD_SIZE,
    )
    return _image_from_payload(external, hw, _merged_legacy_payload(payload))
