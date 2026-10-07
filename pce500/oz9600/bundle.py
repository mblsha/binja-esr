"""Validate OZROM01 captures before configuring a machine; no live RAM load."""

import hashlib
import json
import struct
from .hardware import Hardware, FIXED_ROM_HASH


def from_verified_images(mapped, manifest_bytes, provider):
    if (
        len(mapped) != 0x100000
        or hashlib.sha256(mapped[0xE0000:]).hexdigest() != FIXED_ROM_HASH
    ):
        raise ValueError("fixed system ROM does not match the verified acquisition")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "oz9600-bank-stream-snapshot-1":
        raise ValueError("requires an inspect_bank_stream verified snapshot")
    hw = Hardware()
    for frame in manifest["frames"]:
        selector = frame["selector"]
        original, restored = (
            frame.get("original_selector"),
            frame.get("restored_selector"),
        )
        if (
            type(selector) is not int
            or not 0xF0 <= selector <= 0xFF
            or frame["start"] != 0xC0000
            or frame["length"] != 0x20000
            or type(original) is not int
            or type(restored) is not int
            or not 0 <= original <= 255
            or original != restored
        ):
            raise ValueError("invalid bank metadata")
        name = frame["file"]
        if name != f"bank-{selector:02X}-C0000.bin":
            raise ValueError("unexpected bank filename")
        bank = bytes(provider(name))
        if (
            len(bank) != 0x20000
            or frame["sha256"] != hashlib.sha256(bank).hexdigest()
            or frame["sum16"] != sum(bank) & 65535
            or selector in hw.banks
        ):
            raise ValueError("bank length/hash/checksum or duplicate selector mismatch")
        hw.banks[selector] = bank
    return bytes(mapped[0xE0000:]), hw


def from_rom_bundle(image):
    if len(image) < 52 or image[:8] != b"OZROM01\0":
        raise ValueError("ROM bundle version/header mismatch")
    mapped_size, manifest_size, bank_size = struct.unpack_from("<III", image, 8)
    if (
        mapped_size != 0x100000
        or not 0 < manifest_size <= 0x40000
        or bank_size > 16 * 0x20000
        or bank_size % 0x20000
        or len(image) != 52 + mapped_size + manifest_size + bank_size
    ):
        raise ValueError("ROM bundle length mismatch")
    payload = image[52:]
    if hashlib.sha256(payload).digest() != image[20:52]:
        raise ValueError("ROM bundle payload checksum mismatch")
    offset = mapped_size + manifest_size

    def provider(_):
        nonlocal offset
        bank = payload[offset : offset + 0x20000]
        offset += 0x20000
        return bank

    result = from_verified_images(
        payload[:mapped_size],
        payload[mapped_size : mapped_size + manifest_size],
        provider,
    )
    if offset != len(payload):
        raise ValueError("ROM bundle has unclaimed bank payloads")
    return result
