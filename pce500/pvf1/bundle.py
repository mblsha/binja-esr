"""Read the bounded PVROM01 bundle without importing captured live RAM."""

import hashlib
import json

MAGIC = b"PVROM01\0"
FIXED_SHA256 = "4ba7105476b2da9d28f10df6ff7ae41466357dd0acd3915d592de3ebde3b37a9"
BANKS_SHA256 = "222a81b9abb0092b6db694cc03eb49114a8c16180728853d12669a2ff3f29a80"
PAYLOAD_SIZE = 0x420000


def parse_bundle(data):
    if len(data) < 12 or data[:8] != MAGIC:
        raise ValueError("requires a PVROM01 bundle")
    length = int.from_bytes(data[8:12], "little")
    if not 0 < length <= 0x40000 or len(data) != 12 + length + PAYLOAD_SIZE:
        raise ValueError("PV-F1 bundle length mismatch")
    manifest = json.loads(data[12 : 12 + length])
    if manifest["schema"] != "pvf1-rom-bank-set-1":
        raise ValueError("requires the verified 32-selector PV-F1 manifest")
    payload = data[12 + length :]
    fixed = payload[:0x20000]
    if (
        hashlib.sha256(fixed).hexdigest() != FIXED_SHA256
        or manifest["fixed_rom"]["sha256"] != FIXED_SHA256
        or hashlib.sha256(payload[0x20000:]).hexdigest() != BANKS_SHA256
    ):
        raise ValueError("PV-F1 fixed/bank acquisition identity mismatch")
    if len(manifest["banks"]) != 32:
        raise ValueError("PV-F1 bank inventory length mismatch")
    banks = {}
    for index, entry in enumerate(manifest["banks"]):
        selector = index if index < 16 else index + 224
        offset = index * 0x20000
        bank = payload[0x20000 + offset : 0x40000 + offset]
        if (
            any(
                type(entry[field]) is not int
                for field in ("selector", "logical_image_offset", "bytes")
            )
            or entry["selector"] != selector
            or entry["logical_image_offset"] != offset
            or entry["bytes"] != 0x20000
            or entry["sha256"] != hashlib.sha256(bank).hexdigest()
        ):
            raise ValueError("PV-F1 bank selector/offset/hash mismatch")
        banks[selector] = bank
    if banks[255] != fixed:
        raise ValueError("PV-F1 FF/fixed ROM disagreement")
    return fixed, banks
