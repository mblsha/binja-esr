"""Read the bounded PIROM01 bundle without importing captured live RAM."""

import hashlib
import json

MAGIC = b"PIROM01\0"
FIXED_SHA256 = "855205ac2655e02d670c4cf7c3107241dd3fbef8ec056e4766f97074c5c90009"
BANKS_SHA256 = "a31c087974e5a5a96d45bb25f9f12bcb8e19a075cfe39af5f2074b683d5641b4"
PAYLOAD_SIZE = 0x420000


def parse_bundle(data):
    if len(data) < 12 or data[:8] != MAGIC:
        raise ValueError("requires a PIROM01 bundle")
    length = int.from_bytes(data[8:12], "little")
    if not 0 < length <= 0x40000 or len(data) != 12 + length + PAYLOAD_SIZE:
        raise ValueError("PI bundle length mismatch")
    manifest = json.loads(data[12 : 12 + length])
    if manifest["schema"] != "pi3000-rom-bank-set-2":
        raise ValueError("requires the verified 32-selector PI manifest")
    payload = data[12 + length :]
    fixed = payload[:0x20000]
    if (
        hashlib.sha256(fixed).hexdigest() != FIXED_SHA256
        or manifest["fixed_rom"]["sha256"] != FIXED_SHA256
        or hashlib.sha256(payload[0x20000:]).hexdigest() != BANKS_SHA256
    ):
        raise ValueError("PI fixed/bank acquisition identity mismatch")
    if len(manifest["banks"]) != 32:
        raise ValueError("PI bank inventory length mismatch")
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
            raise ValueError("PI bank selector/offset/hash mismatch")
        banks[selector] = bank
    if banks[255] != fixed:
        raise ValueError("PI FF/fixed ROM disagreement")
    return fixed, banks
