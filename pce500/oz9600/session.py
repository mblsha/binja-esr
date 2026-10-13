"""Portable full-session envelope; RAM-only battery images remain separate.

This independent codec validates transport bounds and identities. Typed CPU,
UART/controller validation and fresh-candidate restoration live in the core.
"""

import gzip
import hashlib
import json
import zlib

MAGIC = b"OZRUN01\0"
MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE = 2 * 1024 * 1024


def decode(image: bytes, identity: bytes) -> dict:
    if len(identity) != 64 or not 104 < len(image) <= MAX_IMAGE or image[:8] != MAGIC:
        raise ValueError("OZ session version/length mismatch")
    if image[8:72] != identity or image[72:104] != hashlib.sha256(image[104:]).digest():
        raise ValueError("OZ session ROM/bank identity or checksum mismatch")
    stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
    body = stream.decompress(image[104:], MAX_BODY + 1)
    if (
        len(body) > MAX_BODY
        or not stream.eof
        or stream.unused_data
        or stream.unconsumed_tail
    ):
        raise ValueError("OZ session compression/length mismatch")
    data = json.loads(body)
    if not isinstance(data, dict) or data.get("schema") not in (1, 2):
        raise ValueError("OZ session layout mismatch")
    _check_recognition_schema(data)
    return data


def encode(data: dict, identity: bytes) -> bytes:
    if len(identity) != 64 or data.get("schema") not in (1, 2):
        raise ValueError("OZ session layout mismatch")
    _check_recognition_schema(data)
    body = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_BODY:
        raise ValueError("Session exceeds decoded size limit")
    payload = gzip.compress(body, compresslevel=1, mtime=0)
    image = MAGIC + identity + hashlib.sha256(payload).digest() + payload
    if len(image) > MAX_IMAGE:
        raise ValueError("Session exceeds encoded size limit")
    return image


def _check_recognition_schema(data: dict) -> None:
    enabled = data.get("timer", {}).get("recognition_delay", False)
    if (data["schema"] == 2) != bool(enabled):
        raise ValueError("OZ session recognition policy/schema mismatch")
