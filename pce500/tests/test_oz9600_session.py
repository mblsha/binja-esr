"""Full session envelopes cannot silently substitute or erase saved work."""

import hashlib
import pytest
from pce500.oz9600.session import MAGIC, decode, encode


def test_roundtrip_and_failed_inputs_preserve_original():
    identity = bytes(range(64))
    payload = {"schema": 1, "draft": "p0 unfinished", "counter": 123456}
    original = encode(payload, identity)
    assert decode(original, identity) == payload
    corrupted = bytearray(original)
    corrupted[-9] ^= 1
    for data in [b"", original[:-1], original + b"tail", bytes(corrupted)]:
        with pytest.raises(ValueError):
            decode(data, identity)
    with pytest.raises(ValueError, match="identity"):
        decode(original, bytes(64))
    assert decode(original, identity) == payload


def test_checksumming_trailing_gzip_data_does_not_make_it_valid():
    identity = bytes(range(64))
    original = encode({"schema": 1}, identity)
    payload = original[104:] + b"trailing member"
    changed = MAGIC + identity + hashlib.sha256(payload).digest() + payload
    with pytest.raises(ValueError, match="compression"):
        decode(changed, identity)


def test_bad_version_and_nonfinite_values_reject():
    with pytest.raises(ValueError):
        encode({"schema": 0}, bytes(64))
    with pytest.raises(ValueError):
        encode({"schema": 1, "time": float("nan")}, bytes(64))
