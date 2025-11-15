from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from sc62015.decoding import decode_map
from sc62015.decoding.bind import PreLatch
from sc62015.decoding.reader import StreamCtx


_STREAM_TEMPLATES = [
    bytes([0x00] * 16),
    bytes([0x04] + [0x00] * 15),  # force r3 register when required
]


def _decode(opcode: int) -> List[Dict[str, object]]:
    last_exc: Exception | None = None
    for template in _STREAM_TEMPLATES:
        ctx = StreamCtx(pc=0, data=template, base_len=1)
        try:
            variants = []
            for variant in decode_map.decode_with_pre_variants(opcode, ctx):
                latch: PreLatch | None = variant.pre_applied
                variants.append(
                    {
                        "opcode": opcode,
                        "mnemonic": variant.mnemonic,
                        "family": variant.family,
                        "length": variant.length,
                        "pre_first": latch.first.value if latch else None,
                        "pre_second": latch.second.value if latch else None,
                    }
                )
            return variants
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return []


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="List PRE-aware decode variants for each opcode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table",
    )
    args = parser.parse_args()

    data: Dict[int, List[Dict[str, object]]] = {}
    for opcode in sorted(decode_map.DECODERS):
        try:
            data[opcode] = _decode(opcode)
        except Exception as exc:  # pragma: no cover - diagnostic helper
            print(f"warning: opcode 0x{opcode:02X} failed to decode ({exc})")

    if args.json:
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    for opcode in sorted(data):
        variants = data[opcode]
        if not variants:
            continue
        mnemonic = variants[0]["mnemonic"]
        family = variants[0]["family"]
        print(f"Opcode 0x{opcode:02X} ({mnemonic}, family={family})")
        for entry in variants:
            pre = (
                f"{entry['pre_first']} → {entry['pre_second']}"
                if entry["pre_first"]
                else "default"
            )
            print(f"  - {pre}")
        print()


if __name__ == "__main__":
    _main()
