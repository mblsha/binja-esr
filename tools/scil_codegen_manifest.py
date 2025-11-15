from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

from sc62015.decoding import decode_map
from sc62015.decoding.bind import DecodedInstr, PreLatch
from sc62015.decoding.reader import StreamCtx
from sc62015.scil import from_decoded, serde
from sc62015.scil.bound_repr import BoundInstrRepr

_STREAM_TEMPLATES = (
    bytes([0x00] * 16),
    bytes([0x04] + [0x00] * 15),
)


def _decode_with_templates(opcode: int):
    last_exc: Exception | None = None
    for template in _STREAM_TEMPLATES:
        ctx = StreamCtx(pc=0, data=template, base_len=1, record_layout=True)
        try:
            variants = decode_map.decode_with_pre_variants(opcode, ctx)
            layout = ctx.snapshot_layout()
            return variants, layout
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return (), ()


def _serialize_layout(layout_entries) -> List[Dict[str, object]]:
    serialized = []
    for entry in layout_entries:
        serialized.append(
            {
                "key": entry.key,
                "kind": entry.kind,
                "meta": entry.meta,
            }
        )
    return serialized


def _entry_from_variant(
    opcode: int, variant: DecodedInstr, layout_entries
) -> Dict[str, object]:
    build = from_decoded.build(variant)
    pre: PreLatch | None = build.pre_applied
    return {
        "opcode": opcode,
        "mnemonic": variant.mnemonic,
        "family": variant.family,
        "length": variant.length,
        "pre": {
            "first": pre.first.value,
            "second": pre.second.value,
        }
        if pre
        else None,
        "instr": serde.instr_to_dict(build.instr),
        "binder": {
            name: serde.expr_to_dict(expr) for name, expr in build.binder.items()
        },
        "bound_repr": BoundInstrRepr.from_decoded(variant).to_dict(),
        "layout": _serialize_layout(layout_entries),
    }


def generate_manifest() -> Tuple[List[Dict[str, object]], List[Tuple[int, str]]]:
    manifest: List[Dict[str, object]] = []
    errors: List[Tuple[int, str]] = []
    entry_id = 0

    for opcode in sorted(decode_map.DECODERS):
        try:
            variants, layout = _decode_with_templates(opcode)
        except Exception as exc:
            errors.append((opcode, f"decode failure: {exc}"))
            continue
        for variant in variants:
            try:
                entry = _entry_from_variant(opcode, variant, layout)
                entry["id"] = entry_id
                entry_id += 1
                manifest.append(entry)
            except Exception as exc:
                errors.append((opcode, f"build failure ({variant.mnemonic}): {exc}"))

    return manifest, errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce SCIL manifest (JSON + bound representation)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Path to write manifest JSON (default stdout)",
    )
    parser.add_argument(
        "--errors",
        type=argparse.FileType("w"),
        default=None,
        help="Optional path to JSON error log",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest, errors = generate_manifest()
    json.dump(manifest, args.output, indent=2)
    args.output.write("\n")
    if args.errors:
        json.dump(
            [{"opcode": opcode, "error": msg} for opcode, msg in errors],
            args.errors,
            indent=2,
        )
        args.errors.write("\n")
    elif errors:
        for opcode, msg in errors:
            print(f"[warn] opcode 0x{opcode:02X}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
