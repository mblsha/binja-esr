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

def _build_stream_templates() -> Tuple[bytes, ...]:
    """Generate operand templates that cover all register and pointer selectors."""

    def _pack(first_byte: int) -> bytes:
        return bytes([first_byte & 0xFF] + [0x00] * 15)

    templates: list[int] = [0x00]

    # Register pair selectors (dst hi nibble, src lo nibble) for each width class.
    reg_groups = (
        (0, 1),  # r1
        (2, 3),  # r2
        (4, 5, 6, 7),  # r3
    )
    for group in reg_groups:
        for dst in group:
            for src in group:
                byte = ((dst & 0x07) << 4) | (src & 0x07)
                if byte & 0x88:
                    continue
                templates.append(byte)

    # External register pointer selectors use the high nibble for modes.
    for raw_mode in (0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF):
        for idx in range(8):
            byte = ((raw_mode & 0x0F) << 4) | (idx & 0x07)
            templates.append(byte | 0x08)  # mark as pointer-only so reg decoders skip

    seen: set[int] = set()
    ordered: list[bytes] = []
    for byte in templates:
        if byte in seen:
            continue
        seen.add(byte)
        ordered.append(_pack(byte))
    return tuple(ordered)


_STREAM_TEMPLATES = _build_stream_templates()


def _build_cross_reg_pair_templates() -> Tuple[bytes, ...]:
    """Construct register-pair templates that mix width classes (r1↔r2↔r3)."""

    def _pack(byte: int) -> bytes:
        return bytes([byte & 0xFF] + [0x00] * 15)

    reg_table = getattr(decode_map, "_REG_TABLE")
    cross_templates: list[int] = []
    for dst_idx, (_, dst_group, _) in enumerate(reg_table):
        for src_idx, (_, src_group, _) in enumerate(reg_table):
            if dst_group == src_group:
                continue
            raw = ((dst_idx & 0x07) << 4) | (src_idx & 0x07)
            if raw & 0x88:
                continue
            cross_templates.append(raw)
    seen: set[int] = set()
    ordered: list[bytes] = []
    for byte in cross_templates:
        if byte in seen:
            continue
        seen.add(byte)
        ordered.append(_pack(byte))
    return tuple(ordered)


_REG_ARITH_OPCODES = frozenset({0x44, 0x45, 0x46, 0x4C, 0x4D, 0x4E, 0xFD})
_CROSS_REG_PAIR_TEMPLATES = _build_cross_reg_pair_templates()


def _decode_with_templates(opcode: int):
    collected: list[DecodedInstr] = []
    layout = ()
    last_exc: Exception | None = None
    for template in _STREAM_TEMPLATES:
        ctx = StreamCtx(pc=0, data=template, base_len=1, record_layout=True)
        try:
            variants = decode_map.decode_with_pre_variants(opcode, ctx)
        except Exception as exc:
            last_exc = exc
            continue
        if not layout:
            layout = ctx.snapshot_layout()
        collected.extend(variants)

    if opcode in _REG_ARITH_OPCODES:
        for template in _CROSS_REG_PAIR_TEMPLATES:
            ctx = StreamCtx(pc=0, data=template, base_len=1, record_layout=True)
            try:
                variants = decode_map.decode_with_pre_variants(opcode, ctx)
            except Exception:
                continue
            if not layout:
                layout = ctx.snapshot_layout()
            collected.extend(variants)

    if not collected:
        if last_exc:
            raise last_exc
        return (), ()
    dedup: dict[tuple, DecodedInstr] = {}
    for variant in collected:
        key = (
            variant.mnemonic,
            variant.length,
            variant.family,
            variant.pre_applied,
            tuple(
                (name, type(value).__name__, getattr(value, "value", repr(value)))
                for name, value in sorted(variant.binds.items())
            ),
        )
        dedup.setdefault(key, variant)
    return tuple(dedup.values()), layout


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
            if variant.family == "pre" or variant.mnemonic.upper().startswith("PRE"):
                continue
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
