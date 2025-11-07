#!/usr/bin/env python3
"""Emit LLIL-driven opcode metadata for the SC62015 architecture.

This utility exercises the existing Python instruction decoder and Binary Ninja
mock LLIL to produce a JSON payload describing each primary opcode. The Rust
core uses this data during build time to seed its decode tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence

# Ensure Binary Ninja imports resolve to the test mocks when available.
os.environ.setdefault("FORCE_BINJA_MOCK", "1")

from binja_test_mocks import binja_api  # noqa: F401  # pylint: disable=unused-import
from binja_test_mocks.coding import Decoder
from binja_test_mocks.mock_llil import (
    MockFlag,
    MockGoto,
    MockIntrinsic,
    MockLabel,
    MockLLIL,
    MockLowLevelILFunction,
    MockReg,
    MockIfExpr,
)
from binja_test_mocks.tokens import asm_str
from binaryninja import InstructionInfo  # type: ignore

from sc62015.pysc62015.instr import decode as decode_instruction
from sc62015.pysc62015.instr.opcode_table import OPCODES


MAX_INSN_LENGTH = 6  # Longest instructions observed today use <=6 bytes.


@dataclass
class OpcodeMetadata:
    opcode: int
    mnemonic: str
    length: int
    asm: str
    il: List[str]
    decoder_consumed: int
    errors: List[str]
    llil: Dict[str, Any]


@dataclass
class MetadataEnvelope:
    version: int
    instructions: List[OpcodeMetadata]


def _render_il(il_ops: Sequence[object]) -> List[str]:
    rendered: List[str] = []
    for op in il_ops:
        try:
            rendered.append(str(op))
        except Exception as exc:  # pragma: no cover - defensive
            rendered.append(f"<repr-error: {exc}>")
    return rendered


def collect_opcode_metadata(opcode: int) -> OpcodeMetadata:
    errors: List[str] = []
    raw = bytearray([opcode]) + bytearray(MAX_INSN_LENGTH)
    decoder = Decoder(raw)
    try:
        instruction = decode_instruction(decoder, 0, OPCODES)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - extremely rare
        errors.append(f"decode-error: {exc}")
        return OpcodeMetadata(
            opcode=opcode,
            mnemonic="??",
            length=1,
            asm="??",
            il=[],
            decoder_consumed=0,
            errors=errors,
            llil={"expressions": [], "nodes": [], "label_count": 0},
        )

    mnemonic = getattr(instruction, "name", lambda: "UNKNOWN")()
    try:
        asm = asm_str(instruction.render())
    except Exception as exc:  # pragma: no cover - render failures should be rare
        errors.append(f"render-error: {exc}")
        asm = mnemonic

    length = max(1, decoder.get_pos())
    info = InstructionInfo()
    try:
        instruction.analyze(info, 0)
        if info.length:
            length = int(info.length)
    except Exception as exc:  # pragma: no cover
        errors.append(f"analyze-error: {exc}")

    il_lines: List[str]
    llil_payload: Dict[str, Any] = {"expressions": [], "nodes": [], "label_count": 0}
    il = MockLowLevelILFunction()
    try:
        instruction.lift(il, 0)
        il_lines = _render_il(il.ils)
        llil_payload = serialize_llil(il)
    except Exception as exc:  # pragma: no cover
        errors.append(f"lift-error: {exc}")
        il_lines = []

    return OpcodeMetadata(
        opcode=opcode,
        mnemonic=mnemonic,
        length=length,
        asm=asm,
        il=il_lines,
        decoder_consumed=decoder.get_pos(),
        errors=errors,
        llil=llil_payload,
    )


def generate_metadata() -> MetadataEnvelope:
    instructions = [collect_opcode_metadata(opcode) for opcode in range(0x100)]
    return MetadataEnvelope(version=1, instructions=instructions)


def serialize_llil(il: MockLowLevelILFunction) -> Dict[str, Any]:
    label_ids: Dict[Any, int] = {}

    def register_label(label: Any) -> int:
        return label_ids.setdefault(label, len(label_ids))

    for node in il.ils:
        if isinstance(node, MockLabel):
            register_label(node.label)
        elif isinstance(node, MockGoto):
            register_label(node.label)
        elif isinstance(node, MockIfExpr):
            register_label(node.t)
            register_label(node.f)

    expressions: List[Dict[str, Any]] = []
    expr_cache: Dict[int, int] = {}

    def serialize_operand(operand: Any) -> Dict[str, Any]:
        if isinstance(operand, MockLLIL):
            expr_index = get_expr_id(operand)
            return {"kind": "expr", "expr": expr_index}
        if isinstance(operand, MockReg):
            return {"kind": "reg", "name": operand.name}
        if isinstance(operand, MockFlag):
            return {"kind": "flag", "name": operand.name}
        if isinstance(operand, int):
            return {"kind": "imm", "value": int(operand)}
        if operand is None:
            return {"kind": "none"}
        raise TypeError(f"Unsupported LLIL operand type: {type(operand)}")

    def get_expr_id(expr: MockLLIL) -> int:
        key = id(expr)
        cached = expr_cache.get(key)
        if cached is not None:
            return cached

        index = len(expressions)
        expr_cache[key] = index
        expressions.append(None)  # Reserve slot so children keep stable indices.

        operands = [serialize_operand(operand) for operand in expr.ops]

        full_op = expr.op
        base_op = full_op.split("{")[0]
        suffix = None
        if "." in base_op:
            suffix = base_op.split(".", 1)[1]

        record: Dict[str, Any] = {
            "op": expr.bare_op(),
            "full_op": full_op,
            "suffix": suffix,
            "width": expr.width(),
            "flags": expr.flags(),
            "operands": operands,
        }

        if isinstance(expr, MockIntrinsic):
            record["intrinsic"] = {"name": expr.name}

        expressions[index] = record
        return index

    nodes: List[Dict[str, Any]] = []

    for node in il.ils:
        if isinstance(node, MockLabel):
            label_index = register_label(node.label)
            nodes.append({"kind": "label", "label": label_index})
        elif isinstance(node, MockGoto):
            label_index = register_label(node.label)
            nodes.append({"kind": "goto", "label": label_index})
        elif isinstance(node, MockIfExpr):
            cond_index = get_expr_id(node.cond)
            true_label = register_label(node.t)
            false_label = register_label(node.f)
            nodes.append(
                {
                    "kind": "if",
                    "cond": cond_index,
                    "true": true_label,
                    "false": false_label,
                }
            )
        elif isinstance(node, MockLLIL):
            expr_index = get_expr_id(node)
            nodes.append({"kind": "expr", "expr": expr_index})
        else:  # pragma: no cover - unexpected node types
            raise TypeError(f"Unsupported LLIL node type: {type(node)}")

    return {
        "expressions": expressions,
        "nodes": nodes,
        "label_count": len(label_ids),
    }


def emit_json(envelope: MetadataEnvelope, pretty: bool) -> str:
    data = asdict(envelope)
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True) + "\n"
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to write the JSON payload. Defaults to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output for human inspection.",
    )
    parser.add_argument(
        "--lowering-plan",
        type=str,
        help="Optional path to emit a lowering-plan JSON for Rust codegen.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    envelope = generate_metadata()
    payload = emit_json(envelope, pretty=args.pretty)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        sys.stdout.write(payload)

    if args.lowering_plan:
        lowering_plan = build_lowering_plan(envelope)
        with open(args.lowering_plan, "w", encoding="utf-8") as fh:
            json.dump(lowering_plan, fh, indent=2, sort_keys=True)
    return 0


SIDE_EFFECT_OPS = {
    "SET_REG",
    "SET_FLAG",
    "STORE",
    "LOAD",  # because it consults memory bus
    "PUSH",
    "POP",
    "CALL",
    "JUMP",
    "RET",
    "INTRINSIC",
    "UNIMPL",
}


def build_lowering_plan(envelope: MetadataEnvelope) -> Dict[str, Any]:
    plan: Dict[str, Any] = {"instructions": []}
    for instr in envelope.instructions:
        expr_plan: List[Dict[str, Any]] = []
        for index, expr in enumerate(instr.llil["expressions"]):
            deps = [
                operand["expr"]
                for operand in expr["operands"]
                if operand.get("kind") == "expr"
            ]
            has_flags = bool(expr.get("flags"))
            expr_plan.append(
                {
                    "index": index,
                    "op": expr["op"],
                    "full_op": expr["full_op"],
                    "width": expr["width"],
                    "flags": expr["flags"],
                    "suffix": expr.get("suffix"),
                    "deps": deps,
                    "side_effect": expr["op"] in SIDE_EFFECT_OPS or has_flags,
                    "intrinsic": expr.get("intrinsic", {}).get("name"),
                    "operands": expr["operands"],
                }
            )
        plan["instructions"].append(
            {
                "opcode": instr.opcode,
                "mnemonic": instr.mnemonic,
                "length": instr.length,
                "expressions": expr_plan,
                "nodes": instr.llil["nodes"],
            }
        )
    return plan


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
