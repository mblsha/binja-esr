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
from typing import Iterable, List, Sequence

# Ensure Binary Ninja imports resolve to the test mocks when available.
os.environ.setdefault("FORCE_BINJA_MOCK", "1")

from binja_test_mocks import binja_api  # noqa: F401  # pylint: disable=unused-import
from binja_test_mocks.coding import Decoder
from binja_test_mocks.mock_llil import MockLowLevelILFunction
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
    il = MockLowLevelILFunction()
    try:
        instruction.lift(il, 0)
        il_lines = _render_il(il.ils)
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
    )


def generate_metadata() -> MetadataEnvelope:
    instructions = [collect_opcode_metadata(opcode) for opcode in range(0x100)]
    return MetadataEnvelope(version=1, instructions=instructions)


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
