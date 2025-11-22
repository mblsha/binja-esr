#!/usr/bin/env python3
"""
Emit a lightweight JSON view of the Python opcode table for Project LLAMA.

This is a helper for the LLIL-less Rust core to mirror Python decode shapes
without pulling in the SCIL manifest. It serializes opcode → instruction class
name + opts (operands, cond, name override) using the existing Python tables.

Usage:
  uv run python tools/llama_opcode_dump.py > /tmp/opcodes.json
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, Dict, List, Tuple

from sc62015.pysc62015.instr.opcodes import Opts
from sc62015.pysc62015.instr.opcode_table import OPCODES


def _serialize_operand(op: Any) -> Dict[str, Any]:
    """Return a JSON-friendly summary of an operand instance."""

    payload: Dict[str, Any] = {"kind": op.__class__.__name__}
    # Opportunistically capture simple attributes if present.
    for attr in (
        "name",
        "reg",
        "reg_name",
        "width",
        "order",
        "allowed_modes",
        "mode",
        "value",
        "offset",
        "min_offset",
        "max_offset",
    ):
        if hasattr(op, attr):
            val = getattr(op, attr)
            if isinstance(val, (str, int, float, type(None))):
                payload[attr] = val
            elif isinstance(val, (list, tuple)):
                payload[attr] = list(val)
    payload["repr"] = repr(op)
    return payload


def _serialize_entry(opcode: int, definition) -> Dict[str, Any]:
    cls, opts = definition if isinstance(definition, tuple) else (definition, Opts())
    ops = [copy.deepcopy(op) for op in (opts.ops or [])]
    return {
        "opcode": opcode,
        "instr_class": cls.__name__,
        "name": opts.name or cls.__name__.split("_")[0],
        "cond": opts.cond,
        "ops_reversed": opts.ops_reversed,
        "operands": [_serialize_operand(op) for op in ops],
    }


def main() -> None:
    entries: List[Dict[str, Any]] = []
    for opcode in sorted(OPCODES):
        entries.append(_serialize_entry(opcode, OPCODES[opcode]))
    try:
        json.dump(entries, fp=sys.stdout, indent=2)
        sys.stdout.write("\n")
    except BrokenPipeError:
        # Allow piping to tools like `head` without noisy tracebacks.
        pass


if __name__ == "__main__":
    main()
