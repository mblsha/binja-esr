#!/usr/bin/env python3
"""
Generate Rust opcode table entries from the Python opcode_table.py source of truth.

This preserves parity between the Python and Rust LLAMA opcode tables and helps
detect drift. The script emits Rust `OpcodeEntry` initializers grouped by opcode.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_opcode_table():
    sys.path.insert(0, str(REPO_ROOT))
    mod = importlib.import_module("sc62015.pysc62015.instr.opcode_table")
    return mod


def _width_bytes(obj: Any) -> int | None:
    width = getattr(obj, "width_bits", None)
    if width is not None:
        # width_bits is expressed in bits; convert to bytes.
        return int(width) // 8
    width_attr = getattr(obj, "width", None)
    if callable(width_attr):
        try:
            return int(width_attr())
        except Exception:
            pass
    elif width_attr is not None:
        return int(width_attr)
    for attr in ["_width", "width_bytes"]:
        val = getattr(obj, attr, None)
        if val is not None:
            return int(val)
    return None


def _width_bits(obj: Any, default: int = 8) -> int:
    width = getattr(obj, "width_bits", None)
    if width is not None:
        return int(width)
    width_bytes = _width_bytes(obj)
    if width_bytes is not None:
        return int(width_bytes) * 8
    return default


@dataclass
class RustOperand:
    kind: str
    args: list[str]

    def render(self) -> str:
        if not self.args:
            return f"OperandKind::{self.kind}"
        joined = ", ".join(self.args)
        return f"OperandKind::{self.kind}({joined})"


def _map_operand(op: Any) -> RustOperand:
    name = op.__class__.__name__
    if name == "Reg":
        reg = op.reg
        width_bits = _width_bits(op)
        return RustOperand("Reg", [f"RegName::{reg}", str(width_bits)])
    if name == "RegIL":
        return RustOperand("RegIL", [])
    if name == "RegIMR":
        return RustOperand("RegIMR", [])
    if name == "RegF":
        return RustOperand("RegF", [])
    if name == "Reg3":
        return RustOperand("Reg3", [])
    if name == "RegPair":
        return RustOperand("RegPair", [str(getattr(op, "size", getattr(op, "size_bytes", 0)))])
    if name == "Imm8":
        return RustOperand("Imm", ["8"])
    if name == "Imm16":
        return RustOperand("Imm", ["16"])
    if name == "Imm20":
        return RustOperand("Imm", ["20"])
    if name == "ImmOffset":
        return RustOperand("ImmOffset", [])
    if name == "IMem8":
        return RustOperand("IMem", ["8"])
    if name == "IMem16":
        return RustOperand("IMem", ["16"])
    if name == "IMem20":
        return RustOperand("IMem", ["20"])
    if name == "IMemWidth":
        return RustOperand("IMemWidth", [str(getattr(op, "width_bytes", 0))])
    if name == "EMemAddr":
        width_bytes = _width_bytes(op)
        if width_bytes is None:
            raise ValueError("EMemAddr missing width")
        return RustOperand("EMemAddrWidth", [str(width_bytes)])
    if name == "EMemAddrWidth":
        return RustOperand("EMemAddrWidth", [str(getattr(op, "width_bytes", 0))])
    if name == "EMemAddrWidthOp":
        return RustOperand("EMemAddrWidthOp", [str(getattr(op, "width_bytes", 0))])
    if name == "EMemReg":
        width_bytes = _width_bytes(op)
        if width_bytes is None:
            raise ValueError("EMemReg missing width")
        allowed_modes = getattr(op, "allowed_modes", None) or []
        allowed = {getattr(m, "name", m) for m in allowed_modes}
        if allowed == {"POST_INC", "PRE_DEC"}:
            return RustOperand("EMemRegModePostPre", [])
        return RustOperand("EMemRegWidth", [str(width_bytes)])
    if name == "EMemRegMode":
        return RustOperand("EMemRegModePostPre", [])
    if name == "EMemRegWidth":
        return RustOperand("EMemRegWidth", [str(getattr(op, "width_bytes", 0))])
    if name == "EMemRegWidthMode":
        return RustOperand("EMemRegWidthMode", [str(getattr(op, "width_bytes", 0))])
    if name == "EMemIMem":
        width_bytes = _width_bytes(op)
        if width_bytes is None:
            raise ValueError("EMemIMem missing width")
        return RustOperand("EMemIMemWidth", [str(width_bytes)])
    if name == "EMemIMemWidth":
        return RustOperand("EMemIMemWidth", [str(getattr(op, "width_bytes", 0))])
    if name == "RegIMemOffset":
        order = getattr(op, "order", None)
        kind = "DestImem"
        if order is not None and getattr(order, "name", "") != "DEST_IMEM":
            kind = "DestRegOffset"
        return RustOperand("RegIMemOffset", [f"RegImemOffsetKind::{kind}"])
    if name == "EMemImemOffsetDestIntMem":
        return RustOperand("EMemImemOffsetDestIntMem", [])
    if name == "EMemImemOffsetDestExtMem":
        return RustOperand("EMemImemOffsetDestExtMem", [])
    if name == "EMemIMemOffset":
        order = getattr(op, "order", None)
        kind = "EMemImemOffsetDestIntMem"
        if order is not None and getattr(order, "name", "") != "DEST_INT_MEM":
            kind = "EMemImemOffsetDestExtMem"
        return RustOperand(kind, [])
    if name == "ImemPtr":
        return RustOperand("ImemPtr", [])
    if name == "Placeholder":
        return RustOperand("Placeholder", [])
    if name == "UnknownOperand":
        return RustOperand("Unknown", [f"\"{op.name}\""])
    if name == "RegB":
        return RustOperand("RegB", [])
    raise ValueError(f"Unsupported operand type: {name}")


def _iter_opcodes(mod) -> Iterable[tuple[int, Any]]:
    table = getattr(mod, "OPCODES")
    for opcode, value in table.items():
        yield opcode, value


def _opcode_entry(opcode: int, entry: Any) -> str:
    name = None
    cond = None
    ops_reversed = None
    operands: list[RustOperand] = []

    # OPCODES entries can be instruction classes or (instr, Opts)
    if isinstance(entry, tuple):
        instr, opts = entry
        name = getattr(opts, "name", None) or getattr(instr, "__name__", "UNK")
        cond = getattr(opts, "cond", None)
        ops_reversed = getattr(opts, "ops_reversed", None)
        operands = [_map_operand(op) for op in getattr(opts, "ops", [])]
    else:
        name = getattr(entry, "__name__", "UNK")

    kind_map = {
        "JP_Abs": "JpAbs",
        "JPF": "JpAbs",
        "JP_Rel": "JpRel",
        "CALL": "Call",
        "CALLF": "Call",
        "RET": "Ret",
        "RETF": "RetF",
        "RETI": "RetI",
        "PUSHU": "PushU",
        "POPU": "PopU",
        "PUSHS": "PushS",
        "POPS": "PopS",
        "ADCL": "Adc",
        "EXP": "Ex",
        "EXW": "Ex",
        "UnknownInstruction": "Unknown",
    }
    kind = kind_map.get(name, name.capitalize() if name else "Unknown")
    cond_str = f'Some("{cond}")' if cond else "None"
    rev_str = "Some(true)" if ops_reversed else "None"
    ops = ", ".join(op.render() for op in operands)
    ops_block = f"&[{ops}]" if ops else "&[]"

    return f"""    OpcodeEntry {{
        opcode: 0x{opcode:02X},
        kind: InstrKind::{kind},
        name: "{name}",
        cond: {cond_str},
        ops_reversed: {rev_str},
        operands: {ops_block},
    }},"""


def main() -> int:
    mod = _import_opcode_table()
    entries = [_opcode_entry(op, val) for op, val in sorted(_iter_opcodes(mod), key=lambda p: p[0])]
    for entry in entries:
        print(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
