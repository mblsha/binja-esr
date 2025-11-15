from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..decoding.bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    ExtRegPtr,
    Imm16,
    Imm24,
    Imm8,
    ImemPtr,
    PreLatch,
    RegSel,
)


OperandValue = Dict[str, Any]


def _encode_operand(value: object) -> OperandValue:
    if isinstance(value, Imm8):
        return {"kind": "imm8", "value": value.value}
    if isinstance(value, Imm16):
        return {"kind": "imm16", "lo": value.lo, "hi": value.hi}
    if isinstance(value, Imm24):
        return {"kind": "imm24", "lo": value.lo, "mid": value.mid, "hi": value.hi}
    if isinstance(value, Disp8):
        return {"kind": "disp8", "value": value.value}
    if isinstance(value, Addr16Page):
        return {
            "kind": "addr16_page",
            "offs16": _encode_operand(value.offs16),
            "page20": value.page20,
        }
    if isinstance(value, Addr24):
        return {"kind": "addr24", "imm24": _encode_operand(value.v)}
    if isinstance(value, RegSel):
        return {
            "kind": "regsel",
            "size_group": value.size_group,
            "name": value.name,
        }
    if isinstance(value, ExtRegPtr):
        payload: OperandValue = {
            "kind": "ext_reg_ptr",
            "ptr": _encode_operand(value.ptr),
            "mode": value.mode,
        }
        if value.disp is not None:
            payload["disp"] = _encode_operand(value.disp)
        return payload
    if isinstance(value, ImemPtr):
        payload = {
            "kind": "imem_ptr",
            "base": _encode_operand(value.base),
            "mode": value.mode,
        }
        if value.disp is not None:
            payload["disp"] = _encode_operand(value.disp)
        return payload
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_operand(v) for v in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_operand(v) for v in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {k: _encode_operand(v) for k, v in value.items()},
        }
    raise TypeError(f"Unsupported operand type: {type(value)!r}")


def _encode_prelatch(pre: Optional[PreLatch]) -> Optional[Dict[str, str]]:
    if pre is None:
        return None
    return {
        "first": pre.first.value,
        "second": pre.second.value,
    }


@dataclass(frozen=True)
class BoundInstrRepr:
    opcode: int
    mnemonic: str
    family: Optional[str]
    length: int
    pre: Optional[Dict[str, str]]
    operands: Dict[str, OperandValue]

    @classmethod
    def from_decoded(cls, decoded: DecodedInstr) -> "BoundInstrRepr":
        operands = {
            name: _encode_operand(value) for name, value in decoded.binds.items()
        }
        return cls(
            opcode=decoded.opcode,
            mnemonic=decoded.mnemonic,
            family=decoded.family,
            length=decoded.length,
            pre=_encode_prelatch(decoded.pre_applied),
            operands=operands,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opcode": self.opcode,
            "mnemonic": self.mnemonic,
            "family": self.family,
            "length": self.length,
            "pre": self.pre,
            "operands": self.operands,
        }

    def pack(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def unpack(cls, payload: str) -> "BoundInstrRepr":
        import json

        data = json.loads(payload)
        return cls(
            opcode=data["opcode"],
            mnemonic=data["mnemonic"],
            family=data.get("family"),
            length=data["length"],
            pre=data.get("pre"),
            operands=data.get("operands", {}),
        )
