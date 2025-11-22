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
    Imm20,
    Imm24,
    Imm8,
    ImemPtr,
    PreLatch,
RegSel,
)


OperandValue = Dict[str, Any]

_REG_BITS = {"r1": 8, "r2": 16, "r3": 24}


def _const_operand(value: int, size: int) -> OperandValue:
    mask = (1 << size) - 1
    return {"type": "const", "value": value & mask, "size": size}


def _reg_operand(sel: RegSel) -> OperandValue:
    size = _REG_BITS.get(sel.size_group, 8)
    return {"type": "reg", "name": sel.name, "size": size, "bank": "gpr"}


def _encode_operand(value: object) -> OperandValue:
    if isinstance(value, Imm8):
        return _const_operand(value.value, 8)
    if isinstance(value, Imm16):
        return _const_operand(value.u16, 16)
    if isinstance(value, Imm20):
        return _const_operand(value.value, 20)
    if isinstance(value, Imm24):
        return _const_operand(value.u24, 24)
    if isinstance(value, Disp8):
        return _const_operand(value.value & 0xFF, 8)
    if isinstance(value, Addr16Page):
        full = (value.page20 << 16) | value.offs16.u16
        return _const_operand(full, 20)
    if isinstance(value, Addr24):
        return _const_operand(value.v.u24, 24)
    if isinstance(value, RegSel):
        return _reg_operand(value)
    if isinstance(value, ExtRegPtr):
        payload: OperandValue = {
            "type": "ext_reg_ptr",
            "ptr": _reg_operand(value.ptr),
            "mode": value.mode,
        }
        if value.disp is not None:
            payload["disp"] = _const_operand(value.disp.value & 0xFF, 8)
        return payload
    if isinstance(value, ImemPtr):
        payload = {
            "type": "imem_ptr",
            "base": _const_operand(value.base.value, 8),
            "mode": value.mode,
        }
        if value.disp is not None:
            payload["disp"] = _const_operand(value.disp.value & 0xFF, 8)
        return payload
    if isinstance(value, bool):
        return _const_operand(1 if value else 0, 8)
    if isinstance(value, int):
        return _const_operand(value, 32)
    if isinstance(value, str):
        return _const_operand(0, 8)
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
