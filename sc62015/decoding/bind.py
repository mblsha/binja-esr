from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional


class IntAddrCalc(str, Enum):
    """Enumeration of supported internal memory addressing calculations."""

    N = "(n)"
    BP_N = "(BP+n)"
    PX_N = "(PX+n)"
    PY_N = "(PY+n)"
    BP_PX = "(BP+PX)"
    BP_PY = "(BP+PY)"


RegSizeGroup = Literal["r1", "r2", "r3", "r4"]


@dataclass(frozen=True, slots=True)
class PreLatch:
    first: IntAddrCalc
    second: IntAddrCalc


@dataclass(frozen=True, slots=True)
class Imm8:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 0xFF:
            raise ValueError(f"Imm8 out of range: {self.value:#x}")


@dataclass(frozen=True, slots=True)
class Disp8:
    value: int  # signed

    def __post_init__(self) -> None:
        if not -0x80 <= self.value <= 0x7F:
            raise ValueError(f"Disp8 out of range: {self.value}")


@dataclass(frozen=True, slots=True)
class Imm16:
    lo: int
    hi: int

    def __post_init__(self) -> None:
        for label, val in (("lo", self.lo), ("hi", self.hi)):
            if not 0 <= val <= 0xFF:
                raise ValueError(f"Imm16 {label} byte out of range: {val:#x}")

    @property
    def u16(self) -> int:
        return (self.hi << 8) | self.lo


@dataclass(frozen=True, slots=True)
class Imm24:
    lo: int
    mid: int
    hi: int

    def __post_init__(self) -> None:
        for label, val in (("lo", self.lo), ("mid", self.mid), ("hi", self.hi)):
            if not 0 <= val <= 0xFF:
                raise ValueError(f"Imm24 {label} byte out of range: {val:#x}")

    @property
    def u24(self) -> int:
        return (self.hi << 16) | (self.mid << 8) | self.lo


@dataclass(frozen=True, slots=True)
class Addr16Page:
    offs16: Imm16
    page20: int  # upper nibble of PC

    def __post_init__(self) -> None:
        if not 0 <= self.page20 <= 0xFF000:
            raise ValueError(f"page20 out of range: {self.page20:#x}")


@dataclass(frozen=True, slots=True)
class Addr24:
    v: Imm24


@dataclass(frozen=True, slots=True)
class RegSel:
    size_group: RegSizeGroup
    name: str


@dataclass(frozen=True, slots=True)
class ExtRegPtr:
    ptr: RegSel
    mode: Literal["simple", "post_inc", "pre_dec", "offset"]
    disp: Optional[Disp8] = None


@dataclass(frozen=True, slots=True)
class ImemPtr:
    base: Imm8
    mode: Literal["simple", "pos", "neg"]
    disp: Optional[Disp8] = None


@dataclass(frozen=True, slots=True)
class DecodedInstr:
    opcode: int
    mnemonic: str
    length: int
    binds: Dict[str, object] = field(default_factory=dict)
    pre_latch: Optional[PreLatch] = None
    pre_applied: Optional[PreLatch] = None
    family: Optional[str] = None
