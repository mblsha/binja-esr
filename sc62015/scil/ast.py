from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

Space = Literal["int", "ext", "code"]
UnaryOp = Literal["neg", "not", "sext", "zext", "low_part", "high_part", "band", "bor", "bxor"]
BinaryOp = Literal[
    "add",
    "sub",
    "and",
    "or",
    "xor",
    "shl",
    "shr",
    "sar",
    "rol",
    "ror",
    "cmp",
]
TernaryOp = Literal["select"]
CondKind = Literal["flag", "eq", "ne", "ltu", "geu", "lts", "ges"]
FetchKind = Literal[
    "u8",
    "s8",
    "u16",
    "u24",
    "disp8",
    "addr16_page",
    "addr16_page_hi",
    "addr24",
]
EffectKind = Literal[
    "push_ret20",
    "push_ret24",
    "pre_latch",
    "far_enter",
    "far_exit",
    "interrupt_enter",
    "interrupt_exit",
]
ExtRegMode = Literal["simple", "post_inc", "pre_dec", "offset"]


def _as_tuple(items: Sequence["Stmt"]) -> Tuple["Stmt", ...]:
    return tuple(items) if not isinstance(items, tuple) else items


@dataclass(frozen=True, slots=True)
class Const:
    value: int
    size: int  # bits


@dataclass(frozen=True, slots=True)
class Tmp:
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class Reg:
    name: str
    size: int
    bank: str = "gpr"


@dataclass(frozen=True, slots=True)
class Flag:
    name: str


@dataclass(frozen=True, slots=True)
class Mem:
    space: Space
    addr: "Expr"
    size: int


@dataclass(frozen=True, slots=True)
class UnOp:
    op: UnaryOp
    a: "Expr"
    out_size: int
    param: Optional[int] = None


@dataclass(frozen=True, slots=True)
class BinOp:
    op: BinaryOp
    a: "Expr"
    b: "Expr"
    out_size: int


@dataclass(frozen=True, slots=True)
class TernOp:
    op: TernaryOp
    cond: "Cond"
    t: "Expr"
    f: "Expr"
    out_size: int


@dataclass(frozen=True, slots=True)
class PcRel:
    base_advance: int
    disp: Optional["Expr"] = None
    out_size: int = 20


@dataclass(frozen=True, slots=True)
class Join24:
    hi: "Expr"
    mid: "Expr"
    lo: "Expr"


Expr = Union[Const, Tmp, Reg, Flag, Mem, UnOp, BinOp, TernOp, PcRel, Join24]


@dataclass(frozen=True, slots=True)
class Cond:
    kind: CondKind
    a: Optional[Expr] = None
    b: Optional[Expr] = None
    flag: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Fetch:
    kind: FetchKind
    dst: Tmp


@dataclass(frozen=True, slots=True)
class SetReg:
    reg: Reg
    value: Expr
    flags: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True, slots=True)
class Store:
    dst: Mem
    value: Expr


@dataclass(frozen=True, slots=True)
class SetFlag:
    flag: str
    value: Expr


@dataclass(frozen=True, slots=True)
class If:
    cond: Cond
    then_ops: Sequence["Stmt"]
    else_ops: Sequence["Stmt"] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "then_ops", _as_tuple(self.then_ops))
        object.__setattr__(self, "else_ops", _as_tuple(self.else_ops))


@dataclass(frozen=True, slots=True)
class Goto:
    target: Expr


@dataclass(frozen=True, slots=True)
class Call:
    target: Expr
    far: bool = False


@dataclass(frozen=True, slots=True)
class Ret:
    far: bool = False
    reti: bool = False


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    args: Tuple[Expr, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtRegLoad:
    dst: Reg
    ptr: Reg
    mode: ExtRegMode
    disp: Optional[Const] = None


@dataclass(frozen=True, slots=True)
class ExtRegStore:
    src: Reg
    ptr: Reg
    mode: ExtRegMode
    disp: Optional[Const] = None


@dataclass(frozen=True, slots=True)
class IntMemSwap:
    left: Expr
    right: Expr
    width: int


@dataclass(frozen=True, slots=True)
class ExtRegToIntMem:
    ptr: Reg
    mode: ExtRegMode
    disp: Optional[Const]
    dst: Mem


@dataclass(frozen=True, slots=True)
class IntMemToExtReg:
    src: Mem
    ptr: Reg
    mode: ExtRegMode
    disp: Optional[Const]


@dataclass(frozen=True, slots=True)
class Label:
    name: str


@dataclass(frozen=True, slots=True)
class Comment:
    text: str


Stmt = Union[
    Fetch,
    SetReg,
    Store,
    SetFlag,
    If,
    Goto,
    Call,
    Ret,
    Effect,
    ExtRegLoad,
    ExtRegStore,
    IntMemSwap,
    ExtRegToIntMem,
    IntMemToExtReg,
    Label,
    Comment,
]


@dataclass(frozen=True, slots=True)
class Instr:
    name: str
    length: int
    semantics: Sequence[Stmt]

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantics", _as_tuple(self.semantics))
