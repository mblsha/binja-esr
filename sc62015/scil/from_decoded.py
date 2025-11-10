from __future__ import annotations

from typing import Callable, Dict, Tuple

from . import ast
from ..decoding.bind import Addr16Page, Addr24, DecodedInstr, Disp8, Imm8
from .specs import examples


Binder = Dict[str, ast.Const]
BuildResult = Tuple[ast.Instr, Binder]


def _const(value: int, bits: int) -> ast.Const:
    mask = (1 << bits) - 1
    return ast.Const(value & mask, bits)


def _mv_a_n(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_a_imm()
    imm = decoded.binds["n"]
    assert isinstance(imm, Imm8)
    binder = {"imm8": _const(imm.value, 8)}
    return spec, binder


def _jrz(decoded: DecodedInstr) -> BuildResult:
    spec = examples.jrz_rel()
    disp = decoded.binds["disp"]
    assert isinstance(disp, Disp8)
    binder = {"disp8": _const(disp.value & 0xFF, 8)}
    return spec, binder


def _jp(decoded: DecodedInstr) -> BuildResult:
    spec = examples.jp_paged()
    addr = decoded.binds["addr16_page"]
    assert isinstance(addr, Addr16Page)
    binder = {
        "addr16": _const(addr.offs16.u16, 16),
        "page_hi": _const(addr.page20, 20),
    }
    return spec, binder


def _mv_a_abs(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_a_abs_ext()
    addr = decoded.binds["addr24"]
    assert isinstance(addr, Addr24)
    binder = {
        "addr_ptr": _const(addr.v.u24, 24),
    }
    return spec, binder


BUILDERS: Dict[str, Callable[[DecodedInstr], BuildResult]] = {
    "MV A,n": _mv_a_n,
    "JRZ ±n": _jrz,
    "JP mn": _jp,
    "MV A,[lmn]": _mv_a_abs,
}


def build(decoded: DecodedInstr) -> Tuple[ast.Instr, Binder]:
    if decoded.mnemonic not in BUILDERS:
        raise KeyError(f"No SCIL builder for mnemonic {decoded.mnemonic}")
    return BUILDERS[decoded.mnemonic](decoded)
