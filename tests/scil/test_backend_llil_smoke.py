from binja_test_mocks.mock_llil import (  # type: ignore
    MockLowLevelILFunction,
    mllil,
    mreg,
)

from sc62015.scil import backend_llil, specs, ast
from sc62015.scil.compat_builder import CompatLLILBuilder


def test_mv_a_imm_emits_set_reg() -> None:
    instr = specs.mv_a_imm()
    il = MockLowLevelILFunction()
    binder = {"imm8": ast.Const(0x5A, 8)}
    backend_llil.emit_llil(il, instr, binder, CompatLLILBuilder(il), 0x1000)
    assert il.ils == [
        mllil(
            "SET_REG.b{0}",
            [
                mreg("A"),
                mllil("CONST.b", [0x5A]),
            ],
        )
    ]


def test_jrz_emits_if_node() -> None:
    instr = specs.jrz_rel()
    il = MockLowLevelILFunction()
    binder = {"disp8": ast.Const(0x02, 8)}
    backend_llil.emit_llil(il, instr, binder, CompatLLILBuilder(il), 0x2000)
    assert il.ils
    assert il.ils[0].op == "IF"
