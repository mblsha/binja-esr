from binja_test_mocks.mock_llil import (  # type: ignore
    MockLowLevelILFunction,
    mllil,
    mreg,
)

from sc62015.scil import backend_llil, specs


class _ImmediateStream:
    def __init__(self, *values: int) -> None:
        self._values = list(values)
        self._index = 0

    def read(self, _kind: str) -> int:
        value = self._values[self._index]
        self._index += 1
        return value


def test_mv_a_imm_emits_set_reg() -> None:
    instr = specs.mv_a_imm()
    il = MockLowLevelILFunction()
    backend_llil.emit_llil(il, instr, _ImmediateStream(0x5A))
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
    backend_llil.emit_llil(il, instr, _ImmediateStream(0x02))
    assert il.ils
    assert il.ils[0].op == "IF"
