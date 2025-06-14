from binja_helpers import binja_api  # noqa: F401
import pytest

if binja_api._has_binja():
    pytest.skip(
        "Skipping architecture tests requiring Binary Ninja license",
        allow_module_level=True,
    )

from sc62015.arch import SC62015
from binja_helpers.binja_helpers.mock_llil import MockLowLevelILFunction


def test_get_instruction_low_level_il_with_bytes() -> None:
    arch = SC62015()
    il = MockLowLevelILFunction()
    length = arch.get_instruction_low_level_il(b"\x00", 0, il)
    assert length == 1
    assert len(il.ils) == 1
    assert il.ils[0].op == "NOP"
