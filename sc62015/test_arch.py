from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore

from .arch import SC62015


def _full_width_reg(reg_info) -> str:
    # Binary Ninja's RegisterInfo uses `full_width_reg`; the test mocks use `name`.
    return getattr(reg_info, "full_width_reg", getattr(reg_info, "name", ""))


def test_subregister_offsets_match_docs() -> None:
    regs = SC62015.regs

    assert _full_width_reg(regs["A"]) == "BA"
    assert _full_width_reg(regs["B"]) == "BA"
    assert regs["A"].offset == 0  # LSB of BA
    assert regs["B"].offset == 1  # MSB of BA

    assert _full_width_reg(regs["IL"]) == "I"
    assert _full_width_reg(regs["IH"]) == "I"
    assert regs["IL"].offset == 0  # LSB of I
    assert regs["IH"].offset == 1  # MSB of I
