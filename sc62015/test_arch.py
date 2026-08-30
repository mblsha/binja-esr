from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from binja_test_mocks.mock_llil import MockLowLevelILFunction

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


def test_all_architecture_hooks_reject_reserved_opcodes() -> None:
    arch = object.__new__(SC62015)

    for data in (bytes([0x20]), bytes([0xBF])):
        assert arch.get_instruction_info(data, 0x1000) is None
        assert arch.get_instruction_text(data, 0x1000) is None
        assert (
            arch.get_instruction_low_level_il(data, 0x1000, MockLowLevelILFunction())
            is None
        )


def test_all_architecture_hooks_reject_unfused_pre() -> None:
    arch = object.__new__(SC62015)
    data = bytes([0x30, 0x31, 0x00])

    assert arch.get_instruction_info(data, 0x1000) is None
    assert arch.get_instruction_text(data, 0x1000) is None
    assert (
        arch.get_instruction_low_level_il(data, 0x1000, MockLowLevelILFunction())
        is None
    )


def test_register_pair_alias_remains_disassemblable() -> None:
    arch = object.__new__(SC62015)

    text, length = arch.get_instruction_text(bytes([0xED, 0x00]), 0x1000)

    assert length == 2
    assert text


def test_disproved_overlapping_pre_alias_is_not_disassemblable() -> None:
    arch = object.__new__(SC62015)

    assert arch.get_instruction_info(bytes.fromhex("23483f"), 0xF0002) is None


def test_table_or_misaligned_aliases_are_not_disassemblable() -> None:
    arch = object.__new__(SC62015)

    for data in (
        bytes.fromhex("053a077c"),  # F003A: dispatch-table bytes, not a BN entry
        bytes.fromhex("257c01"),  # EFE2B: starts in the preceding instruction
    ):
        assert arch.get_instruction_info(data, 0x1000) is None
        assert arch.get_instruction_text(data, 0x1000) is None
        assert (
            arch.get_instruction_low_level_il(data, 0x1000, MockLowLevelILFunction())
            is None
        )
