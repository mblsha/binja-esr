from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from binja_test_mocks.mock_llil import MockLowLevelILFunction
import pytest

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
        bytes.fromhex("257c01"),  # EFE2B: starts in the preceding instruction
    ):
        assert arch.get_instruction_info(data, 0x1000) is None
        assert arch.get_instruction_text(data, 0x1000) is None
        assert (
            arch.get_instruction_low_level_il(data, 0x1000, MockLowLevelILFunction())
            is None
        )


@pytest.mark.parametrize(
    "raw",
    [
        "228020",
        "318020",
        "328020",
        "338020",
        "368020",
        "268000",
        "24308020",
        "30248000",
        "248001",
        "053a077c",
        "0ca55a3c",
        "88000181",
    ],
)
def test_silicon_accepted_raw_aliases_reach_all_architecture_hooks(raw: str) -> None:
    # Hardware acceptance and ROM executable boundaries are separate facts.
    # In particular, a table can contain bytes that also encode a valid CALLF;
    # the decoder must not reject that encoding based on its former provenance.
    arch = object.__new__(SC62015)
    data = bytes.fromhex(raw)
    info = arch.get_instruction_info(data, 0x1000)
    text = arch.get_instruction_text(data, 0x1000)
    assert info is not None and info.length == len(data)
    assert text is not None and text[1] == len(data)
    il = MockLowLevelILFunction()
    assert arch.get_instruction_low_level_il(data, 0x1000, il) == len(data)
    assert il.ils


@pytest.mark.parametrize("raw", ["21c00001", "2230248000", "243000", "80", "bf", "20"])
def test_raw_alias_acceptance_does_not_admit_invalid_encodings(raw: str) -> None:
    arch = object.__new__(SC62015)
    data = bytes.fromhex(raw)
    assert arch.get_instruction_info(data, 0x1000) is None
    assert arch.get_instruction_text(data, 0x1000) is None
    il = MockLowLevelILFunction()
    assert arch.get_instruction_low_level_il(data, 0x1000, il) is None
    assert not il.ils
