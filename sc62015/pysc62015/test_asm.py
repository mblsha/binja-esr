# test_asm_e2e.py
import pytest
from textwrap import dedent
from typing import NamedTuple, List
from lark import exceptions as lark_exceptions
from typing import Optional

from .sc_asm import Assembler, AssemblerError


class AssemblerTestCase(NamedTuple):
    test_id: str
    asm_code: str
    expected_ihex: Optional[str] = None
    expected_ti: Optional[str] = None


assembler_test_cases: List[AssemblerTestCase] = [
    AssemblerTestCase(
        test_id="simple_unambiguous_instructions",
        asm_code="""
            ; A very simple program
            NOP
            SC
            RC
            HALT
        """,
        # Opcodes: NOP=0x00, SC=0x97, RC=0x9F, HALT=0xDE
        expected_ti="""
            @0000
            00 97 9F DE
            q
        """
    ),
    AssemblerTestCase(
        test_id="register_and_stack_ops",
        asm_code="""
            start:
                MV B, A
                SWAP A
                PUSHS F
                POPS F
                RET
        """,
        # Opcodes: MV B,A=0x75, SWAP A=0xEE, PUSHS F=0x4F, POPS F=0x5F, RET=0x06
        expected_ti="""
            @0000
            75 EE 4F 5F 06
            q
        """
    ),
    AssemblerTestCase(
        test_id="user_stack_register_ops",
        asm_code="""
            PUSHU A
            PUSHU IL
            PUSHU BA
            PUSHU I
            PUSHU X
            PUSHU Y
            PUSHU F
            PUSHU IMR
            POPU A
            POPU IL
            POPU BA
            POPU I
            POPU X
            POPU Y
            POPU F
            POPU IMR
        """,
        expected_ti="""
            @0000
            28 29 2A 2B 2C 2D 2E 2F 38 39 3A 3B 3C 3D 3E 3F
            q
        """
    ),
    AssemblerTestCase(
        test_id="data_directive_defb_with_code",
        asm_code="""
            SECTION code
                RETF
            SECTION data
            my_data:
                defb 0xDE, 0xAD, 0xBE, 0xEF
        """,
        # Opcodes: RETF=0x07. Data section starts at 0x80000.
        expected_ti="""
            @0000
            07
            @80000
            DE AD BE EF
            q
        """
    ),
    AssemblerTestCase(
        test_id="defm_and_unambiguous_ops",
        asm_code="""
            SECTION code
                EX A, B
                WAIT
            SECTION data
            message:
                defm "OK"
        """,
        # Opcodes: EX A,B=0xDD, WAIT=0xEF. Data "OK" = 0x4F, 0x4B.
        expected_ti="""
            @0000
            DD EF
            @80000
            4F 4B
            q
        """
    ),
    AssemblerTestCase(
        test_id="bss_section_is_not_in_output",
        asm_code="""
            SECTION code
                NOP

            SECTION bss
            my_buffer:
                defs 256 ; This should not appear in the output file
        """,
        expected_ti="""
            @0000
            00
            q
        """
    ),
    AssemblerTestCase(
        test_id="symbols_in_data_directives",
        asm_code="""
            val1: defb 1
            val2: defb 2

            SECTION data
            pointers:
                defw val1, val2
        """,
        # val1=0x00, val2=0x01. data section starts at 0x80000
        # defw writes 0x0000 (val1), 0x0100 (val2) -> bytes 00 00, 01 00
        expected_ti="""
            @0000
            01 02
            @80000
            00 00 01 00
            q
        """
    ),
    # --- Tests for PRE byte generation in MV instructions ---
    AssemblerTestCase(
        test_id="mv_imem_imem_simple_pre",
        asm_code='MV (0x10), (0x20)',
        # Should generate PRE=0x32, then MV=0xC8, then operands
        expected_ti="""
            @0000
            32 C8 10 20
            q
        """
    ),
    AssemblerTestCase(
        test_id="mv_imem_imem_complex_pre_1",
        asm_code='MV (BP+0x10), (PY+0x20)',
        # Should generate PRE=0x23, then MV=0xC8, then operands
        expected_ti="""
            @0000
            23 C8 10 20
            q
        """
    ),
    AssemblerTestCase(
        test_id="mv_imem_imem_complex_pre_2",
        asm_code="""
            MV (BP+PX), (BP+0x30)
        """,
        # Should generate PRE=0x24, then MV=0xC8, then operands.
        # Note: (BP+PX) has no operand byte.
        expected_ti="""
            @0000
            24 C8 30
            q
        """
    ),
    AssemblerTestCase(
        test_id="mv_imem_imem_invalid_combo",
        asm_code='MV (BP+PX), (BP+PY)'
    ),
    AssemblerTestCase(
        test_id="and_all_forms",
        asm_code="""
            AND A, 0x55
            AND (0x10), 0x01
            AND [0x12345], 0x02
            AND (0x20), A
            AND A, (0x30)
            AND (0x40), (0x50)
        """,
        expected_ti="""
            @0000
            70 55 71 10 01 72 45 23 01 02 73 20 77 30 76 40
            50
            q
        """
    ),
    AssemblerTestCase(  
        test_id="call_and_callf",
        asm_code="""
            CALL 0xAABB
            CALLF 0xAABBCC
        """,
        expected_ti="""
            @0000
            04 BB AA 05 CC BB AA
            q
        """,
    ),
    AssemblerTestCase(
        test_id="inc_reg",
        asm_code="INC A",
        expected_ti="""
            @0000
            6C 00
            q
        """,
    ),
    AssemblerTestCase(
        test_id="inc_imem",
        asm_code="INC (0x10)",
        expected_ti="""
            @0000
            6D 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="dec_reg",
        asm_code="DEC S",
        expected_ti="""
            @0000
            7C 07
            q
        """,
    ),
    AssemblerTestCase(
        test_id="dec_imem",
        asm_code="DEC (0x20)",
        expected_ti="""
            @0000
            7D 20
            q
        """,
    ),
    # --- ADD Instruction Tests ---
    AssemblerTestCase(
        test_id="add_a_imm",
        asm_code="ADD A, 0x01",
        expected_ti="""
            @0000
            40 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="add_imem_imm",
        asm_code="ADD (0x10), 0x02",
        expected_ti="""
            @0000
            41 10 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="add_a_imem",
        asm_code="ADD A, (0x20)",
        expected_ti="""
            @0000
            42 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="add_imem_a",
        asm_code="ADD (0x30), A",
        expected_ti="""
            @0000
            43 30
            q
        """,
    ),
    # --- SUB Instruction Tests ---
    AssemblerTestCase(
        test_id="sub_a_imm",
        asm_code="SUB A, 0x01",
        expected_ti="""
            @0000
            48 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sub_imem_imm",
        asm_code="SUB (0x10), 0x02",
        expected_ti="""
            @0000
            49 10 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sub_a_imem",
        asm_code="SUB A, (0x20)",
        expected_ti="""
            @0000
            4A 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sub_imem_a",
        asm_code="SUB (0x30), A",
        expected_ti="""
            @0000
            4B 30
            q
        """,
    ),
]


@pytest.mark.parametrize("case", assembler_test_cases, ids=lambda c: c.test_id)
def test_assembler_e2e(case: AssemblerTestCase) -> None:
    """
    Runs an end-to-end assembly test case.
    It assembles the provided code and compares the resulting IHEX
    output against the expected golden string.
    """
    assembler = Assembler()

    # dedent is used to remove common leading whitespace from triple-quoted strings,
    # which makes the test cases in the list above much more readable.
    source_code = dedent(case.asm_code)
    expected_ihex = dedent(case.expected_ihex).strip() if case.expected_ihex else ""
    expected_ti = dedent(case.expected_ti).strip() if case.expected_ti else ""

    if "invalid" in case.test_id:
        with pytest.raises(AssemblerError) as exc_info:
            assembler.assemble(source_code)
        assert "Invalid addressing mode combination" in str(exc_info.value)
        return

    try:
        bin_file = assembler.assemble(source_code)
        actual_ihex = bin_file.as_ihex().strip()
        actual_ti = bin_file.as_ti_txt().strip()

        if case.expected_ti:
            expected_ti = dedent(case.expected_ti).strip()
            assert (
                actual_ti.splitlines() == expected_ti.splitlines()
            ), f"TI output mismatch for test '{case.test_id}'"

        if case.expected_ihex:
            assert (
                actual_ihex.splitlines() == expected_ihex.splitlines()
            ), f"IHEX output mismatch for test '{case.test_id}'"

    except (AssemblerError, lark_exceptions.LarkError) as e:
        # Re-raise with a cleaner message for test reports
        pytest.fail(f"Test '{case.test_id}' failed during assembly: {e}", pytrace=False)
    except Exception as e:
        pytest.fail(f"Test '{case.test_id}' raised an unexpected exception: {e}")


def test_assembler_fails_on_ambiguous_instruction() -> None:
    """
    Tests that the assembler fails to parse an instruction not in the grammar.
    """
    assembler = Assembler()
    # MV with an immediate value is not in the simple asm.lark grammar
    source_code = "MV A, 0x42"
    with pytest.raises((AssemblerError, lark_exceptions.LarkError)):
        assembler.assemble(source_code)
