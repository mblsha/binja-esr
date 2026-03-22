# test_asm_e2e.py
import pytest
from textwrap import dedent
from typing import NamedTuple, List

# Assuming the assembler class is in sc_asm.py as created in the previous step
from .sc_asm import Assembler, AssemblerError


# Define a structure for our end-to-end test cases
class AssemblerTestCase(NamedTuple):
    test_id: str
    asm_code: str
    expected_ihex: str


# A list of test cases to run
assembler_test_cases: List[AssemblerTestCase] = [
    AssemblerTestCase(
        test_id="simple_nop_and_halt",
        asm_code="""
            ; A very simple program
            NOP
            HALT
        """,
        expected_ihex="""
            :0200000000DE20
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="code_with_label_and_jump",
        asm_code="""
            start:
                MV A, 0x42      ; Load A with 0x42
                JP start        ; Infinite loop
        """,
        expected_ihex="""
            :050000000842020000AF
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="data_directive_defb",
        asm_code="""
            SECTION data
            my_data:
                defb 0xDE, 0xAD, 0xBE, 0xEF
        """,
        expected_ihex="""
            :020000040008F2
            :04000000DEADBEEFC4
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="defm_string_and_defl_pointer",
        asm_code="""
            SECTION code
                JPF message_printer

            SECTION data
            message_printer:
                MV X, [msg_ptr_loc]
                RET

            msg_ptr_loc:
                defl message

            message:
                defm "OK"
                defb 0
        """,
        expected_ihex="""
            :0400000003000008F1
            :020000040008F2
            :0B0000008C050008060800084F4B00AC
            :00000001FF
        """,
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
        expected_ihex="""
            :0100000000FF
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="forward_reference_in_jump",
        asm_code="""
            JP  there   ; Jump to a label defined later
            NOP         ; This should be at address 0x0003
        there:
            HALT        ; This should be at address 0x0004
        """,
        expected_ihex="""
            :0500000002040000DE17
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="pointer_loop",
        asm_code="""
            table_ptr:
                defl table

            table:
                defb 1, 2, 3, 0

            start:
                MV X, [table_ptr]
        read_next:
                MV A, [X++]
                CMP A, 0
                JPZ done
                JP read_next
        done:
                RET
        """,
        expected_ihex="""
            :16000000030000010203008C00000090246000141500020B000605
            :00000001FF
        """,
    ),
    AssemblerTestCase(
        test_id="high_org_call_label",
        asm_code="""
            .ORG 0x10100
        start:
            CALL emit_char
            RET
        emit_char:
            RET
        """,
        expected_ihex="""
            :020000040001F9
            :050100000404010606E5
            :00000001FF
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
    expected_ihex = dedent(case.expected_ihex).strip()

    try:
        bin_file = assembler.assemble(source_code)
        actual_ihex = bin_file.as_ihex().strip()

        # For easier debugging, compare line by line
        actual_lines = actual_ihex.splitlines()
        expected_lines = expected_ihex.splitlines()

        assert actual_lines == expected_lines, (
            f"IHEX output mismatch for test '{case.test_id}'"
        )

    except AssemblerError as e:
        # Re-raise with a cleaner message for test reports
        pytest.fail(f"Test '{case.test_id}' failed during assembly: {e}", pytrace=False)
    except Exception as e:
        pytest.fail(f"Test '{case.test_id}' raised an unexpected exception: {e}")
