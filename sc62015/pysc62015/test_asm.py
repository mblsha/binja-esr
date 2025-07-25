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
        test_id="mv_reg_imm_8bit",
        asm_code="MV A, 0x42",
        expected_ti="""
            @0000
            08 42
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_reg_imm_16bit",
        asm_code="MV BA, 0x1234",
        expected_ti="""
            @0000
            0A 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_reg_imm_20bit",
        asm_code="MV X, 0x12345",
        expected_ti="""
            @0000
            0C 45 23 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_reg_imem",
        asm_code="MV A, (0x10)",
        expected_ti="""
            @0000
            80 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_imem_reg",
        asm_code="MV (0x10), A",
        expected_ti="""
            @0000
            A0 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_imem_imm",
        asm_code="MV (0x20), 0x55",
        expected_ti="""
            @0000
            CC 20 55
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_reg_emem",
        asm_code="MV A, [0x12345]",
        expected_ti="""
            @0000
            88 45 23 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_emem_reg",
        asm_code="MV [0x12345], A",
        expected_ti="""
            @0000
            A8 45 23 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_imem_imem",
        asm_code="MVW (0x30), (0x40)",
        expected_ti="""
            @0000
            C9 30 40
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_imem_imm",
        asm_code="MVW (0x30), 0x1122",
        expected_ti="""
            @0000
            CD 30 22 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvp_imem_imm",
        asm_code="MVP (0x20), 0x112233",
        expected_ti="""
            @0000
            DC 20 33 22 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_imem_emem_abs",
        asm_code="MV (0x10), [0x12345]",
        expected_ti="""
            @0000
            D0 10 45 23 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_emem_abs_imem",
        asm_code="MV [0x12345], (0x20)",
        expected_ti="""
            @0000
            D8 45 23 01 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_imem_emem_abs",
        asm_code="MVW (0x30), [0x12345]",
        expected_ti="""
            @0000
            D1 30 45 23 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_imem_ememreg_simple",
        asm_code="MV (0x20), [X]",
        expected_ti="""
            @0000
            E0 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_ememreg_imem_simple",
        asm_code="MV [X], (0x20)",
        expected_ti="""
            @0000
            E8 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_imem_ememreg_simple",
        asm_code="MVW (0x20), [X]",
        expected_ti="""
            @0000
            E1 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_ememreg_imem_simple",
        asm_code="MVW [X], (0x20)",
        expected_ti="""
            @0000
            E9 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvp_imem_ememreg_simple",
        asm_code="MVP (0x20), [X]",
        expected_ti="""
            @0000
            E2 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvp_ememreg_imem_simple",
        asm_code="MVP [X], (0x20)",
        expected_ti="""
            @0000
            EA 04 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_imem_ememimem_simple",
        asm_code="MV (0x30), [(0x40)]",
        expected_ti="""
            @0000
            F0 00 30 40
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mv_ememimem_imem_simple",
        asm_code="MV [(0x40)], (0x30)",
        expected_ti="""
            @0000
            F8 00 40 30
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_imem_ememimem_simple",
        asm_code="MVW (0x30), [(0x40)]",
        expected_ti="""
            @0000
            F1 00 30 40
            q
        """,
    ),
    AssemblerTestCase(
        test_id="mvw_ememimem_imem_simple",
        asm_code="MVW [(0x40)], (0x30)",
        expected_ti="""
            @0000
            F9 00 40 30
            q
        """,
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
            70 55 71 10 01 72 45 23 01 02 73 20 77 30 32 76
            40 50
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
        test_id="inc_px_n",
        asm_code="INC (PX+0xEC)",
        expected_ti="""
            @0000
            36 6D EC
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
        test_id="add_a_bp_n",
        asm_code="ADD A, (BP+0x50)",
        expected_ti="""
            @0000
            22 42 50
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
    AssemblerTestCase(
        test_id="add_reg_r1",
        asm_code="ADD A, IL",
        expected_ti="""
            @0000
            46 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="add_reg_r2",
        asm_code="ADD BA, I",
        expected_ti="""
            @0000
            44 23
            q
        """,
    ),
    AssemblerTestCase(
        test_id="add_reg_r3",
        asm_code="ADD X, Y",
        expected_ti="""
            @0000
            45 45
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
    AssemblerTestCase(
        test_id="sub_reg_r1",
        asm_code="SUB A, IL",
        expected_ti="""
            @0000
            4E 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sub_reg_r2",
        asm_code="SUB BA, I",
        expected_ti="""
            @0000
            4C 23
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sub_reg_r3",
        asm_code="SUB X, Y",
        expected_ti="""
            @0000
            4D 45
            q
        """,
    ),
    # --- ADC Instruction Tests ---
    AssemblerTestCase(
        test_id="adc_a_imm",
        asm_code="ADC A, 0x01",
        expected_ti="""
            @0000
            50 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="adc_imem_imm",
        asm_code="ADC (0x10), 0x02",
        expected_ti="""
            @0000
            51 10 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="adc_a_imem",
        asm_code="ADC A, (0x20)",
        expected_ti="""
            @0000
            52 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="adc_imem_a",
        asm_code="ADC (0x30), A",
        expected_ti="""
            @0000
            53 30
            q
        """,
    ),
    # --- SBC Instruction Tests ---
    AssemblerTestCase(
        test_id="sbc_a_imm",
        asm_code="SBC A, 0x01",
        expected_ti="""
            @0000
            58 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sbc_imem_imm",
        asm_code="SBC (0x10), 0x02",
        expected_ti="""
            @0000
            59 10 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sbc_a_imem",
        asm_code="SBC A, (0x20)",
        expected_ti="""
            @0000
            5A 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sbc_imem_a",
        asm_code="SBC (0x30), A",
        expected_ti="""
            @0000
            5B 30
            q
        """,
    ),
    # --- ADCL Instruction Tests ---
    AssemblerTestCase(
        test_id="adcl_imem_imem",
        asm_code="ADCL (0x10), (0x20)",
        expected_ti="""
            @0000
            32 54 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="adcl_imem_a",
        asm_code="ADCL (0x30), A",
        expected_ti="""
            @0000
            55 30
            q
        """,
    ),
    # --- SBCL Instruction Tests ---
    AssemblerTestCase(
        test_id="sbcl_imem_imem",
        asm_code="SBCL (0x40), (0x50)",
        expected_ti="""
            @0000
            32 5C 40 50
            q
        """,
    ),
    AssemblerTestCase(
        test_id="sbcl_imem_a",
        asm_code="SBCL (0x60), A",
        expected_ti="""
            @0000
            5D 60
            q
        """,
    ),
    # --- DADL Instruction Tests ---
    AssemblerTestCase(
        test_id="dadl_imem_imem",
        asm_code="DADL (0x10), (0x20)",
        expected_ti="""
            @0000
            32 C4 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="dadl_imem_a",
        asm_code="DADL (0x30), A",
        expected_ti="""
            @0000
            C5 30
            q
        """,
    ),
    # --- DSBL Instruction Tests ---
    AssemblerTestCase(
        test_id="dsbl_imem_imem",
        asm_code="DSBL (0x40), (0x50)",
        expected_ti="""
            @0000
            32 D4 40 50
            q
        """,
    ),
    AssemblerTestCase(
        test_id="dsbl_imem_a",
        asm_code="DSBL (0x60), A",
        expected_ti="""
            @0000
            D5 60
            q
        """,
    ),
    # --- DSLL/DSRL Instruction Tests ---
    AssemblerTestCase(
        test_id="dsll_imem_direct",
        asm_code="DSLL (0x10)",
        expected_ti="""
            @0000
            EC 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="dsrl_imem_direct",
        asm_code="DSRL (0x20)",
        expected_ti="""
            @0000
            FC 20
            q
        """,
    ),
    # --- ROR/ROL/SHR/SHL Memory Instruction Tests ---
    AssemblerTestCase(
        test_id="ror_imem_n",
        asm_code="ROR (0x10)",
        expected_ti="""
            @0000
            E5 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ror_imem_bp_n",
        asm_code="ROR (BP+0x10)",
        expected_ti="""
            @0000
            22 E5 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ror_imem_px_n",
        asm_code="ROR (PX+0x10)",
        expected_ti="""
            @0000
            36 E5 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ror_imem_py_n",
        asm_code="ROR (PY+0x10)",
        expected_ti="""
            @0000
            33 E5 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ror_imem_bp_px",
        asm_code="ROR (BP+PX)",
        expected_ti="""
            @0000
            26 E5
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ror_imem_bp_py",
        asm_code="ROR (BP+PY)",
        expected_ti="""
            @0000
            31 E5
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_n",
        asm_code="ROL (0x11)",
        expected_ti="""
            @0000
            E7 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_bp_n",
        asm_code="ROL (BP+0x11)",
        expected_ti="""
            @0000
            22 E7 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_px_n",
        asm_code="ROL (PX+0x11)",
        expected_ti="""
            @0000
            36 E7 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_py_n",
        asm_code="ROL (PY+0x11)",
        expected_ti="""
            @0000
            33 E7 11
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_bp_px",
        asm_code="ROL (BP+PX)",
        expected_ti="""
            @0000
            26 E7
            q
        """,
    ),
    AssemblerTestCase(
        test_id="rol_imem_bp_py",
        asm_code="ROL (BP+PY)",
        expected_ti="""
            @0000
            31 E7
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_n",
        asm_code="SHR (0x12)",
        expected_ti="""
            @0000
            F5 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_bp_n",
        asm_code="SHR (BP+0x12)",
        expected_ti="""
            @0000
            22 F5 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_px_n",
        asm_code="SHR (PX+0x12)",
        expected_ti="""
            @0000
            36 F5 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_py_n",
        asm_code="SHR (PY+0x12)",
        expected_ti="""
            @0000
            33 F5 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_bp_px",
        asm_code="SHR (BP+PX)",
        expected_ti="""
            @0000
            26 F5
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shr_imem_bp_py",
        asm_code="SHR (BP+PY)",
        expected_ti="""
            @0000
            31 F5
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_n",
        asm_code="SHL (0x13)",
        expected_ti="""
            @0000
            F7 13
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_bp_n",
        asm_code="SHL (BP+0x13)",
        expected_ti="""
            @0000
            22 F7 13
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_px_n",
        asm_code="SHL (PX+0x13)",
        expected_ti="""
            @0000
            36 F7 13
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_py_n",
        asm_code="SHL (PY+0x13)",
        expected_ti="""
            @0000
            33 F7 13
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_bp_px",
        asm_code="SHL (BP+PX)",
        expected_ti="""
            @0000
            26 F7
            q
        """,
    ),
    AssemblerTestCase(
        test_id="shl_imem_bp_py",
        asm_code="SHL (BP+PY)",
        expected_ti="""
            @0000
            31 F7
            q
        """,
    ),
    # --- JP/JR Instruction Tests ---
    AssemblerTestCase(
        test_id="jp_abs",
        asm_code="JP 0x1234",
        expected_ti="""
            @0000
            02 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jpf_abs",
        asm_code="JPF 0xABCDE",
        expected_ti="""
            @0000
            03 DE BC 0A
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jp_imem",
        asm_code="JP (0x10)",
        expected_ti="""
            @0000
            10 10
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jp_reg",
        asm_code="JP S",
        expected_ti="""
            @0000
            11 07
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jr_plus",
        asm_code="JR +0x05",
        expected_ti="""
            @0000
            12 05
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jr_minus",
        asm_code="JR -0x02",
        expected_ti="""
            @0000
            13 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jpz_abs",
        asm_code="JPZ 0x1234",
        expected_ti="""
            @0000
            14 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jpnz_abs",
        asm_code="JPNZ 0x1234",
        expected_ti="""
            @0000
            15 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jpc_abs",
        asm_code="JPC 0x1234",
        expected_ti="""
            @0000
            16 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jpnc_abs",
        asm_code="JPNC 0x1234",
        expected_ti="""
            @0000
            17 34 12
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrz_plus",
        asm_code="JRZ +0x05",
        expected_ti="""
            @0000
            18 05
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrz_minus",
        asm_code="JRZ -0x02",
        expected_ti="""
            @0000
            19 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrnz_plus",
        asm_code="JRNZ +0x05",
        expected_ti="""
            @0000
            1A 05
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrnz_minus",
        asm_code="JRNZ -0x02",
        expected_ti="""
            @0000
            1B 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrc_plus",
        asm_code="JRC +0x05",
        expected_ti="""
            @0000
            1C 05
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrc_minus",
        asm_code="JRC -0x02",
        expected_ti="""
            @0000
            1D 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrnc_plus",
        asm_code="JRNC +0x05",
        expected_ti="""
            @0000
            1E 05
            q
        """,
    ),
    AssemblerTestCase(
        test_id="jrnc_minus",
        asm_code="JRNC -0x02",
        expected_ti="""
            @0000
            1F 02
            q
        """,
    ),
    # --- PMDF Instruction Tests ---
    AssemblerTestCase(
        test_id="pmdf_imem_imm",
        asm_code="PMDF (0x70), 0x03",
        expected_ti="""
            @0000
            47 70 03
            q
        """,
    ),
    AssemblerTestCase(
        test_id="pmdf_imem_a",
        asm_code="PMDF (0x80), A",
        expected_ti="""
            @0000
            57 80
            q
        """,
    ),
    # --- OR Instruction Tests ---
    AssemblerTestCase(
        test_id="or_a_imm",
        asm_code="OR A, 0x55",
        expected_ti="""
            @0000
            78 55
            q
        """,
    ),
    AssemblerTestCase(
        test_id="or_imem_imm",
        asm_code="OR (0x10), 0x01",
        expected_ti="""
            @0000
            79 10 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="or_emem_imm",
        asm_code="OR [0x12345], 0x02",
        expected_ti="""
            @0000
            7A 45 23 01 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="or_imem_a",
        asm_code="OR (0x20), A",
        expected_ti="""
            @0000
            7B 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="or_a_imem",
        asm_code="OR A, (0x30)",
        expected_ti="""
            @0000
            7F 30
            q
        """,
    ),
    AssemblerTestCase(
        test_id="or_imem_imem",
        asm_code="OR (0x40), (0x50)",
        expected_ti="""
            @0000
            32 7E 40 50
            q
        """,
    ),
    # --- XOR Instruction Tests ---
    AssemblerTestCase(
        test_id="xor_a_imm",
        asm_code="XOR A, 0x55",
        expected_ti="""
            @0000
            68 55
            q
        """,
    ),
    AssemblerTestCase(
        test_id="xor_imem_imm",
        asm_code="XOR (0x10), 0x01",
        expected_ti="""
            @0000
            69 10 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="xor_emem_imm",
        asm_code="XOR [0x12345], 0x02",
        expected_ti="""
            @0000
            6A 45 23 01 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="xor_imem_a",
        asm_code="XOR (0x20), A",
        expected_ti="""
            @0000
            6B 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="xor_a_imem",
        asm_code="XOR A, (0x30)",
        expected_ti="""
            @0000
            6F 30
            q
        """,
    ),
    AssemblerTestCase(
        test_id="xor_imem_imem",
        asm_code="XOR (0x40), (0x50)",
        expected_ti="""
            @0000
            32 6E 40 50
            q
        """,
    ),
    # --- Exchange Instruction Tests ---
    AssemblerTestCase(
        test_id="ex_imem_imem_simple",
        asm_code="EX (0x10), (0x20)",
        expected_ti="""
            @0000
            32 C0 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ex_imem_imem_complex",
        asm_code="EX (BP+0x10), (PY+0x20)",
        expected_ti="""
            @0000
            23 C0 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="exw_imem_imem_simple",
        asm_code="EXW (0x30), (0x40)",
        expected_ti="""
            @0000
            32 C1 30 40
            q
        """,
    ),
    AssemblerTestCase(
        test_id="exp_imem_imem_simple",
        asm_code="EXP (0x50), (0x60)",
        expected_ti="""
            @0000
            32 C2 50 60
            q
        """,
    ),
    AssemblerTestCase(
        test_id="exl_imem_imem_simple",
        asm_code="EXL (0x70), (0x80)",
        expected_ti="""
            @0000
            32 C3 70 80
            q
        """,
    ),
    AssemblerTestCase(
        test_id="ex_reg_reg_simple",
        asm_code="EX X, Y",
        expected_ti="""
            @0000
            ED 45
            q
        """,
    ),
    # --- CMP Instruction Tests ---
    AssemblerTestCase(
        test_id="cmp_a_imm",
        asm_code="CMP A, 0x55",
        expected_ti="""
            @0000
            60 55
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmp_imem_imm",
        asm_code="CMP (0x10), 0x02",
        expected_ti="""
            @0000
            61 10 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmp_emem_imm",
        asm_code="CMP [0x12345], 0x02",
        expected_ti="""
            @0000
            62 45 23 01 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmp_imem_a",
        asm_code="CMP (0x30), A",
        expected_ti="""
            @0000
            63 30
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmp_bp_py_a",
        asm_code="CMP (BP+PY), A",
        expected_ti="""
            @0000
            31 63
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmp_imem_imem",
        asm_code="CMP (0x40), (0x50)",
        expected_ti="""
            @0000
            32 B7 40 50
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmpw_imem_imem",
        asm_code="CMPW (0x10), (0x20)",
        expected_ti="""
            @0000
            C6 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmpw_imem_reg",
        asm_code="CMPW (0x30), BA",
        expected_ti="""
            @0000
            D6 02 30
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmpp_imem_imem",
        asm_code="CMPP (0x10), (0x20)",
        expected_ti="""
            @0000
            C7 10 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="cmpp_imem_reg",
        asm_code="CMPP (0x40), X",
        expected_ti="""
            @0000
            D7 04 40
            q
        """,
    ),
    # --- TEST Instruction Tests ---
    AssemblerTestCase(
        test_id="test_a_imm",
        asm_code="TEST A, 0x55",
        expected_ti="""
            @0000
            64 55
            q
        """,
    ),
    AssemblerTestCase(
        test_id="test_imem_imm",
        asm_code="TEST (0x10), 0x01",
        expected_ti="""
            @0000
            65 10 01
            q
        """,
    ),
    AssemblerTestCase(
        test_id="test_emem_imm",
        asm_code="TEST [0x12345], 0x02",
        expected_ti="""
            @0000
            66 45 23 01 02
            q
        """,
    ),
    AssemblerTestCase(
        test_id="test_imem_a",
        asm_code="TEST (0x20), A",
        expected_ti="""
            @0000
            67 20
            q
        """,
    ),
    AssemblerTestCase(
        test_id="org_changes_address",
        asm_code="""
            NOP
            .ORG 0x10
            HALT
        """,
        expected_ti="""
            @0000
            00
            @0010
            DE
            q
        """,
    ),
    AssemblerTestCase(
        test_id="org_with_label_reference",
        asm_code="""
            .ORG 0x20
        label:
            NOP
            JP label
        """,
        expected_ti="""
            @0020
            00 02 20 00
            q
        """,
    ),
    AssemblerTestCase(
        test_id="org_multiple_addresses",
        asm_code="""
            .ORG 0x10
            NOP
            .ORG 0x20
            HALT
            .ORG 0x05
            NOP
        """,
        expected_ti="""
            @0005
            00
            @0010
            00
            @0020
            DE
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
    """Ensure previously ambiguous instructions now assemble."""
    assembler = Assembler()
    source_code = "MV A, 0x42"
    bin_file = assembler.assemble(source_code)
    assert bin_file.as_ti_txt().strip() == "@0000\n08 42\nq"
