import pytest

from .sc_asm import Assembler, AssemblerError


def test_mv_a_imem_numeric_kil_should_require_name() -> None:
    """
    Assembling an internal memory access to 0xF2 (KIL) using a numeric literal
    should be rejected in favor of the symbolic IMEMRegisters name.

    Expectation: the assembler errors and the message references "KIL" to guide
    the user to use `MV A, (KIL)` instead of `MV A, (0xF2)`.
    """
    assembler = Assembler()
    with pytest.raises(AssemblerError) as excinfo:
        assembler.assemble("MV A, (0xF2)")
    # Keep the assertion flexible: any error text that mentions KIL is acceptable
    assert "KIL" in str(excinfo.value)


def test_mv_a_imem_symbol_kil_ok() -> None:
    """Using the symbolic KIL name should assemble successfully."""
    assembler = Assembler()
    bin_file = assembler.assemble("MV A, (KIL)")
    # Opcode for MV A, (n) is 0x80 followed by the byte.
    assert bin_file.as_ti_txt().strip() == "@0000\n80 F2\nq"
