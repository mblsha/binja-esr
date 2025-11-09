from sc62015.scil import ast, specs, validate


def test_seed_specs_validate_cleanly() -> None:
    for factory in (specs.mv_a_imm, specs.jrz_rel, specs.mv_a_abs_ext):
        instr = factory()
        assert validate.validate(instr) == []


def test_set_reg_width_mismatch_is_reported() -> None:
    instr = ast.Instr(
        name="BAD_WIDTH",
        length=1,
        semantics=(
            ast.SetReg(
                reg=ast.Reg("A", 8),
                value=ast.Const(0, 16),
            ),
        ),
    )
    errors = validate.validate(instr)
    assert any("set_reg" in msg for msg in errors)


def test_external_memory_requires_24_bit_address() -> None:
    instr = ast.Instr(
        name="BAD_MEM",
        length=4,
        semantics=(
            ast.SetReg(
                ast.Reg("A", 8),
                ast.Mem("ext", ast.Const(0x10, 8), 8),
            ),
        ),
    )
    errors = validate.validate(instr)
    assert any("ext address" in msg for msg in errors)
