from sc62015.scil import ast, passes, specs


def test_fold_pcrel_without_disp() -> None:
    expr = ast.PcRel(base_advance=4, out_size=20)
    folded = passes.fold_expr(expr)
    assert isinstance(folded, ast.Const)
    assert folded.value == 4


def test_fold_pcrel_with_const_disp() -> None:
    expr = ast.PcRel(
        base_advance=2,
        disp=ast.Const(0xFE, 8),  # -2
        out_size=20,
    )
    folded = passes.fold_expr(expr)
    assert isinstance(folded, ast.Const)
    assert folded.value == 0  # 2 + (-2)


def test_fold_join24_constants() -> None:
    expr = ast.Join24(
        hi=ast.Const(0x01, 8),
        mid=ast.Const(0x02, 8),
        lo=ast.Const(0x03, 8),
    )
    folded = passes.fold_expr(expr)
    assert isinstance(folded, ast.Const)
    assert folded.value == 0x010203


def test_fold_instr_runs_on_seed_spec() -> None:
    instr = specs.mv_a_abs_ext()
    folded = passes.fold_instr(instr)
    assert folded.length == instr.length
    assert len(folded.semantics) == len(instr.semantics)
