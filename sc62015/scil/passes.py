from __future__ import annotations


from . import ast


def _mask(bits: int) -> int:
    return (1 << bits) - 1


def _to_signed(value: int, bits: int) -> int:
    mask = _mask(bits)
    value &= mask
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def fold_expr(expr: ast.Expr) -> ast.Expr:
    if isinstance(expr, (ast.Const, ast.Tmp, ast.Reg)):
        return expr

    if isinstance(expr, ast.Mem):
        new_addr = fold_expr(expr.addr)
        if new_addr is expr.addr:
            return expr
        return ast.Mem(space=expr.space, addr=new_addr, size=expr.size)

    if isinstance(expr, ast.UnOp):
        new_a = fold_expr(expr.a)
        if new_a is expr.a:
            return expr
        return ast.UnOp(op=expr.op, a=new_a, out_size=expr.out_size, param=expr.param)

    if isinstance(expr, ast.BinOp):
        new_a = fold_expr(expr.a)
        new_b = fold_expr(expr.b)
        if new_a is expr.a and new_b is expr.b:
            return expr
        return ast.BinOp(op=expr.op, a=new_a, b=new_b, out_size=expr.out_size)

    if isinstance(expr, ast.TernOp):
        new_t = fold_expr(expr.t)
        new_f = fold_expr(expr.f)
        new_cond = ast.Cond(
            kind=expr.cond.kind,
            a=fold_expr(expr.cond.a) if expr.cond.a is not None else None,
            b=fold_expr(expr.cond.b) if expr.cond.b is not None else None,
            flag=expr.cond.flag,
        )
        if (
            new_t is expr.t
            and new_f is expr.f
            and new_cond.a is expr.cond.a
            and new_cond.b is expr.cond.b
        ):
            return expr
        return ast.TernOp(
            op=expr.op, cond=new_cond, t=new_t, f=new_f, out_size=expr.out_size
        )

    if isinstance(expr, ast.PcRel):
        disp = fold_expr(expr.disp) if expr.disp is not None else None
        if disp is None:
            return ast.Const(expr.base_advance & _mask(expr.out_size), expr.out_size)
        if isinstance(disp, ast.Const):
            signed_disp = _to_signed(disp.value, disp.size)
            total = (expr.base_advance + signed_disp) & _mask(expr.out_size)
            return ast.Const(total, expr.out_size)
        if disp is expr.disp:
            return expr
        return ast.PcRel(
            base_advance=expr.base_advance, disp=disp, out_size=expr.out_size
        )

    if isinstance(expr, ast.Join24):
        hi = fold_expr(expr.hi)
        mid = fold_expr(expr.mid)
        lo = fold_expr(expr.lo)
        if all(
            isinstance(part, ast.Const) and part.size == 8 for part in (hi, mid, lo)
        ):
            value = (
                ((hi.value & 0xFF) << 16)
                | ((mid.value & 0xFF) << 8)
                | (lo.value & 0xFF)
            )
            return ast.Const(value=value, size=24)
        if hi is expr.hi and mid is expr.mid and lo is expr.lo:
            return expr
        return ast.Join24(hi=hi, mid=mid, lo=lo)

    raise TypeError(f"Unsupported expression {expr!r}")


def fold_stmt(stmt: ast.Stmt) -> ast.Stmt:
    if isinstance(stmt, ast.Fetch):
        return stmt
    if isinstance(stmt, ast.SetReg):
        return ast.SetReg(reg=stmt.reg, value=fold_expr(stmt.value), flags=stmt.flags)
    if isinstance(stmt, ast.Store):
        return ast.Store(
            dst=ast.Mem(stmt.dst.space, fold_expr(stmt.dst.addr), stmt.dst.size),
            value=fold_expr(stmt.value),
        )
    if isinstance(stmt, ast.SetFlag):
        return ast.SetFlag(flag=stmt.flag, value=fold_expr(stmt.value))
    if isinstance(stmt, ast.If):
        folded_then = tuple(fold_stmt(s) for s in stmt.then_ops)
        folded_else = tuple(fold_stmt(s) for s in stmt.else_ops)
        new_cond = ast.Cond(
            kind=stmt.cond.kind,
            a=fold_expr(stmt.cond.a) if stmt.cond.a is not None else None,
            b=fold_expr(stmt.cond.b) if stmt.cond.b is not None else None,
            flag=stmt.cond.flag,
        )
        return ast.If(cond=new_cond, then_ops=folded_then, else_ops=folded_else)
    if isinstance(stmt, ast.Goto):
        return ast.Goto(target=fold_expr(stmt.target))
    if isinstance(stmt, ast.Call):
        return ast.Call(target=fold_expr(stmt.target), far=stmt.far)
    if isinstance(stmt, ast.Ret):
        return stmt
    if isinstance(stmt, ast.Effect):
        return ast.Effect(
            kind=stmt.kind, args=tuple(fold_expr(arg) for arg in stmt.args)
        )
    if isinstance(stmt, ast.Label):
        return stmt
    if isinstance(stmt, ast.Comment):
        return stmt
    return stmt


def fold_instr(instr: ast.Instr) -> ast.Instr:
    return ast.Instr(
        name=instr.name,
        length=instr.length,
        semantics=tuple(fold_stmt(stmt) for stmt in instr.semantics),
    )
