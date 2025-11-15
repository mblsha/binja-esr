from __future__ import annotations

import json
from typing import Any, Dict

from . import ast
from ..decoding.bind import PreLatch

_KIND = "type"


def _const_to_dict(expr: ast.Const) -> Dict[str, Any]:
    return {_KIND: "const", "value": expr.value, "size": expr.size}


def _tmp_to_dict(tmp: ast.Tmp) -> Dict[str, Any]:
    return {_KIND: "tmp", "name": tmp.name, "size": tmp.size}


def _reg_to_dict(reg: ast.Reg) -> Dict[str, Any]:
    return {_KIND: "reg", "name": reg.name, "size": reg.size, "bank": reg.bank}


def _mem_to_dict(mem: ast.Mem) -> Dict[str, Any]:
    return {
        _KIND: "mem",
        "space": mem.space,
        "size": mem.size,
        "addr": expr_to_dict(mem.addr),
    }


def cond_to_dict(cond: ast.Cond) -> Dict[str, Any]:
    data: Dict[str, Any] = {_KIND: "cond", "kind": cond.kind}
    if cond.a is not None:
        data["a"] = expr_to_dict(cond.a)
    if cond.b is not None:
        data["b"] = expr_to_dict(cond.b)
    if cond.flag is not None:
        data["flag"] = cond.flag
    return data


def expr_to_dict(expr: ast.Expr) -> Dict[str, Any]:
    if isinstance(expr, ast.Const):
        return _const_to_dict(expr)
    if isinstance(expr, ast.Tmp):
        return _tmp_to_dict(expr)
    if isinstance(expr, ast.Reg):
        return _reg_to_dict(expr)
    if isinstance(expr, ast.Flag):
        return {_KIND: "flag", "name": expr.name}
    if isinstance(expr, ast.Mem):
        return _mem_to_dict(expr)
    if isinstance(expr, ast.UnOp):
        return {
            _KIND: "unop",
            "op": expr.op,
            "a": expr_to_dict(expr.a),
            "out_size": expr.out_size,
            "param": expr.param,
        }
    if isinstance(expr, ast.BinOp):
        return {
            _KIND: "binop",
            "op": expr.op,
            "a": expr_to_dict(expr.a),
            "b": expr_to_dict(expr.b),
            "out_size": expr.out_size,
        }
    if isinstance(expr, ast.TernOp):
        return {
            _KIND: "ternop",
            "op": expr.op,
            "cond": cond_to_dict(expr.cond),
            "t": expr_to_dict(expr.t),
            "f": expr_to_dict(expr.f),
            "out_size": expr.out_size,
        }
    if isinstance(expr, ast.PcRel):
        data: Dict[str, Any] = {
            _KIND: "pcrel",
            "base": expr.base_advance,
            "out_size": expr.out_size,
        }
        if expr.disp is not None:
            data["disp"] = expr_to_dict(expr.disp)
        return data
    if isinstance(expr, ast.Join24):
        return {
            _KIND: "join24",
            "hi": expr_to_dict(expr.hi),
            "mid": expr_to_dict(expr.mid),
            "lo": expr_to_dict(expr.lo),
        }
    if isinstance(expr, ast.LoopIntPtr):
        return {
            _KIND: "loop_ptr",
            "offset": expr_to_dict(expr.offset),
        }
    raise TypeError(f"Unsupported expression {expr!r}")


def stmt_to_dict(stmt: ast.Stmt) -> Dict[str, Any]:
    if isinstance(stmt, ast.Fetch):
        return {_KIND: "fetch", "kind": stmt.kind, "dst": _tmp_to_dict(stmt.dst)}
    if isinstance(stmt, ast.SetReg):
        data: Dict[str, Any] = {
            _KIND: "set_reg",
            "reg": _reg_to_dict(stmt.reg),
            "value": expr_to_dict(stmt.value),
        }
        if stmt.flags:
            data["flags"] = list(stmt.flags)
        return data
    if isinstance(stmt, ast.Store):
        return {
            _KIND: "store",
            "dst": _mem_to_dict(stmt.dst),
            "value": expr_to_dict(stmt.value),
        }
    if isinstance(stmt, ast.SetFlag):
        return {
            _KIND: "set_flag",
            "flag": stmt.flag,
            "value": expr_to_dict(stmt.value),
        }
    if isinstance(stmt, ast.If):
        return {
            _KIND: "if",
            "cond": cond_to_dict(stmt.cond),
            "then": [stmt_to_dict(s) for s in stmt.then_ops],
            "else": [stmt_to_dict(s) for s in stmt.else_ops],
        }
    if isinstance(stmt, ast.Goto):
        return {_KIND: "goto", "target": expr_to_dict(stmt.target)}
    if isinstance(stmt, ast.Call):
        return {
            _KIND: "call",
            "target": expr_to_dict(stmt.target),
            "far": stmt.far,
        }
    if isinstance(stmt, ast.Ret):
        return {_KIND: "ret", "far": stmt.far, "reti": stmt.reti}
    if isinstance(stmt, ast.ExtRegLoad):
        data: Dict[str, Any] = {
            _KIND: "ext_reg_load",
            "dst": _reg_to_dict(stmt.dst),
            "ptr": _reg_to_dict(stmt.ptr),
            "mode": stmt.mode,
        }
        if stmt.disp is not None:
            data["disp"] = expr_to_dict(stmt.disp)
        return data
    if isinstance(stmt, ast.ExtRegStore):
        data = {
            _KIND: "ext_reg_store",
            "src": _reg_to_dict(stmt.src),
            "ptr": _reg_to_dict(stmt.ptr),
            "mode": stmt.mode,
        }
        if stmt.disp is not None:
            data["disp"] = expr_to_dict(stmt.disp)
        return data
    if isinstance(stmt, ast.IntMemSwap):
        return {
            _KIND: "int_mem_swap",
            "left": expr_to_dict(stmt.left),
            "right": expr_to_dict(stmt.right),
            "width": stmt.width,
        }
    if isinstance(stmt, ast.ExtRegToIntMem):
        data = {
            _KIND: "ext_reg_to_int",
            "ptr": _reg_to_dict(stmt.ptr),
            "mode": stmt.mode,
            "dst": _mem_to_dict(stmt.dst),
        }
        if stmt.disp is not None:
            data["disp"] = expr_to_dict(stmt.disp)
        return data
    if isinstance(stmt, ast.IntMemToExtReg):
        data = {
            _KIND: "int_to_ext_reg",
            "ptr": _reg_to_dict(stmt.ptr),
            "mode": stmt.mode,
            "src": _mem_to_dict(stmt.src),
        }
        if stmt.disp is not None:
            data["disp"] = expr_to_dict(stmt.disp)
        return data
    if isinstance(stmt, ast.Effect):
        return {
            _KIND: "effect",
            "kind": stmt.kind,
            "args": [expr_to_dict(arg) for arg in stmt.args],
        }
    if isinstance(stmt, ast.Label):
        return {_KIND: "label", "name": stmt.name}
    if isinstance(stmt, ast.Comment):
        return {_KIND: "comment", "text": stmt.text}
    raise TypeError(f"Unsupported statement {stmt!r}")


def instr_to_dict(instr: ast.Instr) -> Dict[str, Any]:
    return {
        "name": instr.name,
        "length": instr.length,
        "semantics": [stmt_to_dict(stmt) for stmt in instr.semantics],
    }


def binder_to_dict(binder) -> Dict[str, Any]:
    return {name: expr_to_dict(expr) for name, expr in binder.items()}


def prelatch_to_dict(pre: PreLatch) -> Dict[str, Any]:
    return {"first": pre.first, "second": pre.second}


def to_json(instr: ast.Instr, *, indent: int = 2) -> str:
    return json.dumps(instr_to_dict(instr), indent=indent, sort_keys=True)


def _dict_to_tmp(data: Dict[str, Any]) -> ast.Tmp:
    return ast.Tmp(name=data["name"], size=data["size"])


def _dict_to_reg(data: Dict[str, Any]) -> ast.Reg:
    return ast.Reg(name=data["name"], size=data["size"], bank=data.get("bank", "gpr"))


def _dict_to_mem(data: Dict[str, Any]) -> ast.Mem:
    return ast.Mem(
        space=data["space"],
        size=data["size"],
        addr=dict_to_expr(data["addr"]),
    )


def dict_to_cond(data: Dict[str, Any]) -> ast.Cond:
    return ast.Cond(
        kind=data["kind"],
        a=dict_to_expr(data["a"]) if "a" in data else None,
        b=dict_to_expr(data["b"]) if "b" in data else None,
        flag=data.get("flag"),
    )


def dict_to_expr(data: Dict[str, Any]) -> ast.Expr:
    kind = data[_KIND]
    if kind == "const":
        return ast.Const(value=data["value"], size=data["size"])
    if kind == "tmp":
        return _dict_to_tmp(data)
    if kind == "reg":
        return _dict_to_reg(data)
    if kind == "flag":
        return ast.Flag(name=data["name"])
    if kind == "mem":
        return _dict_to_mem(data)
    if kind == "unop":
        return ast.UnOp(
            op=data["op"],
            a=dict_to_expr(data["a"]),
            out_size=data["out_size"],
            param=data.get("param"),
        )
    if kind == "binop":
        return ast.BinOp(
            op=data["op"],
            a=dict_to_expr(data["a"]),
            b=dict_to_expr(data["b"]),
            out_size=data["out_size"],
        )
    if kind == "ternop":
        return ast.TernOp(
            op=data["op"],
            cond=dict_to_cond(data["cond"]),
            t=dict_to_expr(data["t"]),
            f=dict_to_expr(data["f"]),
            out_size=data["out_size"],
        )
    if kind == "pcrel":
        return ast.PcRel(
            base_advance=data["base"],
            disp=dict_to_expr(data["disp"]) if "disp" in data else None,
            out_size=data.get("out_size", 20),
        )
    if kind == "join24":
        return ast.Join24(
            hi=dict_to_expr(data["hi"]),
            mid=dict_to_expr(data["mid"]),
            lo=dict_to_expr(data["lo"]),
        )
    if kind == "loop_ptr":
        return ast.LoopIntPtr(dict_to_expr(data["offset"]))
    raise ValueError(f"Unknown expression kind {kind}")


def dict_to_stmt(data: Dict[str, Any]) -> ast.Stmt:
    kind = data[_KIND]
    if kind == "fetch":
        return ast.Fetch(kind=data["kind"], dst=_dict_to_tmp(data["dst"]))
    if kind == "set_reg":
        flags = tuple(data.get("flags", ()))
        return ast.SetReg(
            reg=_dict_to_reg(data["reg"]),
            value=dict_to_expr(data["value"]),
            flags=flags if flags else None,
        )
    if kind == "store":
        return ast.Store(
            dst=_dict_to_mem(data["dst"]), value=dict_to_expr(data["value"])
        )
    if kind == "set_flag":
        return ast.SetFlag(flag=data["flag"], value=dict_to_expr(data["value"]))
    if kind == "if":
        return ast.If(
            cond=dict_to_cond(data["cond"]),
            then_ops=[dict_to_stmt(s) for s in data.get("then", ())],
            else_ops=[dict_to_stmt(s) for s in data.get("else", ())],
        )
    if kind == "goto":
        return ast.Goto(target=dict_to_expr(data["target"]))
    if kind == "call":
        return ast.Call(target=dict_to_expr(data["target"]), far=data.get("far", False))
    if kind == "ret":
        return ast.Ret(far=data.get("far", False), reti=data.get("reti", False))
    if kind == "ext_reg_load":
        return ast.ExtRegLoad(
            dst=_dict_to_reg(data["dst"]),
            ptr=_dict_to_reg(data["ptr"]),
            mode=data["mode"],
            disp=dict_to_expr(data["disp"]) if "disp" in data else None,
        )
    if kind == "ext_reg_store":
        return ast.ExtRegStore(
            src=_dict_to_reg(data["src"]),
            ptr=_dict_to_reg(data["ptr"]),
            mode=data["mode"],
            disp=dict_to_expr(data["disp"]) if "disp" in data else None,
        )
    if kind == "int_mem_swap":
        return ast.IntMemSwap(
            left=dict_to_expr(data["left"]),
            right=dict_to_expr(data["right"]),
            width=data["width"],
        )
    if kind == "ext_reg_to_int":
        return ast.ExtRegToIntMem(
            ptr=_dict_to_reg(data["ptr"]),
            mode=data["mode"],
            disp=dict_to_expr(data["disp"]) if "disp" in data else None,
            dst=_dict_to_mem(data["dst"]),
        )
    if kind == "int_to_ext_reg":
        return ast.IntMemToExtReg(
            src=_dict_to_mem(data["src"]),
            ptr=_dict_to_reg(data["ptr"]),
            mode=data["mode"],
            disp=dict_to_expr(data["disp"]) if "disp" in data else None,
        )
    if kind == "effect":
        return ast.Effect(
            kind=data["kind"],
            args=tuple(dict_to_expr(arg) for arg in data.get("args", ())),
        )
    if kind == "label":
        return ast.Label(name=data["name"])
    if kind == "comment":
        return ast.Comment(text=data["text"])
    raise ValueError(f"Unknown statement kind {kind}")


def dict_to_instr(data: Dict[str, Any]) -> ast.Instr:
    semantics = tuple(dict_to_stmt(stmt) for stmt in data.get("semantics", ()))
    return ast.Instr(name=data["name"], length=data["length"], semantics=semantics)


def from_json(payload: str) -> ast.Instr:
    return dict_to_instr(json.loads(payload))
