"""Bounded GUI qualification using retained, view-owned analysis functions.

Load this module once through binja-cli, then call start(case) and inspect(case)
in separate requests. Never creates anonymous LowLevelILFunction objects,
redirects/closes logs, alters existing views, or saves a database. The caller
must capture/check GUI diagnostics separately (binja-cli --fail-on-new-errors).
This is a small integration fixture, not the full headless manifest sweep.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import binaryninja as bn


CASES = {
    "nop": "00",
    "ror_a": "e4",
    "ror_imem": "e520",
    "rol_a": "e6",
    "rol_imem": "e720",
    "shr_a": "f4",
    "shr_imem": "f520",
    "shl_a": "f6",
    "shl_imem": "f720",
    "wait": "ef",
    "reset": "ff",
    "redundant_pre": "228020",
    "pre_alias": "248001",
    "pre_31": "318020",
    "pre_32": "328020",
    "pre_33": "338020",
    "pre_36": "368020",
    "pre_26": "268000",
    "pre_pair_24_30": "24308020",
    "pre_pair_30_24": "30248000",
    "callf_raw_high_nibble": "053a077c",
    "mv_x_raw_high_nibble": "0ca55a3c",
    "mv_emem_raw_high_nibble": "88000181",
    "dsll": "ec20",
    "dsrl": "fc20",
    "ex_bp": "30c0ec30",
    "exw_px": "34c10030",
    "dadl": "c42030",
    "dsbl": "d42030",
}

_fixtures: dict = {}
_architecture = None
_digest = None
_source_loader = None


def _load():
    global _architecture, _digest, _source_loader
    if _architecture is None:
        path = Path(__file__).with_name("live_binja_llil_qualify.py")
        spec = importlib.util.spec_from_file_location(
            "_sc62015_gui_source_loader", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load source qualification helper")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _architecture, _, _digest = module._register_current_architecture()
        _source_loader = module
    elif _source_loader._source_digest() != _digest:
        raise RuntimeError("qualification sources changed; load a fresh fixture module")
    return _architecture


def start(case: str) -> dict:
    if not bn.core_ui_enabled():
        raise RuntimeError("this fixture is intended for the licensed GUI process")
    if case not in CASES:
        raise ValueError(f"unknown bounded fixture: {case}")
    if case in _fixtures:
        raise RuntimeError("fixture already exists; inspect it rather than rebuilding")
    architecture = _load()
    data = bytes.fromhex(CASES[case])
    info = architecture.get_instruction_info(data, 0)
    if info is None or info.length != len(data):
        raise ValueError("fixture target is not exactly one accepted instruction")
    text = architecture.get_instruction_text(data, 0)
    if text is None or text[1] != len(data):
        raise ValueError("fixture instruction text disagrees with metadata")
    rendered = "".join(token.text for token in text[0]).strip()
    if not rendered:
        raise ValueError("fixture instruction text is empty")
    view = bn.BinaryView.new(data + b"\x07")  # RETF successor for normal flow
    if view is None:
        raise RuntimeError("could not create isolated in-memory BinaryView")
    # Retain the view before requesting analysis, including on errors.
    fixture = {
        "view": view,
        "data": data,
        "source_digest": _digest,
        "rendered": rendered,
    }
    _fixtures[case] = fixture
    view.platform = architecture.standalone_platform
    function = view.add_function(0)
    if function is None:
        raise RuntimeError("could not create fixture function")
    fixture["function"] = function
    view.update_analysis()
    return {"case": case, "source_digest": _digest, "analysis_requested": True}


def inspect(case: str) -> dict:
    fixture = _fixtures[case]
    if (
        _source_loader is not None
        and _source_loader._source_digest() != fixture["source_digest"]
    ):
        raise RuntimeError("qualification sources changed since fixture creation")
    view, function = fixture["view"], fixture["function"]
    progress = view.analysis_progress.state.name
    if progress != "IdleState":
        return {"case": case, "pending": True, "analysis_state": progress}
    if function.analysis_skipped:
        raise RuntimeError("fixture analysis was skipped; refusing IL access")
    result = {
        "case": case,
        "source_digest": fixture["source_digest"],
        "target_hex": fixture["data"].hex(),
        "rendered": fixture["rendered"],
        "bn_version": bn.core_version(),
        "view_owned": True,
        "diagnostics_checked": False,
        "stages": {},
    }
    for stage in ("lifted_il", "llil"):
        il = getattr(function, stage)
        if il is None:
            raise RuntimeError(f"missing {stage}")
        # Keep IL wrappers alive with their function/view for the entire session.
        fixture[stage] = il
        expressions = _source_loader._reachable_expressions(il)
        bad = []
        for expression in expressions:
            if expression.operation.name in {
                "LLIL_UNIMPL",
                "LLIL_UNIMPL_MEM",
                "LLIL_UNDEF",
            }:
                bad.append(
                    {"index": expression.expr_index, "expression": str(expression)}
                )
        constants = [
            {
                "expression": expression.expr_index,
                "size": expression.size,
                "value": expression.constant,
            }
            for expression in expressions
            if expression.operation.name == "LLIL_CONST"
        ]
        rows = [
            {
                "address": instruction.address,
                "text": str(instruction),
                "operation": instruction.operation.name,
            }
            for instruction in il.instructions
        ]
        if not rows:
            raise RuntimeError(f"empty {stage}")
        result["stages"][stage] = {
            "rows": rows,
            "bad_expressions": bad,
            "constants": constants,
            "reachable_expression_count": len(expressions),
        }
    result["no_missing_semantics"] = all(
        not s["bad_expressions"] for s in result["stages"].values()
    )
    return result
