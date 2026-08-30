from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRACE_PY = ROOT / "trace_ref_python.trace"
TRACE_LLAMA = ROOT / "trace_ref_llama.trace"
SCRIPT = ROOT / "scripts" / "compare_perfetto_traces.py"
SWEEP = ROOT / "tools" / "llama_parity_sweep.py"
SPEC = importlib.util.spec_from_file_location("public_compare_perfetto_traces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _event(index: int, *, name: str = "Instruction"):
    return compare.TraceEvent(
        track="InstructionTrace",
        name=name,
        timestamp=index,
        event_type=0,
        annotations={
            "pc": 0x1000 + index,
            "opcode": 0x00,
            "op_index": index,
            "mem_imr": 0x80,
            "mem_isr": 0x00,
            "reg_pc": 0x1000 + index,
        },
    )


@pytest.mark.parametrize("field", sorted(compare.REQUIRED_INSTRUCTION_FIELDS))
def test_instruction_index_rejects_missing_required_field(field: str) -> None:
    event = _event(0)
    event.annotations.pop(field)

    with pytest.raises(ValueError, match=field):
        compare._index_instruction_events([event])


def test_instruction_index_rejects_duplicate_op_index() -> None:
    with pytest.raises(ValueError, match="duplicate instruction op_index 3"):
        compare._index_instruction_events(
            [_event(3, name="first"), _event(3, name="second")]
        )


def test_instruction_index_ignores_legacy_execution_diagnostic() -> None:
    canonical = _event(0)
    annotations = dict(canonical.annotations)
    annotations.pop("op_index")
    annotations.pop("mem_imr")
    annotations.pop("mem_isr")
    legacy = compare.TraceEvent(
        track="Execution",
        name=canonical.name,
        timestamp=canonical.timestamp,
        event_type=canonical.event_type,
        annotations=annotations,
    )

    assert compare._index_instruction_events([legacy]) == {}


def test_instruction_comparison_uses_union_of_op_indices() -> None:
    lhs = {0: _event(0)}
    rhs = {0: _event(0), 1: _event(1)}

    index, event_a, event_b, fields = compare.compare_instruction_traces(lhs, rhs)

    assert index == 1
    assert event_a is None
    assert event_b == rhs[1]
    assert fields == ["missing-event"]


def test_complete_matching_instruction_events_compare_equal() -> None:
    event = _event(0)
    assert compare.compare_instruction_traces({0: event}, {0: event}) == (
        None,
        None,
        None,
        [],
    )


def test_synthetic_trace_subprocess_contract(tmp_path: Path) -> None:
    """Generate both backends' traces and validate them through the real CLI."""

    prefix = tmp_path / "synthetic"
    env = os.environ.copy()
    env["FORCE_BINJA_MOCK"] = "1"
    generated = subprocess.run(
        [
            sys.executable,
            str(SWEEP),
            "--limit",
            "0",
            "--emit-traces",
            "--trace-prefix",
            str(prefix),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert generated.returncode == 0, (
        f"stdout:\n{generated.stdout}\nstderr:\n{generated.stderr}"
    )

    python_trace = prefix.with_name(f"{prefix.name}_python.trace")
    llama_trace = prefix.with_name(f"{prefix.name}_llama.trace")
    for trace_path in (python_trace, llama_trace):
        indexed = compare._index_instruction_events(compare._load_trace(trace_path))
        # Five decoded instructions, not one event per program byte.
        assert set(indexed) == set(range(5))
        for event in indexed.values():
            assert compare.REQUIRED_INSTRUCTION_FIELDS <= event.annotations.keys()
            assert any(key.startswith("reg_") for key in event.annotations)

    compared = subprocess.run(
        [sys.executable, str(SCRIPT), str(python_trace), str(llama_trace)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert compared.returncode == 0, (
        f"stdout:\n{compared.stdout}\nstderr:\n{compared.stderr}"
    )


@pytest.mark.skipif(
    not (TRACE_PY.exists() and TRACE_LLAMA.exists()),
    reason="reference traces not available",
)
def test_perfetto_trace_comparison_smoke(tmp_path: Path):
    # Run the comparison script against bundled reference traces; should exit 0.
    out = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            str(TRACE_PY),
            str(TRACE_LLAMA),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if out.returncode != 0:
        pytest.fail(
            f"Perfetto comparison failed (exit {out.returncode})\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
        )
