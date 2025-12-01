from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRACE_PY = ROOT / "trace_ref_python.trace"
TRACE_LLAMA = ROOT / "trace_ref_llama.trace"
SCRIPT = ROOT / "scripts" / "compare_perfetto_traces.py"


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
