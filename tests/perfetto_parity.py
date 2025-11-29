from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import importlib.util


def test_perfetto_reference_traces_match() -> None:
    """Guardrail: ensure bundled Python and LLAMA Perfetto traces compare cleanly."""
    if importlib.util.find_spec("binja_test_mocks") is None:
        pytest.skip("binja_test_mocks not available; skipping perfetto parity check")

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "compare_perfetto_traces.py"
    trace_python = repo_root / "trace_ref_python.trace"
    trace_llama = repo_root / "trace_ref_llama.trace"

    if not script.is_file() or not trace_python.is_file() or not trace_llama.is_file():
        pytest.skip("Perfetto parity fixtures not present")

    result = subprocess.run(
        [sys.executable, str(script), str(trace_python), str(trace_llama)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"perfetto traces diverged: rc={result.returncode}\n{result.stdout}"
