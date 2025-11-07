import importlib

import pytest

sc62015_rustcore = importlib.util.find_spec("_sc62015_rustcore")
if sc62015_rustcore is None:
    pytest.skip("Rust backend not built; skipping parity smoke test", allow_module_level=True)

import sc62015_parity


def test_parity_runs():
    sc62015_parity.run_parity(0x1234, 64)
