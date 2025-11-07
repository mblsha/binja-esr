import pytest

sc62015_parity = pytest.importorskip("sc62015_parity")


def test_parity_runs():
    pytest.importorskip(
        "_sc62015_rustcore", reason="Rust backend not built; skipping parity smoke test"
    )
    sc62015_parity.run_parity(0x1234, 64)
