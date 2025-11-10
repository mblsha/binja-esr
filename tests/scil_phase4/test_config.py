import pytest

from sc62015.pysc62015 import config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in ("BN_ALLOW_LEGACY", "BN_SCIL_TRACE"):
        monkeypatch.delenv(name, raising=False)
    yield


def test_default_config_has_legacy_off() -> None:
    cfg = config.load_scil_config()
    assert cfg.allow_legacy is False
    assert cfg.trace is False


def test_allow_legacy_flag(monkeypatch) -> None:
    monkeypatch.setenv("BN_ALLOW_LEGACY", "1")
    cfg = config.load_scil_config()
    assert cfg.allow_legacy is True


def test_trace_flag(monkeypatch) -> None:
    monkeypatch.setenv("BN_SCIL_TRACE", "true")
    cfg = config.load_scil_config()
    assert cfg.trace is True
