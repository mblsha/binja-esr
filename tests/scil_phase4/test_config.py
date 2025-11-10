import pytest

from sc62015.pysc62015 import config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "BN_USE_SCIL",
        "BN_SCIL_ALLOW",
        "BN_SCIL_BLOCK",
        "BN_SCIL_FAMILIES",
        "BN_SCIL_STRICT_COMPARE",
        "BN_SCIL_TRACE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def test_default_config_is_shadow_mode() -> None:
    cfg = config.load_scil_config()
    assert cfg.mode == "shadow"
    assert not cfg.allow
    assert cfg.strict_compare is True


def test_config_parses_lists(monkeypatch) -> None:
    monkeypatch.setenv("BN_SCIL_ALLOW", "MV A,n;JRZ ±n")
    monkeypatch.setenv("BN_SCIL_FAMILIES", "imm8, rel8 ")
    cfg = config.load_scil_config()
    assert "MV A,n" in cfg.allow
    assert "JRZ ±n" in cfg.allow
    assert cfg.families == {"imm8", "rel8"}


def test_mode_normalization(monkeypatch) -> None:
    monkeypatch.setenv("BN_USE_SCIL", "PROD")
    cfg = config.load_scil_config()
    assert cfg.mode == "prod"

    monkeypatch.setenv("BN_USE_SCIL", "unknown")
    cfg = config.load_scil_config()
    assert cfg.mode == "off"
