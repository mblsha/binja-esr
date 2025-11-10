import pytest
from binja_test_mocks.mock_llil import MockLowLevelILFunction

from sc62015.arch import SC62015


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "BN_USE_SCIL",
        "BN_SCIL_ALLOW",
        "BN_SCIL_BLOCK",
        "BN_SCIL_FAMILIES",
        "BN_SCIL_STRICT_COMPARE",
        "BN_SCIL_TRACE",
        "SC62015_SKIP_BN_INIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SC62015_SKIP_BN_INIT", "1")
    yield


def test_prod_mode_emits_scil_for_allowlisted_instruction(monkeypatch) -> None:
    monkeypatch.setenv("BN_USE_SCIL", "prod")
    monkeypatch.setenv("BN_SCIL_ALLOW", "MV A,n")
    arch = SC62015()
    il = MockLowLevelILFunction()
    length = arch.get_instruction_low_level_il(bytes.fromhex("085A"), 0x4000, il)
    assert length == 2
    counters = arch.get_scil_counters()
    assert counters.get("prod_success") == 1


def test_blocklist_disables_prod_path(monkeypatch) -> None:
    monkeypatch.setenv("BN_USE_SCIL", "prod")
    monkeypatch.setenv("BN_SCIL_ALLOW", "MV A,n")
    monkeypatch.setenv("BN_SCIL_BLOCK", "MV A,n")
    arch = SC62015()
    il = MockLowLevelILFunction()
    arch.get_instruction_low_level_il(bytes.fromhex("085A"), 0x4000, il)
    counters = arch.get_scil_counters()
    assert counters.get("prod_success") is None


def test_family_flag_allows_batch(monkeypatch) -> None:
    monkeypatch.setenv("BN_USE_SCIL", "prod")
    monkeypatch.setenv("BN_SCIL_FAMILIES", "imm8")
    arch = SC62015()
    il = MockLowLevelILFunction()
    arch.get_instruction_low_level_il(bytes.fromhex("4055"), 0x4000, il)
    counters = arch.get_scil_counters()
    assert counters.get("prod_success") == 1
