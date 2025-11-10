import pytest
from binja_test_mocks.mock_llil import MockLowLevelILFunction

from sc62015.arch import SC62015


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in ("BN_ALLOW_LEGACY", "BN_SCIL_TRACE", "SC62015_SKIP_BN_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SC62015_SKIP_BN_INIT", "1")
    yield


def test_scil_is_default() -> None:
    arch = SC62015()
    il = MockLowLevelILFunction()
    length = arch.get_instruction_low_level_il(bytes.fromhex("085A"), 0x4000, il)
    assert length == 2
    counters = arch.get_scil_counters()
    assert counters.get("scil_ok") == 1
    assert counters.get("legacy_rescue") is None


def test_allow_legacy_rescues_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("BN_ALLOW_LEGACY", "1")

    def boom(_decoded):
        raise RuntimeError("boom")

    monkeypatch.setattr("sc62015.arch.from_decoded.build", boom)
    arch = SC62015()
    il = MockLowLevelILFunction()
    length = arch.get_instruction_low_level_il(bytes.fromhex("085A"), 0x4000, il)
    assert length == 2
    counters = arch.get_scil_counters()
    assert counters.get("legacy_rescue") == 1


def test_failure_raises_without_rescue(monkeypatch) -> None:
    def boom(_decoded):
        raise RuntimeError("boom")

    monkeypatch.setattr("sc62015.arch.from_decoded.build", boom)
    arch = SC62015()
    il = MockLowLevelILFunction()
    with pytest.raises(RuntimeError):
        arch.get_instruction_low_level_il(bytes.fromhex("085A"), 0x4000, il)
