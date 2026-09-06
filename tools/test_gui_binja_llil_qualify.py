"""Safety policy for the bounded GUI fixture; not a real-BN qualification."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load(monkeypatch):
    api = SimpleNamespace(core_ui_enabled=lambda: True)
    monkeypatch.setitem(sys.modules, "binaryninja", api)
    path = Path(__file__).with_name("gui_binja_llil_qualify.py")
    spec = importlib.util.spec_from_file_location("gui_fixture_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, api


def test_unknown_and_duplicate_cases_do_not_load_an_architecture(monkeypatch):
    module, _ = _load(monkeypatch)
    monkeypatch.setattr(module, "_load", lambda: pytest.fail("loaded architecture"))
    with pytest.raises(ValueError, match="unknown"):
        module.start("not_a_fixture")
    module._fixtures["nop"] = {}
    with pytest.raises(RuntimeError, match="already exists"):
        module.start("nop")


def test_invalid_target_is_rejected_before_creating_a_view(monkeypatch):
    module, _ = _load(monkeypatch)
    monkeypatch.setattr(
        module, "_load", lambda: SimpleNamespace(get_instruction_info=lambda *_: None)
    )
    with pytest.raises(ValueError, match="exactly one"):
        module.start("nop")


@pytest.mark.parametrize("text", [None, ([], 1), ([SimpleNamespace(text="NOP")], 2)])
def test_inconsistent_text_is_rejected_before_creating_a_view(monkeypatch, text):
    module, _ = _load(monkeypatch)
    monkeypatch.setattr(
        module,
        "_load",
        lambda: SimpleNamespace(
            get_instruction_info=lambda *_: SimpleNamespace(length=1),
            get_instruction_text=lambda *_: text,
        ),
    )
    with pytest.raises(ValueError, match="instruction text"):
        module.start("nop")


def test_source_changes_reject_cached_load_and_inspection(monkeypatch):
    module, _ = _load(monkeypatch)
    module._architecture = object()
    module._digest = "old"
    module._source_loader = SimpleNamespace(_source_digest=lambda: "new")
    module._fixtures["nop"] = {"source_digest": "old"}
    with pytest.raises(RuntimeError, match="sources changed"):
        module._load()
    with pytest.raises(RuntimeError, match="sources changed"):
        module.inspect("nop")


@pytest.mark.parametrize("pending", [False, True])
def test_pending_or_skipped_analysis_never_requests_il(monkeypatch, pending):
    module, _ = _load(monkeypatch)

    class Function:
        analysis_skipped = True

        @property
        def lifted_il(self):
            pytest.fail("accessed skipped/incomplete IL")

    module._fixtures["nop"] = {
        "function": Function(),
        "view": SimpleNamespace(
            analysis_progress=SimpleNamespace(
                state=SimpleNamespace(name="AnalyzeState" if pending else "IdleState")
            )
        ),
    }
    if pending:
        assert module.inspect("nop")["pending"]
    else:
        with pytest.raises(RuntimeError, match="skipped"):
            module.inspect("nop")


def test_gui_inspection_uses_only_attached_expressions(monkeypatch):
    module, api = _load(monkeypatch)
    root = SimpleNamespace(operation=SimpleNamespace(name="LLIL_SET_FLAG"), address=0)
    nested = SimpleNamespace(
        expr_index=7, operation=SimpleNamespace(name="LLIL_UNIMPL")
    )

    class IL:
        instructions = [root]

        def get_expr_count(self):
            pytest.fail("GUI must not scan the expression arena")

        def get_expr(self, _index):
            pytest.fail("GUI must not dereference expression arena slots")

    il = IL()
    module._source_loader = SimpleNamespace(
        _source_digest=lambda: "test",
        _reachable_expressions=lambda value: [nested] if value is il else [],
    )
    api.core_version = lambda: "synthetic policy test"
    module._fixtures["nop"] = {
        "function": SimpleNamespace(analysis_skipped=False, lifted_il=il, llil=il),
        "view": SimpleNamespace(
            analysis_progress=SimpleNamespace(state=SimpleNamespace(name="IdleState"))
        ),
        "data": b"\x00",
        "source_digest": "test",
        "rendered": "NOP",
    }
    result = module.inspect("nop")
    assert not result["no_missing_semantics"]
    assert all(s["bad_expressions"][0]["index"] == 7 for s in result["stages"].values())
