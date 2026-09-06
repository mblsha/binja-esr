"""Test the qualification harness policy without loading the real BN core."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


def _load(
    monkeypatch,
    *,
    ui=False,
    expressions=("LLIL_SET_FLAG", "LLIL_CONST"),
    diagnostics="",
    forbid_arena=False,
):
    class Expression:
        def __init__(self, index, name):
            self.expr_index = index
            self.operation = SimpleNamespace(name=name)

        def traverse(self, callback):
            for index, name in enumerate(expressions):
                yield callback(None if name is None else Expression(index, name))

    class Fragment:
        def __init__(self, _arch):
            self.successor = False

        def nop(self):
            return "successor"

        def append(self, value):
            self.successor = value == "successor"

        def finalize(self):
            assert self.successor

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return Expression(0, expressions[0])

        def get_expr_count(self):
            return len(expressions) + int(forbid_arena)

        def get_expr(self, index):
            if forbid_arena:
                pytest.fail("read expression arena instead of attached operands")
            name = expressions[index]
            return (
                None
                if name is None
                else SimpleNamespace(operation=SimpleNamespace(name=name))
            )

    log_path = None

    def log_to_file(_level, path):
        nonlocal log_path
        log_path = Path(path)
        log_path.write_text("")

    def close_logs():
        # Exercise delayed diagnostics becoming visible only on flush.
        assert log_path is not None
        log_path.write_text(diagnostics)

    monkeypatch.setitem(
        sys.modules,
        "binaryninja",
        SimpleNamespace(
            LowLevelILFunction=Fragment,
            core_ui_enabled=lambda: ui,
            log_to_file=log_to_file,
            close_logs=close_logs,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "binaryninja.enums",
        SimpleNamespace(
            LogLevel=SimpleNamespace(WarningLog=2),
        ),
    )
    return runpy.run_path(str(Path(__file__).with_name("live_binja_llil_qualify.py")))


def _arch():
    return SimpleNamespace(
        get_instruction_info=lambda *_: SimpleNamespace(length=1),
        get_instruction_text=lambda *_: ([SimpleNamespace(text="NOP")], 1),
        get_instruction_low_level_il=lambda *_: 1,
    )


def test_gui_batch_is_rejected_before_architecture_registration(monkeypatch):
    qualifier = _load(monkeypatch, ui=True)
    with pytest.raises(RuntimeError, match="disabled.*GUI"):
        qualifier["_load_current_architecture"]()
    with pytest.raises(RuntimeError, match="disabled.*GUI"):
        qualifier["_qualify_one"](None, b"\x00", 0)


@pytest.mark.parametrize("nested", ["LLIL_UNIMPL", "LLIL_UNIMPL_MEM", None])
def test_nested_missing_semantics_cannot_hide_beneath_set_flag(monkeypatch, nested):
    qualifier = _load(monkeypatch, expressions=("LLIL_SET_FLAG", nested))
    with pytest.raises(ValueError, match="LLIL expression"):
        qualifier["_qualify_one"](_arch(), b"\x00", 0)


def test_valid_fragment_has_successor_before_finalization(monkeypatch):
    qualifier = _load(monkeypatch)
    assert qualifier["_qualify_one"](_arch(), b"\x00", 0)["accepted"]


def test_unused_expression_arena_slots_are_never_dereferenced(monkeypatch):
    qualifier = _load(monkeypatch, forbid_arena=True)
    result = qualifier["_qualify_one"](_arch(), b"\x00", 0)
    assert result["accepted"]
    assert result["llil_reachable_expressions"] == 2


def test_reachable_undef_is_not_treated_as_qualified_semantics(monkeypatch):
    qualifier = _load(monkeypatch, expressions=("LLIL_SET_FLAG", "LLIL_UNDEF"))
    with pytest.raises(ValueError, match="LLIL expression"):
        qualifier["_qualify_one"](_arch(), b"\x00", 0)


def test_native_diagnostic_fails_even_when_api_returns_success(monkeypatch):
    qualifier = _load(monkeypatch, diagnostics="LLIL out of bounds")
    with pytest.raises(RuntimeError, match="diagnostics while qualifying 00 at 00000"):
        qualifier["_qualify_one"](_arch(), b"\x00", 0)


def test_original_failure_is_retained_alongside_native_diagnostic(monkeypatch):
    qualifier = _load(
        monkeypatch, expressions=("LLIL_UNIMPL",), diagnostics="index error"
    )
    with pytest.raises(RuntimeError, match="original_error=ValueError.*unimplemented"):
        qualifier["_qualify_one"](_arch(), b"\x00", 0)


def test_source_digest_tracks_transitive_source_and_file_identity(
    monkeypatch, tmp_path
):
    qualifier = _load(monkeypatch)
    fingerprint = qualifier["_source_digest"]
    with pytest.raises(RuntimeError, match="no Python"):
        fingerprint(tmp_path)
    (tmp_path / "arch.py").write_text("unchanged main architecture")
    dependent = tmp_path / "pysc62015"
    dependent.mkdir()
    intrinsic = dependent / "intrinsics.py"
    intrinsic.write_text("old intrinsic signature")
    old = fingerprint(tmp_path)
    intrinsic.write_text("new intrinsic signature")
    assert fingerprint(tmp_path) != old
    new = fingerprint(tmp_path)
    (dependent / "ignored.pyc").write_bytes(b"bytecode")
    assert fingerprint(tmp_path) == new
    intrinsic.rename(dependent / "renamed_intrinsics.py")
    assert fingerprint(tmp_path) != new


def test_source_digest_is_relocatable_and_has_unambiguous_boundaries(
    monkeypatch, tmp_path
):
    qualifier = _load(monkeypatch)
    fingerprint = qualifier["_source_digest"]
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        (root / "a.py").write_text("a")
        (root / "b.py").write_text("bc")
    assert fingerprint(left) == fingerprint(right)
    (right / "a.py").write_text("ab")
    (right / "b.py").write_text("c")
    assert fingerprint(left) != fingerprint(right)
