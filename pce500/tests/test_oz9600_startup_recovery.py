"""Invalid host saves remain available for diagnosis and manual restoration."""

from pathlib import Path

import pytest

from pce500.oz9600.startup_recovery import action_at, backup, detail_lines


def test_copy_damage_preserves_original_and_previous_copy(tmp_path):
    path = tmp_path / "state.ozbat"
    data = b"invalid saved image\0\xff"
    path.write_bytes(data)
    first, second = backup(path), backup(path)
    assert first != second
    assert path.read_bytes() == first.read_bytes() == second.read_bytes() == data
    with pytest.raises(FileNotFoundError):
        backup(tmp_path / "missing")
    assert path.read_bytes() == data


def test_long_paths_and_error_details_remain_readable():
    rows = detail_lines("checksum mismatch", Path("x" * 5000), "Copy failed")
    assert len(rows) > 26
    assert all(len(row) <= 62 for row in rows)
    assert rows[0] == "checksum mismatch"
    assert rows[-1] == "Copy failed"


def test_failed_backup_removes_partial_copy_and_preserves_source(tmp_path, monkeypatch):
    import pce500.oz9600.startup_recovery as recovery

    path = tmp_path / "state.ozbat"
    data = b"last good or damaged original"
    path.write_bytes(data)
    prior = backup(path)

    def fail_sync(_descriptor):
        raise OSError("test-only disk failure")

    monkeypatch.setattr(recovery.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk failure"):
        backup(path)
    assert path.read_bytes() == prior.read_bytes() == data
    assert set(tmp_path.iterdir()) == {path, prior}


def test_close_label_left_edge_does_not_copy_a_backup():
    assert action_at((277, 490)) == 2
    assert action_at((130, 490)) == 1
    assert action_at((24, 490)) == 0
    assert action_at((277, 450)) is None
    assert action_at(None) is None
