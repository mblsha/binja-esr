"""Host persistence failure and ownership checks; no emulator state injection."""

import pytest

from pce500.oz9600.persistence import RetainedStore, _atomic_write


def test_partial_write_keeps_old_image_and_cleans_temporary(tmp_path):
    path = tmp_path / "saved.ozbat"
    path.write_bytes(b"previous complete image")

    def fail(stream):
        stream.write(b"partial replacement")
        raise OSError("injected write failure")

    with pytest.raises(OSError, match="injected write failure"):
        _atomic_write(path, fail)
    assert path.read_bytes() == b"previous complete image"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_replace_reload_dedup_and_failed_save_retry(tmp_path, monkeypatch):
    path = tmp_path / "saved.ozbat"
    with RetainedStore(path) as store:
        assert store.loaded is None
        assert store.save(b"first")
        assert store.save(b"second complete image")
        assert not store.save(b"second complete image")
        with monkeypatch.context() as patch:
            import pce500.oz9600.persistence as persistence

            def fail(*_):
                raise OSError("injected replace failure")

            patch.setattr(persistence.os, "replace", fail)
            with pytest.raises(OSError, match="replace failure"):
                store.save(b"third")
            assert path.read_bytes() == b"second complete image"
        assert store.save(b"third")
    with RetainedStore(path) as store:
        assert store.loaded == b"third"


def test_writer_lock_excludes_second_owner_and_releases_without_removing_sidecar(
    tmp_path,
):
    path = tmp_path / "saved.ozbat"
    with RetainedStore(path):
        with pytest.raises(OSError):
            RetainedStore(path)
    assert path.with_name(path.name + ".lock").exists()
    with RetainedStore(path) as store:
        assert store.loaded is None
    with pytest.raises(ValueError, match="closed"):
        store.save(b"late write")


def test_existing_unreadable_target_is_not_empty_memory(tmp_path):
    path = tmp_path / "saved.ozbat"
    path.mkdir()
    with pytest.raises(OSError):
        RetainedStore(path)
    assert path.is_dir()
