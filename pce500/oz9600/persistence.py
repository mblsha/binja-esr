"""Host-only atomic image file and advisory single-writer ownership reference.

The caller validates loaded bytes through the core before saving. This module
never reads guest memory, steps a CPU or changes the retained-image format.
"""

from collections.abc import Callable
import hashlib
import os
from pathlib import Path
import tempfile
from typing import BinaryIO


def _atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write(path: Path, image: bytes) -> None:
    _atomic_write(path, lambda stream: stream.write(image))


class RetainedStore:
    """Keep a sidecar OS lock throughout one session, including replacements.

    Closing/crashing releases the advisory lock; the empty sidecar may remain.
    A failure to read an existing image is never treated as empty memory.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = path.with_name(path.name + ".lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.loaded: bytes | None = path.read_bytes()
            except FileNotFoundError:
                self.loaded = None
            self._last_hash = (
                hashlib.sha256(self.loaded).digest()
                if self.loaded is not None
                else None
            )
        except BaseException:
            self._lock.close()
            raise

    def save(self, image: bytes) -> bool:
        """Save validated/exported bytes; failures preserve the old dedup hash."""
        if self._lock.closed:
            raise ValueError("retained store is closed")
        digest = hashlib.sha256(image).digest()
        if digest == self._last_hash:
            return False
        atomic_write(self.path, image)
        self._last_hash = digest
        return True

    def close(self) -> None:
        if self._lock.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock, fcntl.LOCK_UN)
        finally:
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
