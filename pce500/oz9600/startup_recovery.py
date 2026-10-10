"""Host-only failed-startup recovery contract; Rust owns the native OS window.

Retry reconstructs from the same files/profile. Rejected images are never
replaced with empty memory. Backup copies raw original bytes, including damage.
"""

import os
from itertools import count
from pathlib import Path
from time import time_ns

_next_backup = count()


def backup(path: Path) -> Path:
    """Create a distinct durable copy; preserve the source and all prior copies."""
    data = path.read_bytes()
    target = path.with_name(
        f"{path.name}.recovery-{time_ns() // 1_000_000}-{os.getpid()}-{next(_next_backup)}"
    )
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def detail_lines(error: str, path: Path | None, status: str) -> list[str]:
    """Wrap complete host details for a scrollable view; never alter guest pixels."""
    message = (
        f"{error}\n\nSaved file: {path if path is not None else '(none)'}\n\n"
        "No replacement image is being saved.\n"
        "Restore a known-good backup at this path, then Retry.\n"
        "Copy backup preserves the original bytes separately.\n\n"
        f"{status}"
    )
    return [
        chunk
        for line in message.splitlines()
        for chunk in ([line[i : i + 62] for i in range(0, len(line), 62)] or [""])
    ]


def action_at(point: tuple[int, int] | None) -> int | None:
    """Hit-test Retry, Copy and Close including the left edge of their labels."""
    if point is None or not 470 <= point[1] < 505:
        return None
    x = point[0]
    return 0 if x < 110 else 1 if x < 270 else 2
