"""Bounded OS-window startup regression; requires a display (xvfb on Linux).

Uses only task-owned temporary files, no ROM or emulator data from the user.
The headless control rejects immediately; GUI failure stays available for
recovery until the configured host watchdog closes it. No UI input is injected.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


def check(binary: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="oz-startup-failure-") as directory:
        root = Path(directory)
        state = root / "state.ozbat"
        original = b"invalid saved bytes kept for recovery\0\xff"
        state.write_bytes(original)
        replay = root / "input.json"
        replay.write_text('{"steps":[{"boundaries":0}]}\n')
        base = [
            str(binary.resolve()),
            "--rom",
            str(root / "missing.ozrom"),
            "--state",
            str(state),
        ]
        results = {}
        for name, extra in [
            ("headless", ["--headless", "--replay", str(replay)]),
            ("visible_failure", ["--quit-after-ms", "2000", "--scale", "1"]),
        ]:
            start = time.monotonic()
            result = subprocess.run(base + extra, capture_output=True, timeout=15)
            elapsed = time.monotonic() - start
            assert result.returncode == 1, (name, result.stderr)
            assert b"No such file" in result.stderr or b"cannot find" in result.stderr
            assert state.read_bytes() == original, name
            if name == "visible_failure":
                assert elapsed >= 1.9, "GUI exited before its recovery window watchdog"
            results[name] = {"elapsed_seconds": elapsed, "image_unchanged": True}
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().binary)))
