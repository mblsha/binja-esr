#!/usr/bin/env python3
"""
Check that the Rust opcode table matches the Python source of truth.

Run with: `uv run python scripts/check_llama_opcodes.py`
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPCODES_RS = REPO_ROOT / "sc62015/core/src/llama/opcodes.rs"


def generate_expected() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    # Import generator helpers without executing its CLI entrypoint.
    from scripts import generate_llama_opcodes as gen

    mod = gen._import_opcode_table()
    entries = [
        gen._opcode_entry(op, val)
        for op, val in sorted(gen._iter_opcodes(mod), key=lambda pair: pair[0])
    ]
    return "\n".join(entries).strip() + "\n"


def extract_current() -> str:
    content = OPCODES_RS.read_text()
    start_marker = "pub static OPCODES: [OpcodeEntry; 256] = ["
    try:
        start = content.index(start_marker) + len(start_marker)
    except ValueError as exc:
        raise SystemExit(f"opcode table start not found in {OPCODES_RS}") from exc
    remainder = content[start:]
    try:
        end = remainder.index("];")
    except ValueError as exc:
        raise SystemExit(f"opcode table terminator not found in {OPCODES_RS}") from exc
    body = remainder[:end]
    return body.strip() + "\n"


def main() -> int:
    current = extract_current()
    expected = generate_expected()
    if current == expected:
        print("Opcode table matches Python source.")
        return 0

    diff = "\n".join(
        difflib.unified_diff(
            current.splitlines(),
            expected.splitlines(),
            fromfile="opcodes.rs",
            tofile="generated",
            lineterm="",
        )
    )
    print("Opcode table drift detected:\n")
    print(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
