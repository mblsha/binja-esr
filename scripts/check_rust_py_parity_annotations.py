#!/usr/bin/env python3
"""
Verify that every Rust source file declares machine-parseable links back to the
Python implementation it mirrors.

Annotations must use the format:
    // PY_SOURCE: sc62015/pysc62015/<module>.py[:<object>]
"""

from __future__ import annotations

from pathlib import Path
import sys
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
PY_SOURCE_RE = re.compile(r"^// PY_SOURCE: (?P<path>[^\s:]+)(?::(?P<symbol>.+))?$")
SKIP_DIRS = {".git", "target", "node_modules", "venv", ".venv"}


def iter_rust_files() -> list[Path]:
    rust_files: list[Path] = []
    for path in REPO_ROOT.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rust_files.append(path)
    return sorted(rust_files)


def collect_annotations(rust_file: Path) -> list[tuple[int, str, str]]:
    annotations: list[tuple[int, str, str]] = []
    for idx, line in enumerate(
        rust_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = PY_SOURCE_RE.match(line.strip())
        if match:
            rel_path = match.group("path")
            symbol = (match.group("symbol") or "").strip()
            annotations.append((idx, rel_path, symbol))
    return annotations


def validate_annotation_paths(
    rust_file: Path, annotations: list[tuple[int, str, str]]
) -> list[str]:
    issues: list[str] = []
    for lineno, rel_path, _symbol in annotations:
        if Path(rel_path).is_absolute():
            issues.append(
                f"{rust_file}:{lineno}: PY_SOURCE path must be relative to repo root"
            )
            continue

        py_path = (REPO_ROOT / rel_path).resolve()
        if py_path.suffix != ".py":
            issues.append(
                f"{rust_file}:{lineno}: PY_SOURCE must point to a .py file (got {rel_path})"
            )
        if not py_path.exists():
            issues.append(
                f"{rust_file}:{lineno}: referenced Python source not found: {rel_path}"
            )
        elif not py_path.is_file():
            issues.append(
                f"{rust_file}:{lineno}: referenced Python source is not a file: {rel_path}"
            )
    return issues


def main() -> int:
    failures: list[str] = []
    rust_files = iter_rust_files()

    if not rust_files:
        print("No Rust sources found; nothing to check.")
        return 0

    for rust_file in rust_files:
        annotations = collect_annotations(rust_file)
        if not annotations:
            failures.append(f"{rust_file}: missing PY_SOURCE annotation")
            continue
        failures.extend(validate_annotation_paths(rust_file, annotations))

    if failures:
        print("Rust/Python parity annotation check failed:")
        for issue in failures:
            print(f" - {issue}")
        return 1

    print("All Rust sources have valid PY_SOURCE annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
