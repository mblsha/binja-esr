#!/usr/bin/env python3
"""Run the SC62015 pytest suite across multiple CPU backends."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, List


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="python,rust",
        help="Comma-separated list of backends to run (default: python,rust)",
    )
    parser.add_argument(
        "--path",
        default="sc62015/pysc62015",
        help="Pytest target path (default: sc62015/pysc62015)",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest (prefix with --)",
    )
    return parser.parse_args(list(argv))


def _run_pytest(backend: str, target: str, extra_args: List[str]) -> int:
    env = os.environ.copy()
    env["SC62015_CPU_BACKEND"] = backend
    print(f"\n=== Running pytest for backend '{backend}' ===")
    cmd = ["uv", "run", "pytest", target, *extra_args]
    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        print(f"=== Backend '{backend}' passed ===")
    else:
        print(f"=== Backend '{backend}' failed with code {result.returncode} ===")
    return result.returncode


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    backends = [entry.strip() for entry in args.backends.split(",") if entry.strip()]
    if not backends:
        print("error: no backends specified", file=sys.stderr)
        return 2

    extra_args = []
    if args.pytest_args:
        extra_args = args.pytest_args
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]

    target = args.path
    failures = {}
    for backend in backends:
        rc = _run_pytest(backend, target, extra_args)
        if rc != 0:
            failures[backend] = rc

    if failures:
        for backend, code in failures.items():
            print(f"backend '{backend}' failed (exit code {code})")
        return 1

    print("\nAll requested backends passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
