#!/usr/bin/env python3
"""Minimal CLI wrapper around run_pce500 with snapshot-friendly defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pce500.run_pce500 import run_emulator


def _apply_llama_fast_defaults(ns: argparse.Namespace) -> None:
    """Apply fast defaults when backend is LLAMA or preset is requested."""

    backend_env = (os.getenv("SC62015_CPU_BACKEND") or "").lower()
    if ns.preset == "llama-fast" or backend_env == "llama":
        os.environ.setdefault("SC62015_CPU_BACKEND", "llama")
        os.environ.setdefault("RUST_TIMER_IN_RUST", "1")
        if ns.fast_mode is None:
            ns.fast_mode = True
        if ns.timeout_secs is None:
            ns.timeout_secs = 0.0
        if ns.no_perfetto is None:
            ns.no_perfetto = True


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-E500 CLI (snapshot-friendly)")
    parser.add_argument(
        "--rom", type=str, help="Optional ROM override (unused; kept for future)"
    )
    parser.add_argument(
        "--steps", type=int, default=20000, help="Number of instructions to execute"
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable fast mode (default on for LLAMA)",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=None,
        help="Wall clock timeout (default 0 for LLAMA, 10 otherwise)",
    )
    parser.add_argument(
        "--load-snapshot", type=str, help="Load a .pcsnap before stepping"
    )
    parser.add_argument(
        "--save-snapshot", type=str, help="Save a .pcsnap after stepping"
    )
    parser.add_argument(
        "--no-perfetto",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable perfetto tracing (default off for llama-fast)",
    )
    parser.add_argument(
        "--perfetto", action="store_true", help="Enable new perfetto tracing"
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="pc-e500.perfetto-trace",
        help="Perfetto trace path",
    )
    parser.add_argument(
        "--no-lcd", action="store_true", help="Skip saving LCD PNGs on exit"
    )
    parser.add_argument(
        "--preset", choices=["llama-fast"], help="Apply llama-fast defaults"
    )
    args = parser.parse_args()

    _apply_llama_fast_defaults(args)
    perfetto_enabled = False
    if args.no_perfetto is not None:
        perfetto_enabled = not bool(args.no_perfetto)
    if args.perfetto:
        perfetto_enabled = True

    run_emulator(
        num_steps=args.steps,
        perfetto_trace=perfetto_enabled,
        new_perfetto=args.perfetto,
        trace_file=args.trace_file,
        timeout_secs=args.timeout_secs,
        fast_mode=args.fast_mode,
        load_snapshot=args.load_snapshot,
        save_snapshot=args.save_snapshot,
        save_lcd=not args.no_lcd,
        print_stats=True,
    )


if __name__ == "__main__":
    main()
