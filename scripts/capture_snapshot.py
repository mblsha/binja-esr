#!/usr/bin/env python3
"""Capture a PC-E500 emulator snapshot for fast replay/debug loops."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running without PYTHONPATH set
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pce500.run_pce500 import run_emulator


def _apply_preset(args: argparse.Namespace) -> None:
    """Apply optional presets (currently, a Rust-fast profile)."""

    if args.preset == "rust-fast":
        os.environ.setdefault("SC62015_CPU_BACKEND", "rust")
        os.environ.setdefault("RUST_TIMER_IN_RUST", "1")
        if args.fast_mode is None:
            args.fast_mode = True
        if args.timeout_secs is None:
            args.timeout_secs = 0.0
        if args.no_perfetto is None:
            args.no_perfetto = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a .pcsnap bundle")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the snapshot (.pcsnap)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15000,
        help="Instructions to execute before capturing the snapshot (default: 15000)",
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the emulator fast path (default: on for Rust preset/backend)",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=None,
        help="Abort after this many seconds (default: 0 for Rust preset/backend, 10 otherwise)",
    )
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Enable new Perfetto tracing during capture",
    )
    parser.add_argument(
        "--no-perfetto",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable legacy perfetto tracing (default: off; rust-fast preset also forces off unless overridden)",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="pc-e500.perfetto-trace",
        help="Perfetto trace path (only used when --perfetto is set)",
    )
    parser.add_argument(
        "--preset",
        choices=["rust-fast"],
        help="Convenience preset (rust-fast sets backend=rust, timers in Rust, fast_mode on, timeout 0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence run statistics",
    )
    args = parser.parse_args()

    _apply_preset(args)
    perfetto_trace_enabled = False
    if args.no_perfetto is not None:
        perfetto_trace_enabled = not bool(args.no_perfetto)
    if args.perfetto:
        perfetto_trace_enabled = True

    run_emulator(
        num_steps=args.steps,
        perfetto_trace=perfetto_trace_enabled,
        new_perfetto=args.perfetto,
        trace_file=args.trace_file,
        timeout_secs=args.timeout_secs,
        fast_mode=args.fast_mode,
        print_stats=not args.quiet,
        save_snapshot=args.output,
        # snapshots are for speed; avoid extra overhead unless requested
        save_lcd=not args.quiet,
    )


if __name__ == "__main__":
    main()
