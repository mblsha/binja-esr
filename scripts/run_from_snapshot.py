#!/usr/bin/env python3
"""Replay a PC-E500 snapshot and optionally emit a Perfetto trace."""

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
    parser = argparse.ArgumentParser(description="Replay a .pcsnap bundle")
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to a .pcsnap file captured via run_pce500.py or capture_snapshot.py",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of instructions to execute from the snapshot (default: 100)",
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
        help="Enable new Perfetto tracing during replay",
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
        default="snapshot-run.perfetto-trace",
        help="Perfetto trace path (only used when --perfetto is set)",
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        help="Optionally save the post-run snapshot to this path",
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
        # New tracer implies legacy Perfetto stays off to avoid double overhead.
        perfetto_trace_enabled = False

    run_emulator(
        num_steps=args.steps,
        perfetto_trace=perfetto_trace_enabled,
        new_perfetto=args.perfetto,
        trace_file=args.trace_file,
        timeout_secs=args.timeout_secs,
        fast_mode=args.fast_mode,
        print_stats=not args.quiet,
        load_snapshot=args.snapshot,
        save_snapshot=args.save_snapshot,
        # Snapshot replays should not waste time saving LCD PNGs unless asked
        save_lcd=not args.quiet,
    ).close()


if __name__ == "__main__":
    main()
