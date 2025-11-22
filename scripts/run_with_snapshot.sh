#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper around run_pce500.py to exercise snapshot capture/replay.
# Environment variables:
#   SC62015_CPU_BACKEND   Backend selection (rust/python) – defaults to python.
#   FAST_MODE             If set, passes --fast-mode (defaults to 1 for rust).
#   RUST_TIMER_IN_RUST    Forced to 1 for rust backend unless provided.
#   SNAPSHOT_IN           Optional .pcsnap path to load before stepping.
#   SNAPSHOT_OUT          Optional .pcsnap path to capture if SNAPSHOT_IN is absent.
#   SNAPSHOT_SAVE_AFTER   Optional .pcsnap path to save after the replay step.
#   SNAPSHOT_CAPTURE_STEPS  Steps for the capture stage (default: 15000).
#   RUN_STEPS             Steps for the replay stage (default: 500).
#   TIMEOUT_SECS          Override timeout; defaults to 0 for rust, 10 otherwise.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND="${SC62015_CPU_BACKEND:-python}"

# Defaults for Rust: timers in Rust, fast_mode on, no timeout
if [[ "$BACKEND" == "rust" ]]; then
    export RUST_TIMER_IN_RUST="${RUST_TIMER_IN_RUST:-1}"
    FAST_MODE="${FAST_MODE:-1}"
    TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
else
    FAST_MODE="${FAST_MODE:-}"
    TIMEOUT_SECS="${TIMEOUT_SECS:-10}"
fi

base_cmd=(uv run python pce500/run_pce500.py --timeout-secs "$TIMEOUT_SECS" --no-perfetto)
if [[ -n "$FAST_MODE" ]]; then
    base_cmd+=(--fast-mode)
fi

# If we only have SNAPSHOT_OUT, capture first then reuse it for replay.
if [[ -z "${SNAPSHOT_IN:-}" && -n "${SNAPSHOT_OUT:-}" ]]; then
    cap_steps="${SNAPSHOT_CAPTURE_STEPS:-15000}"
    echo "Capturing snapshot to ${SNAPSHOT_OUT} (steps=${cap_steps})..."
    "${base_cmd[@]}" --steps "$cap_steps" --save-snapshot "$SNAPSHOT_OUT"
    SNAPSHOT_IN="$SNAPSHOT_OUT"
fi

run_steps="${RUN_STEPS:-500}"
run_cmd=("${base_cmd[@]}" --steps "$run_steps")
if [[ -n "${SNAPSHOT_IN:-}" ]]; then
    run_cmd+=(--load-snapshot "$SNAPSHOT_IN")
fi
if [[ -n "${SNAPSHOT_SAVE_AFTER:-}" ]]; then
    run_cmd+=(--save-snapshot "$SNAPSHOT_SAVE_AFTER")
fi

echo "Replaying snapshot (steps=${run_steps})..."
FORCE_BINJA_MOCK=1 "${run_cmd[@]}"
