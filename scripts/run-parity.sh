#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_PROFILE="${BUILD_PROFILE:-dev}"
if [[ "$BUILD_PROFILE" == "dev" ]]; then
  TARGET_SUBDIR="debug"
else
  TARGET_SUBDIR="$BUILD_PROFILE"
fi
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
PYTHON_DYLIB="${PYTHON_DYLIB:-/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib}"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/sc62015/rustcore/target}"
EXT_DIR="$TARGET_DIR/$TARGET_SUBDIR"
PARITY_TARGET="$REPO_ROOT/sc62015/parity/target/$TARGET_SUBDIR"

mkdir -p "$EXT_DIR" "$PARITY_TARGET"

pushd "$REPO_ROOT/sc62015/rustcore" >/dev/null
if [[ "$BUILD_PROFILE" == "dev" ]]; then
  cargo build --features enable_rust_cpu
else
  cargo build --profile "$BUILD_PROFILE" --features enable_rust_cpu
fi
popd >/dev/null

pushd "$REPO_ROOT/sc62015/parity" >/dev/null
if [[ "$BUILD_PROFILE" == "dev" ]]; then
  PYO3_PYTHON="$PYTHON_BIN" cargo build
else
  PYO3_PYTHON="$PYTHON_BIN" cargo build --profile "$BUILD_PROFILE"
fi
popd >/dev/null

PY_SUFFIX_DEFAULT="$("$PYTHON_BIN" -c 'import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])' 2>/dev/null || printf '')"
SUFFIX="${PYTHON_EXT_SUFFIX:-$PY_SUFFIX_DEFAULT}"
if [[ -z "$SUFFIX" ]]; then
  SUFFIX=".so"
fi

ln -sf "lib_sc62015_rustcore.dylib" "$EXT_DIR/_sc62015_rustcore$SUFFIX"

export PYTHON_DYLIB
export SC62015_PARITY_DYLIB="$PARITY_TARGET/libsc62015_parity.dylib"
export PYTHONPATH="$EXT_DIR:$REPO_ROOT"
export FORCE_BINJA_MOCK="1"

pushd "$REPO_ROOT/sc62015/parity-harness" >/dev/null
if [[ "$BUILD_PROFILE" == "dev" ]]; then
  cargo test -q -- --nocapture
else
  cargo test --profile "$BUILD_PROFILE" -q -- --nocapture
fi
popd >/dev/null

pushd "$REPO_ROOT" >/dev/null
uv run pytest sc62015/parity/tests/test_runtime.py -q
popd >/dev/null
