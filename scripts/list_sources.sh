#!/usr/bin/env bash
set -euo pipefail

# List Python, Rust, and Markdown sources, excluding tests and generated outputs.
# Relies on `fd` for fast searching.

if ! command -v fd >/dev/null 2>&1; then
  echo "fd is required but not installed" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

(cd "$ROOT" && fd --color=never --type f \
  --extension py --extension rs --extension md \
  --exclude '*/tests/*' \
  --exclude 'tests' \
  --exclude '*/generated/*' \
  --exclude 'generated' \
  --exclude '*/target/*' \
  --exclude 'target' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'venv') \
  | LC_ALL=C sort
