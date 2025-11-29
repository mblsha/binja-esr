#!/usr/bin/env bash
# Lightweight guardrail: compare recorded Python vs. LLAMA Perfetto traces.
# Exits non-zero on the first divergence.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="$ROOT/scripts/compare_perfetto_traces.py"
PAIRS=(
  "trace_ref_python.trace trace_ref_llama.trace"
  "sweep_trace_python.trace sweep_trace_llama.trace"
  "long_trace_python.trace long_trace_llama.trace"
  "trace_latest_python.trace trace_latest_llama.trace"
)

cd "$ROOT"

for pair in "${PAIRS[@]}"; do
  set -- $pair
  py="$1"
  rs="$2"
  if [[ ! -f "$py" || ! -f "$rs" ]]; then
    echo "warning: missing traces $py or $rs, skipping" >&2
    continue
  fi
  echo "Comparing $py vs $rs..."
  uv run python "$COMPARE" "$py" "$rs" --compare-irq
done

echo "Perfetto parity checks completed."
