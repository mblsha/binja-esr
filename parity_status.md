Parity status (Rust ↔ Python)
==============================

Scope: LLAMA CPU core, peripherals (keyboard/LCD/timer), snapshots, tracing.

- All parity fixes applied (IRQ pending, WAIT/HALT timing, cycle_count idle accounting, IMR gating behavior, keyboard/LCD bus wiring, IL/IH semantics, perfetto KEY events, snapshot timer rebasing).
- Test coverage:
  - Rust: `cargo test --manifest-path sc62015/core/Cargo.toml`
  - Python core: `uv run pytest sc62015/pysc62015` (410/410)
  - PCE-500: `uv run pytest pce500/tests` (106/106)
  - Web: `uv run pytest web/tests` (20/20)
- Perfetto comparisons: `scripts/verify_perfetto_parity.sh` (reference/sweep/long traces) show no divergence.
- Parity annotations: `uv run python scripts/check_rust_py_parity_annotations.py` passes.

Status: Parity complete and validated; no pending actions. If a new workload needs tracing, use `scripts/verify_perfetto_parity.sh` as a template.
