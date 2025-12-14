# LLAMA Divergence Audit Runbook

Goal: catch drift between the Rust LLAMA core and the Python emulator early. Run this checklist before releases and when large changes land in bus/peripherals/executor.

## Quick Checks (local)
- `uv sync --extra dev --extra pce500` to ensure tooling is present.
- `FORCE_BINJA_MOCK=1 uv run pytest sc62015/pysc62015` (core/unit coverage).
- `FORCE_BINJA_MOCK=1 uv run pytest pce500/tests` (device-level parity/regressions).
- `uv run pytest sc62015/pysc62015/test_contract_harness.py` (cross-backend contract vectors).
- `uv run pytest pce500/tests/test_snapshot_roundtrip.py` (Python↔LLAMA .pcsnap parity).
- `uv run pytest sc62015/pysc62015/test_perfetto_compare.py` (Perfetto trace smoke).

## Targeted Parity Probes
- **Opcodes/IMEM/EMEM**: run the contract harness with expanded vectors (EX/EXL/DSBL/DSLL/DSRL, IMR/ISR toggles). Add vectors before touching executor/memory.
- **Timers/IRQ cadence**: `uv run pytest sc62015/pysc62015/test_contract_harness.py::test_timer_keyi_parity` and IMR/ISR cadence cases. Confirm KEYI delivery matches Python when enabling IMR.
- **Keyboard/KIO**: verify KOL/KOH/KIL reads/writes via contract vectors; ensure `requires_python` only true when the bridge is disabled. (Rust bus surface: `test_rust_bus_surface.py`.)
- **LCD**: run `test_lcd_status_and_data_parity`; regenerate reference traces if LCD behaviour changes.
- **Snapshots**: save/load .pcsnap in both directions (Python↔LLAMA). If snapshot schema changes, update both serializers and the round-trip tests.
- **Perfetto**: regenerate `trace_ref_python.trace`/`trace_ref_llama.trace` after intentional trace schema changes; rerun `test_perfetto_compare.py`.

## CI Expectations
- Lint/type/tests: `uv run ruff check .`, `uv run pyright sc62015/pysc62015`, `FORCE_BINJA_MOCK=1 uv run pytest --cov` (core + pce500).
- Contract harness/Perfetto smoke wired into CI (ensure `test_contract_harness.py`, `test_perfetto_compare.py`, and snapshot round-trips run in your pipeline).
- Perfetto comparison job should fail on drift; refresh reference traces intentionally and commit updates with rationale.

## When Drift Is Found
- Reproduce with contract vectors or minimal repro script.
- Add a regression test in Python harness (contract vector, snapshot round-trip, or Perfetto compare) before fixing LLAMA.
- Keep Rust/Python behaviour aligned even for quirks; document intentional deviations in `plan.md` and this runbook.
