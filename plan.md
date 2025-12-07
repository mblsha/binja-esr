Resolved
- sc62015/core/src/lib.rs: RETI fallback clearing, pending source prioritization (KEY/ONK), pending/perfetto diagnostics, and KEY overrides are aligned with Python; requires_python paths no longer panic and fall back like Python.
- sc62015/core/src/perfetto.rs: record_irq_check accepts Python parity fields; added CPU/Memory track aliases and optional wall-clock timestamps via PERFETTO_WALL_CLOCK for Python trace parity.
- sc62015/rustcore/src/lib.rs: tick_timers respects kb_irq_enabled before asserting KEYI.
- sc62015/core/src/llama/eval.rs & perfetto.rs: Prefixed instructions now annotate PC with the post-PRE value to match Python trace comparisons.
- sc62015/core/src/lib.rs & lcd.rs: IMEM 0x00–0x0F routes to the LCD controller overlay for parity with Python’s internal remap.
- sc62015/core/src/lib.rs: WAIT no longer spins timers and advances one cycle, matching Python fast-path semantics; regression test added.
- sc62015/core/src/lib.rs: RuntimeBus requires_python accesses now fall back instead of panicking, matching Python tolerance.

Tests/Checks
- cargo test --quiet (sc62015/core)
- uv run pytest sc62015/pysc62015 -q
- uv run pytest pce500/tests -q
- uv run pytest web/tests -q
- Perfetto comparisons: sweep_trace_python.trace vs sweep_trace_llama.trace; long_trace_python.trace vs long_trace_llama.trace; trace_ref_python.trace vs trace_ref_llama.trace; trace_latest_python.trace vs trace_latest_llama.trace (no divergence)
- uv run python scripts/check_rust_py_parity_annotations.py
- uv run python scripts/check_llama_opcodes.py
- uv run python scripts/check_llama_pre_tables.py
