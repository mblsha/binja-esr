Resolved
- sc62015/core/src/lib.rs: RETI fallback clearing, pending source prioritization (KEY/ONK), pending/perfetto diagnostics, and KEY overrides are aligned with Python; requires_python paths no longer panic and fall back like Python.
- sc62015/core/src/perfetto.rs: record_irq_check accepts Python parity fields; added CPU/Memory track aliases and optional wall-clock timestamps via PERFETTO_WALL_CLOCK for Python trace parity.
- sc62015/rustcore/src/lib.rs: tick_timers respects kb_irq_enabled before asserting KEYI.
- sc62015/core/src/llama/eval.rs & perfetto.rs: Prefixed instructions now annotate PC with the post-PRE value to match Python trace comparisons.
- sc62015/core/src/lib.rs & lcd.rs: IMEM 0x00–0x0F routes to the LCD controller overlay for parity with Python’s internal remap.
- sc62015/core/src/lib.rs: WAIT no longer spins timers and advances one cycle, matching Python fast-path semantics; regression test added.
- sc62015/core/src/lib.rs: RuntimeBus requires_python accesses now fall back instead of panicking, matching Python tolerance.
- sc62015/core/src/keyboard.rs, sc62015/core/src/timer.rs, sc62015/rustcore/src/lib.rs: Host-injected matrix events now honor kb_irq_enabled and the KEYI latch is preserved even while keyboard IRQs are disabled so deferred delivery matches Python.
- sc62015/core/src/memory.rs: Perfetto host writes now use the provided cycle or live executor context only; no more fabricated PC/op_index fallbacks that diverge from Python tracer metadata.
- sc62015/core/src/timer.rs: TimerFired perfetto events drop the stale last-pc fallback; they use live executor context or pc_hint only, keeping timestamps/PCs aligned with Python traces.
- sc62015/core/src/lib.rs: ONK (ISR.ONKI) set/clear now records irq_bit_watch transitions for parity with Python.
- sc62015/core/src/timer.rs: KEYI assertions now emit irq_bit_watch transitions even when KEYI was already set, matching Python ISR bit-watch metadata.
- WAIT timing parity re-verified against Python’s _simulate_wait: Rust keeps cycle/timer advances aligned with I loops and flags clear.
- sc62015/core/src/memory.rs: PERFETTO_TRACER access now spins briefly on contention to reduce dropped IMR/ISR/KIO events without risking deadlock.

Tests/Checks
- cargo test --quiet (sc62015/core)
- uv run pytest sc62015/pysc62015 -q
- uv run pytest pce500/tests -q
- uv run pytest web/tests -q
- Perfetto comparisons: sweep_trace_python.trace vs sweep_trace_llama.trace; long_trace_python.trace vs long_trace_llama.trace; trace_ref_python.trace vs trace_ref_llama.trace; trace_latest_python.trace vs trace_latest_llama.trace (no divergence)
- uv run python scripts/check_rust_py_parity_annotations.py
- uv run python scripts/check_llama_opcodes.py
- uv run python scripts/check_llama_pre_tables.py

Findings
- Functional coverage gaps remain in the LLAMA executor: many operand patterns still return Err("...not supported") or fall back to fallback_unknown (e.g., EMemRegMode constraints, several MOV/ALU combinations) in sc62015/core/src/llama/eval.rs. Those opcodes effectively act as NOPs in Rust while the Python emulator implements full semantics, violating the bug-for-bug parity requirement until the missing paths are filled.
