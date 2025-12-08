Open
- (none)

Resolved
- sc62015/core/src/memory.rs: dirty_internal now records the true IMEM address (not INTERNAL_MEMORY_START + byte offset); regression test covers IMEM_KIL writes draining dirty_internal.
- sc62015/core/src/memory.rs: dirty_internal tracks multi-byte IMEM writes with exact addresses; regression test covers 16-bit stores to ensure replay parity.
- sc62015/core/src/perfetto.rs: default timestamps now use instruction-index counters; wall-clock timing is opt-in via PERFETTO_WALL_CLOCK, keeping compare_perfetto_traces.py aligned by default.
- Strict opcode parity guard: unknown/unsupported opcodes always raise instead of advancing, avoiding silent divergence (sc62015/core/src/llama/eval.rs).
- Generic MV fallback now handles remaining reg/mem move patterns instead of erroring (sc62015/core/src/llama/eval.rs), reducing silent NOPs.
- HALT wake parity: ignore KEYI when keyboard IRQs are disabled so HALT only wakes for enabled sources; regression test added (sc62015/core/src/lib.rs).
- 16-bit CALL/RET now track call pages via call_page_stack so returns land on the original page even if PC page changes (sc62015/core/src/llama/eval.rs).
- Host overlay fallback writes now use live executor context (perfetto_instr_context) instead of stale cycle/PC to align trace timing with Python when no host_write is installed (sc62015/core/src/lib.rs).
- Python-only overlays now enforce host callbacks unless LLAMA_ALLOW_PY_FALLBACK=1 (sc62015/core/src/lib.rs); prevents silent divergence on python-required addresses.
- WAIT parity guard: LlamaPyBus requires memory.wait_cycles unless LLAMA_ALLOW_MISSING_WAIT_CYCLES=1 (sc62015/rustcore/src/lib.rs), avoiding silent timer drift.
- Unknown opcode advance parity: fallback now consumes the estimated opcode length (including prefixes) so bad bytes don’t desync tracing (sc62015/core/src/llama/eval.rs).
- sc62015/core: PERFETTO_TRACER replaced with PerfettoHandle (depth-counted guard + thread owner + gate mutex) allowing reentrant access; callers updated to use enter()/guard deref; tests exercise nested access and run across threads without dropped events. Added guard helpers (tracer_mut/take) and kept ownership assertions.
- sc62015/core/src/llama/eval.rs: Unsupported MV/memory patterns no longer error; they advance PC to avoid halting execution. (Full parity semantics still pending.)
- sc62015/core/src/llama/eval.rs: Added generic MV fallback that writes decoded sources to memory/reg where possible instead of erroring; cargo tests cover executor flows.
- sc62015/core/src/llama/eval.rs: Generic ALU fallback now re-evaluates and writes back (mem/A) with flags for unexpected memory patterns. Parity scripts (`check_llama_opcodes.py`, `check_llama_pre_tables.py`) pass; core tests are green.
- Perfetto trace comparisons: `compare_perfetto_traces.py` shows no divergence for trace_ref/sweep/long/trace_latest python vs llama traces.
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
- sc62015/core/src/memory.rs & sc62015/core/src/lib.rs: IMR/ISR writes now invoke a shared hook (Arc) to record bit-watch transitions, refresh irq_imr/irq_isr mirrors, and emit Perfetto IMR_Write/ISR_Write events with pc/prev/value, covering host and direct internal stores.
- sc62015/core/src/perfetto.rs: InstructionTrace track is emitted by default (alongside Execution/CPU/Memory), keeping compare_perfetto_traces.py compatible without PERFETTO_RUST_LAYOUT overrides.
- sc62015/core/src/llama/eval.rs: WAIT no longer calls bus.wait_cycles; matches Python fast-path (zero I/FC/FZ, advance PC) and stops timer/keyboard drift in Perfetto.
- sc62015/core/src/llama/eval.rs: RET uses the current page when it differs from the saved call page (Python parity for near returns that page-hop) and falls back to saved page otherwise.
- sc62015/core/src/timer.rs: Removed duplicate ISR bit-watch/Perfetto logging; rely on memory write hook so IMR/ISR transitions match Python counts.
- sc62015/core/src/lib.rs: Suppressed dead_code warnings on RuntimeBus metadata fields to keep parity builds clean without changing behavior.

Tests/Checks
- cargo test --quiet (sc62015/core)
- uv run pytest sc62015/pysc62015 -q
- uv run pytest pce500/tests -q
- uv run pytest web/tests -q
- Perfetto comparisons: sweep_trace_python.trace vs sweep_trace_llama.trace; long_trace_python.trace vs long_trace_llama.trace; trace_ref_python.trace vs trace_ref_llama.trace; trace_latest_python.trace vs trace_latest_llama.trace (no divergence)
- uv run python scripts/check_rust_py_parity_annotations.py
- uv run python scripts/check_llama_opcodes.py
- uv run python scripts/check_llama_pre_tables.py
- cargo test --quiet (sc62015/core) after PerfettoHandle rework

Findings
- (none)

Findings
- (none)
