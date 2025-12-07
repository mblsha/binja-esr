- sc62015/core/src/lib.rs:927-944 — RETI cleanup only clears ISR when timer.irq_source is set. If the source is lost (e.g., host-set ISR, stale snapshot, or irq_source cleared elsewhere), ISR bits stay stuck and timer.interrupt_stack’s saved mask is thrown away. Python’s delivery path always clears using the active source or the pending bit; Rust never falls back to the stored mask.
- sc62015/core/src/lib.rs:551-589 — Pending-source handling stops once irq_source is non-None; a latched KEY/ONK won’t override an older timer source. Python explicitly prefers KEY/ONK when ISR shows those bits (to avoid keyboard starvation). Perfetto also misses the matching “IRQ_PendingArm” instant and pending-src detail, so traces diverge when IMR/ISR combine multiple bits.
- sc62015/core/src/lib.rs:731-745 — Pending-check perfetto emit lacks the extra fields Python records (pending_flag/kil/imr_reg/pending_src). That makes compare_perfetto_traces.py mismatches likely around masked KEYI/IMR=0 diagnostics.
- sc62015/core/src/lib.rs:634-675 — For addresses flagged requires_python, the runtime bus silently falls back to raw memory when no host callback is installed (external overlays included). Python always routes these through host overlays; the silent fallback can hide missing device hooks and diverge state/Perfetto.
- sc62015/rustcore/src/lib.rs:321-352 — LlamaContractBus::tick_timers asserts KEYI on MTI/STI when FIFO has data without checking kb_irq_enabled. Python (and TimerContext) gate KEYI on that flag. With keyboard IRQs disabled, the Rust contract bus will still raise KEYI/irq_pending and emit trace events.

Resolved
- sc62015/core/src/lib.rs: RETI now clears ISR using the saved mask when irq_source is missing; pending source prioritizes KEY/ONK; pending/perfetto diagnostics include pending_src/kil/imr_reg and emit IRQ_PendingArm; requires_python without a host overlay now panics; added tests for KEY overriding timers and requires_python panic.
- sc62015/core/src/perfetto.rs: record_irq_check accepts Python parity fields.
- sc62015/rustcore/src/lib.rs: tick_timers respects kb_irq_enabled before asserting KEYI.

Tests/Checks
- cargo test --quiet (sc62015/core)
- uv run pytest sc62015/pysc62015 -q
- uv run pytest pce500/tests -q
- uv run pytest web/tests -q
- Perfetto comparisons: sweep_trace_python.trace vs sweep_trace_llama.trace; long_trace_python.trace vs long_trace_llama.trace; trace_ref_python.trace vs trace_ref_llama.trace; trace_latest_python.trace vs trace_latest_llama.trace (no divergence)
- uv run python scripts/check_rust_py_parity_annotations.py
- uv run python scripts/check_llama_opcodes.py
- uv run python scripts/check_llama_pre_tables.py
