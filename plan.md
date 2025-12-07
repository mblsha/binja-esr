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

Findings
- Perfetto clock source drifts from Python: sc62015/core/src/perfetto.rs:61-107 defaults to wall-clock timestamps, so traces depend on host scheduling instead of instruction/cycle ticks. Parity tools that diff Rust vs Python traces will see jittered timestamps even when op_index order matches.
- Perfetto context for off-core events is often stale. Host/timer/keyboard writes fall back to the last recorded op_index/PC when no executor context is active: memory.rs:248-277 (apply_host_write_with_cycle), timer.rs:350-452 (KeyIRQ/KeyScan events) and timer.rs:464-495 (TimerFired) all stamp events with perfetto_last_* instead of the actual cycle/op that caused them (e.g., during HALT/WAIT ticks). This diverges from the Python tracer, which aligns these events to the driving cycle/op.
- WAIT accuracy gap: the executor simply clears I/C/Z and advances PC (llama/eval.rs:1767-1777) and the runtime suppresses all timer ticks when opcode==WAIT (core/src/lib.rs:955-979), always incrementing cycles by 1. If Python/hardware treats WAIT as an idle loop that burns I cycles and lets timers/KEYI advance, Rust will under-run timers and produce different perfetto timing.
- TCL is effectively a no-op: it shares the IR path but when entry.kind == Tcl the code only bumps PC (llama/eval.rs:2185-2224), with no stack activity, IMR/ISR masking, or perfetto IRQ markers. Python’s TCL intrinsic (if modeled as an interrupt/trap) will diverge.
- IRQ delivery ignores the IMR master when KEYI/ONKI bits are set: core/src/lib.rs:1217-1288 treats keyboard/on-key as level-triggered and enables delivery even if IMR bit 7 is clear. If Python/hardware holds off IRQ entry until the master is set, Rust will deliver spurious IRQs (and emit IRQ_Enter/Delivered perfetto events) while the CPU should stay masked.
- LCD address window is truncated on the low mirror: lcd.rs:60-96 rejects accesses above 0x200F even though handles() reports the whole 0x2000..0x200F/0xA000..0xAFFF range. Any firmware writes to the broader 0x2000 window (common HD61202 aliasing) will be dropped here but accepted by Python’s controller wrapper.
- IMR read tracing can silently disappear: memory.rs:342-387 uses try_lock when emitting perfetto IMR_Read events; under contention those reads are ignored. Python emits every IMR read, so counts and counters (IMR_ReadZero/NonZero) will diverge.
