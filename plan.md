Open
- (none)

Resolved
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
- Functional coverage gaps remain in the LLAMA executor: many operand patterns still return Err("...not supported") or fall back to fallback_unknown (e.g., EMemRegMode constraints, several MOV/ALU combinations) in sc62015/core/src/llama/eval.rs. Those opcodes effectively act as NOPs in Rust while the Python emulator implements full semantics, violating the bug-for-bug parity requirement until the missing paths are filled.
- Potential IRQ source priority inversion: sc62015/core/src/lib.rs:1412-1420 delivers ONK before KEY when both ISR bits are set and IMR enables them, whereas the Python path chooses KEY first (pce500/emulator.py:760-879, pending_src order KEY→ONK). If firmware leaves both bits high, LLAMA will jump to the ONK vector while Python goes to the KEY vector, and RETI will clear a different mask bit.
- Keyboard IRQ latching while disabled: sc62015/core/src/timer.rs:480-556 latches key_irq_latched and can assert KEYI/irq_pending even when kb_irq_enabled is false (see refresh_key_irq_latch behaviour in sc62015/core/src/lib.rs:561-629), because latch_active ignores the enable bit. Python only latches/sets KEYI when _kb_irq_enabled is true (pce500/emulator.py:2334-2534, _tick_timers), so disabling keyboard IRQs in Python suppresses KEYI while the Rust core can still raise it.
- Perfetto timeline mismatch in mixed backends: the Rust tracer always runs in instruction-index manual-clock mode and emits extra tracks (InstructionTrace, MemoryWrites, IMR, etc.; sc62015/core/src/perfetto.rs:45-87). Python’s default/legacy tracer uses wall-clock time unless the “new” manual-clock tracing is enabled (pce500/emulator.py:340-379). If users run the legacy Python tracer, Rust traces won’t align temporally or by track naming, which breaks “identical logging” expectations.
- Host-overlay write timing is stamped with a stale cycle/PC: RuntimeBus::store passes self.cycle/self.pc captured before the instruction loop into apply_host_write_with_cycle (sc62015/core/src/lib.rs:853-863). After a long WAIT/halt idle stretch, the emitted Perfetto memory events still carry the pre-idle cycle/PC, whereas Python records them with the live cycle_count/PC at the time of the write. This can desync trace comparisons around host-handled IMEM/overlay updates.
