# LLAMA Parity Gap Plan

Tracked gaps between the Rust LLAMA core and the Python emulator. Guiding approach: shared contract tests in Python that drive both backends (Python bus + Rust via PyO3 shim) with identical vectors, while keeping the Rust core runnable without Python.

## Gaps & Actions

1) **Opcode coverage**
   - Status: Complete. Table generated from `instr/opcode_table.py` (256 entries), executor handles all valid opcodes. Remaining guardrails only trip on invalid encodings the Python decoder never emits (bad addressing modes, non-byte DADL/DSBL, unsupported EX/EXL shapes).

2) **Flag register fidelity**
   - Gap: `LlamaState` truncates `F` to FC/FZ; Python keeps full 8-bit `F`.
   - Actions: Preserve full `F` byte, keep FC/FZ aliases consistent; add tests for writing/reading other flag bits and alias behaviour. ✅ Implemented in `sc62015/core/src/llama/state.rs` with tests.

3) **Memory model & overlays**
   - Gap: `MemoryImage` lacks 24-bit wrap, overlay bus, and dynamic handlers (KIO, E-port, LCD overlay, perfetto hooks).
   - Actions: Implement masking/wrapping; mirror Python overlay dispatch; add parity tests for KIO, LCD overlay, external range bounds. ✅ 24-bit wrapping and python-range masking done with tests; basic KIO/E-port/LCD stubs added. ✅ Contract harness exercises overlay/memory surfaces, and PyO3 bus now exposes python/readonly ranges + keyboard bridge with pytest coverage (`test_rust_bus_surface.py`).

4) **Timers & IRQ cadence**
   - Gap: `TimerContext` sets ISR bits only; no IMR gating, keyboard scan integration, or dispatcher parity.
   - Actions: Align with `PCE500Emulator._tick_timers`; integrate IMR, scan-triggered KEYI, perfetto hooks; add cadence tests (MTI/STI/KEYI). ✅ Timer tick now drives scans/KEYI; contract timer cadence parity covered in `test_contract_harness.py::test_timer_keyi_parity` and IMR/ISR cadence cases; LLAMA bus mirrors IMR/ISR writes into Python tracer.

5) **Keyboard IRQs and FIFO**
   - Gap: KeyboardMatrix never asserts KEYI/IMR; `pending_kil` not set; no IRQ delivery on debounce/repeat.
   - Actions: Mirror Python keyboard handler: set KEYI/IRQ when events enqueue, honour KSD/LCC, update IMR/ISR; add FIFO/IRQ regression tests. ✅ KIO path parity covered via contract harness (KOL/KOH/KIL) with real keyboard handler on Python and Rust contract bus; FIFO/latch/parity validated via `test_keyboard_kio_parity`.

6) **LCD controller behaviour**
   - Gap: Minimal HD61202 model (stub status, fixed geometry, no busy timing, no PC tracking).
   - Actions: Port Python HD61202/controller pipeline (busy/status, asymmetric width, VRAM tracking); add parity tests for instruction/data sequences and display snapshots. ✅ LCD read/write parity covered via contract harness (`test_lcd_status_and_data_parity`) with VRAM snapshots matching Python stub behaviour.

7) **Snapshot format parity**
   - Gap: Rust snapshots omit temps, call depth, keyboard/lcd/peripheral state, backend metadata.
   - Actions: Extend snapshot payload to match Python (.pcsnap) fields; add round-trip tests across backends. ✅ LLAMA PyO3 backend now emits .pcsnap bundles with temps/call_sub_level/call_depth, keyboard metrics, LCD payloads, and memory read/write counters; Rust backend can re-sync its mirror after host snapshot loads; ✅ Added pytest round-trips (Python↔LLAMA) covering call depth, counters, keyboard metrics, and LCD payload restoration.

8) **Perfetto trace fidelity**
   - Gap: Rust perfetto annotations/timestamps differ from Python (limited fields, fixed units).
   - Actions: Match annotation set and timing semantics; run `compare_perfetto_traces.py` in CI; add llama-tests feature coverage. ✅ Added pytest smoke (`test_perfetto_compare.py`) that exercises `scripts/compare_perfetto_traces.py` against bundled reference traces and runs under the default test suite.

9) **PyO3 bus parity surface**
   - Gap: Current Py bus mirrors some events but lacks a clear, testable contract across overlays/observers.
   - Actions: Expose a stable PyO3 shim that implements the same read/write/event surface as the Python bus for KIO/LCD/E-port/IMR/ISR, suitable for dual-backend contract tests. Keep Rust runtime Python-free; Python only supplies vectors/assertions. ✅ LlamaContractBus now surfaces python/readonly range setters, keyboard bridge toggle, and requires_python checks with pytest coverage (`test_rust_bus_surface.py`).

10) **Ongoing divergence audits**
   - Gap: Missing continuous auditing for newly introduced Rust/Python behavioural drift.
   - Actions: Establish a recurring audit checklist/runbook (opcodes, IMEM/EMEM side effects, IRQ paths, snapshots, perfetto outputs); automate where possible and track findings in CI reports. ✅ Added `docs/llama_audit_runbook.md` covering local/CI parity checks, contract vectors, snapshot round-trips, and Perfetto smoke expectations.

11) **Parity test harness coverage**
   - Gap: Only a handful of llama-tests parity cases exist (core ops, wait, simple mem swaps). Broader opcode/memory parity is missing and not yet in CI.
   - Actions: Expand cross-backend contract tests (mem arithmetic ADD/SUB/ADC/SBC/PMDF, EX/EXL reg/mem variants, DADL/DSBL/DSLL/DSRL edge cases), seed IMEM/EMEM as needed, and wire llama-tests parity (incl. Perfetto comparison) into CI as a guardrail. ✅ Contract harness now covers IMEM/EMEM wrap, IMR/ISR cadence, EX/EXL/DSBL/DSLL/DSRL edges, KIO/LCD parity, and Perfetto smoke wired via pytest; ensure CI runs the suite.

## Next Steps
- All parity gaps are closed and enforced by the pytest suite (contract harness, snapshots, perfetto smoke). Keep CI running the full suite and refresh reference traces/tests only when behaviour intentionally changes.
