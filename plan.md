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
   - Actions: Implement masking/wrapping; mirror Python overlay dispatch; add parity tests for KIO, LCD overlay, external range bounds. ✅ 24-bit wrapping and python-range masking done with tests; basic KIO/E-port/LCD stubs added. 🔜 Define a contract-test harness in Python that exercises overlay behaviours against both backends (Python bus vs. Rust bus via PyO3 shim) without requiring Python overlays in the Rust hot path.

4) **Timers & IRQ cadence**
   - Gap: `TimerContext` sets ISR bits only; no IMR gating, keyboard scan integration, or dispatcher parity.
   - Actions: Align with `PCE500Emulator._tick_timers`; integrate IMR, scan-triggered KEYI, perfetto hooks; add cadence tests (MTI/STI/KEYI). ✅ Timer tick now drives keyboard scans and KEYI when events present; further IMR/dispatcher parity still needed. ✅ LLAMA bus now reports IMR/ISR writes into Python `trace_irq_from_rust`; 🔜 Add contract tests that drive timer/IRQ cadence through both backends and assert identical IMR/ISR transitions and KEYI delivery.

5) **Keyboard IRQs and FIFO**
   - Gap: KeyboardMatrix never asserts KEYI/IMR; `pending_kil` not set; no IRQ delivery on debounce/repeat.
   - Actions: Mirror Python keyboard handler: set KEYI/IRQ when events enqueue, honour KSD/LCC, update IMR/ISR; add FIFO/IRQ regression tests. ✅ KEYI assertion on scans/KIO writes, KSD honoured, pending_kil tracked. 🔜 Add cross-backend contract tests for keyboard FIFO/IRQ timing (KSD/LCC, pending_kil, IRQ delivery) using the shared harness.

6) **LCD controller behaviour**
   - Gap: Minimal HD61202 model (stub status, fixed geometry, no busy timing, no PC tracking).
   - Actions: Port Python HD61202/controller pipeline (busy/status, asymmetric width, VRAM tracking); add parity tests for instruction/data sequences and display snapshots. ✅ Reads stubbed to 0xFF to match Python; busy/status/geometry still outstanding. 🔜 Contract tests that issue LCD read/write sequences to both backends and compare status/busy/data and resulting VRAM snapshots.

7) **Snapshot format parity**
   - Gap: Rust snapshots omit temps, call depth, keyboard/lcd/peripheral state, backend metadata.
   - Actions: Extend snapshot payload to match Python (.pcsnap) fields; add round-trip tests across backends.

8) **Perfetto trace fidelity**
   - Gap: Rust perfetto annotations/timestamps differ from Python (limited fields, fixed units).
   - Actions: Match annotation set and timing semantics; run `compare_perfetto_traces.py` in CI; add llama-tests feature coverage. 🔜 Wire LLAMA backend perfetto hooks to the Python comparison harness and add a CI job.

9) **PyO3 bus parity surface**
   - Gap: Current Py bus mirrors some events but lacks a clear, testable contract across overlays/observers.
   - Actions: Expose a stable PyO3 shim that implements the same read/write/event surface as the Python bus for KIO/LCD/E-port/IMR/ISR, suitable for dual-backend contract tests. Keep Rust runtime Python-free; Python only supplies vectors/assertions.

10) **Ongoing divergence audits**
   - Gap: Missing continuous auditing for newly introduced Rust/Python behavioural drift.
   - Actions: Establish a recurring audit checklist/runbook (opcodes, IMEM/EMEM side effects, IRQ paths, snapshots, perfetto outputs); automate where possible and track findings in CI reports.

11) **Parity test harness coverage**
   - Gap: Only a handful of llama-tests parity cases exist (core ops, wait, simple mem swaps). Broader opcode/memory parity is missing and not yet in CI.
   - Actions: Expand cross-backend contract tests (mem arithmetic ADD/SUB/ADC/SBC/PMDF, EX/EXL reg/mem variants, DADL/DSBL/DSLL/DSRL edge cases), seed IMEM/EMEM as needed, and wire llama-tests parity (incl. Perfetto comparison) into CI as a guardrail.

## Next Steps (Option 2: Shared Contract Tests)
- Define the contract-test API in Python (reads/writes/events/VRAM snapshots) and a small PyO3 driver that exposes the Rust bus/peripherals to that API without inserting Python overlays into the Rust hot path.
- Add dual-backend tests for keyboard/KIO, LCD, E-port, IMR/ISR and timer cadence using identical vectors; assert matching traces/state. ✅ Contract harness + tests added.
- Keep Rust runnable standalone; add Rust unit tests for bus/peripherals to mirror the same behaviours the contract harness asserts.
- Wire the contract suite (and Perfetto comparison where applicable) into CI; expand parity vectors for opcodes/memory edge cases once the harness is stable. ✅ Contract harness pytest added to CI. 🔜 Grow vectors for EX/EXL/DSBL/DSLL/DSRL parity and IMR/ISR cadence.
