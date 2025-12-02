# LLAMA Parity Gap Plan

Tracked gaps between the Rust LLAMA core and the Python emulator. Guiding approach: shared contract tests in Python that drive both backends (Python bus + Rust via PyO3 shim) with identical vectors, while keeping the Rust core runnable without Python.

## New Parity Regressions To Fix (must add Python↔Rust parity tests for each)

1) **EMemReg side effects in ADCL/SBCL loops** ✅  
   - Finding: `sc62015/core/src/llama/eval.rs:1386-1441` walked dst/src addresses manually and ignored `side_effect` updates from EMemReg modes, so pre/post-inc/dec pointers and reg offsets never wrote back; register-indirect block adds/subs diverged from Python.
   - Fix: Applied pre/post inc/dec side effects across multi-byte iterations; added unit test (`adcl_ememreg_side_effect_updates_pointers`) to cover register deltas. `sc62015/core/src/llama/eval.rs`, tests in same file.

2) **Perfetto fidelity** ✅  
   - Finding: `sc62015/core/src/perfetto.rs:120-141` masked mem-write values to 8 bits; main LLAMA run loop `sc62015/core/src/bin/pce500.rs:1070-1333` emitted only IRQ events.
   - Fix: Mem writes now mask to operand width; standalone runner emits per-instruction register + mem events (with instruction index) when `--perfetto` is enabled. Files: `sc62015/core/src/perfetto.rs`, `sc62015/core/src/bin/pce500.rs`.

3) **Timer cadence drift** ✅  
   - Finding: `sc62015/core/src/timer.rs:27-156` previously advanced cycles internally; runner ticked once per instruction.
   - Fix: `tick_timers` now takes absolute cycle; runner advances cycles explicitly (including WAIT) and ticks timers accordingly. Files: `sc62015/core/src/timer.rs`, `sc62015/core/src/bin/pce500.rs`.

4) **LCD status/data reads** ✅  
   - Finding: `sc62015/core/src/lcd.rs:264-274` stubbed reads to 0xFF.
   - Fix: Implemented ON/status reads and data reads with Y increment; added unit tests for status and data. File: `sc62015/core/src/lcd.rs`, tests in same file.

5) **Keyboard KEYI latch** ✅  
   - Finding: `sc62015/core/src/keyboard.rs:236-311` did not latch/reassert KEYI with pending FIFO after ISR clear.
   - Fix: Added KEYI latch, reassert on FIFO writes/reads when pending; unit test `keyi_reasserts_when_isr_cleared_with_pending_fifo` in `core/tests/keyboard.rs`.

6) **Snapshot call/temps parity** ✅  
   - Finding: `sc62015/core/src/lib.rs:235-316` omitted call state/temps.
   - Fix: Snapshots now persist call_depth, call_sub_level, and TEMP0..TEMP13; unit test `snapshot_roundtrip_preserves_call_and_temps` covers round-trip. File: `sc62015/core/src/lib.rs`.

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

## Manual audit 2025-02-17

- sc62015/core/src/memory.rs:234-256 treats the top 0x100 bytes of external space (0x0FFF00–0x0FFFFF) as an internal-memory alias. The reset/interrupt vector at 0x0FFFFA therefore resolves to the zeroed IMEM buffer instead of ROM. power_on_reset (and IR/RETI) will jump to 0x000000 on Rust while Python reads the ROM vector, so boot/IRQ paths and perfetto traces diverge immediately.
- sc62015/core/src/llama/eval.rs:1992-2001 infers 16‑ vs 20‑bit absolute jumps from decoded.len == 3. With a PRE prefix (or any extra prefix bytes), a 16‑bit JP is misclassified as 20‑bit and consumes garbage for the high byte, sending PC to the wrong page versus Python’s operand‑width driven decode.
- sc62015/core/src/lcd.rs:451-457 inverts pixels: pixel_on returns 1 when the bit is clear even though the comment says “lit pixels”. The Python HD61202 helper treats a set bit as on, so Rust LCD buffers and decoded text are flipped, affecting UI parity and perfetto snapshots.
- Perfetto parity gap: LLAMA execution never emits InstructionTrace/MemoryWrites events or IMR/ISR annotations (no PERFETTO_TRACER hooks in sc62015/core/src/llama/eval.rs; trace hooks are feature-gated test code). Python’s emulator logs every step, so traces from Rust cores will be empty/misaligned even when tracing is enabled.
- sc62015/core/src/timer.rs:31-105 keeps mirror fields (irq_imr, irq_isr, irq_pending) that aren’t updated from the IMEM writes in tick_timers. Snapshots and perfetto IRQ payloads therefore report zeros even when ISR bits were set, diverging from Python’s _tick_timers which samples real IMR/ISR values for IRQ state and tracing.

### Remediations
- Fix the internal-alias logic (drop aliasing or gate it) and re-read vectors from ROM. ✅
- Key in operand-width for JP absolute (don’t rely on total length) and add PRE coverage tests. ✅
- Flip pixel_on to treat set bits as lit and verify against Python display decoding. ✅
- Wire LLAMA execute path to PERFETTO_TRACER (instr/mem/imr/isr), mirroring Python payloads. ✅
- Keep timer snapshot/trace fields in sync with IMEM-backed IMR/ISR to match Python IRQ reporting. ✅
