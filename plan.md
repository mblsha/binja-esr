# LLAMA Parity Gap Plan

Tracked gaps between the Rust LLAMA core and the Python emulator, with TODOs to close parity and add coverage.

## Gaps & Actions

1) **Opcode coverage**
   - Gap: Rust opcode table/evaluator implements only a subset; unknown opcodes error.
   - Actions: Transcribe full `instr/opcode_table.py`; add executor handlers; add per-opcode regression tests and parity harness cases.
   - Tooling: generate Rust opcode table from Python source with `uv run python scripts/generate_llama_opcodes.py` and replace `sc62015/core/src/llama/opcodes.rs` `OPCODES` block with the output (extend evaluator before swapping in generated entries).
   - Generated opcode entries (from Python source of truth) for reference:
```
    OpcodeEntry {
        opcode: 0x00,
        kind: InstrKind::Nop,
        name: "NOP",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
    OpcodeEntry {
        opcode: 0x01,
        kind: InstrKind::RetI,
        name: "RETI",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
    OpcodeEntry {
        opcode: 0x02,
        kind: InstrKind::JpAbs,
        name: "JP_Abs",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Imm(16)],
    },
    OpcodeEntry {
        opcode: 0x03,
        kind: InstrKind::JpAbs,
        name: "JPF",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Imm(20)],
    },
    OpcodeEntry {
        opcode: 0x04,
        kind: InstrKind::Call,
        name: "CALL",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Imm(16)],
    },
    OpcodeEntry {
        opcode: 0x05,
        kind: InstrKind::Call,
        name: "CALLF",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Imm(20)],
    },
    OpcodeEntry {
        opcode: 0x06,
        kind: InstrKind::Ret,
        name: "RET",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
    OpcodeEntry {
        opcode: 0x07,
        kind: InstrKind::RetF,
        name: "RETF",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
    OpcodeEntry {
        opcode: 0x08,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::A, 8), OperandKind::Imm(8)],
    },
    OpcodeEntry {
        opcode: 0x09,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::RegIL, OperandKind::Imm(8)],
    },
    OpcodeEntry {
        opcode: 0x0A,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::BA, 16), OperandKind::Imm(16)],
    },
    OpcodeEntry {
        opcode: 0x0B,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::I, 16), OperandKind::Imm(16)],
    },
    OpcodeEntry {
        opcode: 0x0C,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::X, 24), OperandKind::Imm(20)],
    },
    OpcodeEntry {
        opcode: 0x0D,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::Y, 24), OperandKind::Imm(20)],
    },
    OpcodeEntry {
        opcode: 0x0E,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::U, 24), OperandKind::Imm(20)],
    },
    OpcodeEntry {
        opcode: 0x0F,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::Reg(RegName::S, 24), OperandKind::Imm(20)],
    },
    ...
    OpcodeEntry {
        opcode: 0xFB,
        kind: InstrKind::Mvl,
        name: "MVL",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::EMemImemOffsetDestExtMem],
    },
    OpcodeEntry {
        opcode: 0xFC,
        kind: InstrKind::Dsrl,
        name: "DSRL",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::IMem(8)],
    },
    OpcodeEntry {
        opcode: 0xFD,
        kind: InstrKind::Mv,
        name: "MV",
        cond: None,
        ops_reversed: None,
        operands: &[OperandKind::RegPair(2)],
    },
    OpcodeEntry {
        opcode: 0xFE,
        kind: InstrKind::Ir,
        name: "IR",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
    OpcodeEntry {
        opcode: 0xFF,
        kind: InstrKind::Reset,
        name: "RESET",
        cond: None,
        ops_reversed: None,
        operands: &[],
    },
```
   - Status: Table is complete (256 entries) and all valid opcodes execute; remaining "unsupported" branches are guardrails for invalid encodings that Python never produces (e.g., bad addressing modes, non-byte DADL/DSBL, EX/EXL odd shapes). Focus can shift to parity tests and behaviour, not new opcode handlers.

2) **Flag register fidelity**
   - Gap: `LlamaState` truncates `F` to FC/FZ; Python keeps full 8-bit `F`.
   - Actions: Preserve full `F` byte, keep FC/FZ aliases consistent; add tests for writing/reading other flag bits and alias behaviour. ✅ Implemented in `sc62015/core/src/llama/state.rs` with tests.

3) **Memory model & overlays**
   - Gap: `MemoryImage` lacks 24-bit wrap, overlay bus, and dynamic handlers (KIO, E-port, LCD overlay, perfetto hooks).
   - Actions: Implement masking/ wrapping; mirror Python overlay dispatch; add parity tests for KIO, LCD overlay, external range bounds. ✅ 24-bit wrapping and python-range masking done with tests; basic KIO/E-port/LCD stubs added. 🔜 Expose LLAMA backend through Python overlay hooks (keyboard/LCD/E-port) so parity tests can exercise stubs vs. Python.

4) **Timers & IRQ cadence**
   - Gap: `TimerContext` sets ISR bits only; no IMR gating, keyboard scan integration, or dispatcher parity.
   - Actions: Align with `PCE500Emulator._tick_timers`; integrate IMR, scan-triggered KEYI, perfetto hooks; add cadence tests (MTI/STI/KEYI). ✅ Timer tick now drives keyboard scans and KEYI when events present; further IMR/dispatcher parity still needed. ✅ LLAMA bus now reports IMR/ISR writes into Python `trace_irq_from_rust`; 🔜 emit timer/IRQ entry/exit payloads and add cross-backend cadence tests via Python dispatcher.

5) **Keyboard IRQs and FIFO**
   - Gap: KeyboardMatrix never asserts KEYI/IMR; `pending_kil` not set; no IRQ delivery on debounce/repeat.
   - Actions: Mirror Python keyboard handler: set KEYI/IRQ when events enqueue, honour KSD/LCC, update IMR/ISR; add FIFO/IRQ regression tests. ✅ KEYI assertion on scans/KIO writes, KSD honoured, pending_kil tracked. 🔜 Add cross-backend parity tests once LLAMA backend can be driven through Python keyboard/irq dispatcher.

6) **LCD controller behaviour**
   - Gap: Minimal HD61202 model (stub status, fixed geometry, no busy timing, no PC tracking).
   - Actions: Port Python HD61202/controller pipeline (busy/status, asymmetric width, VRAM tracking); add parity tests for instruction/data sequences and display snapshots. ✅ Reads stubbed to 0xFF to match Python; busy/status/geometry still outstanding. 🔜 Expose LLAMA LCD overlay via Python emulator to add cross-backend read/write parity tests.

7) **Snapshot format parity**
   - Gap: Rust snapshots omit temps, call depth, keyboard/lcd/peripheral state, backend metadata.
   - Actions: Extend snapshot payload to match Python (.pcsnap) fields; add round-trip tests across backends.

8) **Perfetto trace fidelity**
   - Gap: Rust perfetto annotations/timestamps differ from Python (limited fields, fixed units).
   - Actions: Match annotation set and timing semantics; run `compare_perfetto_traces.py` in CI; add llama-tests feature coverage. 🔜 Wire LLAMA backend perfetto hooks to Python comparison harness and add a CI job.

9) **PyO3 bus side effects**
   - Gap: Py bus bypasses Python overlay callbacks and perfetto/observer hooks; KIO logging differs.
   - Actions: Route via Python bus APIs that enforce overlays/observers; align KIO tracing; add integration tests through the Py module.

10) **Ongoing divergence audits**
   - Gap: Missing continuous auditing for newly introduced Rust/Python behavioural drift.
    - Actions: Establish a recurring, thorough audit checklist/runbook to scan for divergence (opcodes, IMEM/EMEM side effects, IRQ paths, snapshots, perfetto outputs); automate where possible and track findings in CI reports.

11) **Parity test harness coverage**
   - Gap: Only a handful of llama-tests parity cases exist (core ops, wait, simple mem swaps). Broader opcode/memory parity is missing and not yet in CI.
   - Actions: Expand cross-backend parity cases (mem arithmetic ADD/SUB/ADC/SBC/PMDF, EX/EXL reg/mem variants, DADL/DSBL/DSLL/DSRL edge cases), seed IMEM/EMEM as needed, and wire llama-tests parity (incl. Perfetto comparison) into CI as a lower-priority guardrail.

## Next Steps
- Prioritize gaps 1, 3, 4, 5 for instruction/IRQ correctness; follow with 2, 6, 7, 8, 9 for fidelity and tooling.
- Add automated parity tests for each gap; wire into CI (nightly smoke + unit parity). Broader parity harness work (item 11) can follow the core correctness items.
- Extract IRQ pending/delivery logic in `pce500/emulator.py` into reusable helper(s) with a test-only hook to force delivery, then add deterministic MTI/KEYI stack/PC delivery tests for Python and LLAMA backends.
