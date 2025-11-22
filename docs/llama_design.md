# Project LLAMA: Design Notes

Goal: Add a direct-execution Rust CPU core that mirrors the Python LLIL lifter semantics without building LLIL graphs or using the SCIL manifest. It must be drop-in swappable in the existing Rust runtime (core + rustcore) while keeping devices, memory, tracing, and snapshots intact.

## Interfaces to keep untouched
- `MemoryImage`, `HybridBus`, LCD/keyboard/timer devices, `PerfettoTracer`, and snapshot helpers in `sc62015/core`.
- Python facade: `sc62015.pysc62015.CPU` should see LLAMA as another backend option; rustcore FFI surface stays stable.
- Tracing/profile hooks: PC/memory instants and runtime profiling flags should keep working.

## Proposed Rust layout
- `sc62015/core/src/llama/mod.rs`: entry point + runtime struct (state, bus handle, profiler, tracing).
- `sc62015/core/src/llama/opcodes.rs`: opcode table transcribed from Python `instr/opcode_table.py` (mnemonic, operand kinds, length calc hints, call-depth deltas, intrinsic tags).
- `sc62015/core/src/llama/dispatch.rs`: lookup helpers using the transcribed opcode table.
- `sc62015/core/src/llama/eval.rs`: direct evaluator for opcode behaviours (register masking/aliasing, flags, memory ops, call depth, WAIT/HALT/OFF/RESET handling).
- `sc62015/core/src/llama/operands.rs`: helpers for operand decoding (imm8/16/20, IMEM/EMEM addressing, reg selectors).
- `sc62015/core/src/llama/state.rs`: register file with BA<->A/B and I<->IL/IH aliasing, PC mask, FC/FZ shadows aligned with current Rust state.
- `sc62015/core/src/llama/tests/`: parity harness against Python emulator (round-trip decode + execute).
- Opcode mapping: read the Python opcode/operand definitions directly (`instr/opcode_table.py`, `instructions.py`, `opcodes.py`) and rewrite them into Rust dispatch entries; avoid intermediate JSON if possible.
- Parity: see `docs/llama_parity_plan.md` for the oracle-based comparison harness against the Python emulator.

## Data sources for opcode behaviour
- Python tables: `sc62015/pysc62015/instr/opcode_table.py` (dispatch map), `instructions.py` (semantic classes), `opcodes.py` (mode helpers), `intrinsics.py` (HALT/OFF/RESET/TCL).
- Behavioural contract: `docs/sc62015_python_emulator_surface.md` (masking, call depth, WAIT fast path, flags).

## Execution model (LLIL-less)
- Decode: reuse Python opcode shapes (lengths, operands) to feed a Rust dispatch table; no SCIL manifest.
- Eval: for each opcode variant, execute effects directly on Rust state + bus:
  - Apply register/memory reads/writes with masking; keep call-sub-level accounting.
  - Implement intrinsics (HALT/OFF/RESET/TCL) inline.
  - Handle IMEM/EMEM addressing and keyboard/LCD side effects through `HybridBus`.
- PC advance rules mirror Python: use decoded length unless control flow overrides; WAIT fast-path.

## Parity & testing
- Golden oracle: Python emulator. Strategy:
  - Run opcode subsets through Python decode+execute; compare registers/flags/memory deltas to LLAMA results.
  - Reuse existing pytest cases where possible via FFI harness or JSON interchange.
  - Add targeted tests for intrinsics, call-depth tracking, and IMEM/EMEM addressing edge cases.

## Migration plan
- Stage 1: build dispatch schema from Python tables (scriptable extraction to avoid drift).
- Stage 2: implement state/operands/eval modules and wire into `HybridBus`.
- Stage 3: parity harness + CI gating; expose LLAMA backend in rustcore FFI and Python facade.
- Stage 4: perf profiling and optional selection flags.

See `plan_llama.md` for project phases and `AGENTS.md` for repo workflow.
