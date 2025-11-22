# Project LLAMA (LLIL-Free Rust Core)

- Goal: LLAMA is the only Rust CPU core. Remove SCIL payloads/interpreters/codegen and run the emulator stack solely on the direct-execution LLAMA path.
- Scope: reuse the Python opcode shapes as guidance, but re-express them directly in Rust (typed enums/structs, no JSON or SCIL manifest). Keep existing runtime pieces (memory, devices, perfetto tracing, snapshots) and plug LLAMA into them.

## Principles
- No SCIL/LLIL dependencies anywhere in the Rust stack (no manifest generation, no rust_scil crate, no SCIL-backed Python bridge).
- Prefer enums/typed operands over strings for opcode/operand/register representation.
- Keep parity with the Python emulator via opcode sweeps, Perfetto traces, and pytest runs with `--cpu-backend=llama`.
- Stream Perfetto traces directly (binary `retrobus-perfetto`), timestamped by instruction index, with separate threads for instructions and memory writes.

## Current priorities
1) Stabilize the LLAMA-only stack after SCIL removal: ensure core/rustcore build cleanly, snapshots/perfetto hooks still work, and the PyO3 wrapper remains in sync.
2) Keep the Python facade lean: only expose Python + LLAMA backends (SCIL backend removed) and keep the Python emulator as the parity oracle.
3) Preserve parity guardrails: keep the opcode sweep, Perfetto smoke comparisons, and the full pytest matrix running against `--cpu-backend=llama`.
4) Maintain typed tables: continue to mirror Python decode shapes directly in Rust tables using enums for registers/operand modes (avoid string identifiers).

## Integration targets
- Keep `MemoryImage`, buses/devices (LCD/keyboard), snapshots, and Perfetto tracing; LLAMA should drive them without SCIL glue.
- Retain `LlamaCPU` PyO3 surface for the Python facade; remove SCIL-specific exports such as `scil_step_json`.
- Perfetto traces must include instruction events and memory writes (internal/external) so `scripts/compare_perfetto_traces.py` can diff LLAMA vs Python.

## Progress snapshot
- Opcode table: typed enums covering 0x00–0xFF; length estimation embedded in the Rust operands.
- Evaluator: mirrors Python semantics (register aliasing/masking, EMEM reg/IMEM pointer modes, stack ops, flags, intrinsics, PRE handling); parity sweep reports zero mismatches against the Python emulator.
- Tests: LLAMA is the sole native backend; the SCIL path/codegen/tests were removed. Full `sc62015/pysc62015` pytest runs target `--cpu-backend=llama`.
- Tracing: streaming Perfetto writer for instruction and memory threads aligned on instruction-index timestamps; nightly smoke compares LLAMA vs Python for representative opcodes.

## Next steps
- Run/repair the LLAMA pytest/parity matrix after SCIL removal and prune any lingering SCIL-only workflows or scripts.
- Keep LLAMA/parity docs in sync with the new single-core reality.
