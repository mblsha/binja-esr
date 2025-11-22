# Project LLAMA (LLIL-Accelerated Micro-Architecture)

- Goal: build a second Rust CPU core that mirrors the Python LLIL lifter semantics but executes directly (no LLIL graph or SCIL payload).
- Scope: decode path remains shared; execution swaps “lift LLIL + evaluate” for a Rust interpreter of the Python opcode behaviours, and should be drop-in swappable within the existing Rust emulator stack (reuse buses, devices, snapshots, tracing).
- Phases (mimic Python decode/eval as closely as possible):
  1) Map Python LLIL behaviours to a direct evaluator (operand helpers, masking, call depth).
  2) Design Rust opcode dispatch table aligned with `instr/opcode_table.py` and Python decode shapes.
  3) Implement direct evaluator (register/memory helpers, intrinsics, call-depth tracking) that plugs into current Rust buses/devices/snapshots/tracing.
  4) Parity harness against Python emulator across existing opcode suites.
  5) Perf validation and integration into `CPU` facade as selectable backend.

## Integration targets (reuse from current Rust emulator)
- Keep `MemoryImage`, `HybridBus`, `PerfettoTracer`, LCD/keyboard devices, snapshots (`snapshot.rs`), and CLI glue intact; LLAMA core should slot into the same execution surface used by rustcore.
- Reuse the `CPU` Python facade and selection logic; expose LLAMA as another backend option without breaking the SCIL path.
- Preserve tracing hooks (PC/memory instants, runtime profiling) so downstream tooling stays compatible.

## Decode/eval parity checklist
- Decode parity: align opcode shapes with Python `instr/opcode_table.py` and operand helpers; match length computation and addressing interpretations.
- Eval parity: mirror register masking/aliasing (BA<->A/B, I<->IL/IH, PC mask), flag semantics, call depth accounting, WAIT/HALT/OFF/RESET intrinsics, and temp register behaviours.
- Memory semantics: honour `INTERNAL_MEMORY_START` rules and keyboard/LCD side-effects through the existing bus.
- Error/edge handling: match Python fallback behaviours (placeholder decode, PC advance rules) where possible.

## Near-term actions
- **First priority:** implement LLAMA evaluator coverage to pass a full opcode parity sweep (Python vs LLAMA) using `tools/llama_parity_sweep.py` and Perfetto parity traces. Target: reduce “not implemented/mv pattern not supported/unknown opcode” buckets to zero across the opcode generator encodings.
- Also P0: run the full `sc62015/pysc62015` pytest suite with `--cpu-backend=llama` (no SCIL/LLIL) and close gaps so LLAMA can pass the same tests as the Python core.
- Expose LLAMA as a selectable backend in the Python `CPU` facade/`BridgeCPU` so the existing pysc62015/test_* suite can drive it (reuse `MemoryAdapter`, PC/flag masking, backend selection), then run the full pytest matrix against LLAMA. **Status:** `llama` backend exported from rustcore as `LlamaCPU` and recognised by the Python `CPU` facade; uses a Python-memory-backed bus and typed register/flag proxy. Executor coverage is still partial, so tests will surface gaps.
- Read the Python opcode/operand definitions directly (no JSON generation) and rewrite them into a Rust dispatch table mirroring `instr/opcode_table.py` and `instructions.py` semantics. **Status: completed through 0xFF.**
- Sketch the Rust module layout for LLAMA (e.g., `sc62015/core/llama/` with dispatch + evaluator) and how rustcore selects it. **Status: scaffolding in place (`dispatch`, `opcodes`, `state`, `eval`, `operands`, `parity`).**
- Draft parity test plan that reuses existing Python emulator tests as oracles. **Status: see `docs/llama_parity_plan.md`.**
- Write down design notes in `docs/llama_design.md` (layout, interfaces to reuse, parity plan).
- Prefer enums/typed variants over stringly identifiers when transcribing (register names, operand modes, opcode kinds).
- Scaffold evaluator/state modules to consume the typed tables, including masking/aliasing and basic execute entrypoints. **Status: initial evaluator/state implemented; reg+imm/IMEM/EMEM (simple) ops, Z/C flags, PC advance, HALT/OFF/RESET handling, WAIT clears I/flags. Next: EMEM post-inc/pre-dec modes, fuller flag semantics (ADC/SBC across widths), intrinsics parity.**
- Parity harness scaffold: `sc62015/core/src/llama/parity.rs` is stubbed behind the `llama-tests` feature; wire to Python oracle next.
- Build/test facilitation: `LLAMA_SKIP_SCIL=1` in `sc62015/rustcore/build.rs` writes dummy generated files to bypass SCIL codegen (rustcore tests still need stub alignment).
- When mirroring Python structures, prefer concrete enums/structs over string identifiers (e.g., register enums, operand-mode enums) and avoid JSON indirection—read the Python definitions and rewrite them in Rust directly. **Status: opcode table uses enums; LLAMA_SKIP_SCIL dummy types now match manifest structures so rustcore builds in LLAMA-only mode without missing fields.**
- External-memory modes: post-inc/pre-dec/offset EMEM reg modes now decoded and applied with register side-effects, matching Python pointer semantics, without any SCIL/LLIL plumbing. Unit tests cover post-inc/pre-dec/offset paths.
- EMEM-from-IMEM helpers: `[(n)]` pointer modes decoded (simple/pos/neg), MV A,[(n)] executes, and `MV (m),[(n)]` / `MV [(n)],(m)` copy data between external pointers stored in IMEM and internal bytes with correct PC advance; still SCIL-free.
- Reg+IMEM offset ops: 0xE0–0xE3/0xE8–0xEA now decoded as direct transfers between IMEM bytes and EMEM reg pointers (simple/post-inc/pre-dec/offset) with register side-effects preserved. Widths derived from the opcode kind (8/16/24).
- Parity diffing scaffold: `llama::parity` now snapshots registers/flags and mem writes and can diff two snapshots; ready to plug a Python oracle without adding SCIL/LLIL dependencies.
- Python oracle hook (feature-gated): `tools/llama_parity_runner.py` executes a single instruction via the Python emulator and emits a JSON snapshot; `llama::parity::run_python_oracle` can spawn it (with `FORCE_BINJA_MOCK=1`) to drive parity without SCIL/LLIL.
- Perfetto trace writer (streaming) aligns InstructionTrace/MemoryWrites on instruction-index timestamps and annotates memory space (internal/external) via enums; a feature-gated LLAMA step runner can emit trace events + snapshots for parity.
- Parity harness helpers: `run_parity_once` executes LLAMA+Python for a single opcode, writes paired Perfetto traces, and exposes compare_perfetto_traces.py invocation; ready to build a multi-opcode parity sweep once Python deps are available.
- Python side now emits Perfetto traces (when `retrobus-perfetto` is installed via `uv`) with space-tagged MemoryWrites; repo root is auto-added to `sys.path`.
- Integration test scaffold (feature-gated `llama-tests`) validates LLAMA vs Python parity for NOP and compares traces via `scripts/compare_perfetto_traces.py`.

## Current progress snapshot (updated)
- Opcode table: typed enums for regs/operands, full 0x00–0xFF transcription (no stringly identifiers).
- State: masking/aliasing BA↔A/B, I↔IL/IH; PC helpers; halt/reset flags.
- Evaluator: reg+imm and IMEM/EMEM MV/ADD/SUB/AND/OR/XOR/ADC/SBC with Z/C updates; EMEM reg indirect supports simple/offset/post-inc/pre-dec (with register updates); IMEM-stored external pointers decoded/executed for MV A,[(n)] and MV between internal↔external via [(n)]; RegIMemOffset opcodes transfer between IMEM bytes and EMEM reg pointers; WAIT clears I/flags; OFF/HALT set halted; RESET clears state/PC; PRE advances 2 bytes; EX (mem/mem/regpair/A,B) handled; RETI writes IMR back to IMEM.
- Parity sweep status: LLAMA now matches the Python backend across the full opcode sweep (0 mismatches). Use `FORCE_BINJA_MOCK=1 uv run python tools/llama_parity_sweep.py` to regress-check future changes.
- Tests: LLAMA unit tests under `sc62015/core` pass; backend smoke coverage in `test_llama_backend_smoke.py`/`test_cpu_backend_parity.py` passes; full `pysc62015` suite passes with `--cpu-backend=llama`. Use `PYTHONPATH=. FORCE_BINJA_MOCK=1 uv run pytest --cpu-backend=llama sc62015/pysc62015` to exercise LLAMA across the Python matrix.
- Parity tooling: `tools/llama_parity_sweep.py` iterates `opcode_generator()` encodings and compares Python vs LLAMA; `llama::parity` snapshot helpers; streaming Perfetto writer (InstructionTrace + MemoryWrites) with instruction-index timestamps and enums for spaces; `scripts/compare_perfetto_traces.py` expects binary retrobus-perfetto protobufs. Memory writes (internal/external) must be emitted; traces should be written incrementally (no event hoarding).
- Build tooling: `LLAMA_SKIP_SCIL=1` dummy `generated/types.rs` keeps rustcore building without SCIL; opcode/mnemonic/layout fields match rustcore traits. Rust extension exports `LlamaCPU`; Python `CPU` facade accepts `backend=\"llama\"` with typed register/flag proxy on a Python-backed bus.
- CI guardrails: opcode sweep + LLAMA pytest run in the main test workflow; nightly Perfetto smoke compares NOP, CALL, EMEM MV, DSBL, EMEM reg-indirect, and PUSHU traces between Python and LLAMA.

See `AGENTS.md` for repository guidelines and dev commands.
