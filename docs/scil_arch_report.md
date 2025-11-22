# SCIL Architecture & Emulator Notes

## Overview

SCIL is a semantics DSL for the SC62015 CPU. Each instruction has:

- A decoded form (`DecodedInstr`): opcode, mnemonic/family, length, binds (operands), optional PRE latch.
- A bound SCIL AST (`Instr`): sequence of `Stmt`/`Expr` nodes executed by the interpreters.
- Binder defaults (operand expressions) and layout metadata for disassembly/tracing.

## Data Flow (build & runtime)

```
[opcode stream] --> decode_map.decode_with_pre_variants()
                     | (register/pointer selectors, PRE variants)
                     v
              DecodedInstr (mnemonic, binds, length, pre)
                     |
                     | from_decoded.build()
                     v
              Bound Instr + binder defaults (Expr)
                     |
           tools/scil_codegen_rust.py
                     |-- manifest.json        (JSON array of entries)
                     |-- handlers.rs (PAYLOAD: &str of manifest JSON)
                     '-- opcode_index.rs (opcode+PRE -> manifest index)
                     |
 Rust runtime loads PAYLOAD + opcode_index --> lookup ManifestEntry --> eval Instr
```

## Memory & Addressing

- External/code space: 24-bit addresses (0x000000–0xFFFFFF).
- Internal memory (IMEM): 0x100000–0x1000FF, accessed via 8-bit offsets. PRE prefixes remap IMEM addressing modes.
- IMEM modes: `(n)`, `(BP+n)`, `(PX+n)`, `(PY+n)`, `(BP+PX)`, `(BP+PY)` selected via operands; PRE can override with the 4×4 matrix in `pre_modes.py`.
- `resolve_imem_addr` caches per-tmp lookups; must invalidate when tmps change.

## Register Model & Aliases

- 8/16/24-bit registers: A, B, BA (A|B), IL, IH, I (IL|IH), X, Y, U, S, F.
- Flags: C/Z; mirrored into F and also into FC/FZ shadow regs in the Rust state.
- PC is masked to 20 bits (`0x0F_FFFF`).
- Writing BA updates A/B; writing A/B updates BA. Writing IL/IH updates I accordingly.

## Key Components (Rust)

- `emulators/rust_scil/src/state.rs`: register/flag storage with alias handling.
- `emulators/rust_scil/src/eval.rs`: executes SCIL AST over an `Env`:
  - `Env` holds `State`, `Bus`, binder map, tmps, PRE latch, IMEM index, caches.
  - `eval_expr` evaluates `Expr` (const, tmp, reg, mem, ops).
  - `exec_stmt` executes statements (fetch, setreg, store, setflag, if/goto/call, mem swaps, ext/int moves).
  - IMEM last-write cache for read-after-write without reloading bus.
- `bus::Bus` trait: `load/store(space, addr, bits)`. Spaces: Int (IMEM), Ext, Code.
- `pre_modes.rs`: PRE matrix, `needs_pre_variants`, and iterator for all PRE latches.
- `eval::step`: entry point: build `Env` from binder, run semantics list.

## Generation Pipeline

- `tools/scil_codegen_manifest.py`:
  - Builds 121 operand selector templates + 40 cross-width templates for reg-arith opcodes.
  - For each opcode: decode with templates + PRE variants; dedup by (mnemonic, length, family, PRE, binder shape).
  - Emits ~177k entries (every legal reg/pointer/PRE combination).
- `tools/scil_codegen_rust.py`: writes `manifest.json`, `handlers.rs` (PAYLOAD string), `opcode_index.rs`.

### Manifest entry contents

Each entry in `manifest.json` encodes:

- `opcode`: primary opcode byte.
- `mnemonic`: human-readable mnemonic.
- `family`: decoder family tag (drives PRE applicability/semantics grouping).
- `length`: total byte length for this variant.
- `pre`: optional PRE latch (`first`/`second` IMEM addressing modes).
- `instr`: full SCIL AST (`name`, `length`, `semantics` list of statements with nested expressions).
- `binder`: default operand bindings (serialized `Expr` objects) to populate tmps/regs before execution.
- `layout`: operand/disassembly metadata (`key`, `kind`, `meta` with offsets/lengths).
- `bound_repr`: compact pre-bound representation for quick lookup.
- `id`: manifest index (referenced by `opcode_index.rs`).

### Binder (operand binding) model

- Binder is a per-instruction map `name -> Expr` produced at decode time (e.g., `dst`, `src`, `addr`, `ptr`).
- Values are SCIL expressions representing operand sources: register selectors, immediates, displacements, pointer selectors, tmp refs, etc.; whatever the decoder emitted for that operand gets lifted into an `Expr` and stored under its name.
- At runtime, the evaluator clones the binder into the `Env`; `Stmt::Fetch` looks up `dst.name` in the binder, evaluates it, and seeds tmps/regs before executing semantics. Binder itself is immutable during execution; tmps/regs evolve.
- Binder decouples decode from execution: no re-decode is needed—operands come from the serialized binder (in `manifest.json` / `handlers.rs` payload and `bound_repr`), and the same binder is used by Rust and Python interpreters.
- Cache sensitivity: binder-supplied tmps drive IMEM address resolution; when tmps change, related caches (e.g., IMEM addr cache) must be invalidated to avoid stale addresses.
- Parity/testing: binder contents are part of `bound_repr` and passed into `scil_step_json`/Python evaluator; state comparisons must tolerate fields like FC/FZ (flag shadows) that the binder-driven execution can populate in Rust state.

### Family classification

- Family is the decoder/emitter classification string on each decoded variant (`family` in manifest).
- Drives PRE expansion: only families in `pre_modes.IMEM_FAMILIES` get base+16 PRE variants; non-IMEM families do not.
- Reflects operand/behavior shape (e.g., `imem_move`, `imem_const`, `ext_reg`, `imm8`, `rel8`, `loop_move`, `adc_imem`).
- Used at generation time to decide PRE coverage; runtime uses the pre-expanded manifest (family itself is metadata, AST carries semantics).
- Null/missing family means no PRE expansion.

## Gotchas

- PRE explosion: IMEM families get 17 variants (base + 16 PRE modes).
- Internal-offset vs absolute: IMEM loads/stores use 8-bit offsets; addresses ≥0x100000 bypass PRE and map to 0–0xFF.
- Tmp reuse: IMEM cache must be invalidated when tmps mutate (fixed via `Env::set_tmp`).
- Flag shadows: Rust state keeps FC/FZ; Python state ignores them. When comparing states, skip FC/FZ fields.
- Snapshot/export: Python `export_flat_memory` may return 2 or 3 values; Rust bridge now handles both.
- Fallback path: On Rust exec error, bridge restores snapshot and runs Python backend; ensure snapshot restore uses the in-memory struct.
- Large generated artifacts: `sc62015/rustcore/generated/{manifest.json,handlers.rs}` ~425 MB each, `opcode_index.rs` ~20 MB.
- IMEM last-write cache: `eval_expr` consults `imem_last_write` to return just-written values without a bus read.

## Code structure & dependencies

- `sc62015/decoding`: opcode decoders, PRE modes, stream reader; produces `DecodedInstr` + operand layout.
- `sc62015/scil`: SCIL AST (`ast`), builder (`from_decoded`), serde helpers, bound_repr cache.
- `tools/scil_codegen_manifest.py`: runs decoders + SCIL builder to enumerate variants (requires `binja-test-mocks` in PYTHONPATH).
- `tools/scil_codegen_rust.py`: emits `manifest.json`, `handlers.rs` (PAYLOAD string), `opcode_index.rs`.
- `emulators/rust_scil`: Rust SCIL interpreter (state, eval, bus).
- `sc62015/core`: Rust core runtime (HybridBus, Perfetto tracer, snapshots, LCD, keyboard) building on rust_scil.
- `sc62015/rustcore`: pyo3 bridge + CLI (`pce500-cli`), consumes generated payload and core; built with `maturin`.
- `sc62015/pysc62015`: Python bridge (`BridgeCPU`) and emulator; can call into rustcore or pure Python backend.
- Generated artifacts are large and ideally regenerated or moved to LFS rather than pushed.

### Module map (key files)

- `emulators/rust_scil/src/state.rs`: register/flag storage with alias handling.
- `emulators/rust_scil/src/eval.rs`: SCIL evaluator (expr + stmt execution, IMEM caching).
- `emulators/rust_scil/src/bus.rs`: `Bus` trait and space enum (Int/Ext/Code).
- `sc62015/decoding/decode_map.py`: opcode decoders, register/pointer selectors, PRE-aware variants.
- `sc62015/decoding/pre_modes.py`: PRE matrix and applicability (`IMEM_FAMILIES`).
- `sc62015/scil/from_decoded.py`: builds SCIL AST + binder from `DecodedInstr`.
- `sc62015/core/src/exec.rs`: HybridBus, opcode lookup, Rust-side execute_step wrapper.
- `sc62015/core/src/perfetto.rs`: Rust Perfetto tracer (instruction/memory instants).
- `sc62015/core/src/snapshot.rs`: snapshot load/save, range deserialization helpers.
- `sc62015/core/src/lcd.rs`: LCD controller + display buffer export.
- `sc62015/rustcore/src/lib.rs`: pyo3 bindings, manifest loading, bridge glue.
- `sc62015/rustcore/src/bin/pce500-cli.rs`: pure-Rust CLI harness.
- `sc62015/pysc62015/_rust_bridge.py`: Python bridge to rustcore (sync, fallback).
- `sc62015/pysc62015/cpu.py` / `emulator.py`: Python CPU facade and emulator wrapper.
- `tools/scil_codegen_manifest.py` / `tools/scil_codegen_rust.py`: manifest + Rust payload generation.

## Operational constraints & flags

- Address masks: PC is 20 bits; IMEM offsets are 8 bits; IMEM base 0x100000.
- Aliases: BA<->A/B, I<->IL/IH; flags C/Z mirrored into F and FC/FZ (Rust state).
- PRE applies only to families in `IMEM_FAMILIES`.
- Env caches: IMEM addr cache keyed by tmp name (invalidate on tmp writes); IMEM last-write cache short-circuits reads after writes.
- Parity/testing helpers: `FORCE_BINJA_MOCK=1` for Binary Ninja mocks; tracing flags (`RUST_MEM_TRACE`, `RUST_EXT_TRACE`, `IMEM_TRACE`, `LCD_LOOP_TRACE`); parity tests compare Rust/Python states with FC/FZ ignored on Python side.

## Build & regeneration notes

- Python deps: `uv sync --extra dev --extra pce500 --extra web` (needs `binja-test-mocks` for manifest generation).
- Regenerate manifest/payload: `python tools/scil_codegen_rust.py --out-dir sc62015/rustcore/generated` (PYTHONPATH must include `binja-test-mocks`), yields `manifest.json`, `handlers.rs`, `opcode_index.rs`.
- Build Rust SCIL interpreter: `cargo build` / `cargo test` in `emulators/rust_scil`.
- Build rustcore (pyo3 + CLI): `uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml` (Python 3.11+, pyo3 0.22).
- Core/CLI relies on generated payload; `sc62015/rustcore/build.rs` will auto-run `tools/scil_codegen_rust.py` if payload is missing (requires `binja-test-mocks` on PYTHONPATH). If trimming size is needed, regenerate with fewer templates/PRE families or move artifacts to LFS.

## Tracing, snapshots, and perf

- Tracing: legacy `trace_dispatcher` (instants/counters) + `PerfettoTracer` (core/perfetto.rs) for instruction/memory events; new tracer bridges dispatcher events into Perfetto.
- Env flags: `RUST_MEM_TRACE`, `RUST_EXT_TRACE`, `IMEM_TRACE`, `LCD_LOOP_TRACE`, `RUST_PC_TRACE`, `RUST_INTERNAL_WRITE_TRACE` toggle diagnostics.
- Snapshots: Python `export_flat_memory` returns 2 or 3 values (memory + ranges [+ readonly]); Rust bridge tolerates both. Register snapshots include aliases; FC/FZ present only in Rust state. LCD payload optional.
- CLI: `pce500-cli` (rustcore) can load/save snapshots, step, and emit trace JSON (convertible to Perfetto via scripts if needed).

## Minimal Usage Snippets

_Execute one instruction with Rust SCIL evaluator (conceptual)_

```rust
use rust_scil::{ast::Instr, eval, state::State, bus::Bus};

let mut state = State::default();
let mut bus = MyBusImpl::new(); // implements Bus
let binder = /* from manifest entry */ std::collections::HashMap::new();
let instr: Instr = /* from manifest entry */;

eval::step(&mut state, &mut bus, &instr, &binder, None)?;
```

_Apply a PRE mode override_

```rust
use rust_scil::ast::PreLatch;
let pre = Some(PreLatch { first: "(PX+n)".into(), second: "(BP+n)".into() });
eval::step(&mut state, &mut bus, &instr, &binder, pre)?;
```

_Python bridge fallback handling (fixed)_

```python
snapshot = bridge.snapshot_registers()
try:
    opcode, length = bridge._runtime.execute_instruction()
except Exception:
    bridge.restore_snapshot(snapshot)
    opcode, length = bridge._execute_via_fallback(addr)
```

## ASCII Relationship Diagram

```
               +----------------------+
               |  opcode stream       |
               +----------+-----------+
                          |
                  decode_map.decode_with_pre_variants
                          |
                +---------v---------+
                |   DecodedInstr    |
                | (mnemonic, binds, |
                |  family, pre)     |
                +---------+---------+
                          |
                    from_decoded.build
                          |
                +---------v---------+
                | Bound Instr (AST) |
                | + binder defaults |
                +---------+---------+
                          |
             tools/scil_codegen_rust.py
        (manifest.json / handlers.rs / opcode_index.rs)
                          |
         +----------------+----------------+
         |                                 |
 +-------v-------+                 +-------v-------+
 | Rust runtime  |                 | Python runtime|
 | (loads PAYLOAD|                 | (same SCIL AST|
 |  + index)     |                 |  interpreter) |
 +---+-----------+                 +---+-----------+
     |                                   |
     | eval::step                        | execute_build
     v                                   v
 +---+---------------------+    +--------+-----------+
 | Env(State, Bus, PRE,    |    | CPUState, Memory   |
 |  binder, tmps, caches)  |    | Adapter            |
 +------------+------------+    +--------------------+
              | Bus.load/store            | memory read/write
              v                           v
         +----+----+                 +----+----+
         | Memory  |                 | Memory  |
         +---------+                 +---------+
```

## Intermediary Artifacts

- `DecodedInstr` → `BoundInstrRepr` (cached JSON form used in manifest).
- `LayoutEntry` (operand display metadata) captured during decode templates.
- PRE latch (`PreLatch(first, second)`) attached when IMEM family needs it.
- Binder map: name → `Expr` (e.g., register selectors, immediates).

## Rust Emulator Integration

- `sc62015/rustcore/src/lib.rs`:
  - Loads manifest payload and opcode index.
  - `OpcodeLookup` maps `(opcode, pre)` to manifest entry.
  - `execute_step` builds binder, sets up `HybridBus`, and calls `eval::step`.
  - Exposes Python bindings via pyo3 (`_sc62015_rustcore`), used by `BridgeCPU`.
- `HybridBus` bridges Rust memory image with host (Python) memory and devices (LCD, keyboard), and optional Perfetto tracer.
- `PerfettoTracer` (core/perfetto.rs) emits instruction/memory instant events for trace comparison.

## Testing & Parity

- Parity tests compare Rust vs Python execution (`test_rust_execution_matches_pyemu`), with state dicts normalized (skip FC/FZ).
- IMEM cache invalidation and fallback snapshot restore fixes keep parity green.
