# Full-Rust Emulator File Map

Quick map of Rust-side files and how they relate to the SCIL architecture notes in `docs/scil_arch_report.md` (manifest/binder/family, execution, tracing, build).

## SCIL interpreter (emulators/rust_scil)

- `emulators/rust_scil/src/state.rs`: register/flag model and aliasing (ties to register/alias notes and flag shadows FC/FZ).
- `emulators/rust_scil/src/eval.rs`: SCIL evaluator using binder, tmps, PRE, IMEM caches (matches eval and gotchas sections: PRE modes, IMEM caches, tmp invalidation).
- `emulators/rust_scil/src/bus.rs`: `Bus` trait and `Space` enum (Int/Ext/Code) used by eval; aligns with memory/addressing.
- `emulators/rust_scil/src/ast.rs`: SCIL AST node definitions referenced by manifest payload and evaluator.

## Core runtime on top of SCIL (sc62015/core)

- `sc62015/core/src/exec.rs`: HybridBus, opcode lookup, execute_step wrapper around `rust_scil::eval::step`; consumes manifest payload (handlers.rs/opcode_index.rs) and binder.
- `sc62015/core/src/perfetto.rs`: Perfetto tracer for instruction/memory instants (ties to tracing notes).
- `sc62015/core/src/snapshot.rs`: snapshot load/save, range deserialization (constraints section: export variants, readonly ranges).
- `sc62015/core/src/lcd.rs`: LCD device + display buffer export (hardware device layer used by bus; aliasing/constraints around display buffer).
- `sc62015/core/src/lib.rs`: re-exports core types (MemoryImage, PerfettoTracer, execute_step, etc.) and metadata (fast_mode, interrupts); glues core for rustcore.

## Rust bridge / CLI (sc62015/rustcore)

- `sc62015/rustcore/src/lib.rs`: pyo3 bindings exposing the core runtime to Python (`_sc62015_rustcore`), loads generated manifest payload/index, handles binder serialization, exposes execute_step/state access (aligns with manifest/binder/family usage and constraints on FC/FZ).
- `sc62015/rustcore/src/bin/pce500-cli.rs`: pure-Rust CLI harness that loads manifest payload, builds HybridBus, steps instructions, and can emit traces/snapshots (uses build/tracing notes).
- `sc62015/rustcore/generated/{manifest.json,handlers.rs,opcode_index.rs}`: generated payload/index from `tools/scil_codegen_rust.py` (see generation pipeline and size constraints).

## Supporting files and generation

- `tools/scil_codegen_manifest.py`: enumerates decoded variants with templates + PRE, builds manifest entries (manifest/binder/family sections).
- `tools/scil_codegen_rust.py`: writes manifest payload/index consumed by rustcore/core.

## Build/trace touchpoints

- Build rustcore and CLI: `uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml` (pulls in core + generated payload).
- Build SCIL interpreter: `cargo build` in `emulators/rust_scil`.
- Tracing: `sc62015/core/src/perfetto.rs` and dispatcher bridging in `pce500` glue (see tracing section). Environment flags (e.g., `RUST_MEM_TRACE`, `RUST_EXT_TRACE`, `IMEM_TRACE`) map directly to evaluator/device tracing described in the main SCIL report.

## ASCII module chart

```
               +----------------------+
               | tools/scil_codegen_* |
               | (manifest/payload)   |
               +----------+-----------+
                          |
          generated payload (manifest.json /
             handlers.rs / opcode_index.rs)
                          |
          +---------------+-----------------+
          |                                 |
+---------v---------+              +--------v--------+
| emulators/rust_scil|             | sc62015/core    |
| (state, eval, bus) |             | (exec, perfetto,|
|                     |             |  snapshot, lcd) |
+---------+---------+              +--------+--------+
          | eval::step                       | HybridBus/execute_step
          +---------------+------------------+
                          |
                +---------v---------+
                | sc62015/rustcore  |
                | (pyo3 bridge, CLI)|
                +---------+---------+
                          |
             +------------+-------------+
             |                          |
      Python via pyo3             `pce500-cli`
```
