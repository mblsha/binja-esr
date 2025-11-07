# SC62015 Rust Core Scaffold

This repository now includes the scaffolding needed to build an optional Rust implementation of the SC62015 emulator. The backend is **disabled by default**; enabling it requires building the extension with the `enable_rust_cpu` Cargo feature so the Python selector can detect and route traffic to the native runtime.

## Layout
- `sc62015/rustcore/`: Cargo crate that hosts the Rust CPU implementation.
  - `Cargo.toml`: defines a `cdylib` target named `sc62015_rustcore` built with PyO3 (`abi3-py311`).
  - `build.rs`: invokes `sc62015/tools/generate_opcode_metadata.py` to produce opcode metadata plus a structured LLIL program for every opcode. It emits Rust source under `OUT_DIR/opcode_table.rs`, falling back to an empty table if generation fails.
  - `src/constants.rs`: architectural constants mirrored from the Python implementation (address space, PC mask, TEMP register count).
  - `src/state.rs`: register file implementation matching Python semantics, including composite registers, flag bits, TEMP slots, and call-stack tracking.
  - `src/memory.rs`: PyO3 wrapper around the Python memory helper, exposing byte/word accessors and optional performance-tracer hooks.
  - `src/decode.rs`: metadata lookup helpers that classify opcodes (instruction vs prefix) and surface the LLIL program for the Rust code generator.
  - `src/executor.rs`: runtime scaffolding (`LlilRuntime`, error types) that executes generated opcode handlers; currently a stub until the LLIL lowering is implemented.
  - `src/lib.rs`: exposes the stub `CPU` class, generated opcode metadata/LLIL programs, handler dispatch table, and re-exports the supporting modules.
  - `tests/python_parity.rs`: skeleton proptest harness that will cross-check instruction execution once the Rust backend is feature-complete. It currently exits early when the Rust core reports `HAS_CPU_IMPLEMENTATION = False`.
- Python shim:
  - `sc62015/pysc62015/cpu.py`: provides a `CPU` facade that selects between the existing Python `Emulator` and the optional Rust backend. Selection obeys the `SC62015_CPU_BACKEND` environment variable or an explicit `backend=` argument. When Rust is requested but not available, a descriptive error explains how to build the extension.
  - `sc62015/pysc62015/__init__.py`: re-exports `CPU`, backend helpers, and the existing `Emulator`.

## Building the extension
The Rust backend is not compiled automatically. To build it locally (for experimentation or to start filling in the implementation) with the executable CPU enabled:

```bash
uv tool install maturin  # once per environment
uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml --features enable_rust_cpu
```

This command produces a Python extension module named `_sc62015_rustcore` that can be imported from `sc62015.pysc62015`. After building, set `SC62015_CPU_BACKEND=rust` (or pass `backend="rust"` to the `CPU` constructor) to route instantiations through the native module.

Because the module is gated behind the `enable_rust_cpu` feature flag, any attempt to instantiate the Rust `CPU` while running a default build continues to raise `NotImplementedError`. The shim reports availability only when `HAS_CPU_IMPLEMENTATION` is `True`, which happens automatically once you build with the feature enabled.

### Regenerating opcode metadata

The build script automatically runs `sc62015/tools/generate_opcode_metadata.py` (using `FORCE_BINJA_MOCK=1`) to collect LLIL for each primary opcode. You can execute the generator manually for inspection:

```bash
uv run python sc62015/tools/generate_opcode_metadata.py --pretty
```

No JSON artifacts are committed; the output is produced on demand during each build.

### Comparing backends locally

Use `sc62015/tools/compare_cpu_backends.py` to execute individual opcodes through both backends. The tool skips cleanly when the Rust core is unavailable. For deeper semantic coverage, the `sc62015_parity` crate now runs deterministic parity suites that single-step every opcode plus randomized instruction streams, diffing the full machine state after each step.

Every build prints an opcode-lowering coverage summary (specialized vs. LLIL fallback), and importing the `_sc62015_rustcore` module logs the same breakdown exactly once. This makes it easy to spot when new instructions slip back to the LLIL interpreter.

```bash
uv run python sc62015/tools/compare_cpu_backends.py --start 0x00 --end 0x1F
```

### CI coverage

GitHub Actions runs an optional "Optional Rust Backend" job (see `.github/workflows/tests.yml`) that:

1. Installs the Rust toolchain alongside the existing Python environment.
2. Builds the PyO3 extension via `uv run maturin develop`.
3. Executes the backend parity smoke tests with `--cpu-backends python,rust` so the suite is ready to toggle to the native core once implemented. The parity job also builds the extension with `enable_rust_cpu` so the semantics comparator can instantiate both backends.

The job is marked `continue-on-error`, so failures provide signal without blocking merges while the backend is experimental.

## Next steps
- Flesh out the Rust `CPU` implementation so that it mirrors the Python `Emulator` API.
- Decide how to share register state, memory bus abstractions, and LLIL evaluation paths between Python and Rust.
- Update packaging/CI once the backend can run the shared pytest suite.
