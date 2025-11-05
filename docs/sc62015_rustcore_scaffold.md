# SC62015 Rust Core Scaffold

This repository now includes the scaffolding needed to build an optional Rust implementation of the SC62015 emulator. The backend is **disabled by default** and currently ships as a stub so the Python selector can detect whether the native module has been built.

## Layout
- `sc62015/rustcore/`: Cargo crate that will host the Rust CPU implementation.
  - `Cargo.toml`: defines a `cdylib` target named `sc62015_rustcore` built with PyO3 (`abi3-py311`).
  - `build.rs`: invokes `sc62015/tools/generate_opcode_metadata.py` to produce LLIL-driven opcode metadata and emits a Rust source file under `OUT_DIR/opcode_table.rs`. If Python or the generator is unavailable, a stub table is used instead.
  - `src/constants.rs`: architectural constants shared with the Python implementation (address space, PC mask, TEMP register count).
  - `src/state.rs`: register file implementation mirroring the Python semantics (composite registers, flag bits, TEMP slots, call stack tracking).
  - `src/memory.rs`: safe wrapper around the Python `Memory` helper, offering byte/word accessors and tracer discovery.
  - `src/lib.rs`: exposes a minimal `CPU` class and helper functions. The constructor raises `NotImplementedError` for now, but compiling it allows Python to detect the module and decide whether it is ready for use. It also re-exports the generated opcode metadata and state/memory layers for future use.
  - `tests/python_parity.rs`: skeleton proptest harness that will cross-check instruction execution once the Rust backend is feature-complete. It currently exits early when the Rust core reports `HAS_CPU_IMPLEMENTATION = False`.
- Python shim:
  - `sc62015/pysc62015/cpu.py`: provides a `CPU` facade that selects between the existing Python `Emulator` and the optional Rust backend. Selection obeys the `SC62015_CPU_BACKEND` environment variable or an explicit `backend=` argument. When Rust is requested but not available, a descriptive error explains how to build the extension.
  - `sc62015/pysc62015/__init__.py`: re-exports `CPU`, backend helpers, and the existing `Emulator`.

## Building the extension
The Rust backend is not compiled automatically. To build it locally (for experimentation or to start filling in the implementation):

```bash
uv tool install maturin  # once per environment
uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml
```

This command produces a Python extension module named `_sc62015_rustcore` that can be imported from `sc62015.pysc62015`. After building, set `SC62015_CPU_BACKEND=rust` (or pass `backend="rust"` to the `CPU` constructor) to route instantiations through the native module.

Because the module currently exposes a stub implementation, any attempt to instantiate the Rust `CPU` still raises `NotImplementedError`. This is intentional: the selector reports availability only when `HAS_CPU_IMPLEMENTATION` is set to `True`, which will happen once real execution logic is in place.

### Regenerating opcode metadata

The build script automatically runs `sc62015/tools/generate_opcode_metadata.py` (using `FORCE_BINJA_MOCK=1`) to collect LLIL for each primary opcode. You can execute the generator manually for inspection:

```bash
uv run python sc62015/tools/generate_opcode_metadata.py --pretty
```

No JSON artifacts are committed; the output is produced on demand during each build.

### Comparing backends locally

Use `sc62015/tools/compare_cpu_backends.py` to execute individual opcodes through both backends. The tool skips cleanly when the Rust core is unavailable or still stubbed:

```bash
uv run python sc62015/tools/compare_cpu_backends.py --start 0x00 --end 0x1F
```

### CI coverage

GitHub Actions runs an optional "Optional Rust Backend" job (see `.github/workflows/tests.yml`) that:

1. Installs the Rust toolchain alongside the existing Python environment.
2. Builds the PyO3 extension via `uv run maturin develop`.
3. Executes the backend parity smoke tests with `--cpu-backends python,rust` so the suite is ready to toggle to the native core once implemented.

The job is marked `continue-on-error`, so failures provide signal without blocking merges while the backend is experimental.

## Next steps
- Flesh out the Rust `CPU` implementation so that it mirrors the Python `Emulator` API.
- Decide how to share register state, memory bus abstractions, and LLIL evaluation paths between Python and Rust.
- Update packaging/CI once the backend can run the shared pytest suite.
