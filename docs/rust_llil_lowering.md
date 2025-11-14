## Rust LLIL Lowering & Parity Notes

### Environment for parity harness

To rerun the Rust↔Python parity suite you need a matching libpython and the
Rust extension on `PYTHONPATH`. The easiest way is to run
`./scripts/run-parity.sh`, which performs every step below automatically.
To run the commands by hand:

```bash
PYO3_PYTHON=/opt/homebrew/bin/python3.12 \
PYTHON_DYLIB=/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib \
SC62015_PARITY_DYLIB=$PWD/sc62015/parity/target/debug/libsc62015_parity.dylib \
PYTHONPATH=$PWD/sc62015/rustcore/target/debug:$PWD \
FORCE_BINJA_MOCK=1 \
cargo test -p sc62015-parity-harness -q -- --nocapture
```

Build the parity dylib first:

```bash
PYO3_PYTHON=/opt/homebrew/bin/python3.12 cargo build -p sc62015-parity
```

The `_sc62015_rustcore` extension is produced as `lib_sc62015_rustcore.dylib`;
symlink it once into the canonical Python name so imports work: 

```bash
cd sc62015/rustcore/target/debug
ln -sf lib_sc62015_rustcore.dylib _sc62015_rustcore.cpython-312-darwin.so
```

### Lowering coverage snapshot

As of this pass:

- All 256 opcodes lower to direct Rust handlers; no interpreter fallbacks
  remain in `opcode_handlers.rs`.
- Patterns covered: register loads/moves, arithmetic/logic/rotates, LOAD/STORE,
  PUSH/POP, CALL/CALLF, RET/RETF, jumps (absolute + conditional), intrinsics,
  and UNIMPL stubs.
- Stack helpers (`push_value`, `pop_value`, `call_absolute`) mirror the Python
  semantics and are used by generated handlers.
- Conditional jumps lower by evaluating the LLIL predicate (typically
  `CMP_E(FLAG, CONST)`) and updating `PC` only when the predicate is non-zero.

### Follow-ups

- Codify the `_sc62015_rustcore` symlink (e.g., build script step) so parity
  doesn’t rely on a manual `ln -sf`.
- Address the PyO3 deprecation warnings (`Python::import` and
  `GilRefs::function_arg`) when updating to PyO3 ≥0.22.
