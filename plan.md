Env vars across Python + Rust
-----------------------------

Remaining env toggles (post cleanup)

Cross-backend
- `SC62015_CPU_BACKEND`: backend selector (`python`/`llama`).

Build-time
- `PYO3_PYTHON`: chooses Python interpreter during Rust bindings build.
- `OUT_DIR`: cargo build output path (standard Cargo-provided variable).
