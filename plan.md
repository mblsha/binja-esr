Env vars across Python + Rust
-----------------------------

Remaining env toggles (post cleanup)

Cross-backend
- `SC62015_CPU_BACKEND`: backend selector (`python`/`llama`).
- `LLAMA_TIMER_SCALE`: scales MTI/STI periods; keep consistent across backends.
- `FORCE_STROBE_LLAMA`, `FORCE_KEYI_LLAMA`: force keyboard strobe/KEYI for LLAMA debugging; can mask timing issues.

Rust-specific
- `LLAMA_STRICT_OPCODES`: switch unknown-opcode handling to erroring (defaults to Python-style fallback).

Python-specific
- `RUST_PURE_KEYBOARD`, `RUST_PURE_LCD`: force Rust peripherals even on Python backend.
- `RUST_KEYI_BIAS`: adjust KEYI behavior for LLAMA debug.

Build-time
- `PYO3_PYTHON`: chooses Python interpreter during Rust bindings build.
- `OUT_DIR`: cargo build output path (standard Cargo-provided variable).

Web-only
- `PCE500_WEB_ALLOWED_ORIGINS`: CORS allowlist for the Flask web UI.
