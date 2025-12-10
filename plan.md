Env vars across Python + Rust
-----------------------------

Remaining env toggles (post cleanup)

Cross-backend
- `SC62015_CPU_BACKEND`: backend selector (`python`/`llama`).
- `LLAMA_TIMER_SCALE`: scales MTI/STI periods; keep consistent across backends.

Build-time
- `PYO3_PYTHON`: chooses Python interpreter during Rust bindings build.
- `OUT_DIR`: cargo build output path (standard Cargo-provided variable).

Web-only
- `PCE500_WEB_ALLOWED_ORIGINS`: CORS allowlist for the Flask web UI.
