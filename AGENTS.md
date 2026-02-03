# Repository Guidelines

> Tooling: This project standardizes on `uv` for Python package management and running commands. Use `uv sync` to install dependencies and `uv run` to execute all tools and scripts. Prefer `uv` over calling `python -m ...` or tool entry points directly.

## Project Structure & Module Organization
- `sc62015/`: Binary Ninja integration (`arch.py`, `view.py`).
- `sc62015/pysc62015/`: SC62015 assembler/emulator core and unit tests (`test_*.py`).
- `pce500/`: PC‑E500 device model, tools, and tests.
- `data/`: ROMs and traces (e.g., `pc-e500.bin`).
- Root: `plugin.json`, `pyproject.toml`, lint/type configs.

## Reference Docs
- `docs/sc62015_python_emulator_surface.md`: Python emulator surface/tests; keep in sync with the primary Rust LLAMA core.
- `.github/workflows/llama-perfetto-smoke.yml`: nightly/dispatch Perfetto trace smoke (NOP/CALL/EMEM MV/DSBL/EMEM reg-indirect/PUSHU).

## Setup, Build, and Dev Commands
- Environment (preferred): `uv sync` then `uv sync --extra dev [--extra pce500]`
- Install all deps at once: `uv sync --extra dev --extra pce500`  # recommended
- Alternative: `python -m pip install -e .[dev]` (extras: `.[pce500]`)
- Lint: `uv run ruff check .` (format: `uv run ruff format .`)
- Type check: `uv run pyright sc62015/pysc62015` or `uv run python scripts/run_pyright.py`
- Core tests: `FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing`
- PCE‑500 tests: `uv run pytest pce500/tests --cov=pce500 --cov-report=term-missing`
- Watch (fish): `./run-tests.fish` (or `--once`) for continuous testing

## Coding Style & Naming Conventions
- Python 3.11, 4‑space indent, max line length 100.
- Lint with Ruff (`ruff.toml` ignores `F405`, `E701`; excludes `third_party/`).
- Prefer type hints; Pyright in basic mode.
- Files/modules: lowercase_with_underscores; tests named `test_*.py`.

## Rust/Python Parity Notes
- The Rust LLAMA core is primary; keep the Python emulator in lockstep. Annotate each Rust source file with the Python module/class/function it mirrors to aid audits, and keep those references up to date when code moves.
- Annotations must be machine-parseable: add one or more `// PY_SOURCE: sc62015/pysc62015/<module>.py[:<object>]` comment lines near the top of every Rust source file that trace back to the Python implementation.
- Verify annotations with `uv run python scripts/check_rust_py_parity_annotations.py`.
- Keep parity both ways: when Rust changes, update Python to match and cover gaps with tests (parity harnesses, regression tests, and nightly smoke traces).

## Testing Guidelines
- Framework: `pytest` (see `pytest.ini` collects under `sc62015`).
- Coverage: ≥80% (project and patch). CI enforces lint, type check, tests.
- Scope: Unit tests near code; integration in `pce500/tests`.
- Binary Ninja: Most tests run with mocks. Set `FORCE_BINJA_MOCK=1` (recommended even if BN is installed).
- Single test example: `uv run pytest path/to/test_file.py::test_name -q`

## Commit & Pull Request Guidelines
- Commit messages: imperative, concise; optional scope prefix (e.g., `pce500:`).
  - Examples: `Fix load_rom…`, `Add KO/KI labels…`.
- PRs must include: clear description, rationale, linked issues, test plan, coverage results, and screenshots/GIFs for UI or LCD changes.
- CI must pass (lint, type, tests); avoid decreasing coverage.
- Before opening a PR, run formatters locally:
  - Python: `uv run ruff format .` then `uv run ruff check .`
  - Rust: `cargo fmt --manifest-path sc62015/core/Cargo.toml --all` (and `sc62015/rustcore`, `web/emulator-wasm` if touched)
  - Web (TS/Svelte): `cd web && npm run format:check` (or `npm run format` to fix)

## Runtime & Dev Tips
- PC‑E500 ROM: Ensure `data/pc-e500.bin` exists. Without it, some emulator features/tests will be skipped.
- Primary emulator core: use the Rust CLI (`cargo run --manifest-path sc62015/core/Cargo.toml --bin pce500 -- --steps 20000`).
- Legacy Python wrapper: `uv run python pce500/run_pce500.py` (use `--profile-emulator` to emit `emulator-profile.perfetto-trace`).
- Binary Ninja: Not required for dev; mocks auto‑load via `FORCE_BINJA_MOCK=1` or `binja-test-mocks`.
- Native backend: LLAMA is the only Rust core; SCIL/manifest tooling and tests were removed.

- LLAMA is expected to be present in this workspace; do not skip tests on missing LLAMA. If a test checks `available_backends()`, prefer failing loudly instead of skipping.

- **LLAMA tracing/parity:** Perfetto traces (binary `retrobus-perfetto`) from Python and LLAMA cores align on instruction-index timestamps; `scripts/compare_perfetto_traces.py` compares them. Nightly smoke lives in `.github/workflows/llama-perfetto-smoke.yml`.
- **CI coverage (Perfetto included):** All guardrails must run in CI—lint, type checks, pytest suites, parity harnesses, and Perfetto comparison jobs. Ensure Perfetto trace comparison (`scripts/compare_perfetto_traces.py` or the smoke workflow) is wired into CI and kept green.
- **Rust CLI runner (primary):** To boot the ROM and view decoded LCD text:
  - `cargo run --manifest-path sc62015/core/Cargo.toml --bin pce500 -- --steps 20000`
  - Optional LCD logging: `RUST_LCD_TRACE=1 RUST_LCD_TRACE_MAX=2000 ...`
  - Default ROM model: `pc-e500` (uses `data/pc-e500.bin`). Select IQ-7000 with `--model iq-7000` or pass `--rom PATH`.
- **LCD terminal UI:** A live terminal renderer that redraws decoded LCD lines on change:
  - `cargo run --manifest-path sc62015/core/Cargo.toml --bin sc62015-lcd -- --model pc-e500`
  - Keyboard: Ctrl+1..5 or F1..F5 map to PF1..PF5, Enter maps to `=`, Backspace maps to `BS`, Ctrl+C exits.
  - Use `--no-alt-screen` for tmux capture panes, `--force-tty` when running detached, `--input-steps` to reduce input latency, `--pf-numbers` to map digits 1–5 to PF1–PF5, `--bnida PATH` to show function names in the status line, `--force-key-irq` to force KEY interrupts if the ROM stays halted, `--card present|absent` to control the memory card slot state, and `--pf1-twice` to auto-drive the PF1 menu.
- **JS function runner (WASM):** Run an async JS snippet against the same Rust core compiled to WASM:
  - Install deps once: `cd web && npm install`
  - Run a script (auto-builds wasm): `cd web && npm run fnr:cli -- --model pc-e500 path/to/script.js` (or `--eval "<js>"`, or `--stdin`)
  - Script API: `e` is the same `EvalApi` used by the web Function Runner; output JSON is compatible with `FunctionRunnerOutput`.
  - Stubs: `e.stub(0x00F1234, 'demo', (mem, regs, flags) => ({ mem_writes: { 0x2000: 0x41 }, regs: { A: 1 }, flags: { Z: 0, C: 1 }, ret: { kind: 'ret' } }))` intercepts a PC and returns a patch; `mem.read8/read16/read24` are read-only and writes flow through `mem_writes` (array or `{addr: value}` map).
    Return kinds: `ret`, `retf`, `jump`, `stay`.
  - Note: this is separate from the native Rust CLI (no TS wrapper for the Rust CLI); the web UI uses `web/src/lib/wasm/sc62015_wasm.ts` to keep `Pce500Emulator` as an alias for `Sc62015Emulator`.
