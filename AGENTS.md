# Repository Guidelines

> Tooling: This project standardizes on `uv` for Python package management and running commands. Use `uv sync` to install dependencies and `uv run` to execute all tools and scripts. Prefer `uv` over calling `python -m ...` or tool entry points directly.

## Project Structure & Module Organization
- `sc62015/`: Binary Ninja integration (`arch.py`, `view.py`).
- `sc62015/pysc62015/`: SC62015 assembler/emulator core and unit tests (`test_*.py`).
- `pce500/`: PC‑E500 device model, tools, and tests.
- `web/`: Flask UI (app, templates, static) + tests.
- `data/`: ROMs and traces (e.g., `pc-e500.bin`).
- Root: `plugin.json`, `pyproject.toml`, lint/type configs.

## Setup, Build, and Dev Commands
- Environment (preferred): `uv sync` then `uv sync --extra dev [--extra pce500] [--extra web]`
- Install all deps at once: `uv sync --extra dev --extra pce500 --extra web`  # recommended
- Alternative: `python -m pip install -e .[dev]` (extras: `.[pce500]`, `.[web]`)
- Lint: `uv run ruff check .` (format: `uv run ruff format .`)
- Type check: `uv run pyright sc62015/pysc62015` or `uv run python scripts/run_pyright.py`
- Core tests: `FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing`
- PCE‑500 tests: `uv run pytest pce500/tests --cov=pce500 --cov-report=term-missing`
- Web tests: `uv run pytest web/tests --cov=web --cov-report=term-missing`
- Watch (fish): `./run-tests.fish` (or `--once`) for continuous testing

## Coding Style & Naming Conventions
- Python 3.11, 4‑space indent, max line length 100.
- Lint with Ruff (`ruff.toml` ignores `F405`, `E701`; excludes `third_party/`).
- Prefer type hints; Pyright in basic mode.
- Files/modules: lowercase_with_underscores; tests named `test_*.py`.

## Testing Guidelines
- Framework: `pytest` (see `pytest.ini` collects under `sc62015`).
- Coverage: ≥80% (project and patch). CI enforces lint, type check, tests.
- Scope: Unit tests near code; integration in `pce500/tests` and `web/tests`.
- Binary Ninja: Most tests run with mocks. Set `FORCE_BINJA_MOCK=1` (recommended even if BN is installed).
- Single test example: `uv run pytest path/to/test_file.py::test_name -q`

## Commit & Pull Request Guidelines
- Commit messages: imperative, concise; optional scope prefix (e.g., `pce500:`, `web:`).
  - Examples: `Fix load_rom…`, `Add KO/KI labels…`.
- PRs must include: clear description, rationale, linked issues, test plan, coverage results, and screenshots/GIFs for UI or LCD changes.
- CI must pass (lint, type, tests); avoid decreasing coverage.

## Runtime & Dev Tips
- PC‑E500 ROM: Ensure `data/pc-e500.bin` exists. Without it, some emulator features/tests will be skipped.
- Emulator demo: `uv run python pce500/run_pce500.py` (use `--profile-emulator` to emit `emulator-profile.perfetto-trace`).
- Web UI: `uv run python web/run.py` then open the served address.
- Binary Ninja: Not required for dev; mocks auto‑load via `FORCE_BINJA_MOCK=1` or `binja-test-mocks`.

## SCIL Rollout & Gating
- Production flips rely on env vars: `BN_USE_SCIL` (`off`/`shadow`/`prod`),
  `BN_SCIL_ALLOW` (per-mnemonic), `BN_SCIL_BLOCK`, and `BN_SCIL_FAMILIES`
  (comma/semicolon-separated family names). Families map 1:1 to the
  `decode_map` `family` field.
- Phase‑8 families you can now gate via `BN_SCIL_FAMILIES`:
  `loop_move`, `loop_arith` (ADCL/SBCL), `loop_bcd` (DADL/DSBL),
  `decimal_shift` (DSLL/DSRL), `pmdf`, and `system`
  (`HALT`/`OFF`/`RESET`/`WAIT`/`IR`). Existing names from earlier phases
  (e.g., `imm8`, `rel8`, `ext_reg`, `imem_move`, `jp_*`, `stack_sys`,
  `pushu`, `popu`, `call_near`, `ret_near`, etc.) remain valid.
- Example shadow rollout:
  ``BN_USE_SCIL=shadow BN_SCIL_FAMILIES="imm8,rel8,loop_move,loop_arith,loop_bcd,decimal_shift,pmdf,system"``.
  Add/remove entries to toggle specific families without touching per‑mnemonic
  allow/block lists.
