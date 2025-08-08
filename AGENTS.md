# Repository Guidelines

## Project Structure & Modules
- `sc62015/`: Binary Ninja architecture plugin glue (`arch.py`, `view.py`).
- `sc62015/pysc62015/`: Core SC62015 assembler/emulator and tests (`test_*.py`).
- `pce500/`: PC‑E500 device model, tools, and tests.
- `web/`: Flask UI for emulator + tests; static/templates.
- `data/`: ROMs and traces (e.g., `pc-e500.bin`).
- Root: `plugin.json`, `pyproject.toml`, lint/type configs.

## Build, Test, and Dev Commands
- Install (dev): `python -m pip install -e .[dev]`
- Optional extras: `.[pce500]`, `.[web]`
- Lint: `ruff check .`
- Type check: `pyright sc62015/pysc62015` or `python scripts/run_pyright.py`
- Tests (core): `pytest --cov=sc62015/pysc62015 --cov-report=term-missing`
- Tests (all):
  - `pytest pce500/tests --cov=pce500 --cov-report=term-missing`
  - `pytest web/tests --cov=web --cov-report=term-missing`
- Watch (fish): `./run-tests.fish` or `./run-tests.fish --once`

Tip: Most tests run without Binary Ninja by setting `FORCE_BINJA_MOCK=1`.

## Coding Style & Naming
- Python 3.11, 4‑space indentation; max line length 100 (`.flake8`).
- Lint with Ruff; `ruff.toml` ignores `F405`, `E701`, and excludes `third_party/`.
- Prefer type hints; Pyright runs in basic mode.
- Modules and files: lowercase with underscores. Tests named `test_*.py`.

## Testing Guidelines
- Framework: `pytest` with `pytest.ini` collecting under `sc62015`.
- Coverage: Codecov required target 80% (project and patch).
- Add focused unit tests near the code under test; keep integration tests in `pce500/tests` and `web/tests`.

## Commit & PR Guidelines
- Commit style: imperative, concise subjects; optional scope prefix (e.g., `pce500:`, `web:`). Examples: “Fix load_rom…”, “Add KO/KI labels…”.
- PRs include: clear description, rationale, linked issues, test plan, coverage results, and screenshots/GIFs for UI or LCD changes.
- CI must pass (lint, type check, tests); avoid decreasing coverage.

## Runtime & Dev Tips
- Binary Ninja not required for dev: mocks auto‑load with `FORCE_BINJA_MOCK=1` or via `binja-test-mocks`.
- Web UI: `python web/run.py`; Emulator demo: `python pce500/run_pce500.py` (ROM in `data/`).
