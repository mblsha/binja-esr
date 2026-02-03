# Binary Ninja ESR-* plugin

The ESR plugin provides an SC62015 (aka ESR-L) architecture for Binary Ninja.

Currently it only works as a crude disassembler, with the goal to lift all the
instructions and create memory mapping for Sharp PC-E500 and Sharp Organizers.

## Acknowledgements

Overall structure of instruction logic based on
[binja-avnera](https://github.com/whitequark/binja-avnera) plugin by
@whitequark.

## License

Apache License 2.0.

## Development

Install dependencies using uv and run the checks:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv

	# Install all dependencies and create virtual environment
	uv sync --extra dev --extra pce500

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright sc62015/pysc62015

	# Run tests with coverage
	FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml
	```

The CI workflow uploads coverage results to Codecov on each commit.

## CLI emulator (terminal LCD)

The Rust LLAMA CLI is the primary emulator core. Run it with a terminal-rendered LCD view:

```bash
cargo run --manifest-path sc62015/core/Cargo.toml --bin sc62015-lcd -- --model pc-e500
```

Notes:
- Use `--model iq-7000` to switch ROM/profile.
- Use `--refresh-steps 20000` to control redraw cadence.
- Use `--input-steps 1000` to poll for key presses more frequently.
- Use `--no-alt-screen` for tmux capture panes.
- Use `--force-tty` when running detached.
- Use `--pf-numbers` to map digits 1–5 to PF1–PF5 (disables typing those digits).
- Use `--bnida PATH` to show function names in the status line (defaults to `rom-analysis/.../bnida.json` if present).
- Use `--force-key-irq` if the ROM stays halted at the boot menu (forces KEY interrupts on key press).
- Use `--card present|absent` to control memory card slot state (PC-E500).
- Use `--pf1-twice` to auto-drive the boot menu (S2(CARD) → PF1 → S1(MAIN) → PF1).
- Keys: Ctrl+1..5 or F1..F5 → PF1..PF5, Enter → `=`, Backspace → `BS`, Ctrl+C exits.
