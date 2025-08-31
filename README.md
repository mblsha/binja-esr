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
uv sync --extra dev --extra pce500 --extra web

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run pyright sc62015/pysc62015

# Run tests with coverage
FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml
```

The CI workflow uploads coverage results to Codecov on each commit.
