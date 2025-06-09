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

Install the package in editable mode with development dependencies and run the
checks:

```bash
python -m pip install -e .[dev]
ruff check
export MYPYPATH=$(pwd)/stubs  # ensure Binary Ninja stubs are found
mypy sc62015/pysc62015
pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml
```

The CI workflow uploads coverage results to Codecov on each commit.

