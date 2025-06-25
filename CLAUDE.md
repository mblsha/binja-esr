# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install the package in editable mode with development dependencies
python -m pip install -e .[dev]
```

### Code Quality Commands
```bash
# Run linting
ruff check .

# Run type checking (focused on pysc62015 module)
pyright sc62015/pysc62015

# Run tests with coverage
pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml

# Run a single test
pytest path/to/test_file.py::test_function_name

# Continuous testing with file watching (requires Fish shell and fswatch)
./run-tests.fish
```

## Architecture Overview

This is a Binary Ninja plugin for the SC62015 (ESR-L) processor, used in Sharp PC-E500 and Sharp Organizers. The architecture separates the SC62015 implementation from Binary Ninja integration for testability.

### Core Components

1. **Plugin Entry** (`__init__.py`): Registers the SC62015 architecture and binary views with Binary Ninja
2. **Architecture Implementation** (`sc62015/arch.py`): Implements Binary Ninja's Architecture API with three key methods:
   - `get_instruction_info()`: Analyzes branches and calls
   - `get_instruction_text()`: Disassembles to text
   - `get_instruction_low_level_il()`: Lifts to LLIL

3. **Pure Python SC62015** (`sc62015/pysc62015/`): Self-contained implementation of the processor:
   - Instruction decoding/encoding
   - Assembler using Lark parser
   - Emulator for testing
   - Hardware testing utilities

4. **Testing Infrastructure** (`binja_helpers/`): Mock Binary Ninja API for testing without Binary Ninja installation

### Key Design Principles

- The `pysc62015` module is independent of Binary Ninja, making it reusable and testable
- All instruction logic (decode, analyze, render, lift) is implemented in `sc62015/pysc62015/instr/`
- Type safety is enforced with pyright using custom stubs for the Binary Ninja API
- The plugin supports two ROM formats: SC62015RomView (partial) and SC62015FullView (complete memory)

### SC62015 Processor Details

- 8-bit processor with 24-bit addressing
- Little-endian byte order
- Registers: A, B, BA (16-bit A+B), I (index), X/Y (24-bit pointers), U/S (stack pointers), PC
- Flags: Zero (Z) and Carry (C)
- Memory regions: internal (0x000-0x1FF), external, and chip select areas