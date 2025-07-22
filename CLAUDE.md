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

# Run Python code within Binary Ninja process
./mcp/binja-cli.py python "print(bv.functions)"
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

## SC62015 Emulator Usage

The SC62015 emulator (`sc62015/pysc62015/emulator.py`) is a sophisticated LLIL-based implementation that executes instructions by:
1. Decoding them into instruction objects
2. Lifting them to Binary Ninja's Low-Level IL
3. Evaluating the LLIL to perform the actual execution

### Important: Register Access Pattern

The emulator uses getter/setter methods for register access, NOT property syntax:

```python
# CORRECT - Use get/set methods
pc = emu.regs.get(RegisterName.PC)
emu.regs.set(RegisterName.PC, 0x1000)
reg_a = emu.regs.get(RegisterName.A)

# INCORRECT - These will fail
pc = emu.regs.pc  # AttributeError!
emu.regs.pc = 0x1000  # AttributeError!
```

### PC Advancement

The emulator correctly advances PC after each instruction execution. The `execute_instruction()` method:
1. Sets PC to the instruction address
2. Decodes and lifts the instruction
3. Updates PC to `address + instruction_length`
4. Evaluates the LLIL (which may modify PC for jumps/branches)

### Testing the Emulator

When writing tests, always use the correct register access pattern:

```python
from sc62015.pysc62015.emulator import Emulator, Memory, RegisterName

# Set PC
emu.regs.set(RegisterName.PC, 0x1000)

# Execute instruction at current PC
emu.execute_instruction(emu.regs.get(RegisterName.PC))

# Check PC advanced
new_pc = emu.regs.get(RegisterName.PC)
```

## PC-E500 Emulator

The PC-E500 emulator (`pce500/`) wraps the SC62015 emulator with Sharp PC-E500 specific hardware:
- Memory mapper for ROM/RAM regions
- HD61202 LCD controller emulation
- Entry point at 0xFFFFD (3 bytes, little-endian)
- Interrupt vector at 0xFFFFA (3 bytes, little-endian)