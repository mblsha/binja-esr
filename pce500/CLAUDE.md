# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install with development dependencies
python -m pip install -e .[dev]
```

### Code Quality Commands
```bash
# Run linting
ruff check .

# Run type checking
pyright sc62015/pysc62015

# Run tests with coverage
FORCE_BINJA_MOCK=1 pytest --cov=sc62015/pysc62015 --cov-report=term-missing
FORCE_BINJA_MOCK=1 pytest pce500/tests/ --cov=pce500 --cov-report=term-missing

# Run a single test
pytest path/to/test_file.py::test_function_name

# Continuous testing with file watching (requires Fish shell)
./run-tests.fish
```

## Architecture Overview

This is a PC-E500 emulator that provides a complete emulation of the Sharp PC-E500 pocket computer. It wraps the SC62015 CPU emulator from the parent project with PC-E500 specific hardware emulation.

### Core Components

1. **PCE500Emulator** (`emulator.py`): Main emulator class integrating all components
   - Wraps SC62015 CPU emulator with TrackedMemory for statistics
   - Provides context manager support for automatic trace file saving
   - Implements breakpoint and debugging features

2. **PCE500Memory** (`memory.py`): Direct memory implementation
   - Pre-allocated regions for RAM (32KB at 0xB8000) and ROM (256KB at 0xC0000)
   - Optional memory card support (up to 64KB at 0x40000)
   - Memory-mapped I/O for LCD controllers at 0x2xxxx
   - Perfetto tracing integration for memory access

3. **HD61202Controller** (`display/hd61202.py`): LCD controller emulation
   - Dual HD61202 chips for 240x32 pixel display
   - Complex address decoding: CPU address bits determine chip selection and command/data routing
   - Each chip manages 64x32 pixels with 8 pages of 8-bit data

4. **TraceManager** (`trace_manager.py`): Sophisticated tracing infrastructure
   - Singleton pattern for global trace management using retrobus-perfetto
   - Tracks function durations, instant events, and counters
   - Automatic call stack tracking with proper cleanup

### Memory Map

- **Internal ROM**: 0xC0000 - 0xFFFFF (256KB)
- **Internal RAM**: 0xB8000 - 0xBFFFF (32KB)
- **Memory Card**: 0x40000 - 0x4FFFF (up to 64KB)
- **LCD Controllers**: 0x2xxxx (memory-mapped I/O)

System vectors:
- **Entry Point**: 0xFFFFD (3 bytes, little-endian)
- **Interrupt Vector**: 0xFFFFA (3 bytes, little-endian)

### Key Design Patterns

1. **Simplified Architecture**: Recently refactored from complex 4-layer abstraction to clean, maintainable design
2. **Performance Tracking**: Built-in counters for instructions, memory operations, and cycles
3. **Context Manager Support**: Emulator can be used with `with` statement for automatic resource cleanup
4. **Modular Tracing**: Optional Perfetto tracing that can be enabled/disabled at runtime

### Testing Infrastructure

Tests require `FORCE_BINJA_MOCK=1` environment variable to run without Binary Ninja license. The test suite covers:
- Core emulator functionality (initialization, reset, ROM/RAM access)
- LCD controller operations
- Execution tracing and debugging features
- Perfetto trace generation
- Call stack tracking

### Usage Example

```python
# Basic usage from run_pce500.py
from pce500 import PCE500Emulator

# Create emulator with ROM
emu = PCE500Emulator(rom_data=rom_bytes, enable_tracing=True)

# Use as context manager for automatic trace saving
with emu:
    # Run until breakpoint or completion
    emu.run()
    
# Access statistics
print(f"Instructions: {emu.instructions}")
print(f"Memory reads: {emu.memory_reads}")
```