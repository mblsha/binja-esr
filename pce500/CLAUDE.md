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

2. **PCE500Memory** (`memory.py`): Overlay-based memory system
   - Base 1MB writable external memory (0x00000-0xFFFFF) with 0x00 default for unmapped reads
   - Flexible overlay system for ROM, I/O, and special regions
   - ROM at 0xE0000-0xFFFFF (256KB) as read-only overlay
   - Memory card support (up to 64KB at 0x40000) as overlay
   - Memory-mapped I/O for LCD controllers at 0x2xxxx as I/O overlay
   - Perfetto tracing integration for memory access with Internal/External categorization

3. **HD61202Controller** (`display/hd61202.py`): LCD controller emulation
   - Dual HD61202 chips for 240x32 pixel display
   - Complex address decoding: CPU address bits determine chip selection and command/data routing
   - Each chip manages 64x32 pixels with 8 pages of 8-bit data

4. **TraceManager** (`trace_manager.py`): Sophisticated tracing infrastructure
   - Singleton pattern for global trace management using retrobus-perfetto
   - Tracks function durations, instant events, and counters
   - Automatic call stack tracking with proper cleanup

### Memory Map

- **Base External Memory**: 0x00000 - 0xFFFFF (1MB) - fully writable, defaults to 0x00
- **Internal ROM Overlay**: 0xE0000 - 0xFFFFF (256KB) - read-only, mapped over base memory
- **SC62015 Internal Memory**: 0x100000 - 0x1000FF (256B) - separate from external memory
- **Memory Card Overlay**: 0x40000 - 0x4FFFF (up to 64KB) - read-only when inserted
- **LCD Controllers Overlay**: 0x20000 - 0x2FFFF (64KB) - I/O handlers for HD61202 chips

System vectors:
- **Entry Point**: 0xFFFFD (3 bytes, little-endian)
- **Interrupt Vector**: 0xFFFFA (3 bytes, little-endian)

### Key Design Patterns

1. **Overlay Memory System**: Recently refactored from fixed regions to flexible overlay system
   - Base 1MB writable external memory layer
   - Read-only ROM/memory card overlays mapped on top
   - I/O overlays with custom read/write handlers
   - Easy addition/removal of memory regions at runtime
2. **Performance Tracking**: Built-in counters for instructions, memory operations, and cycles
3. **Context Manager Support**: Emulator can be used with `with` statement for automatic resource cleanup
4. **Modular Tracing**: Optional Perfetto tracing that can be enabled/disabled at runtime
5. **Simplified Memory Categories**: All memory operations traced as Internal or External only

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

### Perfetto Tracing System

The emulator includes sophisticated Perfetto tracing that can be visualized at https://ui.perfetto.dev/

#### Trace Threads

The tracing system uses multiple threads to organize different types of events:

1. **Execution Thread**: All instruction executions
   - Shows every instruction with PC, opcode, and flag states
   - Includes C and Z flags before/after each instruction (C_before, Z_before, C_after, Z_after)
   - Helps debug flag-dependent conditional jumps

2. **CPU Thread**: Control flow and system events
   - Function calls/returns with call stack tracking
   - Jump instructions (only taken jumps for conditionals)
   - Interrupts and system events
   - No longer shows individual instruction execution

3. **Memory Thread**: Memory operations
   - Read/write operations with addresses and values
   - Memory-mapped I/O access

4. **I/O Thread**: I/O operations
5. **Display Thread**: LCD controller operations
6. **Interrupt Thread**: Interrupt handling

#### Jump Tracing

Jump instructions are traced with special handling:
- **Unconditional jumps** (JP, JPF): Always traced
- **Conditional jumps** (JRZ, JRNZ, JRC, JRNC): Only traced when taken
- Jump traces show actual PC destinations (where execution went)
- FROM address is the jump instruction location
- TO address is the actual PC after the jump executed

#### Implementation Details

Key files for tracing:
- `trace_manager.py`: Core Perfetto integration using retrobus-perfetto
- `emulator.py`: Instruction and control flow tracing
- `memory.py`: Memory access tracing

To enable tracing:
```python
emu = PCE500Emulator(perfetto_trace=True)
# Trace file saved as pc-e500.trace
```

#### Debugging Tips

1. Use separate threads to filter events in Perfetto UI
2. Look at flag states in Execution thread to understand conditional behavior
3. CPU thread shows program flow without execution noise
4. Correlate memory accesses with instruction execution using timestamps