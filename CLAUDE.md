# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv

# Install dependencies and create virtual environment
uv sync --extra dev --extra pce500 --extra web  # Install ALL dependencies (required for full functionality)
```

### Code Quality Commands
```bash
# Run linting
uv run ruff check .
uv run ruff format .  # Format code

# Run type checking (focused on pysc62015 module)
uv run pyright sc62015/pysc62015

# Run tests with coverage
# Note: FORCE_BINJA_MOCK=1 forces use of mock Binary Ninja API (needed if Binary Ninja is installed)
FORCE_BINJA_MOCK=1 uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing --cov-report=xml

# Run a single test
uv run pytest path/to/test_file.py::test_function_name

# Run tests for specific modules (requires all extras installed)
FORCE_BINJA_MOCK=1 uv run pytest  # SC62015 core tests (353 tests)
FORCE_BINJA_MOCK=1 uv run pytest pce500/ -v  # PCE500 tests (140 tests)
cd web && FORCE_BINJA_MOCK=1 uv run pytest tests/ -v  # Web tests (22 tests)

# Continuous testing with file watching (requires Fish shell and fswatch)
./run-tests.fish

# Run Python code within Binary Ninja process
./mcp/binja-cli.py python "print(bv.functions)"
```

### Git Commands
```bash
# Update master branch (use rebase to maintain clean history)
git pull --rebase origin master

# Create feature branch
git checkout -b feature-name

# Push feature branch and set upstream
git push -u origin feature-name
```

### Important: PC-E500 ROM Requirement

**Alert the user if `data/pc-e500.bin` is not present in the repository.** This ROM file is required for full test coverage of the PC-E500 emulator functionality. Without it, certain integration tests and emulator features cannot be properly tested.

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
from sc62015.pysc62015 import CPU, RegisterName
from binja_test_mocks.eval_llil import Memory

memory = Memory(lambda addr: 0, lambda addr, value: None)
emu = CPU(memory, reset_on_init=False)

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

### ROM Loading

The `pc-e500.bin` file is a 1MB memory dump. When loading:
- Extract bytes from offset 0xC0000 to 0x100000 (256KB) - this is the actual ROM
- The ROM is loaded at address 0xC0000 in the emulator's memory space
- Entry point 0xFFFFD typically contains 0xC2 0x10 0x0F (0x0F10C2 in little-endian)

## Performance Profiling

The emulator includes sophisticated wall-clock performance profiling using Perfetto tracing to identify bottlenecks and optimization opportunities.

### Running Performance Profiling

```bash
# Profile emulator execution
uv run -- env FORCE_BINJA_MOCK=1 python3 pce500/run_pce500.py --profile-emulator

# Output: emulator-profile.perfetto-trace
# View at: https://ui.perfetto.dev/
```

### Architecture

The profiling system uses a clean, zero-conditional approach:
- **Always-on decorators and context managers** that check internally if profiling is enabled
- **No conditional branches** in instrumented code - decorators/context managers are always present
- **Minimal overhead** when disabled (just function call + flag check)
- **Wall-clock timing** via `time.perf_counter()` separate from virtual CPU cycles

### Performance Tracks

The trace includes these specialized tracks:
- **Emulation**: High-level operations (`step()`, `reset()`, `load_rom()`) with `op_num` sequential counter
- **Opcodes**: Individual instruction execution with opcode names and `op_num` counter
- **Memory**: Read/write operations (sampled at 1/100 to reduce overhead)
- **Display**: LCD controller operations
- **Lifting**: SC62015 instruction decoding and LLIL evaluation
- **System**: System-level operations like reset and ROM loading

### Implementation Details

#### Clean Instrumentation Pattern
```python
# Decorator always present, checks internally
@perf_trace("Emulation", include_op_num=True)
def step(self):
    # Context manager always runs, checks internally
    with tracer.slice("Opcodes", opcode_name, {"pc": f"0x{pc:06X}", "op_num": self.instruction_count}):
        eval_info = self.cpu.execute_instruction(pc)
```

#### Sampled Tracing for High-Frequency Operations
```python
@perf_trace("Memory", sample_rate=100)  # Only trace every 100th call
def read_byte(self, address: int) -> int:
    # Implementation
```

#### Key Files
- `pce500/tracing/perfetto_tracing.py`: Core tracing infrastructure with `@perf_trace` decorator and `slice()` context manager
- `pce500/emulator.py`: Instrumented with opcode-level tracing
- `pce500/memory.py`: Sampled memory operation tracing
- `pce500/display/controller_wrapper.py`: Display operation tracing

### Trace Analysis Tips

1. **Operation Numbers**: Each instruction has a sequential `op_num` (0-based) in both Emulation and Opcodes tracks
2. **Sampling**: Memory operations are sampled (1/100) to reduce trace size while maintaining insights
3. **Hierarchical View**: Opcodes are nested under Emulation steps for clear execution flow
4. **Wall-Clock Time**: All measurements are real-time, not virtual CPU cycles

### Performance Optimization Workflow

1. Run with `--profile-emulator` flag
2. Load trace in https://ui.perfetto.dev/
3. Look for:
   - Hot spots in the flame graph
   - Long-duration slices in Opcodes track
   - Memory operation patterns
   - Unexpected delays in Display operations
4. Focus optimization on bottlenecks identified in traces

### Technical Notes

- Uses `retrobus-perfetto` library for protobuf trace generation
- Zero overhead when disabled - all instrumentation checks flag internally
- SC62015 emulator integration via `memory.set_perf_tracer()` for cross-module tracing
- Trace files are binary protobuf format compatible with Perfetto ecosystem

## Web-Based PC-E500 Emulator

The web emulator (`web/`) provides a browser-based interface to the PC-E500 emulator:

### Architecture
- **Backend**: Flask server (`app.py`) managing emulator state and providing REST API
- **Frontend**: JavaScript SPA with virtual keyboard and real-time display updates
- **Keyboard**: Single keyboard handler implementation (matrix via KOL/KOH/KIL)

### Key Implementation Details

1. **State Updates**: Triggered by either:
   - 100ms elapsed (10 FPS target)
   - 100,000 instructions executed

2. **Keyboard Matrix**:
   - Output registers KOL (0xF0) and KOH (0xF1) select keyboard columns (KO0–KO10)
     - KOL bits 0–7 map to KO0–KO7; KOH bits 0–2 map to KO8–KO10
     - Compat semantics: bits set = column active
   - Input register KIL (0xF2) reads row states (KI0–KI7)
   - Keys mapped to specific column/row intersections; columns are outputs, rows are inputs

3. **Critical Initialization**:
   ```python
   # Load only the ROM portion (last 256KB of 1MB file)
   rom_portion = rom_data[0xC0000:0x100000]
   emulator.load_rom(rom_portion)
   emulator.reset()  # Must reset after loading to set PC from entry point
   ```

4. **API Endpoints**:
   - `GET /api/v1/state`: Returns screen (base64 PNG), registers, flags, instruction count
   - `POST /api/v1/control`: Commands: run, pause, step, reset
   - `POST /api/v1/key`: Keyboard input with press/release actions

### Running the Web Emulator

```bash
cd web
pip install -r requirements.txt
FORCE_BINJA_MOCK=1 python run.py
# Open http://localhost:8080
```

### Testing

The web emulator has comprehensive test coverage (51 tests):
```bash
cd web
FORCE_BINJA_MOCK=1 python run_tests.py
```

## Keyboard Implementation Details

The project uses a single keyboard handler implementation (`pce500/keyboard_handler.py`):
- **Column Selection**: KOL (bits 0–7) and KOH (bits 0–2) control columns KO0–KO10 (active-high)
- **Row Reading**: KIL returns row bits KI0–KI7 according to currently strobed columns
- **Debouncing**: Queue-based press/release debouncing with configurable read thresholds
- **Layout**: The layout mapping matches the PC‑E500 matrix and is used by the Web UI

### Virtual Keyboard Layout
The web UI virtual keyboard is split into two sections matching the physical PC-E500:
- **Left Section**: QWERTY keyboard with function keys
- **Right Section**: Scientific calculator layout
- **Special Keys**: Tall Enter key spanning two rows, OFF/ON key pair
- **Superscript Labels**: Keys show secondary functions where applicable

### Debugging Features
The implementation provides visibility for keyboard debugging:
- **Internal Register Watch**: KOL/KOH/KIL accesses tracked with PC addresses
- **Key Queue Display**: Queued keys with KOL/KOH/KIL and read progress
- **Keyboard Statistics**: Read counts and detection of stuck keys
- **Visual Feedback**: Progress bars and status indicators in the web UI
