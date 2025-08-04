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

### ROM Loading

The `pc-e500.bin` file is a 1MB memory dump. When loading:
- Extract bytes from offset 0xC0000 to 0x100000 (256KB) - this is the actual ROM
- The ROM is loaded at address 0xC0000 in the emulator's memory space
- Entry point 0xFFFFD typically contains 0xC2 0x10 0x0F (0x0F10C2 in little-endian)

## Web-Based PC-E500 Emulator

The web emulator (`web/`) provides a browser-based interface to the PC-E500 emulator:

### Architecture
- **Backend**: Flask server (`app.py`) managing emulator state and providing REST API
- **Frontend**: JavaScript SPA with virtual keyboard and real-time display updates
- **Keyboard**: Matrix-based keyboard emulation via KOL/KOH/KIL registers

### Key Implementation Details

1. **State Updates**: Triggered by either:
   - 100ms elapsed (10 FPS target)
   - 100,000 instructions executed

2. **Keyboard Matrix**: 
   - Output registers KOL (0xF0) and KOH (0xF1) select keyboard columns (KO0-KO10)
   - Input register KIL (0xF2) reads row states (KI0-KI7)
   - Keys mapped to specific column/row intersections
   - Hardware matrix correctly implemented with columns as outputs, rows as inputs

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

### Hardware-Accurate Keyboard Matrix
The PC-E500 keyboard uses a matrix scanning system that has been accurately implemented:
- **Column Selection**: KOL (bits 0-7) and KOH (bits 0-2) control which columns are active
- **Row Reading**: KIL reads the state of all 8 rows simultaneously
- **Active Low**: Pressed keys pull their row bits low when their column is selected
- **Visual Layout**: The keyboard matrix in `pce500/keyboard.py` visually matches the hardware

### Key Queue with Debouncing
The keyboard implementation includes realistic key debouncing:
- Each key press is queued with a target read count (default 10 reads)
- Keys must be read the target number of times before being considered "pressed"
- Stuck key detection identifies keys not being read for >1 second
- Queue visualization in web UI shows real-time key state and progress

### Virtual Keyboard Layout
The web UI virtual keyboard is split into two sections matching the physical PC-E500:
- **Left Section**: QWERTY keyboard with function keys
- **Right Section**: Scientific calculator layout
- **Special Keys**: Tall Enter key spanning two rows, OFF/ON key pair
- **Superscript Labels**: Keys show secondary functions where applicable

### Debugging Features
The implementation provides excellent visibility for debugging:
- **Internal Register Watch**: KOL/KOH/KIL registers now tracked with PC addresses
- **Key Queue Display**: Shows queued keys with their KOL/KOH/KIL values and read progress
- **Keyboard Statistics**: Tracks read counts and identifies stuck keys
- **Visual Feedback**: Progress bars and status indicators in the web UI