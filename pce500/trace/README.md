# PC-E500 Emulator Tracing

This directory contains optional Perfetto tracing support for the PC-E500 emulator using [retrobus-perfetto](https://github.com/mblsha/retrobus-perfetto).

## Overview

The tracing module allows you to record detailed execution traces of the emulator that can be visualized in the [Perfetto UI](https://ui.perfetto.dev). This is useful for:

- Performance analysis and optimization
- Debugging emulation issues
- Understanding program execution flow
- Identifying hotspots and bottlenecks

## Setup

Tracing is **optional** and the emulator works without it. To enable tracing:

1. Initialize the git submodule (if not already done):
   ```bash
   git submodule init
   git submodule update
   ```

2. Run the setup script:
   ```bash
   cd pce500
   python setup_tracing.py
   ```

   This will install retrobus-perfetto with its dependencies (protobuf).

## Usage

### Basic Usage

```python
from pce500.emulator import PCE500Emulator
from pce500.trace import Tracer

# Create emulator with tracer
emu = PCE500Emulator()
tracer = Tracer("PC-E500")

# Check if tracing is available
if tracer.is_available():
    tracer.enable()
    print("Tracing enabled")
else:
    print("Tracing not available")

# Run emulation (traces will be recorded automatically if integrated)
# ...

# Save trace
if tracer.enabled:
    tracer.save("my_trace.perfetto-trace")
```

### Integration Pattern

To integrate tracing into the emulator, you can:

1. **Direct Integration**: Modify the emulator class to include tracing calls
2. **Decorator Pattern**: Wrap the emulator with tracing functionality
3. **Inheritance**: Create a traced emulator subclass

See `examples/trace_integration_example.py` for complete examples.

## Trace Events

The tracer supports several event types:

- **Instruction Execution**: CPU instructions with PC, opcode, and timing
- **Memory Access**: Read/write operations with addresses and values  
- **I/O Operations**: Port reads/writes
- **Counters**: Cycle counts, performance metrics

## Viewing Traces

1. Generate a trace file (`.perfetto-trace` extension)
2. Open https://ui.perfetto.dev in Chrome/Edge
3. Click "Open trace file" and select your trace
4. Use the UI to explore execution:
   - Timeline view shows threads and events
   - Selection details show annotations
   - SQL queries for analysis

## Implementation Details

The trace module uses dynamic imports to keep retrobus-perfetto optional:

- If available: Full `EmulatorTracer` class with all functionality
- If not available: `StubTracer` class that safely does nothing
- No exceptions or errors when retrobus-perfetto is missing

This ensures the Binary Ninja plugin continues to work without any tracing dependencies.

## Limitations

- Tracing adds overhead to emulation performance
- Large traces can consume significant disk space
- Not all emulator events are traced by default (extend as needed)