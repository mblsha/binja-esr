# PC-E500 Emulator

A Sharp PC-E500 emulator built on the SC62015 CPU implementation.

## Features

- Full memory mapping with support for 256KB ROM files
- Dual LCD controller emulation (main 160x64 and sub display)
- Debug image rendering of display state
- Configurable memory maps for different PC-E500 variants
- Memory inspection and debugging tools

## Installation

```bash
cd pce500
pip install -e .
```

## Usage

```python
from pce500 import PCE500Emulator
from pce500.debug import DisplayRenderer

# Create and configure emulator
emu = PCE500Emulator()
emu.load_rom("path/to/rom.bin")
emu.reset()

# Run emulation
emu.run(max_cycles=1000)

# Render display to image
renderer = DisplayRenderer()
renderer.save_display(emu.machine.main_lcd, "display.png")
```

See `examples/run_pce500.py` for a complete example.

## Memory Map

Default PC-E500 memory configuration:
- 0x000000-0x001FFF: Internal RAM (8KB)
- 0x007000-0x0077FF: Main LCD controller
- 0x007800-0x0078FF: Sub LCD controller  
- 0x008000-0x00FFFF: External RAM (32KB, expandable)
- 0x040000-0x07FFFF: System ROM (256KB)

## Configuration

Create custom machine configurations:

```python
from pce500.config import MachineConfig

config = MachineConfig.for_model("PC-E500S")  # 64KB RAM variant
config.save("my_config.json")
```