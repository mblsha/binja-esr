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
- 0x00000-0x3FFFF: (Unassigned)
- 0x40000-0x4FFFF: Memory card area
  - 64KB card: 0x40000-0x4FFFF
  - 32KB card: 0x44000-0x4BFFF
  - 16KB card: 0x48000-0x4BFFF
  - 8KB card: 0x48000-0x49FFF
- 0x50000-0xB7FFF: Extension area (for future expansion)
- 0xB8000-0xBFFFF: Internal RAM (32KB)
  - 0xB8000-0xBEBFF: User area
  - 0xBEC00-0xBFE33: Machine code area (0x1234 bytes)
  - 0xBFC00-0xBFFFF: Work area (reserved)
- 0xC0000-0xFFFFF: Internal ROM (256KB)
- 0x2xxxx: LCD controllers (memory-mapped I/O, address-decoded)
  - Main LCD: HD61202U dual-chip (128x64)
  - Sub LCD: HD61700

## Configuration

Create custom machine configurations:

```python
from pce500.config import MachineConfig

config = MachineConfig.for_model("PC-E500S")  # 64KB RAM variant
config.save("my_config.json")
```