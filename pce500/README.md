# PC-E500 Architecture Documentation

## Memory Map

The PC-E500 uses a 20-bit address space (1MB total) with the following memory regions:

### Memory Region Details

- **Internal ROM** (0xC0000 - 0xFFFFF): 256KB, contains system firmware
- **Internal RAM** (0xB8000 - 0xBFFFF): 32KB, user and system memory
- **Memory Card Slot** (0x40000 - 0x4FFFF): Starting at 0x40000, supports up to 64KB
- **LCD Controllers** (0x2xxxx): Memory-mapped I/O for display control

## LCD Display System

The PC-E500 features a single 240×32 pixel monochrome LCD display controlled by two HD61202 column driver chips working together at addresses 0x2xxxx.

### LCD Controller Configuration

The display uses two Hitachi HD61202 chips accessed through memory-mapped I/O. The exact division of display area between the two chips needs to be determined.

### Memory-Mapped I/O

Commands to the LCD controllers are interpreted based on the memory address accessed by the CPU in the 0x2xxxx range.

#### Hardware Control Signal Mapping

| Controller Pin | CPU Connection | Function |
|---------------|----------------|----------|
| | **Left HD61202** | **Right HD61202** | |
| DB0-7 | DIO0-7 | DIO0-7 | Data bus, three-state I/O common terminal |
| E (Enable) | CE5 | CE5 | At write (R/W = low): Data latched at falling edge of E<br>At read (R/W = high): Data appears while E is high |
| R/W (Read/Write) | A0 | A0 | R/W = High: Data read mode<br>R/W = Low: Data write mode |
| D/I (Data/Instruction) | A1 | A1 | D/I = High: DB0-7 is display data<br>D/I = Low: DB0-7 is display control data |
| CS1 | A3 | A2 | Active Low |
| CS2 | A12 | A12 | Active Low |
| CS3 | A13 | A13 | Active High |

#### Address Decoding

The CPU address bits control LCD operations:
- **A0**: R/W control (0 = write, 1 = read)
- **A1**: D/I control (0 = instruction, 1 = data)
- **A3:A2**: Chip selection
  - `00`: Broadcast to both chips
  - `01`: Right chip only
  - `10`: Left chip only
  - `11`: No chips selected
- **A12**: CS2 (active low)
- **A13**: CS3 (active high) - must be 1 for valid LCD access

#### Access Examples

Write command to left chip:
- Address: 0x2xxxx with A13=1, A3:A2=10, A1=0, A0=0

Write data to both chips:
- Address: 0x2xxxx with A13=1, A3:A2=00, A1=1, A0=0

### Display Organization

The 240×32 pixel display is managed by two HD61202 chips. The specific arrangement of how these chips divide the display area (e.g., left/right split, top/bottom split, or interleaved) requires further investigation.

## CPU Information

The PC-E500 uses the SC62015 (ESR-L) processor. For detailed CPU information, see the [pysc62015 documentation](../sc62015/pysc62015/).

### System Vectors

The SC62015 processor uses specific memory locations for system initialization:

- **Interrupt Vector**: Located at 0xFFFFA (3 bytes, little-endian)
- **Entry Point**: Located at 0xFFFFD (3 bytes, little-endian)

On reset, the CPU reads the entry point address from 0xFFFFD and begins execution there. The interrupt vector at 0xFFFFA is called when interrupts occur.
