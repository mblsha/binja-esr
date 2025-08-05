# Emulation Speed Display

The web interface now displays real-time emulation speed when the emulator is running.

## Features

- **Real-time Speed Calculation**: Shows instructions per second (ips) while emulator is running
- **Speed Ratio**: Displays percentage of 2MHz target CPU speed
- **Smart Formatting**: 
  - Under 1K: "500 ips"
  - 1K-1M: "1.5K ips"
  - Over 1M: "1.25M ips"
- **Automatic Updates**: Speed updates with same frequency as display (10 FPS or every 100K instructions)
- **Clean Display**: Shows "-" when emulator is paused or stopped

## Implementation Details

### Backend (app.py)
- Tracks instruction count and time between updates
- Calculates speed = instruction_delta / time_delta
- Compares to 2MHz target speed for percentage
- Clears speed data when paused/stopped

### Frontend (app.js)
- Formats speed with appropriate units (ips, K ips, M ips)
- Shows percentage of target speed
- Updates automatically during polling

### HTML (index.html)
- Added speed display below instruction count in Statistics section

## Example Display

When running:
```
Statistics
Instructions: 1,234,567
Speed: 1.23M ips (61.5%)
```

When paused:
```
Statistics
Instructions: 1,234,567
Speed: -
```