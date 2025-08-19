# PC-E500 Keyboard Matrix Analysis Guide

## Table of Contents
1. [Overview](#overview)
2. [Setting Up the Analysis Environment](#setting-up-the-analysis-environment)
3. [Generating Execution Traces](#generating-execution-traces)
4. [Understanding the Trace Format](#understanding-the-trace-format)
5. [Analyzing Keyboard Scanning Patterns](#analyzing-keyboard-scanning-patterns)
6. [Key Findings from Analysis](#key-findings-from-analysis)
7. [Debugging Methodology](#debugging-methodology)
8. [Advanced Analysis Techniques](#advanced-analysis-techniques)

## Overview

This document details the comprehensive analysis process used to understand the PC-E500's keyboard matrix scanning implementation and diagnose why keyboard input is not being detected in the emulator. The analysis revealed critical mismatches between the ROM's expectations and the emulated hardware behavior.

## Setting Up the Analysis Environment

### Prerequisites

1. **Install Dependencies**:
   ```bash
   # Install uv package manager if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync --extra dev --extra pce500
   ```

2. **Ensure ROM is Available**:
   The PC-E500 ROM file must be present at `data/pc-e500.bin` (1MB file). The emulator will load the last 256KB as the actual ROM.

3. **Set Environment Variable**:
   ```bash
   export FORCE_BINJA_MOCK=1  # Required to use mock Binary Ninja API
   ```

## Generating Execution Traces

### Basic Trace Generation

The disassembly trace feature was implemented to capture detailed execution flow with control flow annotations:

```bash
# Generate trace with default 20,000 instructions
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py --disasm-trace

# Generate shorter trace for quick analysis
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py --disasm-trace --steps 5000

# Generate trace with timeout protection
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py --disasm-trace --timeout-secs 10
```

### Output Files

Traces are saved to `data/execution_trace_YYYYMMDD_HHMMSS.txt` with:
- Header showing total instructions, unique PCs, and control flow edges
- Disassembled instructions with address, bytes, and mnemonic
- Control flow annotations (jumps, calls, returns)
- Register names for memory-mapped I/O operations

### Advanced Trace Options

```bash
# Skip boot sequence to reach specific code
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py \
    --disasm-trace \
    --boot-skip 10000 \    # Skip first 10k instructions
    --steps 5000            # Then trace 5k instructions

# Enable performance profiling alongside tracing
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py \
    --disasm-trace \
    --profile-emulator      # Generates emulator-profile.perfetto-trace
```

## Understanding the Trace Format

### Basic Instruction Format

```assembly
0x0F1119: 30 80 F2     MV    A, (KIL)    ; From: 0x0F1117
```

Components:
- `0x0F1119`: Program counter (PC) address
- `30 80 F2`: Raw instruction bytes
- `MV A, (KIL)`: Disassembled instruction with symbolic register names
- `; From: 0x0F1117`: Control flow annotation showing jump source

### Control Flow Annotations

```assembly
; Entry point                    - Marks ROM entry address
; From: 0xADDRESS               - Shows where jumps come from
; Calls: 0xADDRESS              - Shows call targets
; Returns to: 0xADDR1, 0xADDR2  - Shows multiple return addresses
; Jumps to: 0xADDRESS           - Shows jump destinations
```

### Basic Block Separation

Non-contiguous code blocks are separated by blank lines:

```assembly
0x0F0C43: 06           RET    ; Returns to: 0x0F119F

0x0F0C5C: 28           PUSHU A    ; From: 0x0F0C86
```

## Analyzing Keyboard Scanning Patterns

### Step 1: Identify Keyboard Register Operations

Search for keyboard register usage in the trace:

```bash
# Find all keyboard register operations
grep -E "KOL|KOH|KIL" data/execution_trace_*.txt

# Count occurrences
grep -c -E "KOL|KOH|KIL" data/execution_trace_*.txt

# Find with context
grep -B2 -A2 "KIL" data/execution_trace_*.txt
```

### Step 2: Analyze Scanning Sequences

Look for patterns in column selection and row reading:

```bash
# Find KOL/KOH writes (column selection)
grep -E "MV.*KOL|OR.*KOH|AND.*KOH" trace.txt

# Find KIL reads (row detection)
grep -E "MV.*KIL|CMP.*KIL|TEST.*KIL" trace.txt
```

### Step 3: Identify Critical Branches

Find conditional branches that depend on keyboard state:

```bash
# Find branches after KIL reads
grep -A5 "KIL" trace.txt | grep -E "JRZ|JRNZ|JRC|JRNC|JPZ|JPNZ"
```

### Step 4: Map the Scanning Algorithm

Document the complete scanning sequence:

1. **Column Setup Phase**
2. **Row Reading Phase**
3. **Result Processing Phase**
4. **Branch Decision Phase**

## Key Findings from Analysis

### 1. Keyboard Matrix Structure

The PC-E500 uses an 11×8 keyboard matrix:
- **Columns (KO0-KO10)**: 11 columns controlled by KOL (bits 0-7) and KOH (bits 0-2)
- **Rows (KI0-KI7)**: 8 rows read through KIL (bits 0-7)

### 2. ROM Scanning Pattern

```assembly
; Initial keyboard scan at boot
0x0F110B: MV  (KOL), 00    ; Disable columns 0-7
0x0F110F: OR  (KOH), 08    ; Enable column 11 (bit 3)
0x0F1113: AND (KOH), F8    ; Mask to keep only bit 3+
0x0F1119: MV  A, (KIL)     ; Read keyboard rows
0x0F1120: TEST A, 01       ; Test row 0
0x0F1122: JPNZ 100F        ; Jump if key at (11,0) pressed
```

### 3. Critical Mismatch Discovered

**ROM expects**: Key at column 11, row 0
**PF1 location**: Column 10, row 6
**Result**: PF1 press will never be detected by this scan

### 4. Debouncing Implementation

```assembly
; Key release detection with debouncing
0x0F1CF7: OR  (LCC), 04    ; Set timing flag
0x0F1CFB: CMP (KIL), 00    ; Check if all keys released
0x0F1CFF: JRNZ -06         ; Loop if keys still pressed
0x0F1D01: AND (LCC), FB    ; Clear timing flag
```

### 5. Logic Polarity Issue

- **ROM expects**: KIL = 0x00 when no keys pressed (active-high)
- **Hardware keyboard**: KIL = 0xFF when no keys pressed (active-low)
- **Impact**: Debouncing logic fails, branches are missed

## Debugging Methodology

### 1. Trace-Driven Analysis

```python
# Analyze control flow edges
edges = {}
for line in trace:
    if "From:" in line:
        dest = extract_pc(line)
        source = extract_from_address(line)
        edges[dest] = edges.get(dest, set()).add(source)

# Find never-taken branches
for pc, instruction in trace:
    if is_conditional_branch(instruction):
        if not branch_was_taken(pc, edges):
            print(f"Branch never taken at {pc}")
```

### 2. Register State Tracking

Monitor register values throughout execution:

```bash
# Track KOL/KOH state changes
awk '/KOL|KOH/ {print NR": "$0}' trace.txt

# Find register access patterns
grep -E "Reads:|Writes:" trace.txt | grep -E "KOL|KOH|KIL"
```

### 3. Execution Statistics Analysis

From the emulator output:
```
Keyboard reads: 25 last_cols=[] last KOL=0x00 KOH=0x00 strobe_writes=674
Column strobe histogram: KO0:0, KO1:0, KO2:0, ...
```

This shows keyboard activity but no successful key detection.

## Advanced Analysis Techniques

### 1. Differential Analysis

Compare traces with different keyboard states:

```bash
# Generate trace without key press
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py \
    --disasm-trace --steps 5000 \
    --trace-file trace_nokey.txt

# Generate trace with simulated key press
FORCE_BINJA_MOCK=1 uv run python pce500/run_pce500.py \
    --disasm-trace --steps 5000 \
    --auto-press-key KEY_F1 \
    --auto-press-after-pc 0x0F1119 \
    --trace-file trace_withkey.txt

# Compare the traces
diff trace_nokey.txt trace_withkey.txt
```

### 2. Branch Coverage Analysis

```python
def analyze_branch_coverage(trace_file):
    branches = {}
    taken = set()
    
    with open(trace_file) as f:
        for line in f:
            if any(op in line for op in ['JRZ', 'JRNZ', 'JRC', 'JRNC']):
                pc = extract_pc(line)
                branches[pc] = extract_target(line)
            if 'From:' in line:
                taken.add(extract_pc(line))
    
    coverage = len(taken) / len(branches) * 100
    print(f"Branch coverage: {coverage:.1f}%")
    
    for pc, target in branches.items():
        if target not in taken:
            print(f"Never taken: {pc} -> {target}")
```

### 3. Keyboard Matrix Visualization

```python
def visualize_scan_pattern(trace_file):
    matrix = [[' ' for _ in range(11)] for _ in range(8)]
    
    with open(trace_file) as f:
        for line in f:
            if 'KOL' in line or 'KOH' in line:
                kol, koh = extract_column_state(line)
                for col in range(11):
                    if is_column_active(col, kol, koh):
                        # Mark this column as scanned
                        for row in range(8):
                            matrix[row][col] = '█'
    
    # Print matrix
    print("Scanned positions:")
    print("    " + "".join(f"KO{i:1X}" for i in range(11)))
    for row in range(8):
        print(f"KI{row}: " + "".join(matrix[row]))
```

### 4. Timing Analysis

```bash
# Find NOP instructions (timing delays)
grep "NOP" trace.txt

# Find tight loops (potential timing loops)
grep -E "JRNZ.*-[0-9]+" trace.txt | head -20

# Identify delay patterns
awk '/DEC.*I/ {getline; if (/JRNZ.*-/) print NR": delay loop"}' trace.txt
```

## Troubleshooting Common Issues

### Issue: KIL Always Returns 0xFF

**Diagnosis**: Check active-low vs active-high logic
```bash
grep "KIL" trace.txt | grep -E "Reads:|Writes:"
```

**Solution**: Verify keyboard implementation logic polarity

### Issue: Branches Never Taken

**Diagnosis**: Find never-executed code blocks
```bash
# Extract all PC addresses from trace
awk '/^0x[0-9A-F]+:/ {print $1}' trace.txt | sort -u > executed.txt

# Compare with full ROM disassembly if available
```

**Solution**: Identify what conditions would make branches execute

### Issue: Missing Keyboard Initialization

**Diagnosis**: Check if keyboard handler is installed
```bash
grep "0x100F" trace.txt  # Expected keyboard handler address
```

**Solution**: Ensure initial keyboard detection succeeds

## Conclusion

The trace analysis methodology revealed that the PC-E500 keyboard emulation has fundamental mismatches with the ROM's expectations:

1. **Column/Row Mismatch**: ROM scans column 11, row 0 but PF1 is at column 10, row 6
2. **Logic Polarity**: ROM expects active-high but hardware uses active-low
3. **Missing Keys**: Column 11 has no keys mapped in current layout

These findings demonstrate the power of execution trace analysis for understanding complex hardware/software interactions and diagnosing emulation issues. The systematic approach of generating traces, analyzing patterns, and comparing expected vs actual behavior is essential for accurate emulation development.

## Next Steps

1. Verify the actual PC-E500 keyboard matrix layout from hardware documentation
2. Determine if column 11 is a special hardware signal rather than a keyboard column
3. Test with modified keyboard mappings to match ROM expectations
4. Consider implementing a "compatibility mode" that matches ROM scanning patterns