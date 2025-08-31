# PC‑E500 Emulator Performance Analysis Findings

## Executive Summary
Current performance limitations are primarily in the SC62015 CPU emulator’s
instruction decoding/execution pipeline (excessive memory reads and re‑decoding).
The project now uses a single keyboard implementation (compat); previous hardware
keyboard comparisons are historical and not relevant to current code paths.

## Key Findings

### 1. Excessive Memory Reads
- **Problem**: Each instruction execution causes 30+ memory reads
- **Example**: A single MV instruction at PC=0x0F10C2 causes 33 memory reads:
  - The opcode byte is read 7 times
  - Subsequent instruction bytes are read 6 times each
- **Impact**: 67x slowdown compared to expected performance

### 2. Memory Read Pattern Analysis
For the MV instruction at 0x0F10C2:
```
Address     Reads
0x0F10C2    7 reads (opcode)
0x0F10C3    6 reads (operand 1)
0x0F10C6    6 reads (operand 2) 
0x0F10C9    6 reads (operand 3)
0x0F10C4    2 reads
0x0F10C5    2 reads
0x0F10C7    2 reads
0x0F10C8    2 reads
Total:      33 reads
```

### 3. Performance Metrics (indicative)
- Example observed (pre‑optimization): ~29.6 instructions/second with heavy re‑reads
- Target (order of magnitude): ~2000+ instructions/second

### 4. Root Cause
The issue is in the SC62015 emulator's `execute_instruction` pipeline:
1. Instruction is decoded multiple times
2. Each decode reads the same bytes repeatedly
3. The FetchDecoder doesn't cache fetched bytes
4. Instruction lifting may be re-reading bytes

## Recommended Fixes

### Priority 1: Cache Instruction Bytes
- Add a small cache in FetchDecoder to avoid re-reading the same bytes
- Cache the last 8-16 bytes read to cover most instruction sizes

### Priority 2: Cache Decoded Instructions
- Cache recently decoded instructions by PC address
- Avoid re-decoding the same instruction multiple times

### Priority 3: Optimize Instruction Pipeline
- Decode once, execute once principle
- Pass decoded instruction data through the pipeline instead of re-fetching

## Keyboard
The emulator uses a single compat keyboard implementation with debounced press/release.
Keyboard code is not the performance bottleneck; focus should remain on CPU pipeline
optimizations listed above.

## Next Steps
1. Fix the instruction decoding/execution pipeline in SC62015 emulator
2. Implement byte caching in FetchDecoder
3. Re-test overall performance after CPU fixes
