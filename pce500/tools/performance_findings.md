# PC-E500 Emulator Performance Analysis Findings

## Executive Summary
The performance issue is NOT related to the keyboard implementation. Both `compat` and `hardware` keyboards have identical poor performance because the root cause is in the SC62015 CPU emulator's instruction decoding/execution pipeline.

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

### 3. Performance Metrics
- **Current**: ~29.6 instructions/second (with timeout after 10s)
- **Expected**: ~2000+ instructions/second
- **Slowdown**: 67x

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

## Keyboard Implementation Status
The hardware keyboard implementation itself is fine:
- Optimizations already implemented (caching, fast paths)
- Performance: ~5 million KIL reads/second when tested in isolation
- The issue was misattributed due to the CPU emulator bottleneck

## Next Steps
1. Fix the instruction decoding/execution pipeline in SC62015 emulator
2. Implement byte caching in FetchDecoder
3. Re-test both keyboard implementations after CPU fix
4. Hardware keyboard can become default once CPU performance is fixed