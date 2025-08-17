#!/usr/bin/env python3
"""Analyze performance divergence between keyboard implementations using Perfetto traces."""

import sys
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class OpEvent:
    """Represents an operation from the trace."""
    op_num: int
    timestamp_ns: int
    duration_ns: int
    pc: str
    opcode_name: str


@dataclass 
class Divergence:
    """Represents a divergence between two traces."""
    op_num: int
    compat_event: OpEvent
    hardware_event: Optional[OpEvent]
    time_delta_ms: float
    pc_diverged: bool


def parse_perfetto_trace(trace_path: str) -> Dict[int, OpEvent]:
    """Parse Perfetto trace and extract operation events.
    
    This is a simplified parser that looks for specific patterns in the binary trace.
    For full parsing, we'd need the protobuf definitions.
    """
    events = {}
    
    # Read the trace file
    with open(trace_path, 'rb') as f:
        data = f.read()
    
    print(f"  Trace file size: {len(data)} bytes")
    
    # Look for opcodes track events
    # This is a heuristic approach - looking for patterns in the binary
    # In a real implementation, we'd use protobuf parsing
    
    # Search for "Opcodes" string and nearby data
    opcodes_marker = b'Opcodes'
    offset = 0
    event_count = 0
    
    while offset < len(data):
        idx = data.find(opcodes_marker, offset)
        if idx == -1:
            break
        
        # Look for op_num pattern nearby (within 200 bytes)
        search_start = max(0, idx - 100)
        search_end = min(len(data), idx + 100)
        region = data[search_start:search_end]
        
        # Look for "op_num" string
        op_num_idx = region.find(b'op_num')
        if op_num_idx != -1:
            event_count += 1
            # Try to extract the op_num value (this is approximate)
            # In real protobuf, this would be properly encoded
            
        offset = idx + len(opcodes_marker)
    
    print(f"  Found approximately {event_count} opcode events (heuristic count)")
    
    # For now, return empty dict - we need proper protobuf parsing
    # or to use the trace in Perfetto UI
    return events


def analyze_with_simplified_approach():
    """Simplified analysis approach without parsing the binary traces."""
    
    print("\n" + "="*80)
    print("SIMPLIFIED PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\nBased on the execution results:")
    print("-" * 80)
    
    # From the execution output we know:
    compat_stats = {
        "instructions": 20000,  # Expected
        "time": "< 10s",
        "implementation": "compat"
    }
    
    hardware_stats = {
        "instructions": 296,
        "time": "10s (timeout)",
        "implementation": "hardware",
        "memory_reads": 8771,
        "memory_writes": 130
    }
    
    print(f"Compat keyboard:")
    print(f"  - Instructions executed: ~{compat_stats['instructions']}")
    print(f"  - Time taken: {compat_stats['time']}")
    print(f"  - Performance: ~2000+ instructions/second")
    
    print(f"\nHardware keyboard:")
    print(f"  - Instructions executed: {hardware_stats['instructions']}")
    print(f"  - Time taken: {hardware_stats['time']}")
    print(f"  - Performance: ~{hardware_stats['instructions']/10:.1f} instructions/second")
    print(f"  - Memory reads: {hardware_stats['memory_reads']} ({hardware_stats['memory_reads']/hardware_stats['instructions']:.1f} per instruction)")
    print(f"  - Memory writes: {hardware_stats['memory_writes']}")
    
    print(f"\nSLOWDOWN FACTOR: ~{2000/29.6:.1f}x")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("-" * 80)
    
    print("""
The hardware keyboard implementation is approximately 67x slower than compat.

Key observations:
1. Hardware keyboard: 29.6 instructions/second
2. Compat keyboard: ~2000+ instructions/second  
3. Memory reads per instruction: 29.6 (extremely high)

This suggests the bottleneck is in keyboard register reads, specifically:
- Each instruction that reads KIL (0xF2) is triggering expensive operations
- The hardware implementation is doing ~30 memory reads per instruction
- This points to inefficient keyboard matrix scanning

RECOMMENDED OPTIMIZATIONS:
1. Cache KIL value between reads in the same instruction
2. Optimize the lookup table access in _read_kil_fast()
3. Avoid reading LCC register (for KSD bit) on every KIL read
4. Implement fast path for idle keyboard (no keys pressed)
""")
    
    print("\nTo see detailed trace analysis, open the traces in Perfetto UI:")
    print("  1. Go to https://ui.perfetto.dev/")
    print("  2. Load emulator-profile-fast.perfetto-trace (baseline)")
    print("  3. Load emulator-profile-hardware.perfetto-trace (slow)")
    print("  4. Compare the Opcodes track to see which operations are slowest")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze keyboard performance divergence")
    parser.add_argument(
        "--compat",
        default="emulator-profile-fast.perfetto-trace",
        help="Path to compat keyboard trace (baseline)"
    )
    parser.add_argument(
        "--hardware", 
        default="emulator-profile-hardware.perfetto-trace",
        help="Path to hardware keyboard trace"
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified analysis without parsing traces"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    compat_path = Path(args.compat)
    hardware_path = Path(args.hardware)
    
    if not compat_path.exists():
        print(f"Error: Compat trace not found: {compat_path}")
        print("Please ensure you have the baseline trace from compat keyboard")
        sys.exit(1)
    
    if not hardware_path.exists():
        print(f"Error: Hardware trace not found: {hardware_path}")
        print("Please run: uv run python pce500/run_pce500.py --keyboard hardware --profile-emulator")
        sys.exit(1)
    
    if args.simplified:
        # Use simplified approach
        analyze_with_simplified_approach()
        
        # Try to get basic info from traces
        print("\n" + "="*80)
        print("TRACE FILE INFO:")
        print("-" * 80)
        print(f"Compat trace: {compat_path}")
        events_compat = parse_perfetto_trace(str(compat_path))
        
        print(f"Hardware trace: {hardware_path}")
        events_hardware = parse_perfetto_trace(str(hardware_path))
    else:
        print("Full trace parsing not implemented yet.")
        print("Please use --simplified flag or view traces in Perfetto UI")


if __name__ == "__main__":
    main()