#!/usr/bin/env python3
"""
Perfetto Trace Comparator - Compare execution traces from two Perfetto files

This tool compares two Perfetto traces that should represent the same execution,
extracting events from the "Execution" timeline and comparing their debug annotations.

Usage:
    ./perfetto_trace_compare.py <trace1.perfetto> <trace2.perfetto> [--verbose]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import difflib

# Import perfetto protobuf definitions
sys.path.append(str(Path(__file__).parent.parent / "py"))
from retrobus_perfetto.proto import perfetto_pb2

class ExecutionEvent:
    """Represents a single execution event with its annotations"""
    def __init__(self, name: str, timestamp: int, annotations: Dict[str, str]):
        self.name = name
        self.timestamp = timestamp
        self.annotations = annotations
    
    def annotation_string(self) -> str:
        """Convert annotations to a deterministic string for comparison"""
        # Sort keys to ensure consistent ordering
        sorted_items = sorted(self.annotations.items())
        return "; ".join(f"{k}={v}" for k, v in sorted_items)
    
    def normalized_annotation_string(self) -> str:
        """Convert annotations to normalized string for comparison (handles hex format differences)"""
        normalized = {}
        for k, v in self.annotations.items():
            # Normalize hex values (remove 0x prefix and uppercase)
            if isinstance(v, str) and v.startswith("0x"):
                # Convert "0x0F10C2" to "F10C2" (remove leading zeros)
                normalized[k] = v[2:].upper().lstrip("0") or "0"
            else:
                normalized[k] = v
        sorted_items = sorted(normalized.items())
        return "; ".join(f"{k}={v}" for k, v in sorted_items)

class PerfettoTraceComparator:
    """Compares two Perfetto traces focusing on execution events"""
    
    def __init__(self, trace1_path: Path, trace2_path: Path):
        self.trace1_path = trace1_path
        self.trace2_path = trace2_path
        self.trace1_events: List[ExecutionEvent] = []
        self.trace2_events: List[ExecutionEvent] = []
    
    def load_trace(self, trace_path: Path) -> perfetto_pb2.Trace:
        """Load a Perfetto trace from file"""
        trace = perfetto_pb2.Trace()
        with open(trace_path, 'rb') as f:
            trace.ParseFromString(f.read())
        return trace
    
    def extract_execution_events(self, trace: perfetto_pb2.Trace) -> List[ExecutionEvent]:
        """Extract all Exec@ events from the Execution thread"""
        events = []
        
        # First, build a map of track UUIDs to track names
        track_names = {}
        for packet in trace.packet:
            if packet.HasField('track_descriptor'):
                track_names[packet.track_descriptor.uuid] = packet.track_descriptor.name
        
        # Find the Execution thread UUID
        execution_uuid = None
        
        # First try to find by name
        for uuid, name in track_names.items():
            if name == "Execution":
                execution_uuid = uuid
                break
        
        # If not found by name, try to find by event content
        if not execution_uuid:
            # Sample events from each track to find Exec@ events
            track_samples = {}
            for packet in trace.packet:
                if packet.HasField('track_event') and packet.track_event.HasField('name'):
                    uuid = packet.track_event.track_uuid
                    if uuid not in track_samples:
                        track_samples[uuid] = []
                    if len(track_samples[uuid]) < 10:
                        track_samples[uuid].append(packet.track_event.name)
            
            # Find track with Exec@ events
            for uuid, samples in track_samples.items():
                if any(name.startswith("Exec@") for name in samples):
                    execution_uuid = uuid
                    print(f"  Note: Found Exec@ events on track {uuid} (thread name: '{track_names.get(uuid, 'unnamed')}')")
                    break
        
        if not execution_uuid:
            raise ValueError("No 'Execution' thread or Exec@ events found in trace")
        
        # Extract events from the Execution thread
        for packet in trace.packet:
            if packet.HasField('track_event'):
                event = packet.track_event
                
                # Check if this event is on the Execution thread
                if event.track_uuid != execution_uuid:
                    continue
                
                # Check if this is an instant event (not slice begin/end)
                if event.type != perfetto_pb2.TrackEvent.TYPE_INSTANT:
                    continue
                
                # Get event name
                event_name = event.name if event.HasField('name') else ""
                
                # Filter for Exec@ events
                if not event_name.startswith("Exec@"):
                    continue
                
                # Extract annotations
                annotations = {}
                for annotation in event.debug_annotations:
                    # Handle different annotation value types
                    ann_name = annotation.name if annotation.HasField('name') else ""
                    
                    if annotation.HasField('string_value'):
                        ann_value = annotation.string_value
                    elif annotation.HasField('int_value'):
                        ann_value = str(annotation.int_value)
                    elif annotation.HasField('uint_value'):
                        ann_value = str(annotation.uint_value)
                    elif annotation.HasField('bool_value'):
                        ann_value = str(annotation.bool_value).lower()
                    elif annotation.HasField('double_value'):
                        ann_value = str(annotation.double_value)
                    elif annotation.HasField('pointer_value'):
                        ann_value = f"0x{annotation.pointer_value:X}"
                    else:
                        ann_value = "<unknown>"
                    
                    if ann_name:
                        annotations[ann_name] = ann_value
                
                events.append(ExecutionEvent(
                    name=event_name,
                    timestamp=packet.timestamp,
                    annotations=annotations
                ))
        
        # Sort by timestamp to ensure correct ordering
        events.sort(key=lambda e: e.timestamp)
        return events
    
    def compare_events(self) -> Iterator[Tuple[int, Optional[ExecutionEvent], Optional[ExecutionEvent], Optional[str]]]:
        """
        Compare events from both traces, yielding differences.
        
        Yields:
            (index, event1, event2, error_message)
        """
        max_len = max(len(self.trace1_events), len(self.trace2_events))
        
        for i in range(max_len):
            event1 = self.trace1_events[i] if i < len(self.trace1_events) else None
            event2 = self.trace2_events[i] if i < len(self.trace2_events) else None
            
            # Check for length mismatch
            if event1 is None:
                yield (i, event1, event2, "Trace 1 ended but Trace 2 has more events")
                return
            elif event2 is None:
                yield (i, event1, event2, "Trace 2 ended but Trace 1 has more events")
                return
            
            # Compare event names
            if event1.name != event2.name:
                yield (i, event1, event2, f"Event name mismatch: '{event1.name}' vs '{event2.name}'")
                return
            
            # Compare annotations
            ann_str1 = event1.annotation_string()
            ann_str2 = event2.annotation_string()
            
            if ann_str1 != ann_str2:
                yield (i, event1, event2, "Annotation mismatch")
                return
    
    def run_comparison(self, verbose: bool = False) -> bool:
        """
        Run the full comparison process.
        
        Returns:
            True if traces match, False otherwise
        """
        print(f"Loading trace 1: {self.trace1_path}")
        trace1 = self.load_trace(self.trace1_path)
        self.trace1_events = self.extract_execution_events(trace1)
        print(f"  Found {len(self.trace1_events)} Exec@ events")
        
        print(f"Loading trace 2: {self.trace2_path}")
        trace2 = self.load_trace(self.trace2_path)
        self.trace2_events = self.extract_execution_events(trace2)
        print(f"  Found {len(self.trace2_events)} Exec@ events")
        
        print("\nComparing traces...")
        
        # Compare events
        for i, event1, event2, error in self.compare_events():
            if error:
                print(f"\n❌ Difference found at event #{i}:")
                print(f"   Error: {error}")
                
                # Show execution flow context (up to 5 previous PCs)
                print("\n   Execution flow context:")
                context_size = min(5, i)
                for j in range(i - context_size, i + 1):
                    if j >= 0 and j < len(self.trace1_events):
                        pc1 = self.trace1_events[j].annotations.get('pc', 'N/A')
                        pc2 = self.trace2_events[j].annotations.get('pc', 'N/A') if j < len(self.trace2_events) else 'N/A'
                        op1 = self.trace1_events[j].annotations.get('opcode', 'N/A')
                        op2 = self.trace2_events[j].annotations.get('opcode', 'N/A') if j < len(self.trace2_events) else 'N/A'
                        marker = " -> " if j == i else "    "
                        # Show PC and opcode, marking differences
                        pc_match = pc1 == pc2
                        op_match = op1 == op2
                        print(f"   {marker}Event #{j}: PC={pc1} opcode={op1} {'✓' if pc_match and op_match else '✗' if not pc_match or not op_match else ''}")
                
                if event1:
                    print(f"\n   Trace 1 - {event1.name}:")
                    print(f"   {event1.annotation_string()}")
                
                if event2:
                    print(f"\n   Trace 2 - {event2.name}:")
                    print(f"   {event2.annotation_string()}")
                
                # Show detailed diff if annotation mismatch
                if "Annotation mismatch" in error and event1 and event2:
                    print("\n   Detailed differences:")
                    lines1 = event1.annotation_string().split("; ")
                    lines2 = event2.annotation_string().split("; ")
                    
                    for line in difflib.unified_diff(lines1, lines2, fromfile="Trace 1", tofile="Trace 2", lineterm=""):
                        print(f"   {line}")
                
                return False
            
            if verbose and i % 1000 == 0:
                print(f"  Compared {i} events...")
        
        print(f"\n✅ Traces match! ({len(self.trace1_events)} events compared)")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Compare two Perfetto traces for execution differences"
    )
    parser.add_argument("trace1", type=Path, help="First Perfetto trace file")
    parser.add_argument("trace2", type=Path, help="Second Perfetto trace file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress during comparison")
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.trace1.exists():
        print(f"Error: Trace file not found: {args.trace1}")
        sys.exit(1)
    
    if not args.trace2.exists():
        print(f"Error: Trace file not found: {args.trace2}")
        sys.exit(1)
    
    # Run comparison
    comparator = PerfettoTraceComparator(args.trace1, args.trace2)
    
    try:
        matches = comparator.run_comparison(args.verbose)
        sys.exit(0 if matches else 1)
    except Exception as e:
        print(f"\nError during comparison: {e}")
        sys.exit(2)

if __name__ == '__main__':
    main()