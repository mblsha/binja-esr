"""Optional Perfetto tracing support for PC-E500 emulator."""

import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# Try to import retrobus-perfetto from submodule
TRACING_AVAILABLE = False
try:
    # Add submodule to path
    retrobus_path = Path(__file__).parent.parent / "third_party" / "retrobus-perfetto" / "py"
    if retrobus_path.exists():
        sys.path.insert(0, str(retrobus_path))
        from retrobus_perfetto import PerfettoTraceBuilder
        TRACING_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING or TRACING_AVAILABLE:
    from retrobus_perfetto import PerfettoTraceBuilder


class EmulatorTracer:
    """Handles Perfetto trace generation for the PC-E500 emulator."""
    
    def __init__(self, emulator_name: str = "PC-E500"):
        self.enabled = False
        self.builder: Optional['PerfettoTraceBuilder'] = None
        self.cpu_thread = None
        self.io_thread = None
        self.memory_thread = None
        
        if TRACING_AVAILABLE:
            self.builder = PerfettoTraceBuilder(emulator_name)
            self.cpu_thread = self.builder.add_thread("CPU")
            self.io_thread = self.builder.add_thread("I/O") 
            self.memory_thread = self.builder.add_thread("Memory")
    
    def is_available(self) -> bool:
        """Check if tracing is available."""
        return TRACING_AVAILABLE
    
    def enable(self) -> bool:
        """Enable tracing if available."""
        if TRACING_AVAILABLE:
            self.enabled = True
            return True
        return False
    
    def disable(self) -> None:
        """Disable tracing."""
        self.enabled = False
    
    def trace_instruction(self, pc: int, opcode: int, mnemonic: str, 
                         timestamp_ns: int, duration_ns: Optional[int] = None) -> None:
        """Trace a CPU instruction execution."""
        if not self.enabled or not self.builder:
            return
            
        if duration_ns:
            # Duration event
            event = self.builder.begin_slice(self.cpu_thread, mnemonic, timestamp=timestamp_ns)
            event.add_annotations({
                "pc": pc,
                "opcode": opcode,
                "mnemonic": mnemonic
            })
            self.builder.end_slice(self.cpu_thread, timestamp=timestamp_ns + duration_ns)
        else:
            # Instant event
            event = self.builder.add_instant_event(self.cpu_thread, mnemonic, timestamp=timestamp_ns)
            event.add_annotations({
                "pc": pc,
                "opcode": opcode
            })
    
    def trace_memory_access(self, address: int, value: int, is_write: bool, 
                           timestamp_ns: int) -> None:
        """Trace memory read/write operations."""
        if not self.enabled or not self.builder:
            return
            
        event_name = "Memory Write" if is_write else "Memory Read"
        event = self.builder.add_instant_event(self.memory_thread, event_name, 
                                             timestamp=timestamp_ns)
        event.add_annotations({
            "address": address,
            "value": value,
            "operation": "write" if is_write else "read"
        })
    
    def trace_io_operation(self, port: int, value: int, is_write: bool,
                          timestamp_ns: int) -> None:
        """Trace I/O port operations."""
        if not self.enabled or not self.builder:
            return
            
        event_name = "Port Write" if is_write else "Port Read"
        event = self.builder.add_instant_event(self.io_thread, event_name,
                                             timestamp=timestamp_ns)
        event.add_annotations({
            "port": port,
            "value": value,
            "operation": "write" if is_write else "read"
        })
    
    def add_counter(self, name: str, value: int, timestamp_ns: int) -> None:
        """Add a counter value (e.g., cycle count, memory usage)."""
        if not self.enabled or not self.builder:
            return
            
        if not hasattr(self, f'_counter_{name}'):
            setattr(self, f'_counter_{name}', self.builder.add_counter(name))
        
        counter = getattr(self, f'_counter_{name}')
        self.builder.add_counter_sample(counter, value, timestamp=timestamp_ns)
    
    def save(self, filename: str) -> bool:
        """Save the trace to a file."""
        if not self.builder:
            return False
            
        try:
            self.builder.save(filename)
            return True
        except Exception:
            return False


# Stub class for when tracing is not available
class StubTracer:
    """Stub tracer that does nothing when retrobus-perfetto is not available."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def is_available(self) -> bool:
        return False
    
    def enable(self) -> bool:
        return False
    
    def disable(self) -> None:
        pass
    
    def trace_instruction(self, *args, **kwargs) -> None:
        pass
    
    def trace_memory_access(self, *args, **kwargs) -> None:
        pass
    
    def trace_io_operation(self, *args, **kwargs) -> None:
        pass
    
    def add_counter(self, *args, **kwargs) -> None:
        pass
    
    def save(self, *args, **kwargs) -> bool:
        return False


# Export the appropriate tracer
Tracer = EmulatorTracer if TRACING_AVAILABLE else StubTracer