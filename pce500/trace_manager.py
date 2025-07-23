"""Perfetto trace manager for PC-E500 emulator using retrobus-perfetto.

This module provides a singleton TraceManager class that handles trace collection
for CPU execution, peripherals, and memory operations with Perfetto format output.
"""

import collections
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Union

from retrobus_perfetto import PerfettoTraceBuilder

# Import configuration
try:
    from .tracing_config import TracingConfig
except ImportError:
    # Fallback if config module not available
    class TracingConfig:
        @staticmethod
        def is_enabled():
            return True

# Configuration flag to enable/disable tracing at compile time
ENABLE_PERFETTO_TRACING = TracingConfig.is_enabled()


class TraceEventType(Enum):
    """Types of trace events."""
    SLICE_BEGIN = "B"
    SLICE_END = "E"
    INSTANT = "I"
    COUNTER = "C"


@dataclass
class CallStackFrame:
    """Represents a frame in the call stack."""
    pc: int
    caller_pc: int
    name: Optional[str] = None
    timestamp: int = 0
    event_sent: bool = False


class TraceManager:
    """Singleton manager for Perfetto tracing using retrobus-perfetto."""
    
    _instance: Optional['TraceManager'] = None
    _lock = threading.Lock()
    
    # Configuration
    MAX_CALL_DEPTH = 50  # Maximum call stack depth to track
    STALE_FRAME_TIMEOUT = 1_000_000_000  # 1 second in nanoseconds
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._trace_builder: Optional[PerfettoTraceBuilder] = None
        self._trace_file: Optional[Path] = None
        self._start_time = time.perf_counter()
        self._tracing_enabled = False
        
        # Track mapping for different components
        self._track_uuids: Dict[str, int] = {}
        self._counter_tracks: Dict[str, int] = {}
        
        # Call stack tracking for CPU
        self._call_stacks: Dict[str, Deque[CallStackFrame]] = {
            "CPU": collections.deque(maxlen=self.MAX_CALL_DEPTH)
        }
        
        # For thread safety
        self._rlock = threading.RLock()
        
    def start_tracing(self, output_path: Union[str, Path]) -> bool:
        """Start tracing to a file."""
        if not ENABLE_PERFETTO_TRACING:
            return False
            
        with self._rlock:
            if self._tracing_enabled:
                return False
                
            try:
                output_path = Path(output_path)
                
                # Create trace builder
                self._trace_builder = PerfettoTraceBuilder("PC-E500 Emulator")
                
                # Create thread tracks
                self._track_uuids["CPU"] = self._trace_builder.add_thread("CPU")
                self._track_uuids["Memory"] = self._trace_builder.add_thread("Memory")
                self._track_uuids["I/O"] = self._trace_builder.add_thread("I/O")
                self._track_uuids["Display"] = self._trace_builder.add_thread("Display")
                self._track_uuids["Interrupt"] = self._trace_builder.add_thread("Interrupt")
                
                # Create counter tracks
                self._counter_tracks["instructions"] = self._trace_builder.add_counter_track(
                    "Instructions", "count"
                )
                self._counter_tracks["call_depth"] = self._trace_builder.add_counter_track(
                    "Call Depth", "level"
                )
                self._counter_tracks["memory_reads"] = self._trace_builder.add_counter_track(
                    "Memory Reads", "ops/s"
                )
                self._counter_tracks["memory_writes"] = self._trace_builder.add_counter_track(
                    "Memory Writes", "ops/s"
                )
                
                self._trace_file = output_path
                self._tracing_enabled = True
                self._start_time = time.perf_counter()
                
                return True
                
            except Exception as e:
                raise RuntimeError(f"Failed to start tracing: {e}") from e
    
    def stop_tracing(self) -> bool:
        """Stop tracing and save the file."""
        if not self._tracing_enabled or not self._trace_builder:
            return False
            
        with self._rlock:
            try:
                # Clean up any pending call stack frames
                for thread, stack in self._call_stacks.items():
                    while stack:
                        frame = stack.pop()
                        if frame.event_sent and thread in self._track_uuids:
                            self._trace_builder.end_slice(
                                self._track_uuids[thread],
                                self._get_timestamp()
                            )
                
                # Save the trace
                if self._trace_file:
                    self._trace_builder.save(str(self._trace_file))
                
                # Reset state
                self._tracing_enabled = False
                self._trace_builder = None
                self._trace_file = None
                self._track_uuids.clear()
                self._counter_tracks.clear()
                
                return True
                
            except Exception as e:
                raise RuntimeError(f"Failed to stop tracing: {e}") from e
    
    def is_tracing(self) -> bool:
        """Check if tracing is currently active."""
        return self._tracing_enabled
    
    def get_call_depth(self, thread_id: int) -> int:
        """Get the current call stack depth for a thread."""
        if not self._tracing_enabled:
            return 0
        thread = str(thread_id)
        if thread in self._call_stacks:
            return len(self._call_stacks[thread])
        return 0
    
    def get_timestamp(self) -> int:
        """Get current timestamp in nanoseconds."""
        if not self._tracing_enabled:
            return 0
        return self._get_timestamp()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in nanoseconds."""
        if not self._tracing_enabled:
            return 0
        return int((time.perf_counter() - self._start_time) * 1_000_000_000)
    
    def begin_function(self, thread: str, pc: int, caller_pc: int, name: Optional[str] = None) -> None:
        """Begin a function duration event."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        with self._rlock:
            # Clean up old frames
            self._cleanup_stale_frames(thread)
            
            # Create new frame
            frame = CallStackFrame(
                pc=pc,
                caller_pc=caller_pc,
                name=name or f"func_0x{pc:06X}",
                timestamp=self._get_timestamp(),
                event_sent=True
            )
            
            # Add to call stack
            if thread not in self._call_stacks:
                self._call_stacks[thread] = collections.deque(maxlen=self.MAX_CALL_DEPTH)
            self._call_stacks[thread].append(frame)
            
            # Send trace event
            event = self._trace_builder.begin_slice(
                self._track_uuids[thread],
                frame.name,
                frame.timestamp
            )
            
            # Add annotations
            event.add_annotations({
                "pc": f"0x{pc:06X}",
                "caller": f"0x{caller_pc:06X}"
            })
    
    def end_function(self, thread: str, pc: int) -> None:
        """End a function duration event."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        with self._rlock:
            stack = self._call_stacks.get(thread)
            if not stack:
                return
                
            # Try to find matching frame
            for i in range(len(stack) - 1, -1, -1):
                frame = stack[i]
                if frame.pc == pc and frame.event_sent:
                    # Remove this frame and all frames above it
                    while len(stack) > i:
                        popped = stack.pop()
                        if popped.event_sent:
                            self._trace_builder.end_slice(
                                self._track_uuids[thread],
                                self._get_timestamp()
                            )
                    return
            
            # If no matching frame found, just pop the top
            if stack and stack[-1].event_sent:
                stack.pop()
                self._trace_builder.end_slice(
                    self._track_uuids[thread],
                    self._get_timestamp()
                )
    
    def trace_instant(self, thread: str, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """Add an instant event."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        with self._rlock:
            event = self._trace_builder.add_instant_event(
                self._track_uuids[thread],
                name,
                self._get_timestamp()
            )
            
            if args:
                event.add_annotations(args)
    
    def trace_counter(self, thread: str, name: str, value: float) -> None:
        """Add a counter event.
        
        Args:
            thread: Thread name (for compatibility, not used in counter tracks)
            name: Counter name (must match a created counter track)
            value: Counter value
        """
        if not self.is_tracing() or name not in self._counter_tracks:
            return
            
        with self._rlock:
            self._trace_builder.update_counter(
                self._counter_tracks[name],
                value,
                self._get_timestamp()
            )
    
    def begin_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        """Begin a flow to connect events across threads."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        with self._rlock:
            self._trace_builder.add_flow(
                self._track_uuids[thread],
                name,
                self._get_timestamp(),
                flow_id,
                terminating=False
            )
    
    def end_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        """End a flow."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        with self._rlock:
            self._trace_builder.add_flow(
                self._track_uuids[thread],
                name,
                self._get_timestamp(),
                flow_id,
                terminating=True
            )
    
    def trace_memory_access(self, thread: str, address: int, size: int, 
                          is_write: bool, value: Optional[int] = None) -> None:
        """Trace memory access operations."""
        if not self.is_tracing() or thread not in self._track_uuids:
            return
            
        name = "Write" if is_write else "Read"
        args = {
            "address": f"0x{address:06X}",
            "size": size
        }
        
        if value is not None:
            args["value"] = f"0x{value:02X}" if size == 1 else f"0x{value:04X}"
            
        self.trace_instant(thread, f"Memory_{name}", args)
    
    def trace_jump(self, thread: str, from_pc: int, to_pc: int) -> None:
        """Trace a jump instruction."""
        self.trace_instant(
            thread, 
            f"Jump_0x{from_pc:06X}_to_0x{to_pc:06X}",
            {
                "from": f"0x{from_pc:06X}",
                "to": f"0x{to_pc:06X}"
            }
        )
    
    def trace_interrupt(self, name: str, **kwargs) -> None:
        """Trace interrupt-related events."""
        self.trace_instant("Interrupt", name, kwargs)
    
    def _cleanup_stale_frames(self, thread: str) -> None:
        """Remove stale frames from call stack."""
        stack = self._call_stacks.get(thread)
        if not stack:
            return
            
        current_time = self._get_timestamp()
        
        # Remove frames older than timeout
        while stack:
            if current_time - stack[0].timestamp > self.STALE_FRAME_TIMEOUT:
                frame = stack.popleft()
                if frame.event_sent:
                    self._trace_builder.end_slice(
                        self._track_uuids[thread],
                        current_time
                    )
            else:
                break


# Global singleton instance
g_tracer = TraceManager()


# Decorator for tracing functions
def trace_function(thread: str):
    """Decorator to automatically trace function execution."""
    def decorator(func):
        if not ENABLE_PERFETTO_TRACING:
            return func
            
        def wrapper(*args, **kwargs):
            # Extract PC if available from self
            pc = 0
            caller_pc = 0
            if args and hasattr(args[0], '_current_pc'):
                pc = args[0]._current_pc
                caller_pc = getattr(args[0], '_last_pc', 0)
                
            g_tracer.begin_function(thread, pc, caller_pc, func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                g_tracer.end_function(thread, pc)
                
        return wrapper
    return decorator