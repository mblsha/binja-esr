"""Perfetto trace observer for the PC-E500 emulator.

The observer implements the generic tracing observer interface defined in
``pce500.tracing.dispatcher`` and translates emitted events into Perfetto trace
records using ``retrobus-perfetto``.
"""

import collections
import logging
import threading
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, TypeVar

from retrobus_perfetto import PerfettoTraceBuilder

from .tracing.dispatcher import (
    TraceEvent,
    TraceEventType,
    TraceObserver,
    trace_dispatcher,
)

logger = logging.getLogger(__name__)


@dataclass
class CallStackFrame:
    """Represents a frame in the call stack."""

    pc: int
    caller_pc: int
    name: Optional[str] = None
    timestamp: int = 0
    event_sent: bool = False


class PerfettoObserver(TraceObserver):
    """Observer that records tracing events to Perfetto traces."""

    # Configuration
    MAX_CALL_DEPTH = 50  # Maximum call stack depth to track
    STALE_FRAME_TIMEOUT = 1_000_000_000  # 1 second in nanoseconds

    def __init__(self) -> None:
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

    def _require_active_track(self, thread: str) -> Optional[int]:
        """Return the active track UUID for ``thread`` if tracing is enabled."""

        if not self._tracing_enabled or not self._trace_builder:
            return None
        return self._track_uuids.get(thread)

    def _require_counter_track(self, name: str) -> Optional[int]:
        """Return the counter track UUID when tracing is active."""

        if not self._tracing_enabled or not self._trace_builder:
            return None
        return self._counter_tracks.get(name)

    def _start_tracing(self, output_path: Path) -> bool:
        """Start tracing to a file."""
        logger.debug("start_tracing called with output_path=%s", output_path)

        with self._rlock:
            if self._tracing_enabled:
                logger.debug("Tracing already enabled")
                return False

            try:
                # Create trace builder
                self._trace_builder = PerfettoTraceBuilder("PC-E500 Emulator")

                # Create thread tracks
                self._track_uuids["CPU"] = self._trace_builder.add_thread("CPU")
                self._track_uuids["Execution"] = self._trace_builder.add_thread(
                    "Execution"
                )
                self._track_uuids["Memory"] = self._trace_builder.add_thread("Memory")
                self._track_uuids["Memory_Internal"] = self._trace_builder.add_thread(
                    "Memory_Internal"
                )  # SC62015 CPU internal RAM (0x100000+)
                self._track_uuids["Memory_External"] = self._trace_builder.add_thread(
                    "Memory_External"
                )  # All external memory (<0x100000)
                self._track_uuids["I/O"] = self._trace_builder.add_thread("I/O")
                self._track_uuids["Display"] = self._trace_builder.add_thread("Display")
                self._track_uuids["Interrupt"] = self._trace_builder.add_thread(
                    "Interrupt"
                )

                # Create counter tracks
                self._counter_tracks["instructions"] = (
                    self._trace_builder.add_counter_track("Instructions", "count")
                )
                self._counter_tracks["call_depth"] = (
                    self._trace_builder.add_counter_track("Call Depth", "level")
                )
                self._counter_tracks["memory_reads"] = (
                    self._trace_builder.add_counter_track("Memory Reads", "ops/s")
                )
                self._counter_tracks["memory_writes"] = (
                    self._trace_builder.add_counter_track("Memory Writes", "ops/s")
                )

                self._trace_file = output_path
                self._tracing_enabled = True
                self._start_time = time.perf_counter()

                logger.debug(
                    "Tracing started successfully; file will be saved to %s",
                    self._trace_file,
                )
                return True

            except Exception as e:
                logger.exception("Exception in start_tracing")
                raise RuntimeError(f"Failed to start tracing: {e}") from e

    def _stop_tracing(self) -> bool:
        """Stop tracing and save the file."""
        logger.debug(
            "stop_tracing called; _tracing_enabled=%s, _trace_builder=%s",
            self._tracing_enabled,
            self._trace_builder is not None,
        )
        if not self._tracing_enabled or not self._trace_builder:
            logger.debug("Tracing not enabled or no trace builder")
            return False

        with self._rlock:
            try:
                # Clean up any pending call stack frames
                for thread, stack in self._call_stacks.items():
                    while stack:
                        frame = stack.pop()
                        if frame.event_sent and thread in self._track_uuids:
                            self._trace_builder.end_slice(
                                self._track_uuids[thread], self._get_timestamp()
                            )

                # Save the trace
                if self._trace_file:
                    logger.debug("Saving trace to %s", self._trace_file)
                    self._trace_builder.save(str(self._trace_file))
                    logger.debug("Trace saved successfully")
                else:
                    logger.debug("No trace file path set")

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

    def _begin_function(
        self,
        thread: str,
        pc: int,
        caller_pc: int,
        name: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Begin a function duration event.

        Returns:
            The event object if tracing is enabled, None otherwise.
        """
        with self._rlock:
            track_uuid = self._require_active_track(thread)
            if track_uuid is None:
                return None

            # Clean up old frames
            self._cleanup_stale_frames(thread, track_uuid)

            # Create new frame
            frame = CallStackFrame(
                pc=pc,
                caller_pc=caller_pc,
                name=name or f"func_0x{pc:06X}",
                timestamp=self._get_timestamp(),
                event_sent=True,
            )

            # Add to call stack
            if thread not in self._call_stacks:
                self._call_stacks[thread] = collections.deque(
                    maxlen=self.MAX_CALL_DEPTH
                )
            self._call_stacks[thread].append(frame)

            # Send trace event
            event = self._trace_builder.begin_slice(
                track_uuid, frame.name, frame.timestamp
            )

            # Add annotations
            event.add_annotations({"pc": f"0x{pc:06X}", "caller": f"0x{caller_pc:06X}"})
            if annotations:
                event.add_annotations(annotations)

            return event

    def _end_function(self, thread: str, pc: int) -> None:
        """End a function duration event."""
        with self._rlock:
            track_uuid = self._require_active_track(thread)
            if track_uuid is None:
                return

            stack = self._call_stacks.get(thread)
            if not stack:
                return

            # Try to find matching frame
            for i, frame in reversed(list(enumerate(stack))):
                if frame.pc == pc and frame.event_sent:
                    # Remove this frame and all frames above it
                    while len(stack) > i:
                        popped = stack.pop()
                        if popped.event_sent:
                            self._trace_builder.end_slice(
                                track_uuid, self._get_timestamp()
                            )
                    return

            # If no matching frame found, just pop the top
            if stack and stack[-1].event_sent:
                stack.pop()
                self._trace_builder.end_slice(track_uuid, self._get_timestamp())

    def _trace_instant(
        self, thread: str, name: str, args: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Add an instant event.

        Returns:
            The event object if tracing is enabled, None otherwise.
        """
        with self._rlock:
            track_uuid = self._require_active_track(thread)
            if track_uuid is None:
                return None

            event = self._trace_builder.add_instant_event(
                track_uuid, name, self._get_timestamp()
            )

            if args:
                event.add_annotations(args)

            return event

    def _trace_counter(self, thread: str, name: str, value: float) -> None:
        """Add a counter event.

        Args:
            thread: Thread name (for compatibility, not used in counter tracks)
            name: Counter name (must match a created counter track)
            value: Counter value
        """
        with self._rlock:
            counter_track = self._require_counter_track(name)
            if counter_track is None:
                return

            self._trace_builder.update_counter(
                counter_track, value, self._get_timestamp()
            )

    def _begin_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        """Begin a flow to connect events across threads."""
        with self._rlock:
            track_uuid = self._require_active_track(thread)
            if track_uuid is None:
                return

            self._trace_builder.add_flow(
                track_uuid,
                name,
                self._get_timestamp(),
                flow_id,
                terminating=False,
            )

    def _end_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        """End a flow."""
        with self._rlock:
            track_uuid = self._require_active_track(thread)
            if track_uuid is None:
                return

            self._trace_builder.add_flow(
                track_uuid,
                name,
                self._get_timestamp(),
                flow_id,
                terminating=True,
            )

    def _trace_memory_access(
        self,
        thread: str,
        address: int,
        size: int,
        is_write: bool,
        value: Optional[int] = None,
    ) -> None:
        """Trace memory access operations."""
        name = "Write" if is_write else "Read"
        args = {"address": f"0x{address:06X}", "size": size}

        if value is not None:
            args["value"] = f"0x{value:02X}" if size == 1 else f"0x{value:04X}"

        self._trace_instant(thread, f"Memory_{name}", args)

    def _trace_jump(self, thread: str, from_pc: int, to_pc: int) -> None:
        """Trace a jump instruction."""
        self._trace_instant(
            thread,
            f"Jump_0x{from_pc:06X}_to_0x{to_pc:06X}",
            {"from": f"0x{from_pc:06X}", "to": f"0x{to_pc:06X}"},
        )

    def _trace_interrupt(self, name: str, **kwargs) -> None:
        """Trace interrupt-related events."""
        self._trace_instant("Interrupt", name, kwargs)

    def _cleanup_stale_frames(self, thread: str, track_id: int) -> None:
        """Remove stale frames from call stack."""
        stack = self._call_stacks.get(thread)
        if not stack:
            return

        track_uuid = self._require_active_track(thread)
        if track_uuid is None:
            stack.clear()
            return

        current_time = self._get_timestamp()

        # Remove frames older than timeout
        while stack:
            if current_time - stack[0].timestamp > self.STALE_FRAME_TIMEOUT:
                frame = stack.popleft()
                if frame.event_sent:
                    self._trace_builder.end_slice(track_uuid, current_time)
            else:
                break

    # ------------------------------------------------------------------ #
    # TraceObserver interface
    # ------------------------------------------------------------------ #
    def handle_event(self, event: TraceEvent) -> None:
        """Dispatch a tracing event emitted by the emulator."""

        if event.type is TraceEventType.START:
            output_path = event.payload.get("output_path")
            if output_path is None:
                return
            self._start_tracing(Path(output_path))
            return

        if event.type is TraceEventType.STOP:
            self._stop_tracing()
            return

        if not self._tracing_enabled:
            return

        if event.type is TraceEventType.INSTANT:
            if event.thread and event.name:
                self._trace_instant(event.thread, event.name, event.payload)
            return

        if event.type is TraceEventType.COUNTER:
            name = event.name
            if name is not None:
                value = event.payload.get("value")
                if value is not None:
                    self._trace_counter(event.thread or "CPU", name, value)
            return

        if event.type is TraceEventType.FUNCTION_BEGIN:
            if event.thread:
                pc = int(event.payload.get("pc", 0))
                caller_pc = int(event.payload.get("caller_pc", 0))
                annotations = event.payload.get("annotations") or {}
                self._begin_function(
                    event.thread,
                    pc,
                    caller_pc,
                    event.name,
                    annotations,
                )
            return

        if event.type is TraceEventType.FUNCTION_END:
            if event.thread:
                pc = int(event.payload.get("pc", 0))
                self._end_function(event.thread, pc)
            return

        if event.type is TraceEventType.FLOW_BEGIN:
            if event.thread:
                flow_id = int(event.payload.get("flow_id", 0))
                self._begin_flow(event.thread, flow_id, event.name or "Flow")
            return

        if event.type is TraceEventType.FLOW_END:
            if event.thread:
                flow_id = int(event.payload.get("flow_id", 0))
                self._end_flow(event.thread, flow_id, event.name or "Flow")
            return


# Decorator for tracing functions
T = TypeVar("T")


def trace_function(thread: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to automatically trace function execution."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Extract PC if available from self
            pc = 0
            caller_pc = 0
            if args and hasattr(args[0], "_current_pc"):
                pc = args[0]._current_pc
                caller_pc = getattr(args[0], "_last_pc", 0)

            trace_dispatcher.begin_function(thread, pc, caller_pc, func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                trace_dispatcher.end_function(thread, pc)

        return wrapper

    return decorator


# Default Perfetto observer registered with the dispatcher.
perfetto_observer = PerfettoObserver()
trace_dispatcher.register(perfetto_observer)

__all__ = ["PerfettoObserver", "perfetto_observer", "trace_function"]
