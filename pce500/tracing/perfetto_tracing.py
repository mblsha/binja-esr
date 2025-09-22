# pce500/tracing/perfetto_tracing.py
import atexit
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, cast

from retrobus_perfetto import PerfettoTraceBuilder


class PerfettoTracer:
    """
    Perfetto tracer using retrobus-perfetto protobuf format.
    Wall-clock timestamps via time.perf_counter().
    Off by default; safe to leave in production.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._start = 0.0
        self._builder: Optional[Any] = None
        self._path: Optional[str] = None
        self._track_uuids: Dict[str, int] = {}
        self._counter_tracks: Dict[str, int] = {}
        self._slice_stacks: Dict[str, list] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _now_ns(self) -> int:
        """Get current timestamp in nanoseconds."""
        return int((time.perf_counter() - self._start) * 1_000_000_000)

    def _ensure_track(self, name: str) -> int:
        """Ensure a track exists and return its UUID."""
        with self._lock:
            if name in self._track_uuids:
                return self._track_uuids[name]

            if not self._builder:
                return 0

            # Create a new thread track
            uuid = self._builder.add_thread(name)
            self._track_uuids[name] = uuid
            self._slice_stacks[name] = []
            return uuid

    def _ensure_counter_track(self, name: str, unit: str = "count") -> int:
        """Ensure a counter track exists and return its UUID."""
        with self._lock:
            if name in self._counter_tracks:
                return self._counter_tracks[name]

            if not self._builder:
                return 0

            # Create a new counter track
            uuid = self._builder.add_counter_track(name, unit)
            self._counter_tracks[name] = uuid
            return uuid

    def _is_active(self) -> bool:
        """Return ``True`` when tracing is enabled and a builder is available."""

        return self._enabled and self._builder is not None

    def _get_builder(self) -> Optional[PerfettoTraceBuilder]:
        """Return the active trace builder if tracing is currently enabled."""

        if not self._is_active():
            return None

        return cast(PerfettoTraceBuilder, self._builder)

    def start(self, path: str = "pc-e500.perfetto-trace") -> None:
        """Start tracing to the specified file."""
        with self._lock:
            if self._enabled:
                return

            self._enabled = True
            self._path = path
            self._start = time.perf_counter()
            self._track_uuids.clear()
            self._counter_tracks.clear()
            self._slice_stacks.clear()

            # Create the trace builder
            self._builder = PerfettoTraceBuilder("PC-E500 Emulator")

            # Pre-create common tracks
            for t in ("CPU", "Execution", "I/O"):
                self._ensure_track(t)

            # Add performance profiling tracks
            for t in ("Emulation", "Opcodes", "Memory", "Display", "Lifting", "System"):
                self._ensure_track(t)

            # Pre-create counter track
            self._ensure_counter_track("instructions", "count")

            atexit.register(self.safe_stop)

    def safe_stop(self) -> None:
        """Safe stop for atexit – ignore errors if already stopped."""
        try:
            self.stop()
        except Exception:
            pass

    def stop(self) -> None:
        """Stop tracing and save the file."""
        with self._lock:
            if not self._enabled or not self._builder:
                return

            # End any open slices
            for track_name, stack in self._slice_stacks.items():
                if track_name in self._track_uuids:
                    track_uuid = self._track_uuids[track_name]
                    while stack:
                        stack.pop()
                        self._builder.end_slice(track_uuid, self._now_ns())

            # Save the trace
            path = self._path or "pc-e500.perfetto-trace"
            self._builder.save(path)

            # Clean up
            self._enabled = False
            self._builder = None
            self._track_uuids.clear()
            self._counter_tracks.clear()
            self._slice_stacks.clear()
            self._path = None

    # ---- Event APIs ----

    def instant(
        self, track: str, name: str, args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an instant event to the trace."""
        builder = self._get_builder()
        if not builder:
            return

        track_uuid = self._ensure_track(track)
        event = builder.add_instant_event(track_uuid, name, self._now_ns())

        if args:
            event.add_annotations(args)

    def counter(
        self,
        track: str,
        name: str,
        value: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a counter value to the trace."""
        builder = self._get_builder()
        if not builder:
            return

        # For counters, we use the counter track
        track_uuid = self._ensure_counter_track(name)

        # retrobus-perfetto uses update_counter
        builder.update_counter(track_uuid, value, self._now_ns())

    def begin_slice(
        self, track: str, name: str, args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Begin a duration slice."""
        builder = self._get_builder()
        if not builder:
            return

        track_uuid = self._ensure_track(track)

        # Track the slice name for matching end_slice
        if track not in self._slice_stacks:
            self._slice_stacks[track] = []
        self._slice_stacks[track].append(name)

        event = builder.begin_slice(track_uuid, name, self._now_ns())

        if args:
            event.add_annotations(args)

    def end_slice(self, track: str) -> None:
        """End a duration slice."""
        builder = self._get_builder()
        if not builder:
            return

        track_uuid = self._ensure_track(track)

        # Pop the slice name
        if track in self._slice_stacks and self._slice_stacks[track]:
            self._slice_stacks[track].pop()

        builder.end_slice(track_uuid, self._now_ns())

    @contextmanager
    def slice(self, track: str, name: str, args: Optional[Dict[str, Any]] = None):
        """Context manager for duration slices.

        Always runs but checks internally if tracing is enabled.
        When disabled, this is a no-op with minimal overhead.
        """
        if not self._is_active():
            yield
            return

        self.begin_slice(track, name, args)
        try:
            yield
        finally:
            self.end_slice(track)


# Global tracer for convenience
tracer = PerfettoTracer()


def perf_trace(
    track: str,
    sample_rate: int = 1,
    extract_args: Optional[Callable[[Any], Dict[str, Any]]] = None,
    include_op_num: bool = False,
) -> Callable:
    """Decorator for automatic performance tracing.

    Always wraps the function but checks internally if tracing is enabled.

    Args:
        track: Perfetto track name for this function
        sample_rate: Only trace every Nth call (1 = trace all)
        extract_args: Optional function to extract trace arguments from function args
        include_op_num: Include instruction_count as op_num in trace args (for Emulation track)
    """

    def decorator(func: Callable) -> Callable:
        call_count = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Check inside wrapper - minimal overhead when disabled
            if not tracer.enabled or (call_count % sample_rate) != 0:
                return func(*args, **kwargs)

            # Extract trace arguments
            trace_args = {}
            if extract_args:
                try:
                    trace_args = extract_args(*args, **kwargs)
                except Exception:
                    pass  # Ignore extraction errors

            # Add operation number if requested (for step function)
            if include_op_num and args and hasattr(args[0], "instruction_count"):
                trace_args["op_num"] = args[0].instruction_count

            func_name = func.__name__
            with tracer.slice(track, func_name, trace_args):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def enable_profiling(path: str = "emulator-profile.perfetto-trace") -> None:
    """Enable performance profiling globally."""
    tracer.start(path)
    print(f"Performance profiling enabled → {path}")


def disable_profiling() -> None:
    """Disable performance profiling and save trace."""
    tracer.stop()
    print("Performance profiling stopped")
