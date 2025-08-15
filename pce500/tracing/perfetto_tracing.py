# pce500/tracing/perfetto_tracing.py
import atexit
import threading
import time
from typing import Any, Dict, Optional

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

    def instant(self, track: str, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """Add an instant event to the trace."""
        if not self._enabled or not self._builder:
            return
        
        track_uuid = self._ensure_track(track)
        event = self._builder.add_instant_event(track_uuid, name, self._now_ns())
        
        if args:
            event.add_annotations(args)

    def counter(self, track: str, name: str, value: float, extra: Optional[Dict[str, Any]] = None) -> None:
        """Add a counter value to the trace."""
        if not self._enabled or not self._builder:
            return
        
        # For counters, we use the counter track
        track_uuid = self._ensure_counter_track(name)
        
        # retrobus-perfetto uses update_counter
        self._builder.update_counter(track_uuid, value, self._now_ns())

    def begin_slice(self, track: str, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """Begin a duration slice."""
        if not self._enabled or not self._builder:
            return
        
        track_uuid = self._ensure_track(track)
        
        # Track the slice name for matching end_slice
        if track not in self._slice_stacks:
            self._slice_stacks[track] = []
        self._slice_stacks[track].append(name)
        
        event = self._builder.begin_slice(track_uuid, name, self._now_ns())
        
        if args:
            event.add_annotations(args)

    def end_slice(self, track: str) -> None:
        """End a duration slice."""
        if not self._enabled or not self._builder:
            return
        
        track_uuid = self._ensure_track(track)
        
        # Pop the slice name
        if track in self._slice_stacks and self._slice_stacks[track]:
            self._slice_stacks[track].pop()
        
        self._builder.end_slice(track_uuid, self._now_ns())


# Global tracer for convenience
tracer = PerfettoTracer()