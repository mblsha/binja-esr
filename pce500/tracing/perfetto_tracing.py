# pce500/tracing/perfetto_tracing.py
import atexit
import json
import threading
import time
from typing import Any, Dict, Optional

class PerfettoTracer:
    """
    Minimal Perfetto (Chrome trace events) writer.
    Wall-clock timestamps via time.perf_counter().
    Off by default; safe to leave in production.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events = []
        self._enabled = False
        self._start = 0.0
        self._pid = 1
        self._next_tid = 1000
        self._tracks = {}  # name -> tid
        self._slice_stacks = {}  # tid -> [name,...]
        self._path: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _now_us(self) -> int:
        # perf_counter is monotonic and high-res
        return int((time.perf_counter() - self._start) * 1_000_000)

    def _emit(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(ev)

    def _ensure_track(self, name: str) -> int:
        with self._lock:
            if name in self._tracks:
                return self._tracks[name]
            tid = self._next_tid
            self._next_tid += 1
            self._tracks[name] = tid
            self._slice_stacks[tid] = []

            # Thread (track) name metadata
            self._emit({
                "ph": "M", "name": "thread_name",
                "pid": self._pid, "tid": tid,
                "args": {"name": name}
            })
            return tid

    def start(self, path: str = "pc-e500.trace.json") -> None:
        with self._lock:
            if self._enabled:
                return
            self._enabled = True
            self._path = path
            self._start = time.perf_counter()
            self._events.clear()
            self._tracks.clear()
            self._slice_stacks.clear()

            # Process name metadata
            self._emit({
                "ph": "M", "name": "process_name",
                "pid": self._pid, "args": {"name": "PC-E500 Emulator"}
            })

            # Pre-create common tracks
            for t in ("CPU", "Execution", "I/O", "Counters"):
                self._ensure_track(t)

            atexit.register(self.safe_stop)

    def safe_stop(self) -> None:
        # For atexit safety – ignore errors if already stopped
        try:
            self.stop()
        except Exception:
            pass

    def stop(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            path = self._path or "pc-e500.trace.json"
            out = {
                "traceEvents": self._events,
                # Perfetto understands Chrome trace timestamps in microseconds
                "displayTimeUnit": "us",
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
            self._events.clear()
            self._tracks.clear()
            self._slice_stacks.clear()
            self._path = None

    # ---- Event APIs ----

    def instant(self, track: str, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        if not self._enabled:
            return
        tid = self._ensure_track(track)
        self._emit({
            "ph": "i",       # instant event
            "s": "t",        # thread-scoped
            "name": name,
            "pid": self._pid,
            "tid": tid,
            "ts": self._now_us(),
            "args": args or {}
        })

    def counter(self, track: str, name: str, value: float, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._enabled:
            return
        tid = self._ensure_track(track)
        args = {name: value}
        if extra:
            args.update(extra)
        self._emit({
            "ph": "C",
            "name": name,
            "pid": self._pid,
            "tid": tid,
            "ts": self._now_us(),
            "args": args
        })

    def begin_slice(self, track: str, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        if not self._enabled:
            return
        tid = self._ensure_track(track)
        self._slice_stacks[tid].append(name)
        self._emit({
            "ph": "B", "name": name,
            "pid": self._pid, "tid": tid,
            "ts": self._now_us(),
            "args": args or {}
        })

    def end_slice(self, track: str) -> None:
        if not self._enabled:
            return
        tid = self._ensure_track(track)
        name = self._slice_stacks[tid].pop() if self._slice_stacks[tid] else ""
        self._emit({
            "ph": "E", "name": name,
            "pid": self._pid, "tid": tid,
            "ts": self._now_us(),
            "args": {}
        })


# Global tracer for convenience
tracer = PerfettoTracer()