"""Tracing event dispatcher and observer interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


class TraceEventType(Enum):
    """Kinds of tracing events emitted by the emulator."""

    START = "start"
    STOP = "stop"
    INSTANT = "instant"
    COUNTER = "counter"
    FUNCTION_BEGIN = "function_begin"
    FUNCTION_END = "function_end"
    FLOW_BEGIN = "flow_begin"
    FLOW_END = "flow_end"


@dataclass
class TraceEvent:
    """Structured tracing event."""

    type: TraceEventType
    thread: Optional[str] = None
    name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class TraceObserver(Protocol):
    """Interface for tracing observers."""

    def handle_event(self, event: TraceEvent) -> None: ...


class TraceDispatcher:
    """Dispatches tracing events to registered observers."""

    def __init__(self) -> None:
        self._observers: List[TraceObserver] = []

    # ------------------------------------------------------------------ #
    # Observer management
    # ------------------------------------------------------------------ #
    def register(self, observer: TraceObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister(self, observer: TraceObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def observers(self) -> Iterable[TraceObserver]:
        return tuple(self._observers)

    def has_observers(self) -> bool:
        """Return True when any observers are registered."""
        return bool(self._observers)

    # ------------------------------------------------------------------ #
    # Convenience emission helpers
    # ------------------------------------------------------------------ #
    def start_trace(self, output_path: Path | str) -> None:
        path = Path(output_path)
        self._emit(
            TraceEvent(
                TraceEventType.START,
                payload={"output_path": path},
            )
        )

    def stop_trace(self) -> None:
        self._emit(TraceEvent(TraceEventType.STOP))

    def record_instant(
        self,
        thread: str,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            TraceEvent(
                TraceEventType.INSTANT,
                thread=thread,
                name=name,
                payload=payload or {},
            )
        )

    def record_counter(self, name: str, value: float, *, thread: str = "CPU") -> None:
        self._emit(
            TraceEvent(
                TraceEventType.COUNTER,
                thread=thread,
                name=name,
                payload={"value": value},
            )
        )

    def begin_function(
        self,
        thread: str,
        pc: int,
        caller_pc: int,
        name: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            TraceEvent(
                TraceEventType.FUNCTION_BEGIN,
                thread=thread,
                name=name,
                payload={
                    "pc": pc,
                    "caller_pc": caller_pc,
                    "annotations": annotations or {},
                },
            )
        )

    def end_function(self, thread: str, pc: int) -> None:
        self._emit(
            TraceEvent(
                TraceEventType.FUNCTION_END,
                thread=thread,
                payload={"pc": pc},
            )
        )

    def begin_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        self._emit(
            TraceEvent(
                TraceEventType.FLOW_BEGIN,
                thread=thread,
                name=name,
                payload={"flow_id": flow_id},
            )
        )

    def end_flow(self, thread: str, flow_id: int, name: str = "Flow") -> None:
        self._emit(
            TraceEvent(
                TraceEventType.FLOW_END,
                thread=thread,
                name=name,
                payload={"flow_id": flow_id},
            )
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _emit(self, event: TraceEvent) -> None:
        for observer in tuple(self._observers):
            observer.handle_event(event)


# Global dispatcher used by the emulator.
trace_dispatcher = TraceDispatcher()

__all__ = [
    "TraceDispatcher",
    "TraceObserver",
    "TraceEvent",
    "TraceEventType",
    "trace_dispatcher",
]
