"""Tracing utilities for the PC-E500 emulator."""

from .dispatcher import (
    TraceDispatcher,
    TraceEvent,
    TraceEventType,
    TraceObserver,
    trace_dispatcher,
)

# Import perfetto tracer to register its observer with the dispatcher.
from . import perfetto_tracing  # noqa: F401

__all__ = [
    "TraceDispatcher",
    "TraceEvent",
    "TraceEventType",
    "TraceObserver",
    "trace_dispatcher",
]
