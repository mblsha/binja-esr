# pce500/tracing/__init__.py
"""Tracing infrastructure for PC-E500 emulator performance analysis."""

from .perfetto_tracing import PerfettoTracer, tracer

__all__ = ["PerfettoTracer", "tracer"]
