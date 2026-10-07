"""Reference peripheral-boundary driver mirrored by the primary Rust core.

This does not implement an OZ device or replace the Python CPU/backend. A
backend supplies a restricted context and its existing CPU-only boundary API.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BoundaryContext:
    memory: object
    state: object
    timer: object
    cycles: int
    instructions: int
    elapsed_timing_units: int


class BoundaryDevice(Protocol):
    def before_boundary(self, context: BoundaryContext) -> bool | None: ...

    def after_boundary(self, context: BoundaryContext, elapsed: int) -> None: ...


class BoundaryRuntime(Protocol):
    def boundary_context(self) -> BoundaryContext: ...

    def step_cpu_boundaries(self, boundaries: int) -> None: ...

    def external_interrupt_level(self) -> bool: ...

    def set_external_interrupt_level(self, level: bool) -> None: ...

    def power_on_reset(self) -> None: ...


class BoundaryScheduler:
    """One device callback pair per public boundary, including HALT/OFF.

    Internal IRQ handler continuations use the CPU-only path. A fault before
    or during CPU execution keeps the device installed and skips its after
    callback. A failed after callback poisons committed progress until reset.
    Machine boundary device state is not representable by snapshot v4.
    """

    def __init__(self, runtime: BoundaryRuntime) -> None:
        self.runtime = runtime
        self.device: BoundaryDevice | None = None
        self.poisoned: str | None = None

    def install_boundary_device(self, device: BoundaryDevice) -> None:
        if self.device is not None:
            raise RuntimeError("a boundary device is already installed")
        self.device = device

    def step_scheduler_boundaries(self, boundaries: int) -> None:
        if boundaries < 0:
            raise ValueError("boundary count must be nonnegative")
        if self.poisoned is not None:
            raise RuntimeError(
                f"runtime is poisoned; power-on reset required: {self.poisoned}"
            )
        device = self.device
        if device is None:
            self.runtime.step_cpu_boundaries(boundaries)
            return
        self.device = None
        try:
            for _ in range(boundaries):
                context = self.runtime.boundary_context()
                level = device.before_boundary(context)
                if (
                    level is not None
                    and self.runtime.external_interrupt_level() != level
                ):
                    self.runtime.set_external_interrupt_level(level)
                before = context.cycles
                self.runtime.step_cpu_boundaries(1)
                context = self.runtime.boundary_context()
                elapsed = (context.cycles - before) & ((1 << 64) - 1)
                try:
                    device.after_boundary(context, elapsed)
                except Exception as error:
                    self.poisoned = f"boundary device after CPU progress: {error}"
                    raise
        finally:
            self.device = device

    def power_on_reset(self) -> None:
        self.runtime.power_on_reset()
        self.poisoned = None

    def snapshot_extension_names(self) -> tuple[str, ...]:
        return ("machine boundary device state",) if self.device is not None else ()
