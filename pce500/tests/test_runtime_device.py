"""Peripheral-boundary contract fixtures shared with Rust; not ROM boot proof."""

import pytest

from pce500.runtime_device import BoundaryContext, BoundaryScheduler


class RuntimeFixture:
    def __init__(self, power: str = "running") -> None:
        self.power = power
        self.cycles = 0
        self.instructions = 0
        self.elapsed_timing_units = 0
        self.level = False
        self.decode_fault = False
        self.memory: dict[str, int] = {}

    def boundary_context(self) -> BoundaryContext:
        return BoundaryContext(
            self.memory,
            self,
            object(),
            self.cycles,
            self.instructions,
            self.elapsed_timing_units,
        )

    def step_cpu_boundaries(self, boundaries: int) -> None:
        if self.decode_fault:
            raise RuntimeError("decode fixture")
        for _ in range(boundaries):
            self.instructions += self.power == "running"
            self.cycles += self.power != "off"
            self.elapsed_timing_units += 1

    def external_interrupt_level(self) -> bool:
        return self.level

    def set_external_interrupt_level(self, level: bool) -> None:
        self.level = level

    def power_on_reset(self) -> None:
        self.power = "running"


class DeviceFixture:
    def __init__(self) -> None:
        self.log: list[tuple[str, int, int]] = []
        self.before_error_once = False
        self.after_error_once = False
        self.level: bool | None = None
        self.timing: list[int] = []

    def before_boundary(self, context: BoundaryContext) -> bool | None:
        self.log.append(("before", context.instructions, context.cycles))
        if self.before_error_once:
            self.before_error_once = False
            raise RuntimeError("before fixture")
        return self.level

    def after_boundary(self, context: BoundaryContext, elapsed: int) -> None:
        self.log.append(("after", context.instructions, elapsed))
        self.timing.append(context.elapsed_timing_units)
        if self.after_error_once:
            self.after_error_once = False
            raise RuntimeError("after fixture")


@pytest.mark.parametrize("power", ["running", "halted", "off"])
def test_every_public_boundary_including_idle_ticks_device(power: str) -> None:
    runtime = RuntimeFixture(power)
    scheduler = BoundaryScheduler(runtime)
    device = DeviceFixture()
    scheduler.install_boundary_device(device)
    scheduler.step_scheduler_boundaries(0)
    assert device.log == []
    scheduler.step_scheduler_boundaries(5)
    assert len(device.log) == 10
    assert device.timing == [1, 2, 3, 4, 5]
    for before, after in zip(device.log[::2], device.log[1::2], strict=True):
        assert before[0] == "before"
        assert after[0] == "after"
        assert after[2] == int(power != "off")
    assert runtime.instructions == (5 if power == "running" else 0)


def test_pre_boundary_fault_consumes_no_cpu_and_preserves_device() -> None:
    runtime = RuntimeFixture()
    scheduler = BoundaryScheduler(runtime)
    device = DeviceFixture()
    device.before_error_once = True
    scheduler.install_boundary_device(device)
    with pytest.raises(RuntimeError, match="before fixture"):
        scheduler.step_scheduler_boundaries(2)
    assert (runtime.instructions, runtime.cycles) == (0, 0)
    scheduler.step_scheduler_boundaries(1)
    assert len(device.log) == 3
    assert scheduler.device is device


def test_decode_fault_skips_after_and_preserves_device() -> None:
    runtime = RuntimeFixture()
    scheduler = BoundaryScheduler(runtime)
    device = DeviceFixture()
    scheduler.install_boundary_device(device)
    runtime.decode_fault = True
    with pytest.raises(RuntimeError, match="decode fixture"):
        scheduler.step_scheduler_boundaries(1)
    assert len(device.log) == 1
    runtime.decode_fault = False
    scheduler.step_scheduler_boundaries(1)
    assert len(device.log) == 3


def test_after_fault_poison_prevents_callbacks_until_reset() -> None:
    runtime = RuntimeFixture()
    scheduler = BoundaryScheduler(runtime)
    device = DeviceFixture()
    device.after_error_once = True
    scheduler.install_boundary_device(device)
    with pytest.raises(RuntimeError, match="after fixture"):
        scheduler.step_scheduler_boundaries(2)
    assert runtime.instructions == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        scheduler.step_scheduler_boundaries(1)
    assert len(device.log) == 2
    scheduler.power_on_reset()
    scheduler.step_scheduler_boundaries(1)
    assert len(device.log) == 4


def test_irq_internal_continuation_does_not_tick_twice() -> None:
    class IRQFixture(RuntimeFixture):
        def step_cpu_boundaries(self, boundaries: int) -> None:
            if self.level:
                self.level = False
                scheduler.step_scheduler_boundaries(1)
            else:
                super().step_cpu_boundaries(boundaries)

    runtime = IRQFixture()
    scheduler = BoundaryScheduler(runtime)
    device = DeviceFixture()
    device.level = True
    scheduler.install_boundary_device(device)
    scheduler.step_scheduler_boundaries(1)
    assert runtime.instructions == 1
    assert len(device.log) == 2
    assert scheduler.snapshot_extension_names() == ("machine boundary device state",)
    with pytest.raises(RuntimeError, match="already installed"):
        scheduler.install_boundary_device(DeviceFixture())
