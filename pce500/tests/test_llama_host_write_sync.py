from __future__ import annotations

import pytest

from pce500.emulator import PCE500Emulator
from pce500.memory import INTERNAL_MEMORY_START, PCE500Memory
from pce500.memory_bus import MemoryOverlay
from sc62015.pysc62015.emulator import RegisterName


class _RecordingBackend:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.events: list[tuple[int, int]] = []
        self.error = error

    def notify_host_write(self, address: int, value: int) -> None:
        self.events.append((address, value))
        if self.error is not None:
            raise self.error


class _LlamaFacade:
    backend = "llama"

    def __init__(self, impl: object) -> None:
        self._impl = impl

    def unwrap(self) -> object:
        return self._impl


def _memory_with_backend(backend: object) -> PCE500Memory:
    memory = PCE500Memory()
    memory.set_cpu(_LlamaFacade(backend))
    return memory


def test_host_writes_sync_base_keyboard_and_writable_overlay_paths() -> None:
    backend = _RecordingBackend()
    memory = _memory_with_backend(backend)

    memory.write_byte(0x20, 0x11)
    # PCE500Memory aliases any raw address above the external 20-bit bus into
    # the 256-byte internal window; the native notification must use the same
    # canonical address.
    memory.write_byte(0x200020, 0x12)

    keyboard_writes: list[tuple[int, int]] = []
    memory.set_keyboard_handler(
        lambda _address, _pc: 0,
        lambda address, value, _pc: keyboard_writes.append((address, value)),
    )
    memory.write_byte(INTERNAL_MEMORY_START + 0xF0, 0x22)

    overlay_data = bytearray([0])
    memory.add_overlay(
        MemoryOverlay(
            start=0x3000,
            end=0x3000,
            name="host_ram",
            data=overlay_data,
            read_only=False,
        )
    )
    memory.write_byte(0x3000, 0x33)

    assert keyboard_writes == [(INTERNAL_MEMORY_START + 0xF0, 0x22)]
    assert overlay_data == b"\x33"
    assert backend.events == [
        (0x20, 0x11),
        (INTERNAL_MEMORY_START + 0x20, 0x12),
        (INTERNAL_MEMORY_START + 0xF0, 0x22),
        (0x3000, 0x33),
    ]


def test_ignored_read_only_and_cpu_originated_writes_do_not_resync() -> None:
    backend = _RecordingBackend()
    memory = _memory_with_backend(backend)
    read_only_data = bytearray([0x44])
    memory.add_overlay(
        MemoryOverlay(
            start=0x3100,
            end=0x3100,
            name="host_rom",
            data=read_only_data,
            read_only=True,
        )
    )

    memory.write_byte(0x3100, 0x55)
    memory.write_byte(0x20, 0x66, cpu_pc=0x1234)

    assert read_only_data == b"\x44"
    assert backend.events == []


def test_host_write_notifier_errors_propagate() -> None:
    memory = _memory_with_backend(_RecordingBackend(error=RuntimeError("sync boom")))

    with pytest.raises(RuntimeError, match="sync boom"):
        memory.write_byte(0x20, 0x77)


def test_missing_llama_host_write_notifier_fails_closed() -> None:
    memory = _memory_with_backend(object())

    with pytest.raises(RuntimeError, match="required notify_host_write"):
        memory.write_byte(0x20, 0x88)


def test_wait_callback_defers_host_sync_until_native_borrow_is_released() -> None:
    backend = _RecordingBackend()
    memory = _memory_with_backend(backend)

    class _WaitEmulator:
        cpu = memory.cpu
        memory_write_count = 0

        def _simulate_wait(self, cycles: int) -> None:
            assert cycles == 2
            memory.write_byte(INTERNAL_MEMORY_START + 0xFC, 0x01)
            memory.write_byte(0x20, 0x42)
            assert backend.events == []

    memory._emulator = _WaitEmulator()

    memory.wait_cycles(2)
    assert backend.events == []

    memory.flush_deferred_llama_host_writes()
    assert backend.events == [
        (INTERNAL_MEMORY_START + 0xFC, 0x01),
        (0x20, 0x42),
    ]


def test_failed_wait_reset_discards_stale_deferred_host_writes() -> None:
    backend = _RecordingBackend()
    memory = _memory_with_backend(backend)

    class _FailingWaitEmulator:
        cpu = memory.cpu
        memory_read_count = 0
        memory_write_count = 0

        def _simulate_wait(self, cycles: int) -> None:
            assert cycles == 2
            memory.write_byte(0x20, 0x42)
            raise RuntimeError("WAIT callback failed")

    memory._emulator = _FailingWaitEmulator()

    with pytest.raises(RuntimeError, match="WAIT callback failed"):
        memory.wait_cycles(2)

    assert memory.read_byte(0x20) == 0x42
    assert backend.events == []

    memory.reset()
    memory.discard_deferred_llama_host_writes()
    memory.flush_deferred_llama_host_writes()

    assert memory.read_byte(0x20) == 0
    assert backend.events == []


def test_failed_native_execute_poison_is_cleared_only_by_complete_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.memory.write_byte(0x1000, 0x00)
    emu.cpu.regs.set(RegisterName.PC, 0x1000)

    # Exercise the wrapper's native failure contract without requiring a
    # specially built PyO3 test double.  During the callback, host writes use
    # the same deferred path as a real LLAMA WAIT borrow.
    emu.cpu.backend = "llama"

    def fail_wait(_cycles: int) -> None:
        emu.memory.write_byte(0x20, 0x42)
        raise RuntimeError("native callback failed")

    monkeypatch.setattr(emu, "_simulate_wait", fail_wait)
    monkeypatch.setattr(
        emu.cpu,
        "execute_instruction",
        lambda _pc: emu.memory.wait_cycles(1),
    )

    with pytest.raises(RuntimeError, match="native callback failed"):
        emu.step()

    assert emu._poisoned is not None  # type: ignore[attr-defined]
    assert emu.memory._deferred_llama_writes == [(0x20, 0x42)]  # type: ignore[attr-defined]

    # Restore the real facade mode so its Python reset path can complete; a
    # successful machine reset then supersedes and discards the stale queue.
    emu.cpu.backend = "python"
    original_reset = emu.cpu.power_on_reset
    original_poison = emu._poisoned  # type: ignore[attr-defined]
    monkeypatch.setattr(
        emu.cpu,
        "power_on_reset",
        lambda: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        emu.reset()

    assert emu._poisoned == original_poison  # type: ignore[attr-defined]
    assert emu.memory._deferred_llama_writes == [(0x20, 0x42)]  # type: ignore[attr-defined]

    monkeypatch.setattr(emu.cpu, "power_on_reset", original_reset)
    emu.reset()

    assert emu._poisoned is None  # type: ignore[attr-defined]
    assert emu.memory._deferred_llama_writes == []  # type: ignore[attr-defined]
    emu.memory.flush_deferred_llama_host_writes()
    assert emu.memory.read_byte(0x20) == 0


def test_failed_post_execute_deferred_flush_poisons_without_replacing_first_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu.cpu.backend = "llama"
    monkeypatch.setattr(emu.cpu, "execute_instruction", lambda _pc: object())
    emu.memory._deferred_llama_writes.append((0x20, 0x42))  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="required notify_host_write"):
        emu._execute_instruction_and_flush(0x1000)  # type: ignore[attr-defined]

    assert "deferred host-write flush failed" in str(emu._poisoned)  # type: ignore[attr-defined]
    first_reason = emu._poisoned  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="required notify_host_write"):
        emu._execute_instruction_and_flush(0x1000)  # type: ignore[attr-defined]

    assert emu._poisoned == first_reason  # type: ignore[attr-defined]


def test_keyboard_read_body_type_error_is_propagated_without_retry() -> None:
    memory = PCE500Memory()
    calls: list[tuple[int, int | None]] = []

    def broken_read(address: int, cpu_pc: int | None) -> int:
        calls.append((address, cpu_pc))
        raise TypeError("keyboard callback body failed")

    memory.set_keyboard_handler(broken_read, lambda _address, _value, _pc: None)

    address = INTERNAL_MEMORY_START + 0xF2
    with pytest.raises(TypeError, match="keyboard callback body failed"):
        memory.read_byte(address, cpu_pc=0x12345)

    assert calls == [(address, 0x12345)]


def test_legacy_single_argument_keyboard_read_is_selected_without_probe_call() -> None:
    memory = PCE500Memory()
    calls: list[int] = []

    def legacy_read(address: int) -> int:
        calls.append(address)
        return 0x5A

    memory.set_keyboard_handler(legacy_read, lambda _address, _value, _pc: None)  # type: ignore[arg-type]

    address = INTERNAL_MEMORY_START + 0xF2
    assert memory.read_byte(address, cpu_pc=0x12345) == 0x5A
    assert calls == [address]
