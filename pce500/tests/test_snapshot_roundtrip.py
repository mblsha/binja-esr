from __future__ import annotations

import json
import os
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pce500.emulator import IRQSource, PCE500Emulator
from sc62015.pysc62015.cpu import available_backends
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.stepper import CPURegistersSnapshot


_CALL_DEPTH = 1
_CALL_SUB_LEVEL = 1
_MEM_READS = 7
_MEM_WRITES = 5
_KB_IRQ_COUNT = 2
_KB_STROBE_COUNT = 4
_KB_COL_HIST = [1, 2] + [0] * 9
_KB_LAST_COLS = [0, 1]
_KB_LAST_KOL = 0x12
_KB_LAST_KOH = 0x03
_KB_KIL_READS = 1


def _seed_state(emu: PCE500Emulator) -> None:
    # Prime counters/metadata that should round-trip through .pcsnap.
    emu.call_depth = _CALL_DEPTH
    emu.cpu.regs.call_sub_level = _CALL_SUB_LEVEL
    # Seed keyboard metrics and sync into snapshot metadata.
    emu._kb_irq_count = _KB_IRQ_COUNT
    emu._kb_strobe_count = _KB_STROBE_COUNT
    emu._kb_col_hist = list(_KB_COL_HIST)
    emu._last_kil_columns = list(_KB_LAST_COLS)
    emu._last_kol = _KB_LAST_KOL
    emu._last_koh = _KB_LAST_KOH
    emu._kil_read_count = _KB_KIL_READS
    emu._kb_irq_enabled = True

    # Seed LCD payload by issuing a couple of writes into the overlay window.
    emu.memory.write_byte(0x2000, 0x3F)  # instruction
    emu.memory.write_byte(0x2002, 0xAB)  # data

    # Snapshot counters after all synthetic writes above.
    emu.memory_read_count = _MEM_READS
    emu.memory_write_count = _MEM_WRITES
    if getattr(emu.cpu, "backend", "python") == "llama":
        impl = emu.cpu.unwrap()
        try:
            impl.call_depth = _CALL_DEPTH
            impl.restore_keyboard_snapshot(emu._capture_keyboard_snapshot_metadata())
            impl.memory_reads = _MEM_READS
            impl.memory_writes = _MEM_WRITES
        except Exception:
            pass

    # Touch a few registers to ensure state is non-zero.
    emu.cpu.regs.set(RegisterName.Y, 0x1234)


def _assert_state(emu: PCE500Emulator) -> None:
    assert emu.call_depth == _CALL_DEPTH
    assert emu.cpu.regs.call_sub_level == _CALL_SUB_LEVEL
    assert emu.memory_read_count == _MEM_READS
    assert emu.memory_write_count == _MEM_WRITES

    assert emu._kb_irq_count == _KB_IRQ_COUNT
    assert emu._kb_strobe_count == _KB_STROBE_COUNT
    assert list(emu._kb_col_hist) == _KB_COL_HIST
    assert list(emu._last_kil_columns) == _KB_LAST_COLS
    assert emu._last_kol == _KB_LAST_KOL
    assert emu._last_koh == _KB_LAST_KOH
    assert emu._kil_read_count == _KB_KIL_READS
    assert emu._kb_irq_enabled is True

    # LCD payload should have been restored.
    snap = emu.lcd.get_snapshot()
    assert any(0xAB in row for chip in snap.chips for row in chip.vram)


def _has_llama_backend() -> bool:
    return "llama" in available_backends()


def _rewrite_snapshot_metadata(
    source: Path,
    target: Path,
    mutate,
) -> None:
    with zipfile.ZipFile(source, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    metadata = json.loads(entries["snapshot.json"])
    mutate(metadata)
    entries["snapshot.json"] = json.dumps(metadata).encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)


def _read_snapshot_metadata(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("snapshot.json"))
    assert isinstance(metadata, dict)
    return metadata


def _state_fingerprint(emu: PCE500Emulator) -> dict[str, object]:
    return {
        "external": bytes(emu.memory.external_memory),
        "overlays": tuple(
            (overlay.name, bytes(overlay.data) if overlay.data is not None else None)
            for overlay in emu.memory.overlays
        ),
        "registers": emu.cpu.snapshot_registers(),
        "lcd": emu.lcd.get_snapshot(),
        "keyboard": deepcopy(emu.keyboard.snapshot_state()),
        "counts": (
            emu.instruction_count,
            emu.cycle_count,
            emu.memory_read_count,
            emu.memory_write_count,
            emu.call_depth,
        ),
        "timer": (
            emu._scheduler.enabled,
            emu._scheduler.mti_period,
            emu._scheduler.sti_period,
            emu._scheduler.next_mti,
            emu._scheduler.next_sti,
        ),
        "interrupts": deepcopy(
            (
                emu._irq_pending,
                emu._in_interrupt,
                emu._key_irq_latched,
                emu._irq_source,
                emu._interrupt_stack,
                emu._next_interrupt_id,
                emu.irq_counts,
                emu.last_irq,
                emu.irq_bit_watch,
            )
        ),
        "power": (
            emu.cpu.state.halted,
            emu.cpu.state.power_state,
        ),
        "poisoned": emu._poisoned,
    }


def test_explicit_llama_snapshot_failure_does_not_fall_back(tmp_path: Path) -> None:
    class FailingLlama:
        @staticmethod
        def is_memory_synced() -> bool:
            return True

        @staticmethod
        def synchronize_host_snapshot_state(*_args: object) -> None:
            return None

        @staticmethod
        def save_snapshot(_path: str) -> None:
            raise ValueError("internal memory length mismatch")

    class LlamaCpuProxy:
        backend = "llama"

        @staticmethod
        def unwrap() -> FailingLlama:
            return FailingLlama()

    emulator = object.__new__(PCE500Emulator)
    emulator.cpu = LlamaCpuProxy()
    emulator._synchronize_llama_snapshot_shadow = lambda _impl: None
    target = tmp_path / "must_not_exist.pcsnap"

    with pytest.raises(ValueError, match="internal memory length mismatch"):
        emulator.save_snapshot(target)

    assert not target.exists()


@pytest.mark.parametrize("poison_source", ["wrapper", "facade", "backend"])
def test_snapshot_save_rejects_poisoned_state_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_source: str,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    target = tmp_path / f"poisoned-{poison_source}.pcsnap"
    emu.save_snapshot(target)
    original = target.read_bytes()

    if poison_source == "wrapper":
        emu._poisoned = "wrapper failure"
    elif poison_source == "facade":
        emu.cpu._contract_poisoned = "facade contract failure"
    else:
        emu.cpu.unwrap()._poisoned = "backend side-effect failure"
    before = _state_fingerprint(emu)

    with pytest.raises(RuntimeError, match="cannot save snapshot.*poisoned"):
        emu.save_snapshot(target)

    assert _state_fingerprint(emu) == before
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("cpu_backend", ["python", "llama"])
def test_snapshot_save_validation_preserves_target_on_inexact_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_backend: str,
) -> None:
    if cpu_backend == "llama" and not _has_llama_backend():
        pytest.skip("LLAMA backend unavailable")
    monkeypatch.setenv("SC62015_CPU_BACKEND", cpu_backend)
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    target = tmp_path / f"inexact-{cpu_backend}.pcsnap"
    emu.save_snapshot(target)
    original = target.read_bytes()

    emu.keyboard._last_kol ^= 0x01
    before = _state_fingerprint(emu)
    with pytest.raises(ValueError, match="keyboard KOL mirrors disagree"):
        emu.save_snapshot(target)

    assert _state_fingerprint(emu) == before
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_llama_metadata_rewrite_failure_preserves_original_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "native.pcsnap"
    host_interrupts = {
        "pending": False,
        "in_interrupt": False,
        "source": None,
        "stack": [],
        "next_id": 1,
        "imr": 0,
        "isr": 0,
        "irq_counts": {"total": 0, "KEY": 0, "MTI": 0, "STI": 0},
        "last_irq": {"src": None, "pc": None, "vector": None},
        "irq_bit_watch": {
            register: {str(bit): {"set": [], "clear": []} for bit in range(8)}
            for register in ("IMR", "ISR")
        },
        "delivered_masks": [],
        "key_irq_latched": False,
        "last_fired": None,
    }
    native_keyboard_fields = {
        "keyi_on_any_press": False,
        "raw_kil": False,
        "emit_events": True,
        "repeat_enabled": True,
        "keyi_latch": False,
        "kil_read_count": 0,
    }
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "snapshot.json",
            json.dumps(
                {
                    "magic": "native",
                    "instruction_count": 0,
                    "cycle_count": 0,
                    "memory_reads": 0,
                    "memory_writes": 0,
                    "pc": 0,
                    "temps": {},
                    "power_state": "running",
                    "call_depth": 0,
                    "call_sub_level": 0,
                    "call_stack": [],
                    "call_page_stack": [],
                    "call_return_widths": [],
                    "interrupts": host_interrupts,
                    "keyboard": native_keyboard_fields,
                }
            ),
        )
        zf.writestr("registers.bin", b"registers")
        zf.writestr("imem.bin", bytes(0x100))
    original = target.read_bytes()

    emulator = object.__new__(PCE500Emulator)
    emulator.cpu = SimpleNamespace(
        backend="llama",
        snapshot_registers=lambda: CPURegistersSnapshot(pc=0),
        state=SimpleNamespace(power_state="running"),
    )
    emulator._capture_scheduler_snapshot_metadata = lambda *_args, **_kwargs: (
        {},
        host_interrupts,
    )
    emulator._capture_keyboard_snapshot_metadata = lambda: {
        "matrix": dict(native_keyboard_fields)
    }

    def fail_replace(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        raise OSError("replace denied")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="atomically rewrite"):
        emulator._patch_llama_snapshot_metadata(target)

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_llama_metadata_parse_failure_is_not_silently_accepted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "malformed.pcsnap"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("snapshot.json", b"not JSON")
    original = target.read_bytes()

    emulator = object.__new__(PCE500Emulator)
    with pytest.raises(RuntimeError, match="parse LLAMA snapshot metadata"):
        emulator._patch_llama_snapshot_metadata(target)

    assert target.read_bytes() == original


def test_snapshot_rejects_late_malformed_fields_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu.cpu.regs.set(RegisterName.PC, 0x12345)
    emu.memory.write_byte(0x3456, 0xA5)
    source = tmp_path / "valid.pcsnap"
    emu.save_snapshot(source)
    before = _state_fingerprint(emu)

    mutations = (
        lambda metadata: metadata["timer"].__setitem__("next_sti", "late-bad"),
        lambda metadata: metadata["interrupts"].__setitem__("key_irq_latched", 1),
        lambda metadata: metadata["interrupts"]["irq_bit_watch"]["ISR"]["7"][
            "clear"
        ].append(0x100000),
        lambda metadata: metadata["keyboard"]["matrix"]["key_states"][
            "KEY_Z"
        ].__setitem__("repeat_ticks", 256),
        lambda metadata: metadata["lcd"]["chips"][1].__setitem__("on_off_count", -1),
        lambda metadata: metadata["kb_metrics"].__setitem__("last_koh", 0x10),
    )
    for index, mutate in enumerate(mutations):
        malformed = tmp_path / f"malformed-{index}.pcsnap"
        _rewrite_snapshot_metadata(source, malformed, mutate)
        with pytest.raises((TypeError, ValueError)):
            emu.load_snapshot(malformed)
        assert _state_fingerprint(emu) == before


def test_python_destination_rejects_llama_only_state_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    source = tmp_path / "python-canonical.pcsnap"
    emu.save_snapshot(source)
    before = _state_fingerprint(emu)

    mutations = (
        lambda metadata: (
            metadata["call_stack"].append(0x12345),
            metadata["call_return_widths"].append(24),
        ),
        lambda metadata: metadata["timer"].__setitem__(
            "instruction_start_cycle", metadata["cycle_count"] + 1
        ),
        lambda metadata: metadata["timer"].update(
            {
                "last_mti_fire_cycle": metadata["cycle_count"],
                "fired_mti_since_boundary": True,
            }
        ),
        lambda metadata: metadata["timer"].__setitem__("preserve_phase", False),
        lambda metadata: metadata["interrupts"].__setitem__("last_fired", "MTI"),
        lambda metadata: metadata["interrupts"]["delivered_masks"].append(0x01),
    )
    for index, mutate in enumerate(mutations):
        target = tmp_path / f"llama-only-{index}.pcsnap"

        def mark_as_llama(metadata: dict[str, object]) -> None:
            metadata["backend"] = "llama"
            mutate(metadata)

        _rewrite_snapshot_metadata(source, target, mark_as_llama)
        with pytest.raises(ValueError, match="Python destination"):
            emu.load_snapshot(target)
        assert _state_fingerprint(emu) == before


def test_snapshot_rejects_inexact_keyboard_duplicates_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    source = tmp_path / "keyboard-canonical.pcsnap"
    emu.save_snapshot(source)
    before = _state_fingerprint(emu)

    mutations = (
        lambda metadata: metadata["keyboard"].__setitem__("ignored", True),
        lambda metadata: metadata["keyboard"].__setitem__(
            "last_kol", metadata["keyboard"]["matrix"]["kol"] ^ 0x01
        ),
        lambda metadata: metadata["keyboard"].__setitem__(
            "last_koh", metadata["keyboard"]["matrix"]["koh"] ^ 0x01
        ),
        lambda metadata: metadata["keyboard"].__setitem__(
            "scan_enabled", not metadata["keyboard"]["matrix"]["scan_enabled"]
        ),
        lambda metadata: metadata["keyboard"]["matrix"].__setitem__(
            "kil_latch", metadata["keyboard"]["matrix"]["kil_latch"] ^ 0x01
        ),
    )
    for index, mutate in enumerate(mutations):
        target = tmp_path / f"keyboard-inexact-{index}.pcsnap"
        _rewrite_snapshot_metadata(source, target, mutate)
        with pytest.raises(ValueError, match="keyboard"):
            emu.load_snapshot(target)
        assert _state_fingerprint(emu) == before


def test_snapshot_rejects_read_only_overlay_replacement_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu.memory.add_rom(0x90000, b"ROM GROUND TRUTH", "snapshot_rom")
    source = tmp_path / "rom.pcsnap"
    emu.save_snapshot(source)
    before = _state_fingerprint(emu)

    with zipfile.ZipFile(source, "r") as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    corrupted = bytearray(entries["external_ram.bin"])
    corrupted[0x90000] ^= 0xFF
    entries["external_ram.bin"] = bytes(corrupted)
    target = tmp_path / "rom-corrupt.pcsnap"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)

    with pytest.raises(ValueError, match="read-only overlay"):
        emu.load_snapshot(target)
    assert _state_fingerprint(emu) == before


def test_snapshot_rejects_unexpected_archive_entry_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    source = tmp_path / "valid.pcsnap"
    emu.save_snapshot(source)
    before = _state_fingerprint(emu)
    with zipfile.ZipFile(source, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ignored-state.bin", b"must not be ignored")

    with pytest.raises(ValueError, match="unexpected entries"):
        emu.load_snapshot(source)
    assert _state_fingerprint(emu) == before


def test_snapshot_roundtrip_preserves_exact_large_timer_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu._scheduler.next_mti = (1 << 40) + 123
    emu._scheduler.next_sti = (1 << 40) + 456
    target = tmp_path / "large-timers.pcsnap"
    emu.save_snapshot(target)
    emu._scheduler.next_mti = 1
    emu._scheduler.next_sti = 2

    emu.load_snapshot(target)

    assert emu._scheduler.next_mti == (1 << 40) + 123
    assert emu._scheduler.next_sti == (1 << 40) + 456


@pytest.mark.parametrize("power_state", ["halted", "off"])
def test_snapshot_roundtrip_restores_stopped_cpu_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    power_state: str,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu.cpu.state.halted = True
    emu.cpu.state.power_state = power_state
    target = tmp_path / f"{power_state}.pcsnap"
    emu.save_snapshot(target)

    emu.cpu.state.halted = False
    emu.cpu.state.power_state = "running"
    emu.load_snapshot(target)

    assert emu.cpu.state.halted is True
    assert emu.cpu.state.power_state == power_state


@pytest.mark.parametrize("latched", [False, True])
def test_snapshot_roundtrip_restores_host_key_irq_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    latched: bool,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu._key_irq_latched = latched
    if latched:
        emu._set_isr_bits(0x04)
        emu._irq_pending = True
        emu._irq_source = IRQSource.KEY
    target = tmp_path / f"key-latch-{latched}.pcsnap"
    emu.save_snapshot(target)

    emu._key_irq_latched = not latched
    emu.load_snapshot(target)

    assert emu._key_irq_latched is latched


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_snapshot_roundtrip_llama_to_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Build and seed a LLAMA-backed emulator snapshot.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu_llama = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    _seed_state(emu_llama)
    snap_path = tmp_path / "llama_snapshot.pcsnap"
    emu_llama.save_snapshot(snap_path)

    # Load into a Python-backed emulator and verify parity of metadata/state.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu_python = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu_python.load_snapshot(snap_path, backend="python")
    _assert_state(emu_python)


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_llama_facade_snapshot_replaces_stale_native_scheduler_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    impl = emu.cpu.unwrap()
    host_timer, host_interrupts = emu._capture_scheduler_snapshot_metadata(
        emu.memory.get_internal_memory_bytes()
    )
    stale_timer = dict(host_timer)
    stale_timer.update(
        {
            "enabled": False,
            "mti_period": 0,
            "sti_period": 0,
            "next_mti": 0,
            "next_sti": 0,
        }
    )
    impl.restore_scheduler_snapshot(stale_timer, host_interrupts, emu.cycle_count)

    native_only = tmp_path / "native-shadow.pcsnap"
    impl.save_snapshot(str(native_only))
    assert _read_snapshot_metadata(native_only)["timer"] == stale_timer

    facade = tmp_path / "facade.pcsnap"
    emu.save_snapshot(facade)
    assert _read_snapshot_metadata(facade)["timer"] == host_timer


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_llama_host_shadow_sync_rejects_all_candidates_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    impl = emu.cpu.unwrap()
    before_path = tmp_path / "before.pcsnap"
    impl.save_snapshot(str(before_path))
    before = _read_snapshot_metadata(before_path)

    timer, interrupts = emu._capture_scheduler_snapshot_metadata(
        emu.memory.get_internal_memory_bytes()
    )
    malformed_interrupts = deepcopy(interrupts)
    malformed_interrupts.update({"pending": True, "source": "MTI", "isr": 0})
    with pytest.raises(ValueError, match="pending with ISR == 0"):
        impl.synchronize_host_snapshot_state(
            timer,
            malformed_interrupts,
            emu._capture_keyboard_snapshot_metadata(),
            99,
            98,
            97,
            96,
        )

    after_path = tmp_path / "after.pcsnap"
    impl.save_snapshot(str(after_path))
    after = _read_snapshot_metadata(after_path)
    for field in (
        "instruction_count",
        "cycle_count",
        "memory_reads",
        "memory_writes",
        "timer",
        "interrupts",
        "keyboard",
    ):
        assert after[field] == before[field]


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_snapshot_roundtrip_python_to_llama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Build and seed a Python-backed emulator snapshot.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu_python = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    _seed_state(emu_python)
    snap_path = tmp_path / "python_snapshot.pcsnap"
    emu_python.save_snapshot(snap_path)

    # Load into a LLAMA-backed emulator and verify parity of metadata/state.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu_llama = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu_llama.load_snapshot(snap_path, backend="llama")
    _assert_state(emu_llama)
