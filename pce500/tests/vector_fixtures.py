"""Stable vector fixtures for synthetic full-machine emulator tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pce500 import PCE500Emulator


ROM_START = 0xC0000
ROM_SIZE = 0x40000
RESET_VECTOR = 0xFFFFD
INTERRUPT_VECTOR = 0xFFFFA
DEFAULT_RESET_TARGET = 0xC0100
DEFAULT_INTERRUPT_TARGET = 0xC0200


def install_static_machine_vectors(
    emulator: "PCE500Emulator",
    *,
    reset_target: int = DEFAULT_RESET_TARGET,
    interrupt_target: int = DEFAULT_INTERRUPT_TARGET,
    reset_target_bytes: bytes = b"\x00",
    interrupt_target_bytes: bytes = b"\x00",
) -> None:
    """Install immutable RESET/IRQ vectors and callback-free target code.

    Synthetic tests used to leave the vector slots in writable flat memory,
    where they alias the SC62015 internal-register backing.  That made the
    selected target depend on IMR/ISR test data.  Use a small static ROM image
    instead so these tests exercise the intended scheduler behavior without
    weakening the production vector-provenance checks.
    """

    for name, target in (
        ("reset_target", reset_target),
        ("interrupt_target", interrupt_target),
    ):
        if isinstance(target, bool) or not 0 <= target <= 0xFFFFF:
            raise ValueError(f"{name} must be a canonical 20-bit address")

    rom = bytearray(ROM_SIZE)
    rom[RESET_VECTOR - ROM_START : RESET_VECTOR - ROM_START + 3] = (
        reset_target.to_bytes(3, "little")
    )
    rom[INTERRUPT_VECTOR - ROM_START : INTERRUPT_VECTOR - ROM_START + 3] = (
        interrupt_target.to_bytes(3, "little")
    )

    external_targets: list[tuple[str, int, bytes]] = []
    for name, target, payload in (
        ("reset", reset_target, reset_target_bytes),
        ("interrupt", interrupt_target, interrupt_target_bytes),
    ):
        if not payload:
            raise ValueError(f"{name} target payload must not be empty")
        if ROM_START <= target and target + len(payload) <= ROM_START + ROM_SIZE:
            offset = target - ROM_START
            rom[offset : offset + len(payload)] = payload
        else:
            external_targets.append((name, target, payload))

    emulator.load_rom(bytes(rom), start_address=ROM_START)
    for name, target, payload in external_targets:
        emulator.memory.add_rom(
            target,
            payload,
            f"synthetic_{name}_vector_target_{target:05x}",
        )
