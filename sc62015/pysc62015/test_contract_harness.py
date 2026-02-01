from __future__ import annotations

import pytest

from sc62015.pysc62015.contract_harness import (
    AccessVector,
    PythonContractBackend,
    RustContractBackend,
    run_dual,
)


def _init_backends():
    py_backend = PythonContractBackend()
    try:
        rust_backend = RustContractBackend(host_memory=py_backend.memory)
    except (
        RuntimeError
    ) as exc:  # pragma: no cover - exercised in CI when rustcore exists
        pytest.skip(str(exc))

    # Preload external/internal blobs so reads have deterministic seeds.
    external = bytes(range(0x30)) * 0x100  # at least 0x3000 bytes
    internal = bytes([0xAA] * 256)
    py_backend.load_memory(external=external, internal=internal)
    rust_backend.load_memory(external=external, internal=internal)
    return py_backend, rust_backend


def test_contract_vectors_match_between_backends():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x2000, 0x12, pc=0x010203),
        AccessVector("read", 0x2000, pc=0x010203),
        AccessVector("write", 0xA005, 0x34, pc=0x010210),
        AccessVector("read", 0xA005, pc=0x010210),
        AccessVector("write", 0x100000 + 0xF0, 0x55, pc=0x010300),
        AccessVector("read", 0x100000 + 0xF0, pc=0x010300),
        AccessVector("write", 0x100000 + 0xFB, 0xC3, pc=0x010308),  # IMR
        AccessVector("read", 0x100000 + 0xFB, pc=0x010310),  # IMR readback
        AccessVector("write", 0x100000 + 0xFC, 0x0F, pc=0x010320),  # ISR
        AccessVector("read", 0x100000 + 0xFC, pc=0x010322),  # ISR readback
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value, e.pc) for e in rs_run.events]
    assert py_events == rs_events

    # Internal memory snapshots should match byte-for-byte.
    assert py_run.snapshot.internal == rs_run.snapshot.internal

    # Check a few external addresses that were touched.
    assert py_run.snapshot.external is not None
    assert rs_run.snapshot.external is not None
    py_ext = py_run.snapshot.external
    rs_ext = rs_run.snapshot.external
    for addr in (0x2000, 0x2001, 0xA005):
        assert py_ext[addr] == rs_ext[addr]

    # Internal bytes (IMR/KIO) should align after the vector sequence.
    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    assert py_int[0xFB] == rs_int[0xFB]  # IMR read value from writes
    assert py_int[0xFC] == rs_int[0xFC]  # ISR read value from writes
    assert py_int[0xF0] == rs_int[0xF0]  # KOL write mirrored

    # IMR/ISR surfaced in snapshots should align.
    assert py_run.snapshot.imr == rs_run.snapshot.imr
    assert py_run.snapshot.isr == rs_run.snapshot.isr

    # LCD-facing events should align via snapshot metadata.
    py_lcd = [(e.kind, e.address, e.value, e.pc) for e in py_run.snapshot.lcd_events]
    rs_lcd = [(e.kind, e.address, e.value, e.pc) for e in rs_run.snapshot.lcd_events]
    assert py_lcd == rs_lcd


def test_irq_and_lcd_contract_vectors():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x100000 + 0xFB, 0x00, pc=0x020000),  # IMR clear
        AccessVector("write", 0x100000 + 0xFC, 0x00, pc=0x020002),  # ISR clear
        AccessVector("write", 0x100000 + 0xFB, 0x04, pc=0x020010),  # IMR KEYI enable
        AccessVector("write", 0x100000 + 0xFC, 0x04, pc=0x020012),  # ISR KEYI set
        AccessVector("read", 0x100000 + 0xFB, pc=0x020014),  # IMR readback
        AccessVector("read", 0x100000 + 0xFC, pc=0x020016),  # ISR readback
        AccessVector("write", 0x2000, 0x77, pc=0x020100),  # LCD low window
        AccessVector("read", 0x2000, pc=0x020102),
        AccessVector("write", 0xA000, 0x99, pc=0x020110),  # LCD high window
        AccessVector("read", 0xA000, pc=0x020112),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value, e.pc) for e in rs_run.events]
    assert py_events == rs_events

    # IMR/ISR parity from snapshots.
    assert py_run.snapshot.imr == rs_run.snapshot.imr == 0x04
    assert py_run.snapshot.isr == rs_run.snapshot.isr == 0x04

    # LCD-facing events captured in snapshots should align and include both windows.
    py_lcd = [(e.kind, e.address, e.value, e.pc) for e in py_run.snapshot.lcd_events]
    rs_lcd = [(e.kind, e.address, e.value, e.pc) for e in rs_run.snapshot.lcd_events]
    assert py_lcd == rs_lcd
    assert {addr for _, addr, _, _ in py_lcd} == {0x2000, 0xA000}


def test_timer_keyi_parity():
    py_backend, rust_backend = _init_backends()
    py_backend.configure_timer(mti_period=1, sti_period=2, enabled=True)
    rust_backend.configure_timer(mti_period=1, sti_period=2, enabled=True)

    # Prime IMR to enable both MTI/STI bits; ISR starts at 0.
    for backend in (py_backend, rust_backend):
        backend.write(0x100000 + 0xFB, 0x03, pc=0x030000)
        backend.write(0x100000 + 0xFC, 0x00, pc=0x030002)

    # Tick a few cycles; MTI fires at 1,3,5,...; STI at 2,4,...
    for _ in range(3):
        py_backend.tick_timers()
        rust_backend.tick_timers()

    py_snap = py_backend.snapshot()
    rs_snap = rust_backend.snapshot()

    # Both should have ISR bits set for the latest MTI/STI ticks.
    assert py_snap.isr == rs_snap.isr
    assert py_snap.isr & 0x03 == 0x03

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_backend.drain_events()]
    rs_events = [
        (e.kind, e.address, e.value, e.pc) for e in rust_backend.drain_events()
    ]
    assert py_events == rs_events


def test_onk_press_sets_pending_parity():
    py_backend, rust_backend = _init_backends()
    py_backend.press_on_key()
    rust_backend.press_on_key()

    py_snap = py_backend.snapshot()
    rs_snap = rust_backend.snapshot()

    assert py_snap.isr & 0x08 == rs_snap.isr & 0x08 == 0x08
    assert py_snap.metadata.get("irq_pending") is True
    assert rs_snap.metadata.get("irq_pending") is True
    assert (
        py_snap.metadata.get("irq_source")
        == rs_snap.metadata.get("irq_source")
        == "ONK"
    )

    py_events = [(e.kind, e.address, e.value) for e in py_backend.drain_events()]
    rs_events = [(e.kind, e.address, e.value) for e in rust_backend.drain_events()]
    assert py_events == rs_events


def test_keyboard_press_helper_sets_pending_parity():
    py_backend, rust_backend = _init_backends()
    assert py_backend.press_matrix_code(0x00)
    assert rust_backend.press_matrix_code(0x00)

    py_snap = py_backend.snapshot()
    rs_snap = rust_backend.snapshot()

    assert py_snap.isr & 0x04 == rs_snap.isr & 0x04 == 0x04
    assert py_snap.metadata.get("irq_pending") is True
    assert rs_snap.metadata.get("irq_pending") is True
    assert (
        py_snap.metadata.get("irq_source")
        == rs_snap.metadata.get("irq_source")
        == "KEY"
    )

    py_events = [(e.kind, e.address, e.value) for e in py_backend.drain_events()]
    rs_events = [(e.kind, e.address, e.value) for e in rust_backend.drain_events()]
    assert py_events == rs_events


def test_external_wraparound_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x0FFFFF, 0xAB, pc=0x050000),
        AccessVector("write", 0x1000000, 0xCD, pc=0x050002),  # wraps to 0x000000
        AccessVector("read", 0x0FFFFF, pc=0x050010),
        AccessVector("read", 0x000000, pc=0x050012),
        AccessVector("write", 0x1FFFFF, 0xEF, pc=0x050020),  # wraps to 0x0FFFFF
        AccessVector("read", 0x0FFFFF, pc=0x050022),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    assert py_run.snapshot.external is not None
    assert rs_run.snapshot.external is not None
    py_ext = py_run.snapshot.external
    rs_ext = rs_run.snapshot.external
    assert py_ext[0x0FFFFF] == rs_ext[0x0FFFFF] == 0xEF
    assert py_ext[0x000000] == rs_ext[0x000000] == 0xCD


def test_imem_ex_register_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        # Write BA low/high via EX/EXL-style address patterns.
        AccessVector("write", 0x100000 + 0xB8, 0x11, pc=0x060000),  # BA low
        AccessVector("write", 0x100000 + 0xB9, 0x22, pc=0x060002),  # BA high
        AccessVector("read", 0x100000 + 0xB8, pc=0x060010),
        AccessVector("read", 0x100000 + 0xB9, pc=0x060012),
        # DSLL/DSRL edge: write then read IMEM scratch (e.g., 0xB0/0xB1).
        AccessVector("write", 0x100000 + 0xB0, 0xF0, pc=0x060020),
        AccessVector("write", 0x100000 + 0xB1, 0x0F, pc=0x060022),
        AccessVector("read", 0x100000 + 0xB0, pc=0x060030),
        AccessVector("read", 0x100000 + 0xB1, pc=0x060032),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    # IMEM bytes should match.
    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    for offset in (0xB0, 0xB1, 0xB8, 0xB9):
        assert py_int[offset] == rs_int[offset]


def test_dsll_dsrll_imem_cadence():
    py_backend, rust_backend = _init_backends()
    vectors = [
        # Seed IMEM scratch bytes for DSLL/DSRL edge cases.
        AccessVector("write", 0x100000 + 0xB2, 0xF1, pc=0x070000),
        AccessVector("write", 0x100000 + 0xB3, 0x0E, pc=0x070002),
        # Simulate DSLL/DSRL by writing/reading back the same offsets.
        AccessVector("read", 0x100000 + 0xB2, pc=0x070010),
        AccessVector("read", 0x100000 + 0xB3, pc=0x070012),
        # Update and read again to mimic cadence changes.
        AccessVector("write", 0x100000 + 0xB2, 0xA5, pc=0x070020),
        AccessVector("write", 0x100000 + 0xB3, 0x5A, pc=0x070022),
        AccessVector("read", 0x100000 + 0xB2, pc=0x070030),
        AccessVector("read", 0x100000 + 0xB3, pc=0x070032),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_int = runs["python"].snapshot.internal
    rs_int = runs["llama"].snapshot.internal
    for offset in (0xB2, 0xB3):
        assert py_int[offset] == rs_int[offset]


def test_requires_python_overlays_are_delegated():
    py_backend, rust_backend = _init_backends()
    # Default keyboard bridge is disabled, so KIO and ON/ONK should route via host memory.
    vectors = [
        AccessVector("read", 0x100000 + 0xF5, pc=0x080000),  # ON
        AccessVector("read", 0x100000 + 0xF6, pc=0x080002),  # ONK
        AccessVector("write", 0x100000 + 0xF0, 0xFF, pc=0x080010),  # KOL
        AccessVector("write", 0x100000 + 0xF1, 0x07, pc=0x080012),  # KOH
        AccessVector("read", 0x100000 + 0xF2, pc=0x080020),  # KIL latch
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value, e.pc) for e in rs_run.events]
    assert py_events == rs_events

    # KIO/ON/ONK reads should match exactly.
    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    for offset in (0xF0, 0xF1, 0xF2, 0xF5, 0xF6):
        assert py_int[offset] == rs_int[offset]


def test_lcd_reads_parity():
    py_backend, rust_backend = _init_backends()
    fallback_addr = 0xA001
    fallback_expected = py_backend.memory.external_memory[fallback_addr & 0xFFFFF]
    vectors = [
        AccessVector("write", 0x2000, 0x82, pc=0x090000),  # Set page=2
        AccessVector("write", 0x2000, 0x41, pc=0x090002),  # Set Y=1
        AccessVector("write", 0x2002, 0xAA, pc=0x090004),  # Data write (left)
        AccessVector("read", 0x200B, pc=0x090006),  # Data read (buffered)
        AccessVector("read", fallback_addr, pc=0x090008),  # CS=both fallback
        AccessVector("read", 0x2009, pc=0x09000A),  # Status read (left)
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value, e.pc) for e in rs_run.events]
    assert py_events == rs_events

    data_read = next(
        e.value for e in py_run.events if e.kind == "read" and e.address == 0x200B
    )
    assert data_read == 0xAA
    fallback_read = next(
        e.value
        for e in py_run.events
        if e.kind == "read" and e.address == fallback_addr
    )
    assert fallback_read == fallback_expected
    status_read = next(
        e.value for e in py_run.events if e.kind == "read" and e.address == 0x2009
    )
    assert status_read == 0xA0


def test_exl_dsbl_cadence_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        # Seed IMEM offsets to mimic EXL/DSBL effects.
        AccessVector("write", 0x100000 + 0xBE, 0x10, pc=0x080000),  # scratch low
        AccessVector("write", 0x100000 + 0xBF, 0x32, pc=0x080002),  # scratch high
        AccessVector("write", 0x100000 + 0xBB, 0x55, pc=0x080004),  # DSBL target
        AccessVector("write", 0x100000 + 0xBC, 0xAA, pc=0x080006),  # DSBL target
        # Read back in EXL order (low/high) to ensure parity.
        AccessVector("read", 0x100000 + 0xBE, pc=0x080010),
        AccessVector("read", 0x100000 + 0xBF, pc=0x080012),
        AccessVector("read", 0x100000 + 0xBB, pc=0x080014),
        AccessVector("read", 0x100000 + 0xBC, pc=0x080016),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_int = runs["python"].snapshot.internal
    rs_int = runs["llama"].snapshot.internal
    for offset in (0xBB, 0xBC, 0xBE, 0xBF):
        assert py_int[offset] == rs_int[offset]


def test_imr_isr_cadence_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x100000 + 0xFB, 0x00, pc=0x090000),
        AccessVector("write", 0x100000 + 0xFC, 0x00, pc=0x090002),
        AccessVector(
            "write", 0x100000 + 0xFB, 0x07, pc=0x090010
        ),  # enable MTI/STI/KEYI
        AccessVector("write", 0x100000 + 0xFC, 0x05, pc=0x090012),  # set ISR bits
        AccessVector("read", 0x100000 + 0xFB, pc=0x090020),
        AccessVector("read", 0x100000 + 0xFC, pc=0x090022),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_int = runs["python"].snapshot.internal
    rs_int = runs["llama"].snapshot.internal
    assert py_int[0xFB] == rs_int[0xFB] == 0x07
    assert py_int[0xFC] == rs_int[0xFC] == 0x05


def test_imr_isr_toggle_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x100000 + 0xFB, 0xFF, pc=0x0B0000),
        AccessVector("write", 0x100000 + 0xFC, 0x00, pc=0x0B0002),
        AccessVector("read", 0x100000 + 0xFB, pc=0x0B0010),
        AccessVector("write", 0x100000 + 0xFB, 0x0F, pc=0x0B0020),
        AccessVector("write", 0x100000 + 0xFC, 0xF0, pc=0x0B0022),
        AccessVector("read", 0x100000 + 0xFB, pc=0x0B0030),
        AccessVector("read", 0x100000 + 0xFC, pc=0x0B0032),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_int = runs["python"].snapshot.internal
    rs_int = runs["llama"].snapshot.internal
    assert py_int[0xFB] == rs_int[0xFB] == 0x0F
    assert py_int[0xFC] == rs_int[0xFC] == 0xF0


def test_keyboard_kio_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x100000 + 0xF0, 0xFF, pc=0x0C1000),  # KOL
        AccessVector("write", 0x100000 + 0xF1, 0x07, pc=0x0C1002),  # KOH
        AccessVector("read", 0x100000 + 0xF0, pc=0x0C1010),  # KOL readback
        AccessVector("read", 0x100000 + 0xF1, pc=0x0C1012),  # KOH readback
        AccessVector("read", 0x100000 + 0xF2, pc=0x0C1014),  # KIL (no keys pressed)
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value) for e in rs_run.events]
    assert py_events == rs_events

    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    # KIO writes hit the overlay but do not persist into internal RAM; parity only matters across backends.
    assert py_int[0xF0] == rs_int[0xF0] == 0xAA
    assert py_int[0xF1] == rs_int[0xF1] == 0xAA
    assert py_int[0xF2] == rs_int[0xF2] == 0xAA


def test_keyboard_irq_delivery_parity():
    py_backend, rust_backend = _init_backends()
    # Enable KEYI in IMR.
    for backend in (py_backend, rust_backend):
        backend.write(0x100000 + 0xFB, 0x04, pc=0x0D0000)
        backend.write(0x100000 + 0xFC, 0x00, pc=0x0D0002)

    # Simulate a key press by strobing KOL/KOH; both backends should set KIL. KEYI assertion
    # requires an actual matrix event; keep expectation limited to matching ISR/KIL snapshots.
    vectors = [
        AccessVector("write", 0x100000 + 0xF0, 0x01, pc=0x0D0010),
        AccessVector("write", 0x100000 + 0xF1, 0x00, pc=0x0D0012),
        AccessVector("read", 0x100000 + 0xF2, pc=0x0D0014),
    ]
    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    assert py_int[0xF2] == rs_int[0xF2]
    # ISR parity (KEYI bit) should match across backends even if not set.
    assert (py_int[0xFC] & 0x04) == (rs_int[0xFC] & 0x04)


def test_ex_memory_and_dsbl_seeds_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        # EX-style memory swap across two addresses.
        AccessVector("write", 0x000100, 0xAA, pc=0x0C0000),
        AccessVector("write", 0x000101, 0xBB, pc=0x0C0002),
        AccessVector("write", 0x000102, 0xCC, pc=0x0C0004),
        AccessVector("write", 0x000103, 0xDD, pc=0x0C0006),
        AccessVector("read", 0x000100, pc=0x0C0010),
        AccessVector("read", 0x000101, pc=0x0C0012),
        AccessVector("read", 0x000102, pc=0x0C0014),
        AccessVector("read", 0x000103, pc=0x0C0016),
        # DSBL seed mutations in IMEM.
        AccessVector("write", 0x100000 + 0xBD, 0x12, pc=0x0C0020),
        AccessVector("write", 0x100000 + 0xBD, 0x34, pc=0x0C0022),
        AccessVector("read", 0x100000 + 0xBD, pc=0x0C0030),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_ext = runs["python"].snapshot.external
    rs_ext = runs["llama"].snapshot.external
    assert py_ext is not None and rs_ext is not None
    for addr in (0x000100, 0x000101, 0x000102, 0x000103):
        assert py_ext[addr] == rs_ext[addr]
    py_int = runs["python"].snapshot.internal
    rs_int = runs["llama"].snapshot.internal
    assert py_int[0xBD] == rs_int[0xBD] == 0x34


def test_exl_memory_and_dsbl_edges():
    py_backend, rust_backend = _init_backends()
    vectors = [
        # EXL memory pattern: write a word across boundary and read back both bytes.
        AccessVector("write", 0x0A_FFFE, 0x12, pc=0x0A0000),
        AccessVector("write", 0x0A_FFFF, 0x34, pc=0x0A0002),
        AccessVector("read", 0x0A_FFFE, pc=0x0A0010),
        AccessVector("read", 0x0A_FFFF, pc=0x0A0012),
        # DSBL-like edge toggles in IMEM scratch.
        AccessVector("write", 0x100000 + 0xBC, 0x00, pc=0x0A0020),
        AccessVector("write", 0x100000 + 0xBC, 0xFF, pc=0x0A0022),
        AccessVector("read", 0x100000 + 0xBC, pc=0x0A0030),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    assert py_run.snapshot.external is not None
    assert rs_run.snapshot.external is not None
    py_ext = py_run.snapshot.external
    rs_ext = rs_run.snapshot.external
    for addr in (0x0A_FFFE, 0x0A_FFFF):
        assert py_ext[addr] == rs_ext[addr]

    py_int = py_run.snapshot.internal
    rs_int = rs_run.snapshot.internal
    assert py_int[0xBC] == rs_int[0xBC] == 0xFF


def test_lcd_status_and_data_parity():
    py_backend, rust_backend = _init_backends()
    vectors = [
        AccessVector("write", 0x2000, 0x3F, pc=0x040000),  # Instruction write
        AccessVector("write", 0x2002, 0x12, pc=0x040002),  # Data write (cs=left)
        AccessVector("write", 0xA001, 0x7F, pc=0x040010),  # Instruction write (cs=both)
        AccessVector("write", 0xA003, 0x34, pc=0x040012),  # Data write
        AccessVector("write", 0x2000, 0x82, pc=0x040014),  # Set page=2 (both)
        AccessVector("write", 0x2000, 0x41, pc=0x040016),  # Set Y=0x01
        AccessVector(
            "write", 0x2002, 0xAA, pc=0x040018
        ),  # Data write (left chip page2,y1)
        AccessVector(
            "write", 0x2004, 0xBB, pc=0x04001A
        ),  # Data write (right chip page2,y2)
        AccessVector("write", 0x2000, 0x7E, pc=0x04001C),  # Set Y near wrap (0x3E)
        AccessVector("write", 0x2002, 0xCC, pc=0x04001E),  # Data write at y=0x3E
        AccessVector(
            "write", 0x2002, 0xDD, pc=0x04001F
        ),  # Data write at y=0x3F -> wraps to 0
        AccessVector("read", 0x2001, pc=0x040020),  # Status/busy read
        AccessVector("read", 0xA001, pc=0x040022),  # Status/busy read
        AccessVector(
            "read", 0x2002, pc=0x040030
        ),  # Data readback (mirrors written byte)
        AccessVector("read", 0xA003, pc=0x040032),
    ]

    runs = run_dual(vectors, python_backend=py_backend, rust_backend=rust_backend)
    py_run = runs["python"]
    rs_run = runs["llama"]

    py_events = [(e.kind, e.address, e.value, e.pc) for e in py_run.events]
    rs_events = [(e.kind, e.address, e.value, e.pc) for e in rs_run.events]
    assert py_events == rs_events

    # External memory should mirror the writes at the LCD addresses.
    assert py_run.snapshot.external is not None
    assert rs_run.snapshot.external is not None
    for addr in (0x2000, 0x2002, 0xA001, 0xA003):
        assert py_run.snapshot.external[addr] == rs_run.snapshot.external[addr]

    # LCD write log parity.
    assert py_run.snapshot.lcd_log == rs_run.snapshot.lcd_log

    # LCD events captured in snapshot should align.
    py_lcd = [(e.kind, e.address, e.value, e.pc) for e in py_run.snapshot.lcd_events]
    rs_lcd = [(e.kind, e.address, e.value, e.pc) for e in rs_run.snapshot.lcd_events]
    assert py_lcd == rs_lcd

    # Status/busy reads should match across backends and surface via snapshot.
    status_reads = [
        e for e in py_run.events if e.kind == "read" and e.address in (0x2001, 0xA001)
    ]
    rust_status_reads = [
        e for e in rs_run.events if e.kind == "read" and e.address in (0x2001, 0xA001)
    ]
    assert [(e.address, e.value) for e in status_reads] == [
        (e.address, e.value) for e in rust_status_reads
    ]
    assert py_run.snapshot.lcd_status == rs_run.snapshot.lcd_status

    # VRAM snapshots should align.
    assert py_run.snapshot.lcd_vram is not None
    assert rs_run.snapshot.lcd_vram is not None
    assert py_run.snapshot.lcd_vram == rs_run.snapshot.lcd_vram
