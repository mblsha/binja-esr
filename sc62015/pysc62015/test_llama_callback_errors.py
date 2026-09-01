from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from binja_test_mocks.eval_llil import Memory
from _sc62015_rustcore import LlamaCPU as RawLlamaCPU  # pyright: ignore[reportAttributeAccessIssue]

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.stepper import CPURegistersSnapshot


pytestmark = pytest.mark.skipif(
    "llama" not in available_backends(),
    reason="LLAMA backend not available in this runtime",
)


def _attach_preflight_peek(memory: object, backing: bytearray) -> None:
    def peek(address: int, _pc: int | None = None) -> int:
        return backing[address & 0xFFFFFF]

    setattr(memory, "peek_byte_for_preflight", peek)


def _memory(
    program: bytes,
    *,
    fail_reads: bool = False,
    fail_writes: bool = False,
) -> Memory:
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[: len(program)] = program

    def read(address: int) -> int:
        if fail_reads:
            raise RuntimeError("read boom")
        return backing[address & 0xFFFFFF]

    def write(address: int, value: int) -> None:
        if fail_writes:
            raise RuntimeError("write boom")
        backing[address & 0xFFFFFF] = value & 0xFF

    memory = Memory(read, write)
    _attach_preflight_peek(memory, backing)
    return memory


def _raw_llama_cpu(memory: Memory) -> tuple[CPU, Any]:
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    return cpu, cpu.unwrap()


def test_llama_bridge_propagates_memory_read_errors() -> None:
    _cpu, raw_cpu = _raw_llama_cpu(_memory(b"\x00", fail_reads=True))

    with pytest.raises(RuntimeError, match="read boom"):
        raw_cpu.execute_instruction(0)


def test_llama_bridge_propagates_memory_write_errors() -> None:
    # MV (0x10),0x42
    _cpu, raw_cpu = _raw_llama_cpu(_memory(b"\xcc\x10\x42", fail_writes=True))

    with pytest.raises(RuntimeError, match="write boom"):
        raw_cpu.execute_instruction(0)


def test_llama_bridge_propagates_wait_hook_errors() -> None:
    memory = _memory(b"\xef")

    def wait_cycles(_cycles: int) -> None:
        raise RuntimeError("wait boom")

    setattr(memory, "wait_cycles", wait_cycles)
    cpu, raw_cpu = _raw_llama_cpu(memory)
    cpu.regs.set(RegisterName.I, 1)

    with pytest.raises(RuntimeError, match="wait boom"):
        raw_cpu.execute_instruction(0)


def test_llama_bridge_propagates_wait_fallback_read_errors() -> None:
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[0] = 0xEF

    def read(address: int) -> int:
        if address >= INTERNAL_MEMORY_START:
            raise RuntimeError("fallback read boom")
        return backing[address & 0xFFFFFF]

    memory = Memory(read, lambda address, value: None)
    _attach_preflight_peek(memory, backing)
    cpu, raw_cpu = _raw_llama_cpu(memory)
    cpu.regs.set(RegisterName.I, 1)

    with pytest.raises(RuntimeError, match="fallback read boom"):
        raw_cpu.execute_instruction(0)


def test_llama_bridge_propagates_wait_fallback_write_errors() -> None:
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[0] = 0xEF
    # Make the fallback's IMR mirror sync dirty so it must flush a host write.
    backing[INTERNAL_MEMORY_START + 0xFB] = 0x01

    def write(address: int, value: int) -> None:
        if address >= INTERNAL_MEMORY_START:
            raise RuntimeError("fallback write boom")
        backing[address & 0xFFFFFF] = value & 0xFF

    memory = Memory(lambda address: backing[address & 0xFFFFFF], write)
    _attach_preflight_peek(memory, backing)
    cpu, raw_cpu = _raw_llama_cpu(memory)
    cpu.regs.set(RegisterName.I, 1)

    with pytest.raises(RuntimeError, match="fallback write boom"):
        raw_cpu.execute_instruction(0)


def test_llama_bridge_propagates_lcd_hook_errors() -> None:
    # OR [0x0A000],1
    memory = _memory(bytes([0x7A, 0x00, 0xA0, 0x00, 0x01]))

    def lcd_hook(_address: int, _value: int, _pc: int) -> None:
        raise RuntimeError("lcd boom")

    setattr(memory, "_llama_lcd_write", lcd_hook)
    _cpu, raw_cpu = _raw_llama_cpu(memory)

    with pytest.raises(RuntimeError, match="lcd boom"):
        raw_cpu.execute_instruction(0)


def test_llama_reset_propagates_memory_read_errors() -> None:
    memory = _memory(b"\x00", fail_reads=True)

    with pytest.raises(RuntimeError, match="read boom"):
        RawLlamaCPU(memory=memory, reset_on_init=True)


def test_llama_reset_propagates_memory_write_errors() -> None:
    memory = _memory(b"\x00", fail_writes=True)

    with pytest.raises(RuntimeError, match="write boom"):
        RawLlamaCPU(memory=memory, reset_on_init=True)


def test_llama_reset_writes_each_reset_field_once() -> None:
    class CountingMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[0xFFFFD:0x100000] = b"\x00\x00\x00"
            self.writes: list[tuple[int, int]] = []
            self.callback_pcs: list[int] = []

        def read_byte(self, address: int, pc: int) -> int:
            self.callback_pcs.append(pc)
            return self.raw[address & 0xFFFFFF]

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, pc: int) -> None:
            self.callback_pcs.append(pc)
            address &= 0xFFFFFF
            value &= 0xFF
            self.writes.append((address, value))
            self.raw[address] = value

    memory = CountingMemory()
    RawLlamaCPU(memory=memory, reset_on_init=True)

    reset_addresses = {
        INTERNAL_MEMORY_START + offset
        for offset in (0xF7, 0xF8, 0xFC, 0xFD, 0xFE, 0xFF)
    }
    assert {address for address, _value in memory.writes} == reset_addresses
    assert len(memory.writes) == len(reset_addresses)
    assert memory.callback_pcs and set(memory.callback_pcs) == {0}


def test_llama_wait_fallback_supports_modern_only_memory_callbacks() -> None:
    class ModernMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[0] = 0xEF
            self.raw[INTERNAL_MEMORY_START + 0xFB] = 0x01
            self.read_pcs: list[int] = []
            self.write_pcs: list[int] = []

        def read_byte(self, address: int, pc: int) -> int:
            self.read_pcs.append(pc)
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, pc: int) -> None:
            self.write_pcs.append(pc)
            self.raw[address & 0xFFFFFF] = value & 0xFF

        def peek_byte_for_preflight(self, address: int, _pc: int) -> int:
            return self.raw[address & 0xFFFFFF]

    memory = ModernMemory()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)
    raw_cpu.write_register("I", 1)

    raw_cpu.execute_instruction(0)

    assert memory.read_pcs and set(memory.read_pcs) == {0}
    assert memory.write_pcs and set(memory.write_pcs) == {0}


def test_llama_mirror_sync_propagates_memory_write_errors() -> None:
    memory = _memory(b"\x00", fail_writes=True)
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)

    with pytest.raises(RuntimeError, match="write boom"):
        raw_cpu.keyboard_press_on_key()


def test_llama_notify_host_write_does_not_repeat_host_memory_write() -> None:
    raw_cpu = RawLlamaCPU(
        memory=_memory(b"\x00", fail_writes=True), reset_on_init=False
    )

    raw_cpu.notify_host_write(0x20, 0x42)

    # The notification is a post-commit mirror update, not another host write,
    # and therefore must neither invoke the failing callback nor poison the CPU.
    raw_cpu.execute_instruction(0)


def test_llama_bridge_round_trips_temp15_snapshots() -> None:
    raw_cpu = RawLlamaCPU(memory=_memory(b"\x00"), reset_on_init=False)
    raw_cpu.write_register("TEMP15", 0x123456)

    snapshot = raw_cpu.snapshot_cpu_registers()
    assert snapshot.temps[15] == 0x123456
    assert raw_cpu.read_register("TEMP15") == 0x123456

    raw_cpu.load_cpu_snapshot(CPURegistersSnapshot(pc=0, temps={15: 0x654321}))
    assert raw_cpu.read_register("TEMP15") == 0x654321

    raw_cpu.load_cpu_snapshot(CPURegistersSnapshot(pc=0, temps={}))
    assert raw_cpu.read_register("TEMP15") == 0

    with pytest.raises(ValueError, match="unknown register"):
        raw_cpu.write_register("TEMP16", 1)


_MISSING = object()


@pytest.mark.parametrize(
    ("field", "malformed", "error_type"),
    [
        ("f", _MISSING, AttributeError),
        ("f", "not-an-integer", TypeError),
        ("pc", _MISSING, AttributeError),
        ("s", "not-an-integer", TypeError),
        ("temps", [], TypeError),
        ("temps", {16: 1}, ValueError),
        ("call_sub_level", "not-an-integer", TypeError),
        ("call_sub_level", -1, ValueError),
    ],
)
def test_llama_malformed_snapshot_is_rejected_atomically(
    field: str, malformed: object, error_type: type[Exception]
) -> None:
    raw_cpu = RawLlamaCPU(memory=_memory(b"\x00"), reset_on_init=False)
    for name, value in {
        "PC": 0x12345,
        "BA": 0x5678,
        "I": 0x9ABC,
        "X": 0x13579,
        "Y": 0x24680,
        "U": 0x11111,
        "S": 0x22222,
        "F": 0x03,
        "TEMP15": 0x654321,
    }.items():
        raw_cpu.write_register(name, value)
    raw_cpu.call_sub_level = 4
    before = raw_cpu.snapshot_cpu_registers()
    candidate = SimpleNamespace(
        pc=0xAAAAA,
        ba=0xBBBB,
        i=0xCCCC,
        x=0xDDDDD,
        y=0xEEEEE,
        u=0x33333,
        s=0x44444,
        f=0x01,
        temps={15: 0x123456},
        call_sub_level=2,
    )
    if malformed is _MISSING:
        delattr(candidate, field)
    else:
        setattr(candidate, field, malformed)

    with pytest.raises(error_type):
        raw_cpu.load_cpu_snapshot(candidate)

    assert raw_cpu.snapshot_cpu_registers() == before


@pytest.mark.parametrize(
    "malformed_interrupts",
    [
        SimpleNamespace(
            imr="not-an-integer",
            isr=0x02,
            pending=True,
            in_interrupt=False,
            source="KEY",
            stack=[1],
            next_id=2,
            irq_counts={"total": 1, "KEY": 1, "MTI": 0, "STI": 0},
            last_irq={"src": "KEY", "pc": 0x12345, "vector": 0xFFFFA},
        ),
        SimpleNamespace(
            imr=0x01,
            isr=0x02,
            pending=True,
            in_interrupt=False,
            source="KEY",
            stack=[1],
            next_id=2,
            irq_counts={"total": 1, "KEY": 1, "MTI": 0, "STI": 0},
            last_irq={"src": "KEY", "pc": "not-an-integer", "vector": 0xFFFFA},
        ),
        SimpleNamespace(
            imr=0x01,
            isr=0x02,
            pending=True,
            in_interrupt=False,
            source="KEY",
            stack=[1],
            next_id=2,
            irq_counts={"total": 1, "KEY": 1, "MTI": 0},
            last_irq={"src": "KEY", "pc": 0x12345, "vector": 0xFFFFA},
        ),
    ],
)
def test_llama_malformed_interrupt_snapshot_is_rejected_atomically(
    malformed_interrupts: object,
) -> None:
    raw_cpu = RawLlamaCPU(memory=_memory(b"\x00"), reset_on_init=False)
    raw_cpu.write_register("PC", 0x12345)
    raw_cpu.write_register("F", 0x03)
    raw_cpu.write_register("TEMP15", 0x654321)
    raw_cpu.call_sub_level = 4
    before = raw_cpu.snapshot_cpu_registers()
    candidate = SimpleNamespace(
        pc=0xAAAAA,
        ba=0xBBBB,
        i=0xCCCC,
        x=0xDDDDD,
        y=0xEEEEE,
        u=0x33333,
        s=0x44444,
        f=0x01,
        temps={15: 0x123456},
        call_sub_level=2,
        interrupts=malformed_interrupts,
    )

    with pytest.raises((TypeError, ValueError)):
        raw_cpu.load_cpu_snapshot(candidate)

    assert raw_cpu.snapshot_cpu_registers() == before


def test_llama_power_state_proxy_distinguishes_halt_off_and_running() -> None:
    halt_cpu, halt_raw = _raw_llama_cpu(_memory(b"\xde"))
    halt_raw.execute_instruction(0)
    assert halt_cpu.state.halted
    assert halt_cpu.state.power_state == "halted"

    off_cpu, off_raw = _raw_llama_cpu(_memory(b"\xdf"))
    off_raw.execute_instruction(0)
    assert off_cpu.state.halted
    assert off_cpu.state.power_state == "off"

    off_cpu.state.power_state = "running"
    assert not off_cpu.state.halted
    assert off_cpu.state.power_state == "running"

    with pytest.raises(ValueError, match="unknown power state"):
        off_cpu.state.power_state = "bogus"


def test_llama_read_body_type_error_is_not_retried_as_legacy_signature() -> None:
    class MemoryWithBodyTypeError:
        def __init__(self) -> None:
            self.read_calls = 0

        def read_byte(self, _address: int, _pc: int | None = None) -> int:
            self.read_calls += 1
            raise TypeError("read body boom")

        def write_byte(
            self, _address: int, _value: int, _pc: int | None = None
        ) -> None:
            return None

        def peek_byte_for_preflight(self, _address: int, _pc: int | None = None) -> int:
            return 0

    memory = MemoryWithBodyTypeError()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)

    with pytest.raises(TypeError, match="read body boom"):
        raw_cpu.execute_instruction(0)
    assert memory.read_calls == 1

    with pytest.raises(RuntimeError, match="CPU is poisoned"):
        raw_cpu.execute_instruction(0)
    assert memory.read_calls == 1


def test_llama_write_body_type_error_rolls_back_native_state_and_poisons() -> None:
    class MemoryWithBodyTypeError:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[:3] = bytes([0xCC, 0x10, 0x42])  # MV (0x10),0x42
            self.write_calls = 0

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            self.write_calls += 1
            self.raw[address & 0xFFFFFF] = value & 0xFF
            raise TypeError("write body boom")

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

    memory = MemoryWithBodyTypeError()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)

    with pytest.raises(TypeError, match="write body boom"):
        raw_cpu.execute_instruction(0)

    assert memory.write_calls == 1
    assert raw_cpu.read_register("PC") == 0
    with pytest.raises(RuntimeError, match="CPU is poisoned"):
        raw_cpu.execute_instruction(0)
    assert memory.write_calls == 1


def test_llama_multibyte_write_stops_after_first_mutating_callback_error() -> None:
    class MutatingFailMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[:4] = bytes.fromhex("AC200000")  # MV [00020],X
            self.writes: list[tuple[int, int]] = []

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            address &= 0xFFFFFF
            value &= 0xFF
            self.raw[address] = value
            self.writes.append((address, value))
            raise RuntimeError("first byte committed then failed")

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

    memory = MutatingFailMemory()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)
    raw_cpu.write_register("PC", 0)
    raw_cpu.write_register("X", 0x23456)

    with pytest.raises(RuntimeError, match="first byte committed then failed"):
        raw_cpu.execute_instruction(0)

    assert memory.writes == [(0x20, 0x56)]
    assert memory.raw[0x20:0x23] == bytes.fromhex("560000")
    assert raw_cpu.read_register("PC") == 0
    assert raw_cpu.read_register("X") == 0x23456
    with pytest.raises(RuntimeError, match="CPU is poisoned"):
        raw_cpu.execute_instruction(0)
    assert memory.writes == [(0x20, 0x56)]


def test_llama_reset_stops_after_first_mutating_callback_error() -> None:
    class MutatingFailMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.reads: list[int] = []
            self.writes: list[tuple[int, int]] = []

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            address &= 0xFFFFFF
            self.reads.append(address)
            return self.raw[address]

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def instruction_byte_is_callback_free(self, _address: int) -> bool:
            return True

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            address &= 0xFFFFFF
            value &= 0xFF
            self.raw[address] = value
            self.writes.append((address, value))
            raise RuntimeError("reset first field committed then failed")

    memory = MutatingFailMemory()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)
    raw_cpu.write_register("PC", 0x12345)
    raw_cpu.prepare_vector_transfer(
        0xFFFFD,
        0x12345,
        require_immutable=False,
        scope="machine_reset",
    )

    with pytest.raises(RuntimeError, match="reset first field committed then failed"):
        raw_cpu.power_on_reset()

    assert memory.reads == [
        0xFFFFD,
        0xFFFFE,
        0xFFFFF,
        INTERNAL_MEMORY_START + 0xFE,
    ]
    assert memory.writes == [(INTERNAL_MEMORY_START + 0xFE, 0)]
    assert raw_cpu.read_register("PC") == 0x12345
    with pytest.raises(RuntimeError, match="CPU is poisoned"):
        raw_cpu.execute_instruction(0)
    assert memory.reads == [
        0xFFFFD,
        0xFFFFE,
        0xFFFFF,
        INTERNAL_MEMORY_START + 0xFE,
    ]
    assert memory.writes == [(INTERNAL_MEMORY_START + 0xFE, 0)]


def test_llama_present_broken_preflight_peek_does_not_fall_back() -> None:
    class MemoryWithBrokenHelper:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            self.raw[address & 0xFFFFFF] = value & 0xFF

        def peek_byte_for_preflight(self, _address: int, _pc: int | None = None) -> int:
            raise RuntimeError("silent helper boom")

    raw_cpu = RawLlamaCPU(memory=MemoryWithBrokenHelper(), reset_on_init=False)

    with pytest.raises(RuntimeError, match="silent helper boom"):
        raw_cpu.execute_instruction(0)


def test_llama_exp_preflight_does_not_peek_mutable_operand_data() -> None:
    class CountingMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[:4] = bytes.fromhex("32C22050")  # PRE (n),(n); EXP (20),(50)
            self.raw[INTERNAL_MEMORY_START + 0x20 : INTERNAL_MEMORY_START + 0x23] = (
                bytes.fromhex("1122F0")
            )
            self.raw[INTERNAL_MEMORY_START + 0x50 : INTERNAL_MEMORY_START + 0x53] = (
                bytes.fromhex("334410")
            )
            self.normal_reads: list[int] = []
            self.preflight_reads: list[int] = []

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            address &= 0xFFFFFF
            self.normal_reads.append(address)
            return self.raw[address]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            self.raw[address & 0xFFFFFF] = value & 0xFF

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            address &= 0xFFFFFF
            self.preflight_reads.append(address)
            return self.raw[address]

    memory = CountingMemory()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)

    raw_cpu.execute_instruction(0)

    assert memory.raw[
        INTERNAL_MEMORY_START + 0x20 : INTERNAL_MEMORY_START + 0x23
    ] == bytes.fromhex("334410")
    assert memory.raw[
        INTERNAL_MEMORY_START + 0x50 : INTERNAL_MEMORY_START + 0x53
    ] == bytes.fromhex("1122F0")
    assert INTERNAL_MEMORY_START + 0x22 not in memory.preflight_reads
    assert INTERNAL_MEMORY_START + 0x52 not in memory.preflight_reads
    assert INTERNAL_MEMORY_START + 0x22 in memory.normal_reads
    assert INTERNAL_MEMORY_START + 0x52 in memory.normal_reads


def test_llama_facade_render_decode_does_not_duplicate_instruction_fetch() -> None:
    class CountingMemory:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.normal_reads: list[int] = []
            self.preflight_reads: list[int] = []

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            address &= 0xFFFFFF
            self.normal_reads.append(address)
            return self.raw[address]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            self.raw[address & 0xFFFFFF] = value & 0xFF

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            address &= 0xFFFFFF
            self.preflight_reads.append(address)
            return self.raw[address]

    memory = CountingMemory()
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    memory.normal_reads.clear()
    memory.preflight_reads.clear()

    cpu.execute_instruction(0)

    assert memory.normal_reads == [0]
    assert memory.preflight_reads


def test_llama_missing_preflight_peek_fails_before_normal_read() -> None:
    class MemoryWithoutPeek:
        def __init__(self) -> None:
            self.read_calls = 0

        def read_byte(self, _address: int, _pc: int | None = None) -> int:
            self.read_calls += 1
            return 0

        def write_byte(
            self, _address: int, _value: int, _pc: int | None = None
        ) -> None:
            return None

    memory = MemoryWithoutPeek()
    raw_cpu = RawLlamaCPU(memory=memory, reset_on_init=False)

    with pytest.raises(RuntimeError, match="peek_byte_for_preflight"):
        raw_cpu.execute_instruction(0)

    assert memory.read_calls == 0


def test_llama_present_broken_kio_trace_hook_propagates() -> None:
    class MemoryWithBrokenTrace:
        def __init__(self) -> None:
            self.raw = bytearray(ADDRESS_SPACE_SIZE)
            self.raw[:2] = bytes([0xA0, 0xF0])  # MV (KOL),A

        def read_byte(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def write_byte(self, address: int, value: int, _pc: int | None = None) -> None:
            self.raw[address & 0xFFFFFF] = value & 0xFF

        def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
            return self.raw[address & 0xFFFFFF]

        def trace_kio_from_rust(
            self, _offset: int, _value: int, _pc: int | None = None
        ) -> None:
            raise RuntimeError("kio trace boom")

    raw_cpu = RawLlamaCPU(memory=MemoryWithBrokenTrace(), reset_on_init=False)
    raw_cpu.write_register("A", 0x12)

    with pytest.raises(RuntimeError, match="kio trace boom"):
        raw_cpu.execute_instruction(0)


def test_llama_present_noncallable_optional_hook_is_configuration_error() -> None:
    memory = _memory(b"\x00")
    setattr(memory, "trace_kio_from_rust", 42)
    _cpu, raw_cpu = _raw_llama_cpu(memory)

    with pytest.raises(TypeError, match="present but is not callable"):
        raw_cpu.execute_instruction(0)
