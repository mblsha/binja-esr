"""Address diagnostics must not introduce observable reads of unused bases."""

from binja_test_mocks.eval_llil import Memory
import pytest

# The bundled type stubs omit this supported runtime API and generated Trace.
from retrobus_perfetto import resolve_interned_trace  # pyright: ignore[reportAttributeAccessIssue]
from retrobus_perfetto.proto import perfetto_pb2

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


@pytest.mark.parametrize("backend", ["python", "llama", "llama-traced"])
@pytest.mark.parametrize("bp", [0, 0xF0])
@pytest.mark.parametrize(
    "mode,prefix,selector",
    [
        ("N", b"\x30", 0x10),
        ("BpN", b"", 0x10),
        ("PxN", b"\x34", 0x10),
        ("BpPx", b"\x24", 0),
    ],
)
def test_only_required_imem_bases_are_read_and_reported(
    tmp_path, backend, bp, mode, prefix, selector
):
    values = {0xEC: bp, 0xED: 0x21, 0xEE: 0x32}
    used = {"N": [], "BpN": [0xEC], "PxN": [0xED], "BpPx": [0xEC, 0xED]}[mode]
    address = (selector + sum(values[offset] for offset in used)) & 255
    raw = bytearray(ADDRESS_SPACE_SIZE)
    program = prefix + bytes([0x80, selector])  # MV A,(selector)
    raw[0x1000 : 0x1000 + len(program)] = program
    for offset, value in values.items():
        raw[INTERNAL_MEMORY_START + offset] = value
    raw[INTERNAL_MEMORY_START + address] = 0xA5
    reads = []

    def read(location):
        if INTERNAL_MEMORY_START <= location < INTERNAL_MEMORY_START + 256:
            reads.append((location - INTERNAL_MEMORY_START, raw[location]))
        return raw[location]

    memory = Memory(read, raw.__setitem__)
    setattr(memory, "peek_byte_for_preflight", lambda location, _pc=None: raw[location])
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend="python" if backend == "python" else "llama",
    )
    cpu.regs.set(RegisterName.PC, 0x1000)
    trace_path = tmp_path / "base-reads.perfetto-trace"
    traced = backend == "llama-traced"
    if traced:
        cpu.set_perfetto_trace(str(trace_path))
    try:
        cpu.execute_instruction(0x1000)
    finally:
        if traced:
            cpu.flush_perfetto()
            cpu.set_perfetto_trace(None)
    assert cpu.regs.get(RegisterName.A) == 0xA5
    assert reads == [(offset, values[offset]) for offset in used] + [(address, 0xA5)]
    if traced:
        trace = perfetto_pb2.Trace()  # pyright: ignore[reportAttributeAccessIssue]
        trace.ParseFromString(trace_path.read_bytes())
        resolve_interned_trace(trace, inplace=True)
        events = [
            packet.track_event
            for packet in trace.packet
            if packet.HasField("track_event")
            and packet.track_event.name == "IMEM_EffectiveAddr"
        ]
        assert len(events) == 1
        annotations = {
            annotation.name: annotation for annotation in events[0].debug_annotations
        }
        assert annotations["mode"].string_value == mode
        assert annotations["base"].uint_value == address
        names = {0xEC: "bp", 0xED: "px", 0xEE: "py"}
        assert set(annotations) & {"bp", "px", "py"} == {
            names[offset] for offset in used
        }
        for offset in used:
            # A genuinely sampled zero must remain distinguishable from absent.
            assert annotations[names[offset]].HasField("uint_value")
            assert annotations[names[offset]].uint_value == values[offset]
