"""Test that execution events trace all registers correctly."""

from pce500.emulator import PCE500Emulator
from pce500.tracing import TraceEventType, trace_dispatcher, TraceObserver


class CollectingObserver(TraceObserver):
    def __init__(self) -> None:
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


def test_execution_traces_all_registers():
    """Test that execution events include all register values."""
    # Create a simple ROM that does some register operations
    rom_data = bytes(
        [
            # At 0xC0000 - LD A, 0x42
            0x90,
            0x42,
            # At 0xC0002 - LD B, 0x33
            0x91,
            0x33,
            # At 0xC0004 - NOP
            0x00,
        ]
    ) + bytes(0x3FFF9)  # Pad to 256KB

    # Add entry point at 0xFFFFD pointing to 0xC0000
    rom_data = rom_data[:0x3FFFD] + bytes([0x00, 0x00, 0x0C])  # Little-endian 0xC0000

    observer = CollectingObserver()
    trace_dispatcher.register(observer)
    try:
        emu = PCE500Emulator(perfetto_trace=False)
        emu.load_rom(rom_data)

        # Execute 3 instructions
        emu.run(max_instructions=3)

        exec_events = [
            event
            for event in observer.events
            if event.type is TraceEventType.INSTANT
            and event.thread == "Execution"
            and event.name
            and event.name.startswith("Exec@")
        ]
        assert len(exec_events) >= 3

        annotations = exec_events[0].payload
        expected_registers = [
            "reg_A",
            "reg_B",
            "reg_BA",
            "reg_I",
            "reg_X",
            "reg_Y",
            "reg_U",
            "reg_S",
            "reg_PC",
            "flag_C",
            "flag_Z",
        ]

        for reg in expected_registers:
            assert reg in annotations, f"Missing register {reg} in annotations"

        assert annotations["reg_A"].startswith("0x")
        assert len(annotations["reg_A"]) == 4  # 0xXX
        assert annotations["reg_BA"].startswith("0x")
        assert len(annotations["reg_BA"]) == 6  # 0xXXXX
        assert annotations["reg_PC"].startswith("0x")
        assert len(annotations["reg_PC"]) == 8  # 0xXXXXXX

        assert isinstance(annotations["flag_C"], int)
        assert isinstance(annotations["flag_Z"], int)
    finally:
        trace_dispatcher.unregister(observer)


def test_no_before_flags_in_execution():
    """Test that execution events don't include _before flags."""
    rom_data = bytes([0x00]) + bytes(0x3FFFF)  # NOP + padding
    rom_data = rom_data[:0x3FFFD] + bytes([0x00, 0x00, 0x0C])  # Entry point

    observer = CollectingObserver()
    trace_dispatcher.register(observer)
    try:
        emu = PCE500Emulator(perfetto_trace=False)
        emu.load_rom(rom_data)
        emu.step()

        exec_events = [
            event
            for event in observer.events
            if event.type is TraceEventType.INSTANT and event.thread == "Execution"
        ]
        assert exec_events

        payload = exec_events[0].payload
        assert "C_before" not in payload
        assert "Z_before" not in payload
        assert "C_after" not in payload
        assert "Z_after" not in payload
    finally:
        trace_dispatcher.unregister(observer)
