import importlib.util
from pathlib import Path
import types
import pytest

# Utility to load the orchestrator module from its script path

ORCH_PATH = Path(__file__).parent / "hw-test" / "orchestrator.py"


def import_orchestrator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("orchestrator", ORCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


class MockSerial:
    def __init__(self, responses):
        self.responses = list(responses)
        self.written = []
        self.is_open = True

    def write(self, data):
        self.written.append(data.decode("ascii"))

    def readline(self):
        if not self.responses:
            return b""
        return self.responses.pop(0)

    def close(self):
        self.is_open = False


@pytest.fixture
def orch():
    return import_orchestrator()


def test_hardware_interface_ping(monkeypatch, orch):
    responses = [b"PONG\n", b"OK\n"]
    ms = MockSerial(responses)
    hw = orch.HardwareInterface(
        "dummy", serial_cls=lambda *a, **k: ms
    )
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    with hw:
        assert hw.ping() is True
    assert ms.written == ["P\n"]


def test_run_test_case_success(monkeypatch, orch):
    # Prepare expected final state
    final_state = {"F": 0, "BA": 0x11, "I": 0, "X": 0, "Y": 0, "S": 0}
    final_bytes = []
    for reg, size in orch.TestRunner.DUMP_REGISTERS:
        final_bytes.extend(final_state[reg].to_bytes(size, "little"))

    responses = [
        b"OK\n",  # load_code
        b"OK\n",  # execute_code
        *[f"{b:02X}\n".encode("ascii") for b in final_bytes],
        b"OK\n",  # read_memory
    ]
    ms = MockSerial(responses)
    hw = orch.HardwareInterface("dummy", serial_cls=lambda *a, **k: ms)

    class DummyAssembler:
        def __init__(self):
            self.source = None

        def assemble(self, source: str):
            self.source = source
            return bytearray([0xDE, 0xAD])

    assembler = DummyAssembler()

    class DummyEmulator:
        pass

    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    with hw:
        runner = orch.TestRunner(hw, assembler, DummyEmulator())
        monkeypatch.setattr(runner, "run_on_emulator", lambda mc, init: final_state)
        success = runner.run_test_case("ADD A, #$10", {"A": 1})

    assert success is True
    assert "ADD A, #$10" in assembler.source
    expected_writes = [
        "L\n",
        "&H8000\n",
        "2\n",
        "DE\n",
        "AD\n",
        "X\n",
        "&H8000\n",
        "R\n",
        "&H9000\n",
        f"{len(final_bytes)}\n",
    ]
    assert ms.written == expected_writes
