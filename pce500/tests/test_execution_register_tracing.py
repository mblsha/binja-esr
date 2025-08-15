"""Test that execution events trace all registers correctly."""

from unittest.mock import patch, MagicMock
from pce500.emulator import PCE500Emulator
from pce500.trace_manager import g_tracer


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

    # Mock the trace manager methods
    with patch.object(g_tracer, "start_tracing") as mock_start:
        with patch.object(g_tracer, "trace_instant") as mock_instant:
            mock_start.return_value = True

            # Create a mock event that has add_annotations method
            mock_event = MagicMock()
            mock_instant.return_value = mock_event

            # Create emulator with perfetto tracing
            emu = PCE500Emulator(perfetto_trace=True)
            emu.load_rom(rom_data)

            # Execute 3 instructions
            emu.run(max_instructions=3)

            # Check that trace_instant was called for each execution
            assert mock_instant.call_count >= 3

            # Check that execution events were traced with correct format
            exec_calls = [
                call
                for call in mock_instant.call_args_list
                if call[0][0] == "Execution" and "Exec@" in call[0][1]
            ]

            assert len(exec_calls) >= 3

            # Check that add_annotations was called for each execution event
            # with all the expected registers
            assert mock_event.add_annotations.call_count >= 3

            # Check the annotations for one of the calls
            annotations = mock_event.add_annotations.call_args_list[0][0][0]

            # Verify all registers are present with correct naming
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

            # Verify format matches function call format (hex with proper width)
            assert annotations["reg_A"].startswith("0x")
            assert len(annotations["reg_A"]) == 4  # 0xXX
            assert annotations["reg_BA"].startswith("0x")
            assert len(annotations["reg_BA"]) == 6  # 0xXXXX
            assert annotations["reg_PC"].startswith("0x")
            assert len(annotations["reg_PC"]) == 8  # 0xXXXXXX

            # Verify flags are boolean values
            assert isinstance(annotations["flag_C"], int)
            assert isinstance(annotations["flag_Z"], int)


def test_no_before_flags_in_execution():
    """Test that execution events don't include _before flags."""
    rom_data = bytes([0x00]) + bytes(0x3FFFF)  # NOP + padding
    rom_data = rom_data[:0x3FFFD] + bytes([0x00, 0x00, 0x0C])  # Entry point

    with patch.object(g_tracer, "start_tracing") as mock_start:
        with patch.object(g_tracer, "trace_instant") as mock_instant:
            mock_start.return_value = True
            mock_event = MagicMock()
            mock_instant.return_value = mock_event

            emu = PCE500Emulator(perfetto_trace=True)
            emu.load_rom(rom_data)
            emu.step()

            # Find execution event
            exec_calls = [
                call
                for call in mock_instant.call_args_list
                if call[0][0] == "Execution"
            ]

            assert len(exec_calls) >= 1

            # Check initial args don't have _before flags
            initial_args = exec_calls[0][0][2]
            assert "C_before" not in initial_args
            assert "Z_before" not in initial_args
            assert "C_after" not in initial_args
            assert "Z_after" not in initial_args
