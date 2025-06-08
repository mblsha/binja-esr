#!/usr/bin/env python

import json
import time
from typing import Any, Dict, Optional

import serial
from plumbum import cli

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import Emulator

# from sc62015.pysc62015.emulator import Registers, RegisterName


class HardwareInterface:
    """Handles all serial communication with the BASIC harness."""

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: int = 3,
        serial_cls=serial.Serial,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_cls = serial_cls
        self.conn: Optional[serial.Serial] = None

    def __enter__(self):
        print(f"Connecting to hardware on {self.port}...")
        self.conn = self.serial_cls(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2)  # Wait for BASIC program and serial port to be ready
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn and self.conn.is_open:
            self.conn.close()
        print("Hardware connection closed.")

    def _send_command(self, cmd: str, *args: Any):
        if not self.conn:
            raise ConnectionError("Serial port not open.")
        self.conn.write(f"{cmd}\n".encode("ascii"))
        for arg in args:
            self.conn.write(f"{arg}\n".encode("ascii"))

    def _wait_for_ok(self):
        if not self.conn:
            raise ConnectionError("Serial port not open.")
        response = self.conn.readline().decode("ascii").strip()
        if response != "OK":
            raise IOError(f"Protocol error: Expected 'OK', but got '{response}'")

    def ping(self) -> bool:
        self._send_command("P")
        response = self.conn.readline().decode("ascii").strip()
        if response != "PONG":
            return False
        self._wait_for_ok()
        return True

    def load_code(self, address: int, code: bytearray):
        print(f"Loading {len(code)} bytes to 0x{address:X}...")
        self._send_command("L", f"&H{address:X}", str(len(code)))
        for byte in code:
            self.conn.write(f"{byte:02X}\n".encode("ascii"))
            time.sleep(0.01)  # Small delay to avoid overwhelming BASIC's input buffer
        self._wait_for_ok()
        print("Load successful.")

    def execute_code(self, address: int):
        print(f"Executing code at 0x{address:X}...")
        self._send_command("X", f"&H{address:X}")
        self._wait_for_ok()
        print("Execution finished.")

    def read_memory(self, address: int, length: int) -> bytearray:
        print(f"Reading {length} bytes from 0x{address:X}...")
        self._send_command("R", f"&H{address:X}", str(length))
        data = bytearray()
        for _ in range(length):
            byte_hex = self.conn.readline().decode("ascii").strip()
            data.append(int(byte_hex, 16))
        self._wait_for_ok()
        print("Read successful.")
        return data


class TestRunner:
    """Orchestrates the test generation, execution, and comparison."""

    TEST_WRAPPER_ADDR = 0x8000
    STATE_DUMP_AREA = 0x9000
    DUMP_REGISTERS = [("F", 1), ("BA", 2), ("I", 2), ("X", 3), ("Y", 3), ("S", 3)]

    def __init__(
        self, hw_interface: HardwareInterface, assembler: Assembler, emulator: Emulator
    ):
        self.hw = hw_interface
        self.assembler = assembler
        self.emulator = emulator

    def _generate_test_wrapper(
        self, instruction_to_test: str, initial_state: Dict[str, int]
    ) -> str:
        """Dynamically creates the assembly code for a test case."""
        setup_lines = []
        for reg_name, value in initial_state.items():
            if reg_name.upper() == "F":
                if value & 1:
                    setup_lines.append("SC ; Set Carry Flag")
                else:
                    setup_lines.append("RC ; Reset Carry Flag")
                if value & 2:
                    # No direct SET Z, so we use a sequence that guarantees Z=1
                    setup_lines.append("SUB A, A ; Set Zero Flag")
            else:
                setup_lines.append(f"MV {reg_name.upper()}, #&H{value:X}")

        save_lines = [f"MV U, #&H{self.STATE_DUMP_AREA} ; Point U to dump area"]
        for reg_name, _ in self.DUMP_REGISTERS:
            save_lines.append(f"PUSHU {reg_name}")

        return "\n".join(
            [
                f"SECTION code, &H{self.TEST_WRAPPER_ADDR}",
                "MV BA, U          ; Save original U to BA",
                *setup_lines,
                "; --- Instruction Under Test ---",
                instruction_to_test,
                "; --- Save Final State ---",
                *save_lines,
                "MV U, BA          ; Restore original U",
                "RET",
            ]
        )

    def _parse_dump(self, dumped_bytes: bytearray) -> Dict[str, int]:
        """Parses the raw byte dump from hardware into a dictionary of register values."""
        final_state = {}
        offset = 0
        for reg_name, size in self.DUMP_REGISTERS:
            value = int.from_bytes(dumped_bytes[offset : offset + size], "little")
            final_state[reg_name] = value
            offset += size
        return final_state

    def run_on_hardware(self, machine_code: bytearray) -> Dict[str, int]:
        """Loads and runs code on hardware, then reads the result state."""
        self.hw.load_code(self.TEST_WRAPPER_ADDR, machine_code)
        self.hw.execute_code(self.TEST_WRAPPER_ADDR)
        dump_size = sum(size for _, size in self.DUMP_REGISTERS)
        dumped_bytes = self.hw.read_memory(self.STATE_DUMP_AREA, dump_size)
        return self._parse_dump(dumped_bytes)

    def run_on_emulator(
        self, machine_code: bytearray, initial_state: Dict[str, int]
    ) -> Dict[str, int]:
        """Sets up and runs the test on the emulator."""
        print("Running on emulator...")
        # Your real emulator logic goes here
        # self.emulator.reset()
        # for reg_str, val in initial_state.items():
        #     self.emulator.regs.set(RegisterName[reg_str.upper()], val)
        # self.emulator.memory.write(self.TEST_WRAPPER_ADDR, machine_code)
        # self.emulator.execute(self.TEST_WRAPPER_ADDR) # This would run until RET

        final_state = {}
        # for reg_name, _ in self.DUMP_REGISTERS:
        #     final_state[reg_name] = self.emulator.regs.get(RegisterName[reg_name.upper()])
        print("Emulator run complete (simulated).")
        # Returning a dummy "correct" result for demonstration
        result_a = initial_state.get("A", 0) + 0x10
        final_state = initial_state.copy()
        final_state["BA"] = (initial_state.get("BA", 0) & 0xFF00) | (result_a & 0xFF)
        final_state["F"] = 0  # No flags set
        return final_state

    def run_test_case(self, instruction: str, initial_state: Dict[str, int]) -> bool:
        """Runs a complete test case on both platforms and compares results."""
        print(f"\n===== Running Test Case: {instruction} =====")
        print(f"Initial State: {initial_state}")

        wrapper_asm = self._generate_test_wrapper(instruction, initial_state)
        machine_code = self.assembler.assemble(wrapper_asm)

        hw_state = self.run_on_hardware(machine_code)
        emu_state = self.run_on_emulator(machine_code, initial_state)

        print("\n--- Comparison ---")
        print(f"{'Register':<5} | {'Hardware':<10} | {'Emulator':<10} | {'Status'}")
        print("-" * 42)

        has_failed = False
        all_regs = sorted(hw_state.keys())
        for reg_name in all_regs:
            hw_val = hw_state.get(reg_name, -1)
            emu_val = emu_state.get(reg_name, -1)
            status = "PASS" if hw_val == emu_val else "FAIL"
            if status == "FAIL":
                has_failed = True
            print(f"{reg_name:<5} | 0x{hw_val:<8X} | 0x{emu_val:<8X} | {status}")

        summary = "TEST FAILED" if has_failed else "TEST PASSED"
        print("\n" + f"{'='*15} {summary} {'='*16}")
        return not has_failed


class OrchestratorApp(cli.Application):
    """
    SC62015 Hardware-in-the-Loop Test Orchestrator

    This application runs a specific instruction test case on both the
    real hardware (via a serial harness) and a Python emulator, then
    compares the results to verify correctness.
    """

    instruction = cli.SwitchAttr(
        ["-i", "--instruction"],
        str,
        mandatory=True,
        help="The assembly instruction to test (e.g., 'ADD A, #$10').",
    )

    initial_state_json = cli.SwitchAttr(
        ["-s", "--state"],
        str,
        default='{"A": 5, "F": 0}',
        help="A JSON string representing the initial register state.",
    )

    def main(self, port: str):
        """
        Args:
            port: The serial port for the hardware harness (e.g., COM3, /dev/ttyUSB0).
        """
        try:
            initial_state = json.loads(self.initial_state_json)
        except json.JSONDecodeError:
            print(
                f"Error: Invalid JSON format for initial state: {self.initial_state_json}"
            )
            return 1

        # --- Setup ---
        assembler = Assembler()
        emulator = Emulator(mem=None)  # Your real emulator initialization

        try:
            with HardwareInterface(port) as hw:
                if not hw.ping():
                    print(
                        "Error: Could not ping the hardware harness. Check connection and BASIC program."
                    )
                    return 1

                print("Hardware harness ping successful.")
                runner = TestRunner(hw, assembler, emulator)

                # --- Run the specified test case ---
                success = runner.run_test_case(self.instruction, initial_state)
                return 0 if success else 1

        except serial.SerialException as e:
            print(f"\nSerial Error: {e}")
            return 1
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            return 1


if __name__ == "__main__":
    OrchestratorApp.run()
