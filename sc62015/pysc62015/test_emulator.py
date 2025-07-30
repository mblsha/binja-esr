
from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from .emulator import (
    Registers,
    RegisterName,
    Emulator,
    Memory,
)
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START, PC_MASK
from .instr.opcodes import IMEMRegisters
from binja_test_mocks.mock_llil import MockLowLevelILFunction
from .test_instr import opcode_generator
from typing import Dict, Tuple, List, NamedTuple, Optional
from binja_test_mocks.tokens import asm_str
from dataclasses import dataclass, field
import pytest

# Preallocate a single memory buffer for unit tests to reuse. This avoids
# repeatedly allocating large bytearrays in many test cases and speeds up the
# overall test suite.
_SHARED_MEMORY = bytearray(ADDRESS_SPACE_SIZE)


def test_registers() -> None:
    regs = Registers()

    # Test initial state
    assert regs.get(RegisterName.A) == 0
    assert regs.get(RegisterName.B) == 0
    assert regs.get(RegisterName.IL) == 0
    assert regs.get(RegisterName.IH) == 0
    assert regs.get(RegisterName.I) == 0
    assert regs.get(RegisterName.BA) == 0
    assert regs.get(RegisterName.X) == 0
    assert regs.get(RegisterName.Y) == 0
    assert regs.get(RegisterName.U) == 0
    assert regs.get(RegisterName.S) == 0
    assert regs.get(RegisterName.PC) == 0
    assert regs.get(RegisterName.FC) == 0
    assert regs.get(RegisterName.FZ) == 0
    assert regs.get(RegisterName.F) == 0

    regs.set(RegisterName.A, 0x42)
    assert regs.get(RegisterName.A) == 0x42
    assert regs.get(RegisterName.BA) == 0x42
    regs.set(RegisterName.B, 0x84)
    assert regs.get(RegisterName.B) == 0x84
    assert regs.get(RegisterName.BA) == 0x8442

    regs.set(RegisterName.IL, 0x12)
    assert regs.get(RegisterName.IL) == 0x12
    assert regs.get(RegisterName.I) == 0x12
    regs.set(RegisterName.IH, 0x34)
    assert regs.get(RegisterName.IH) == 0x34
    assert regs.get(RegisterName.I) == 0x3412

    regs.set(RegisterName.FC, 1)
    assert regs.get(RegisterName.FC) == 1
    assert regs.get(RegisterName.F) == 1
    regs.set(RegisterName.FZ, 1)
    assert regs.get(RegisterName.FZ) == 1
    assert regs.get(RegisterName.F) == 3  # FC + FZ bits set


def test_pc_mask() -> None:
    regs = Registers()

    # Setting a value with bits above 20 should wrap around
    regs.set(RegisterName.PC, PC_MASK + 1 + 0x12345)
    assert regs.get(RegisterName.PC) == 0x12345

    # Verify masking occurs on retrieval as well
    regs.set(RegisterName.PC, 0x1234567)
    assert regs.get(RegisterName.PC) == 0x34567


def _make_cpu_and_mem(
    size: int, init_data: Dict[int, int], instr_bytes: bytes, instr_addr: int = 0
) -> Tuple[Emulator, bytearray, List[int], List[Tuple[int, int]]]:
    """
    Create a bytearray-backed mock memory, preload it with `init_data` and
    `instr_bytes`, then return (cpu, raw_memory, read_log, write_log).
    """
    assert size <= ADDRESS_SPACE_SIZE
    raw = _SHARED_MEMORY
    raw[:size] = b"\x00" * size
    for addr, val in init_data.items():
        raw[addr] = val & 0xFF

    # Place instruction bytes at the specified address
    raw[instr_addr : instr_addr + len(instr_bytes)] = instr_bytes

    reads: List[int] = []
    writes: List[Tuple[int, int]] = []

    def read_mem(addr: int) -> int:
        reads.append(addr)
        if addr >= INTERNAL_MEMORY_START:
            addr = INTERNAL_MEMORY_START + ((addr - INTERNAL_MEMORY_START) & 0xFF)
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:04x} out of bounds")
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        # print(f"Writing {value:02x} to address {addr:04x}") # Uncomment for debugging
        if addr >= INTERNAL_MEMORY_START:
            addr = INTERNAL_MEMORY_START + ((addr - INTERNAL_MEMORY_START) & 0xFF)
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:04x} out of bounds")
        raw[addr] = value & 0xFF

    cpu = Emulator(Memory(read_mem, write_mem), reset_on_init=False)
    return cpu, raw, reads, writes


def debug_instruction(cpu: Emulator, address: int) -> None:
    il = MockLowLevelILFunction()
    instr = cpu.decode_instruction(address)
    assert instr is not None, f"Failed to decode instruction at {address:04x}"
    instr.lift(il, address)

    rendered = asm_str(instr.render())

    print(f"Decoded instruction at {address:04x}: {rendered}")
    for llil in il.ils:
        print(f"  {llil}")


@dataclass
class InstructionTestCase:
    """A structured container for a single instruction test case."""

    test_id: str
    instr_bytes: bytes
    init_regs: Dict[RegisterName, int] = field(default_factory=dict)
    init_mem: Dict[int, int] = field(default_factory=dict)
    expected_regs: Dict[RegisterName, int] = field(default_factory=dict)
    expected_mem_writes: Optional[List[Tuple[int, int]]] = None
    expected_mem_state: Dict[int, int] = field(default_factory=dict)
    initial_pc: int = 0x00
    expected_asm_str: Optional[str] = None


instruction_test_cases: List[InstructionTestCase] = [
    # --- MV (Load/Store) Instructions ---
    InstructionTestCase(
        test_id="MV_A_from_ext_mem",
        instr_bytes=bytes.fromhex("88100000"),
        init_mem={0x10: 0xAB},
        expected_regs={RegisterName.A: 0xAB},
        expected_asm_str="MV    A, [00010]",
    ),
    InstructionTestCase(
        test_id="MV_BA_from_ext_mem",
        instr_bytes=bytes.fromhex("8A200000"),
        init_mem={0x20: 0x12, 0x21: 0x34},
        expected_regs={RegisterName.BA: 0x3412},
        expected_asm_str="MV    BA, [00020]",
    ),
    InstructionTestCase(
        test_id="MV_X_from_ext_mem",
        instr_bytes=bytes.fromhex("8C300000"),
        init_mem={0x30: 0x01, 0x31: 0x02, 0x32: 0x03},
        expected_regs={RegisterName.X: 0x030201},
        expected_asm_str="MV    X, [00030]",
    ),
    InstructionTestCase(
        test_id="MV_A_to_ext_mem",
        instr_bytes=bytes.fromhex("A8200000"),
        init_regs={RegisterName.A: 0xCD},
        expected_mem_writes=[(0x20, 0xCD)],
        expected_mem_state={0x20: 0xCD},
        expected_asm_str="MV    [00020], A",
    ),
    InstructionTestCase(
        test_id="MV_BA_to_ext_mem",
        instr_bytes=bytes.fromhex("AA200000"),
        init_regs={RegisterName.BA: 0x1234},
        expected_mem_writes=[(0x20, 0x34), (0x21, 0x12)],
        expected_mem_state={0x20: 0x34, 0x21: 0x12},
        expected_asm_str="MV    [00020], BA",
    ),
    InstructionTestCase(
        test_id="MV_X_to_ext_mem",
        instr_bytes=bytes.fromhex("AC200000"),
        init_regs={RegisterName.X: 0x010203},
        expected_mem_writes=[(0x20, 0x03), (0x21, 0x02), (0x22, 0x01)],
        expected_mem_state={0x20: 0x03, 0x21: 0x02, 0x22: 0x01},
        expected_asm_str="MV    [00020], X",
    ),
    # External memory via register pointer
    InstructionTestCase(
        test_id="MV_A_from_emem_reg",
        instr_bytes=bytes.fromhex("9004"),
        init_regs={RegisterName.X: 0x0040},
        init_mem={0x40: 0xAA},
        expected_regs={RegisterName.A: 0xAA},
        expected_asm_str="MV    A, [X]",
    ),
    InstructionTestCase(
        test_id="MV_BA_from_emem_reg",
        instr_bytes=bytes.fromhex("9204"),
        init_regs={RegisterName.X: 0x0050},
        init_mem={0x50: 0x11, 0x51: 0x22},
        expected_regs={RegisterName.BA: 0x2211},
        expected_asm_str="MV    BA, [X]",
    ),
    InstructionTestCase(
        test_id="MV_X_to_emem_reg",
        instr_bytes=bytes.fromhex("B405"),
        init_regs={RegisterName.X: 0x010203, RegisterName.Y: 0x0060},
        expected_mem_writes=[(0x60, 0x03), (0x61, 0x02), (0x62, 0x01)],
        expected_mem_state={0x60: 0x03, 0x61: 0x02, 0x62: 0x01},
        expected_asm_str="MV    [Y], X",
    ),
    InstructionTestCase(
        test_id="MV_preserves_flags",
        instr_bytes=bytes.fromhex("88100000"),
        init_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        init_mem={0x10: 0x55},
        expected_regs={
            RegisterName.A: 0x55,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        expected_asm_str="MV    A, [00010]",
    ),
    InstructionTestCase(
        test_id="MV_[Y++]_A", 
        instr_bytes=bytes.fromhex("B025"),
        init_regs={
            RegisterName.A: 0x42,
            RegisterName.Y: 0x1000,
        },
        expected_regs={
            RegisterName.A: 0x42,  # A unchanged
            RegisterName.Y: 0x1001,  # Y incremented by 1 (data width for A)
        },
        expected_mem_writes=[(0x1000, 0x42)],  # Write happens at Y, then increment
        expected_mem_state={0x1000: 0x42},
        expected_asm_str="MV    [Y++], A",
    ),
    InstructionTestCase(
        test_id="MV_[Y++]_BA",
        instr_bytes=bytes.fromhex("B225"),  # B2 is MV [EMemReg], BA
        init_regs={
            RegisterName.BA: 0x1234,
            RegisterName.Y: 0x2000,
        },
        expected_regs={
            RegisterName.BA: 0x1234,  # BA unchanged
            RegisterName.Y: 0x2002,  # Y incremented by 2 (data width for BA)
        },
        expected_mem_writes=[(0x2000, 0x34), (0x2001, 0x12)],  # Write BA at Y, then increment
        expected_mem_state={0x2000: 0x34, 0x2001: 0x12},
        expected_asm_str="MV    [Y++], BA",
    ),
    InstructionTestCase(
        test_id="JP_preserves_flags",
        instr_bytes=bytes.fromhex("023412"),
        init_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_regs={
            RegisterName.PC: 0x1234,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        expected_asm_str="JP    1234",
    ),
    InstructionTestCase(
        test_id="PUSHU_preserves_flags",
        instr_bytes=bytes.fromhex("2E"),
        init_regs={RegisterName.U: 0x20, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_mem_writes=[(0x1F, 0x03)],
        expected_mem_state={0x1F: 0x03},
        expected_regs={
            RegisterName.U: 0x1F,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        expected_asm_str="PUSHU F",
    ),
    # --- ADD Instructions ---
    InstructionTestCase(
        test_id="ADD_A_imm_simple",
        instr_bytes=bytes.fromhex("4001"),
        init_regs={RegisterName.A: 0x10},
        expected_regs={RegisterName.A: 0x11, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="ADD   A, 01",
    ),
    InstructionTestCase(
        test_id="ADD_A_imm_carry_zero",
        instr_bytes=bytes.fromhex("4001"),
        init_regs={RegisterName.A: 0xFF},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},
        expected_asm_str="ADD   A, 01",
    ),
    # --- SUB Instructions ---
    InstructionTestCase(
        test_id="SUB_A_imm_simple",
        instr_bytes=bytes.fromhex("4801"),
        init_regs={RegisterName.A: 0x10},
        expected_regs={RegisterName.A: 0x0F, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="SUB   A, 01",
    ),
    InstructionTestCase(
        test_id="SUB_A_imm_borrow",
        instr_bytes=bytes.fromhex("4801"),
        init_regs={RegisterName.A: 0x00},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SUB   A, 01",
    ),
    InstructionTestCase(
        test_id="SUB_A_imm_zero",
        instr_bytes=bytes.fromhex("4801"),
        init_regs={RegisterName.A: 0x01},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 0},
        expected_asm_str="SUB   A, 01",
    ),
    # --- SWAP Instructions ---
    InstructionTestCase(
        test_id="SWAP_A_non_zero_to_non_zero",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0x12, RegisterName.FC: 0, RegisterName.FZ: 1},
        expected_regs={RegisterName.A: 0x21, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="SWAP  A",
    ),
    InstructionTestCase(
        test_id="SWAP_A_non_zero_FC_unaffected",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0xAB, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_regs={RegisterName.A: 0xBA, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SWAP  A",
    ),
    InstructionTestCase(
        test_id="SWAP_A_zero_to_zero_sets_FZ",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},
        expected_asm_str="SWAP  A",
    ),
    InstructionTestCase(
        test_id="SWAP_A_edge_case_F0",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 0, RegisterName.FZ: 1},
        expected_regs={RegisterName.A: 0x0F, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="SWAP  A",
    ),
    InstructionTestCase(
        test_id="SWAP_A_edge_case_0F",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0x0F, RegisterName.FC: 0, RegisterName.FZ: 1},
        expected_regs={RegisterName.A: 0xF0, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="SWAP  A",
    ),
    InstructionTestCase(
        test_id="SWAP_A_edge_case_FF",
        instr_bytes=bytes.fromhex("EE"),
        init_regs={RegisterName.A: 0xFF, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SWAP  A",
    ),
    # --- MVL/MVLD Edge Cases ---
    InstructionTestCase(
        test_id="MVL_(m)_(n)_I_is_zero",
        instr_bytes=bytes.fromhex("CB50A0"),  # MVL (50), (A0)
        init_regs={RegisterName.I: 0},
        init_mem={
            INTERNAL_MEMORY_START + 0xA0: 0xDE,  # Source
            INTERNAL_MEMORY_START + 0x50: 0xAD,  # Destination
        },
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            INTERNAL_MEMORY_START + 0xA0: 0xDE,
            INTERNAL_MEMORY_START + 0x50: 0xAD,  # Should remain unchanged
        },
        expected_asm_str="MVL   (BP+50), (BP+A0)",
    ),
    InstructionTestCase(
        test_id="MVL_imem_overlap_fwd_clobber",
        instr_bytes=bytes.fromhex("CB5150"),  # MVL (51), (50)
        init_regs={RegisterName.I: 3},
        init_mem={
            INTERNAL_MEMORY_START + 0x50: 0xAA,
            INTERNAL_MEMORY_START + 0x51: 0xBB,
            INTERNAL_MEMORY_START + 0x52: 0xCC,
        },
        # A naive forward copy clobbers the source.
        # Expected: mem[51]=mem[50]=AA; mem[52]=mem[51]=AA; mem[53]=mem[52]=AA
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            INTERNAL_MEMORY_START + 0x50: 0xAA,
            INTERNAL_MEMORY_START + 0x51: 0xAA,
            INTERNAL_MEMORY_START + 0x52: 0xAA,
            INTERNAL_MEMORY_START + 0x53: 0xAA,
        },
        expected_asm_str="MVL   (BP+51), (BP+50)",
    ),
    InstructionTestCase(
        test_id="MVLD_imem_overlap_bwd_correct",
        instr_bytes=bytes.fromhex("CF5150"),  # MVLD (51), (50)
        init_regs={RegisterName.I: 3},
        # Dst ends at 0x51, Src ends at 0x50.
        # Copies from {50, 4F, 4E} to {51, 50, 4F}.
        init_mem={
            INTERNAL_MEMORY_START + 0x50: 0xAA,
            INTERNAL_MEMORY_START + 0x4F: 0xBB,
            INTERNAL_MEMORY_START + 0x4E: 0xCC,
        },
        # A backward copy handles this overlap correctly.
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            INTERNAL_MEMORY_START + 0x51: 0xAA,
            INTERNAL_MEMORY_START + 0x50: 0xBB,
            INTERNAL_MEMORY_START + 0x4F: 0xCC,
        },
        expected_asm_str="MVLD  (BP+51), (BP+50)",
    ),
    InstructionTestCase(
        test_id="MVL_imem_to_imem_wrap_around",
        instr_bytes=bytes.fromhex("CBFEF0"),  # MVL (FE), (F0)
        init_regs={RegisterName.I: 4},
        init_mem={
            INTERNAL_MEMORY_START + 0xF0: 0x11,
            INTERNAL_MEMORY_START + 0xF1: 0x22,
            INTERNAL_MEMORY_START + 0xF2: 0x33,
            INTERNAL_MEMORY_START + 0xF3: 0x44,
        },
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            INTERNAL_MEMORY_START + 0xFE: 0x11,
            INTERNAL_MEMORY_START + 0xFF: 0x22,
            INTERNAL_MEMORY_START + 0x00: 0x33,
            INTERNAL_MEMORY_START + 0x01: 0x44,
        },
        expected_asm_str="MVL   (BP+FE), (BP+F0)",
    ),
    InstructionTestCase(
        test_id="MVL_(imem)_[--X]",
        instr_bytes=bytes.fromhex("E33452"),  # MVL (52), [--X]
        init_regs={
            RegisterName.I: 2,
            RegisterName.X: 0x2002,  # Start X pointing after the source data
        },
        init_mem={
            0x2000: 0xBE,
            0x2001: 0xEF,
        },
        expected_regs={
            RegisterName.I: 0,
            RegisterName.X: 0x2000,
        },
        expected_mem_state={
            INTERNAL_MEMORY_START + 0x52: 0xEF,
            INTERNAL_MEMORY_START + 0x51: 0xBE,
        },
        expected_asm_str="MVL   (BP+52), [--X]",
    ),
    InstructionTestCase(
        test_id="MVL_(02)_[--X]_I5_X2000",
        instr_bytes=bytes.fromhex("E33402"),  # MVL (02), [--X]
        init_regs={
            RegisterName.I: 5,
            RegisterName.X: 0x2000,  # X points to address 2000
        },
        init_mem={
            # Source data at external memory (will be read from addresses 1FFB-1FFF due to pre-decrement)
            0x1FFB: 0x55,
            0x1FFC: 0x11,
            0x1FFD: 0x22,
            0x1FFE: 0x33,
            0x1FFF: 0x44,
        },
        expected_regs={
            RegisterName.I: 0,
            RegisterName.X: 0x1FFB,  # X decremented by 5
        },
        expected_mem_state={
            # MVL with pre-decrement source causes destination to decrement too
            # Writes go to: 0x02, 0x01, 0x00, 0xFF, 0xFE (wrapped)
            INTERNAL_MEMORY_START + 0x02: 0x44,  # First byte from 0x1FFF
            INTERNAL_MEMORY_START + 0x01: 0x33,  # Second byte from 0x1FFE
            INTERNAL_MEMORY_START + 0x00: 0x22,  # Third byte from 0x1FFD
            INTERNAL_MEMORY_START + 0xFF: 0x11,  # Fourth byte from 0x1FFC (wrapped from -1)
            INTERNAL_MEMORY_START + 0xFE: 0x55,  # Fifth byte from 0x1FFB (wrapped from -2)
        },
        expected_asm_str="MVL   (BP+02), [--X]",
    ),
    InstructionTestCase(
        test_id="MVL_(FE)_(50)_I5_wrap",
        instr_bytes=bytes.fromhex("CBFE50"),  # MVL (FE), (50) - no PRE, both use BP_N by default
        init_regs={
            RegisterName.I: 5,
        },
        init_mem={
            # Source data at internal memory starting at BP+50
            INTERNAL_MEMORY_START + 0x50: 0xAA,
            INTERNAL_MEMORY_START + 0x51: 0xBB,
            INTERNAL_MEMORY_START + 0x52: 0xCC,
            INTERNAL_MEMORY_START + 0x53: 0xDD,
            INTERNAL_MEMORY_START + 0x54: 0xEE,
        },
        expected_regs={
            RegisterName.I: 0,
        },
        expected_mem_state={
            # MVL copies from (BP+50) to (BP+FE) with incrementing addresses
            # Destination addresses: 0xFE, 0xFF, 0x00 (wrapped), 0x01, 0x02
            INTERNAL_MEMORY_START + 0xFE: 0xAA,  # From BP+50
            INTERNAL_MEMORY_START + 0xFF: 0xBB,  # From BP+51
            INTERNAL_MEMORY_START + 0x00: 0xCC,  # From BP+52 (wrapped from 0x100)
            INTERNAL_MEMORY_START + 0x01: 0xDD,  # From BP+53 (wrapped from 0x101)
            INTERNAL_MEMORY_START + 0x02: 0xEE,  # From BP+54 (wrapped from 0x102)
            # Source data remains unchanged
            INTERNAL_MEMORY_START + 0x50: 0xAA,
            INTERNAL_MEMORY_START + 0x51: 0xBB,
            INTERNAL_MEMORY_START + 0x52: 0xCC,
            INTERNAL_MEMORY_START + 0x53: 0xDD,
            INTERNAL_MEMORY_START + 0x54: 0xEE,
        },
        expected_asm_str="MVL   (BP+FE), (BP+50)",
    ),
    InstructionTestCase(
        test_id="MVL_(00)_[--X]_BP2_I5",
        instr_bytes=bytes.fromhex("E33400"),  # MVL (00), [--X] with BP=2
        init_regs={
            RegisterName.I: 5,
            RegisterName.X: 0x2000,
        },
        init_mem={
            # BP register at internal memory
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x02,  # BP = 2
            # Source data at external memory
            0x1FFB: 0x55,
            0x1FFC: 0x44,
            0x1FFD: 0x33,
            0x1FFE: 0x22,
            0x1FFF: 0x11,
        },
        expected_regs={
            RegisterName.I: 0,
            RegisterName.X: 0x1FFB,  # X decremented by 5
        },
        expected_mem_state={
            # BP=2, so (BP+00) = address 0x02
            # MVL with pre-decrement source causes destination to decrement too
            # Writes go to: 0x02, 0x01, 0x00, 0xFF (wrapped), 0xFE (wrapped)
            INTERNAL_MEMORY_START + 0x02: 0x11,  # From 0x1FFF
            INTERNAL_MEMORY_START + 0x01: 0x22,  # From 0x1FFE
            INTERNAL_MEMORY_START + 0x00: 0x33,  # From 0x1FFD
            INTERNAL_MEMORY_START + 0xFF: 0x44,  # From 0x1FFC (wrapped from -1)
            INTERNAL_MEMORY_START + 0xFE: 0x55,  # From 0x1FFB (wrapped from -2)
            # BP remains unchanged
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x02,
        },
        expected_asm_str="MVL   (BP+00), [--X]",
    ),
    InstructionTestCase(
        test_id="MVL_(00)_(50)_BP_FE_I3",
        instr_bytes=bytes.fromhex("CB0050"),  # MVL (00), (50) with BP=0xFE
        init_regs={
            RegisterName.I: 3,
        },
        init_mem={
            # BP register at internal memory
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0xFE,  # BP = 0xFE
            # Source data at internal memory (BP+50)
            # With BP=0xFE, (BP+50) = 0xFE + 0x50 = 0x14E, wrapped to 0x4E
            INTERNAL_MEMORY_START + 0x4E: 0xAA,
            INTERNAL_MEMORY_START + 0x4F: 0xBB,
            INTERNAL_MEMORY_START + 0x50: 0xCC,
        },
        expected_regs={
            RegisterName.I: 0,
        },
        expected_mem_state={
            # BP=0xFE, so (BP+00) = address 0xFE
            # MVL copies from (BP+50) to (BP+00) with incrementing addresses
            # Source: 0x4E, 0x4F, 0x50
            # Destination: 0xFE, 0xFF, 0x00 (wrapped)
            INTERNAL_MEMORY_START + 0xFE: 0xAA,  # From BP+50 (0x4E)
            INTERNAL_MEMORY_START + 0xFF: 0xBB,  # From BP+51 (0x4F)
            INTERNAL_MEMORY_START + 0x00: 0xCC,  # From BP+52 (0x50), wrapped from 0x100
            # Source and BP remain unchanged
            INTERNAL_MEMORY_START + 0x4E: 0xAA,
            INTERNAL_MEMORY_START + 0x4F: 0xBB,
            INTERNAL_MEMORY_START + 0x50: 0xCC,
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0xFE,
        },
        expected_asm_str="MVL   (BP+00), (BP+50)",
    ),
    # --- SHL/SHR Instructions ---
    # SHL A (0xF6)
    InstructionTestCase(
        test_id="SHL_A_simple",
        instr_bytes=bytes.fromhex("F6"),
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0xAA, RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_asm_str="SHL   A",
    ),
    InstructionTestCase(
        test_id="SHL_A_carry_in",
        instr_bytes=bytes.fromhex("F6"),
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x55, RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_asm_str="SHL   A",
    ),
    InstructionTestCase(
        test_id="SHL_A_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F6"),
        init_regs={RegisterName.A: 0x80, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x00, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHL   A",
    ),
    # SHR A (0xF4)
    InstructionTestCase(
        test_id="SHR_A_simple",
        instr_bytes=bytes.fromhex("F4"),
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x2A, RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_asm_str="SHR   A",
    ),
    InstructionTestCase(
        test_id="SHR_A_carry_in",
        instr_bytes=bytes.fromhex("F4"),
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0xD5, RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_asm_str="SHR   A",
    ),
    InstructionTestCase(
        test_id="SHR_A_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F4"),
        init_regs={RegisterName.A: 0x01, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x00, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHR   A",
    ),
    # SHL (n) (0xF7)
    InstructionTestCase(
        test_id="SHL_mem_simple",
        instr_bytes=bytes.fromhex("F710"),
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x55},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0xAA)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0xAA},
        expected_regs={RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_asm_str="SHL   (BP+10)",
    ),
    InstructionTestCase(
        test_id="SHL_mem_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F710"),
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x80},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x00)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x00},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHL   (BP+10)",
    ),
    # SHR (n) (0xF5)
    InstructionTestCase(
        test_id="SHR_mem_simple",
        instr_bytes=bytes.fromhex("F510"),
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x55},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x2A)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x2A},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_asm_str="SHR   (BP+10)",
    ),
    InstructionTestCase(
        test_id="SHR_mem_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F510"),
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x01},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x00)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x00},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHR   (BP+10)",
    ),
    # --- MV Instruction for Internal Memory Register ---
    InstructionTestCase(
        test_id="MV_BP_register_immediate",
        instr_bytes=bytes.fromhex("30ccecc2"),
        init_mem={INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x00},  # Initial BP = 0x00
        expected_mem_state={INTERNAL_MEMORY_START + IMEMRegisters.BP: 0xC2},  # BP = 0xC2
        expected_asm_str="MV    (EC), C2",
    ),
    # --- AND Instructions ---
    # 0x70: AND A, imm8
    InstructionTestCase(
        test_id="AND_A_imm_zero_result",
        instr_bytes=bytes.fromhex("7000"),  # AND A, 00
        init_regs={RegisterName.A: 0xFF, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="AND   A, 00",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("700F"),  # AND A, 0F
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x05, RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="AND   A, 0F",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_all_ones",
        instr_bytes=bytes.fromhex("70FF"),  # AND A, FF
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0xAA, RegisterName.FZ: 0, RegisterName.FC: 1},  # A unchanged, FC unchanged
        expected_asm_str="AND   A, FF",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_zero_operand",
        instr_bytes=bytes.fromhex("700F"),  # AND A, 0F
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 0},  # Zero result
        expected_asm_str="AND   A, 0F",
    ),
    # 0x71: AND (n), imm8
    InstructionTestCase(
        test_id="AND_mem_imm_zero_result",
        instr_bytes=bytes.fromhex("711000"),  # AND (BP+10), 00
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xFF},
        init_regs={RegisterName.FC: 1},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x00)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x00},
        expected_regs={RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="AND   (BP+10), 00",
    ),
    InstructionTestCase(
        test_id="AND_mem_imm_non_zero_result",
        instr_bytes=bytes.fromhex("71100F"),  # AND (BP+10), 0F
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x55},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x05)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x05},
        expected_regs={RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="AND   (BP+10), 0F",
    ),
    # --- OR Instructions ---
    # 0x78: OR A, imm8
    InstructionTestCase(
        test_id="OR_A_imm_zero_result",
        instr_bytes=bytes.fromhex("7800"),  # OR A, 00
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="OR    A, 00",
    ),
    InstructionTestCase(
        test_id="OR_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("780F"),  # OR A, 0F
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x0F, RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="OR    A, 0F",
    ),
    InstructionTestCase(
        test_id="OR_A_imm_all_ones",
        instr_bytes=bytes.fromhex("78FF"),  # OR A, FF
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="OR    A, FF",
    ),
    InstructionTestCase(
        test_id="OR_A_imm_combine_bits",
        instr_bytes=bytes.fromhex("78F0"),  # OR A, F0
        init_regs={RegisterName.A: 0x0F, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="OR    A, F0",
    ),
    # 0x79: OR (n), imm8
    InstructionTestCase(
        test_id="OR_mem_imm_zero_result",
        instr_bytes=bytes.fromhex("791000"),  # OR (BP+10), 00
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x00},
        init_regs={RegisterName.FC: 1},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x00)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x00},
        expected_regs={RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="OR    (BP+10), 00",
    ),
    InstructionTestCase(
        test_id="OR_mem_imm_non_zero_result",
        instr_bytes=bytes.fromhex("79100F"),  # OR (BP+10), 0F
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xF0},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0xFF)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0xFF},
        expected_regs={RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="OR    (BP+10), 0F",
    ),
    # --- XOR Instructions ---
    # 0x68: XOR A, imm8
    InstructionTestCase(
        test_id="XOR_A_imm_zero_result_same",
        instr_bytes=bytes.fromhex("68FF"),  # XOR A, FF
        init_regs={RegisterName.A: 0xFF, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="XOR   A, FF",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_zero_result_pattern",
        instr_bytes=bytes.fromhex("6855"),  # XOR A, 55
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="XOR   A, 55",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("6855"),  # XOR A, 55
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="XOR   A, 55",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_with_zero",
        instr_bytes=bytes.fromhex("6800"),  # XOR A, 00
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 0},
        expected_regs={RegisterName.A: 0xAA, RegisterName.FZ: 0, RegisterName.FC: 0},  # A unchanged
        expected_asm_str="XOR   A, 00",
    ),
    # 0x69: XOR (n), imm8
    InstructionTestCase(
        test_id="XOR_mem_imm_zero_result",
        instr_bytes=bytes.fromhex("6910FF"),  # XOR (BP+10), FF
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xFF},
        init_regs={RegisterName.FC: 1},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0x00)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0x00},
        expected_regs={RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="XOR   (BP+10), FF",
    ),
    InstructionTestCase(
        test_id="XOR_mem_imm_non_zero_result",
        instr_bytes=bytes.fromhex("69100F"),  # XOR (BP+10), 0F
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xF0},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(INTERNAL_MEMORY_START + 0x10, 0xFF)],
        expected_mem_state={INTERNAL_MEMORY_START + 0x10: 0xFF},
        expected_regs={RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="XOR   (BP+10), 0F",
    ),
    # Test register-to-register variants
    # 0x77: AND A, (n)
    InstructionTestCase(
        test_id="AND_A_mem_zero_result",
        instr_bytes=bytes.fromhex("7710"),  # AND A, (BP+10)
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 1},
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x0F},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="AND   A, (BP+10)",
    ),
    # 0x7F: OR A, (n)
    InstructionTestCase(
        test_id="OR_A_mem_non_zero_result",
        instr_bytes=bytes.fromhex("7F10"),  # OR A, (BP+10)
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 0},
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x0F},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 0},  # FC unchanged
        expected_asm_str="OR    A, (BP+10)",
    ),
    # 0x6F: XOR A, (n)
    InstructionTestCase(
        test_id="XOR_A_mem_zero_result",
        instr_bytes=bytes.fromhex("6F10"),  # XOR A, (BP+10)
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xAA},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},  # FC unchanged
        expected_asm_str="XOR   A, (BP+10)",
    ),
    
    # Test case for b204 instruction with BA=0x5AA5
    InstructionTestCase(
        test_id="MV_emem_BA_b204",
        instr_bytes=bytes.fromhex("B204"),  # MV [X], BA
        init_regs={
            RegisterName.BA: 0x5AA5, 
            RegisterName.X: 0xBE000,  # X points to address 0xBE000
            RegisterName.FC: 0, 
            RegisterName.FZ: 0
        },
        init_mem={0xBE000: 0x00, 0xBE001: 0x00},  # Clear external memory at 0xBE000-0xBE001
        expected_regs={
            RegisterName.BA: 0x5AA5, 
            RegisterName.X: 0xBE000,  # X unchanged
            RegisterName.FC: 0, 
            RegisterName.FZ: 0
        },
        expected_mem_writes=[(0xBE000, 0xA5), (0xBE001, 0x5A)],  # Little-endian: LSB first
        expected_mem_state={0xBE000: 0xA5, 0xBE001: 0x5A},  # BA=0x5AA5 stored as A5 5A
        expected_asm_str="MV    [X], BA",
    ),
    
    # Test case for 30e904d4 instruction - MVW [X], (BP+D4)
    InstructionTestCase(
        test_id="MVW_X_indirect_from_BP_30e904d4",
        instr_bytes=bytes.fromhex("30E904D4"),  # MVW [X], (BP+D4)
        init_regs={
            RegisterName.X: 0x080000,  # X points to external memory address 0x080000
            RegisterName.FC: 0, 
            RegisterName.FZ: 0
        },
        init_mem={
            # Initialize internal memory at BP+D4 with test data
            INTERNAL_MEMORY_START + 0xD4: 0x34,  # Low byte of word
            INTERNAL_MEMORY_START + 0xD5: 0x12,  # High byte of word
            # Clear destination memory
            0x080000: 0x00,
            0x080001: 0x00,
        },
        expected_regs={
            RegisterName.X: 0x080000,  # X unchanged
            RegisterName.FC: 0,  # Flags unchanged
            RegisterName.FZ: 0
        },
        expected_mem_writes=[
            (0x080000, 0x34),  # Low byte written first
            (0x080001, 0x12),  # High byte written second
        ],
        expected_mem_state={
            0x080000: 0x34,  # Word 0x1234 stored little-endian
            0x080001: 0x12,
        },
        expected_asm_str="MVW   [X], (BP+D4)",
    ),
    InstructionTestCase(
        test_id="MV_Y_from_BP_plus_E6_PRE30_no_wrap",
        instr_bytes=bytes.fromhex("3085E6"),
        init_mem={
            # BP register at internal memory
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,  # BP = 0x10, so BP+0xE6 = 0xF6 (no wrap)
            # Place test data at internal memory 0xF6-0xF8
            INTERNAL_MEMORY_START + 0xF6: 0x11,
            INTERNAL_MEMORY_START + 0xF7: 0x22,
            INTERNAL_MEMORY_START + 0xF8: 0x33,
        },
        expected_regs={
            RegisterName.Y: 0x332211,  # Little-endian: low byte first
        },
        expected_asm_str="MV    Y, (BP+E6)",
    ),
    InstructionTestCase(
        test_id="MV_Y_from_BP_plus_E6_PRE30_with_wraparound",
        instr_bytes=bytes.fromhex("3085E6"),
        init_mem={
            # BP register at internal memory
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x20,  # BP = 0x20, so BP+0xE6 = 0x106, wraps to 0x06
            # Place test data at internal memory 0x06-0x08 (after wraparound)
            INTERNAL_MEMORY_START + 0x06: 0x11,
            INTERNAL_MEMORY_START + 0x07: 0x22,
            INTERNAL_MEMORY_START + 0x08: 0x33,
        },
        expected_regs={
            RegisterName.Y: 0x332211,  # Little-endian: low byte first
        },
        expected_asm_str="MV    Y, (BP+E6)",
    ),
]

# --- New Centralized Test Runner ---


@pytest.mark.parametrize(
    "case",
    instruction_test_cases,
    ids=[case.test_id for case in instruction_test_cases],
)
def test_instruction_execution(case: InstructionTestCase) -> None:
    """
    A generic, parameterized test function that runs a single instruction case.
    """
    # 1. Setup Phase
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, case.init_mem, case.instr_bytes, case.initial_pc
    )

    for reg, val in case.init_regs.items():
        cpu.regs.set(reg, val)

    # 2. Decode Phase - verify disassembly if expected
    decoded = cpu.decode_instruction(case.initial_pc)
    actual_asm = asm_str(decoded.render())
    if case.expected_asm_str is not None:
        assert actual_asm == case.expected_asm_str, (
            f"[{case.test_id}] Assembly mismatch:\n"
            f"  Expected: '{case.expected_asm_str}'\n"
            f"  Actual  : '{actual_asm}'"
        )

    # 3. Execution Phase
    _ = cpu.execute_instruction(case.initial_pc)
    # 4. Assertion Phase
    # Check register states
    for reg, expected_val in case.expected_regs.items():
        actual_val = cpu.regs.get(reg)
        assert actual_val == expected_val, (
            f"[{case.test_id}] Register {reg.name} mismatch: "
            f"Expected 0x{expected_val:X}, Got 0x{actual_val:X}"
        )

    # Check memory write log
    if case.expected_mem_writes is not None:
        # Sort both lists to make comparison order-independent if necessary
        assert sorted(writes) == sorted(case.expected_mem_writes), (
            f"[{case.test_id}] Memory write log mismatch: "
            f"Expected {case.expected_mem_writes}, Got {writes}"
        )

    # Check final memory state
    for addr, expected_val in case.expected_mem_state.items():
        actual_val = raw[addr]
        assert actual_val == expected_val, (
            f"[{case.test_id}] Memory state at 0x{addr:X} mismatch: "
            f"Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
        )


def test_pushs_pops() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("4F"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHS F"

    cpu.regs.set(RegisterName.F, 0x0)
    cpu.regs.set(RegisterName.S, 0x20)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1F
    assert writes == [(0x1F, 0x0)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1E
    assert writes == [(0x1E, 0x2)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 0)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1D
    assert writes == [(0x1D, 0x1)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1C
    assert writes == [(0x1C, 0x3)]
    writes.clear()

    cpu.regs.set(RegisterName.F, 0)
    raw[0] = 0x5F  # Change to POPS instruction
    assert asm_str(cpu.decode_instruction(0x00).render()) == "POPS  F"
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1D
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1E
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x1F
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x20
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 0


def test_pushu_popu() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("2E"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHU F"

    cpu.regs.set(RegisterName.F, 0x0)
    cpu.regs.set(RegisterName.U, 0x20)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1F
    assert writes == [(0x1F, 0x0)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1E
    assert writes == [(0x1E, 0x2)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 0)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1D
    assert writes == [(0x1D, 0x1)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1C
    assert writes == [(0x1C, 0x3)]
    writes.clear()

    cpu.regs.set(RegisterName.F, 0)
    raw[0] = 0x3E  # POPU instruction
    assert asm_str(cpu.decode_instruction(0x00).render()) == "POPU  F"
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1D
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1E
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x1F
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0


def test_pushu_popu_r2() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("2A"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHU BA"

    cpu.regs.set(RegisterName.BA, 0x1234)
    cpu.regs.set(RegisterName.U, 0x30)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x2E
    assert writes == [(0x2E, 0x34), (0x2F, 0x12)]
    writes.clear()

    raw[0] = 0x3A  # POPU BA
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x30
    assert cpu.regs.get(RegisterName.BA) == 0x1234

def test_call_ret() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("042000"))
    raw[0x20] = 0x06
    assert asm_str(cpu.decode_instruction(0x00).render()) == "CALL  0020"
    assert asm_str(cpu.decode_instruction(0x20).render()) == "RET"

    cpu.regs.set(RegisterName.S, 0x30)  # Set stack pointer to a valid location
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.PC) == 0x20
    assert cpu.regs.get(RegisterName.S) == 0x2E
    assert writes == [(0x2E, 0x03), (0x2F, 0x00)]
    writes.clear()

    _ = cpu.execute_instruction(cpu.regs.get(RegisterName.PC))
    assert cpu.regs.get(RegisterName.PC) == 0x03
    assert cpu.regs.get(RegisterName.S) == 0x30
    assert writes == []


def test_call_ret_high_page() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        0x40000, {}, bytes.fromhex("042000"), instr_addr=0x30000
    )
    raw[0x30020] = 0x06
    assert asm_str(cpu.decode_instruction(0x30000).render()) == "CALL  0020"
    assert asm_str(cpu.decode_instruction(0x30020).render()) == "RET"

    cpu.regs.set(RegisterName.S, 0x30)
    _ = cpu.execute_instruction(0x30000)
    assert cpu.regs.get(RegisterName.PC) == 0x30020
    assert cpu.regs.get(RegisterName.S) == 0x2E
    assert writes == [(0x2E, 0x03), (0x2F, 0x00)]
    writes.clear()

    _ = cpu.execute_instruction(cpu.regs.get(RegisterName.PC))
    assert cpu.regs.get(RegisterName.PC) == 0x30003
    assert cpu.regs.get(RegisterName.S) == 0x30
    assert writes == []


def test_callf_retf() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("05200000"))
    raw[0x20] = 0x07
    assert asm_str(cpu.decode_instruction(0x00).render()) == "CALLF 00020"
    assert asm_str(cpu.decode_instruction(0x20).render()) == "RETF"

    cpu.regs.set(RegisterName.S, 0x30)  # Set stack pointer to a valid location
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.PC) == 0x20
    assert cpu.regs.get(RegisterName.S) == 0x2D
    assert writes == [(0x2D, 0x04), (0x2E, 0x00), (0x2F, 0x00)]
    writes.clear()

    _ = cpu.execute_instruction(cpu.regs.get(RegisterName.PC))
    assert cpu.regs.get(RegisterName.PC) == 0x04
    assert cpu.regs.get(RegisterName.S) == 0x30
    assert writes == []


def test_rol_ror_a() -> None:
    cpu, _, _, _writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("E6"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "ROL   A"

    # Case 1: A = 0x55 (01010101)
    cpu.regs.set(RegisterName.A, 0x55)
    _ = cpu.execute_instruction(0x00)
    # MSB is 0. (01010101 << 1) | 0 = 10101010
    assert cpu.regs.get(RegisterName.A) == 0xAA
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == 0

    # Case 2: A = 0xAA (10101010)
    cpu.regs.set(RegisterName.A, 0xAA)
    _ = cpu.execute_instruction(0x00)
    # MSB is 1. (10101010 << 1) | 1 = 010101010 | 1 = 01010101
    assert cpu.regs.get(RegisterName.A) == 0x55
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0

    cpu, _, _, _writes = _make_cpu_and_mem(0x40, {}, bytes.fromhex("E4"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "ROR   A"

    # Case 1: A = 0x55 (01010101)
    cpu.regs.set(RegisterName.A, 0x55)
    _ = cpu.execute_instruction(0x00)
    # LSB is 1. (01010101 >> 1) | (1 << 7) = 00101010 | 10000000 = 10101010
    assert cpu.regs.get(RegisterName.A) == 0xAA
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0

    # Case 2: A = 0xAA (10101010)
    cpu.regs.set(RegisterName.A, 0xAA)
    _ = cpu.execute_instruction(0x00)
    # LSB is 0. (10101010 >> 1) | (0 << 7) = 01010101
    assert cpu.regs.get(RegisterName.A) == 0x55
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == 0



class PreTestCase(NamedTuple):
    test_id: str  # Descriptive name for the test case
    instr_bytes: bytes  # The full instruction byte sequence (PRE + MV + operands)
    init_memory_state: Dict[int, int]  # Initial values in memory {address: value}
    init_register_state: Dict[
        RegisterName, int
    ]  # Initial register values {reg_name: value}
    expected_asm_str: str  # Expected assembly string after decoding
    expected_pre_val_in_instr: int  # The PRE byte value itself, as stored in the decoded instr

    # For tests like MV A, (mem_source)
    expected_A_val_after: Optional[int] = None

    # For tests like MV (mem_dest), A
    expected_mem_writes_after: Optional[
        List[Tuple[int, int]]
    ] = None  # List of (address, value)


def get_pre_test_cases() -> List[PreTestCase]:
    # Operand 'n' in (n), (BP+n), etc.
    N_OPERAND_VAL = 0x05

    # Value to write to memory or load into A
    OPERAND_A_VAL = 0x77
    OPERAND_MEM_VAL = 0xCC  # Value initially in memory if A is being loaded

    # Initial values for internal RAM pointer registers
    BP_REG_VAL = 0x10
    PX_REG_VAL = 0x20
    PY_REG_VAL = 0x30

    # Base opcodes for MV instructions involving one internal memory operand and register A
    MV_MEM_DEST_A_SRC_OPCODE = 0xA0  # MV (n), A
    MV_A_DEST_MEM_SRC_OPCODE = 0x80  # MV A, (n)

    STATIC_PRE_TEST_CASES: List[PreTestCase] = [
        # --- Test Group: PRE affecting 1st operand (Destination) ---
        # Example: MV (dest_mode), A
        PreTestCase(
            test_id="PRE_0x32_Op1_N_MV_(n)_A",
            instr_bytes=bytes([0x32, MV_MEM_DEST_A_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={},  # No BP/PX/PY needed for (n)
            init_register_state={RegisterName.A: OPERAND_A_VAL},
            expected_asm_str=f"MV    ({N_OPERAND_VAL:02X}), A",
            expected_pre_val_in_instr=0x32,
            expected_mem_writes_after=[
                (INTERNAL_MEMORY_START + N_OPERAND_VAL, OPERAND_A_VAL)
            ],
        ),
        PreTestCase(
            test_id="PRE_0x22_Op1_BP_N_MV_(BP+n)_A",
            instr_bytes=bytes([0x22, MV_MEM_DEST_A_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
            },
            init_register_state={RegisterName.A: OPERAND_A_VAL},
            expected_asm_str=f"MV    (BP+{N_OPERAND_VAL:02X}), A",
            expected_pre_val_in_instr=0x22,
            expected_mem_writes_after=[
                (
                    INTERNAL_MEMORY_START + ((BP_REG_VAL + N_OPERAND_VAL) & 0xFF),
                    OPERAND_A_VAL,
                )
            ],
        ),
        PreTestCase(
            test_id="PRE_0x36_Op1_PX_N_MV_(PX+n)_A",
            instr_bytes=bytes([0x36, MV_MEM_DEST_A_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.PX: PX_REG_VAL,
            },
            init_register_state={RegisterName.A: OPERAND_A_VAL},
            expected_asm_str=f"MV    (PX+{N_OPERAND_VAL:02X}), A",
            expected_pre_val_in_instr=0x36,
            expected_mem_writes_after=[
                (
                    INTERNAL_MEMORY_START + ((PX_REG_VAL + N_OPERAND_VAL) & 0xFF),
                    OPERAND_A_VAL,
                )
            ],
        ),
        PreTestCase(
            test_id="PRE_0x26_Op1_BP_PX_MV_(BP+PX)_A",
            instr_bytes=bytes(
                [0x26, MV_MEM_DEST_A_SRC_OPCODE, N_OPERAND_VAL]
            ),  # N_OPERAND_VAL is present but ignored by (BP+PX) mode for destination calculation
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
                INTERNAL_MEMORY_START + IMEMRegisters.PX: PX_REG_VAL,
            },
            init_register_state={RegisterName.A: OPERAND_A_VAL},
            expected_asm_str="MV    (BP+PX), A",
            expected_pre_val_in_instr=0x26,
            expected_mem_writes_after=[
                (
                    INTERNAL_MEMORY_START + ((BP_REG_VAL + PX_REG_VAL) & 0xFF),
                    OPERAND_A_VAL,
                )
            ],
        ),
        # --- Test Group: PRE affecting 2nd operand (Source) ---
        # Example: MV A, (src_mode)
        PreTestCase(
            test_id="PRE_0x32_Op2_N_MV_A_(n)",
            instr_bytes=bytes([0x32, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + N_OPERAND_VAL: OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},  # To ensure A gets overwritten
            expected_asm_str=f"MV    A, ({N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x32,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
        PreTestCase(
            test_id="PRE_0x30_Op2_BP_N_MV_A_(BP+n)",  # 0x30 for 2nd op (BP+n) if 1st op is (n)
            instr_bytes=bytes([0x30, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
                INTERNAL_MEMORY_START
                + ((BP_REG_VAL + N_OPERAND_VAL) & 0xFF): OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str=f"MV    A, (BP+{N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x30,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
        PreTestCase(
            test_id="PRE_0x33_Op2_PY_N_MV_A_(PY+n)",  # 0x33 for 2nd op (PY+n) if 1st op is (n)
            instr_bytes=bytes([0x33, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.PY: PY_REG_VAL,
                INTERNAL_MEMORY_START
                + ((PY_REG_VAL + N_OPERAND_VAL) & 0xFF): OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str=f"MV    A, (PY+{N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x33,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
        PreTestCase(
            test_id="PRE_0x31_Op2_BP_PY_MV_A_(BP+PY)",  # 0x31 for 2nd op (BP+PY) if 1st op is (n)
            instr_bytes=bytes(
                [0x31, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]
            ),  # N_OPERAND_VAL ignored for (BP+PY) source
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
                INTERNAL_MEMORY_START + IMEMRegisters.PY: PY_REG_VAL,
                INTERNAL_MEMORY_START
                + ((BP_REG_VAL + PY_REG_VAL) & 0xFF): OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str="MV    A, (BP+PY)",
            expected_pre_val_in_instr=0x31,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
    ]
    return STATIC_PRE_TEST_CASES


@pytest.mark.parametrize(
    "tc",  # tc (test_case) will be an instance of PreTestCase
    get_pre_test_cases(),
    ids=[
        case.test_id for case in get_pre_test_cases()
    ],  # Use test_id for readable test names
)
def test_pre_addressing_modes(tc: PreTestCase) -> None:
    cpu, raw_memory_array, _logged_reads, logged_writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        tc.init_memory_state,
        tc.instr_bytes,
    )

    for reg, val in tc.init_register_state.items():
        cpu.regs.set(reg, val)

    # --- Decode and Verify Assembly and PRE Byte ---
    decoded_instr = cpu.decode_instruction(0x00)  # Instructions are at address 0x00
    assert (
        decoded_instr is not None
    ), f"Test '{tc.test_id}': Failed to decode instruction bytes: {tc.instr_bytes.hex()}"

    assert decoded_instr._pre == tc.expected_pre_val_in_instr, (
        f"Test '{tc.test_id}': Decoded instruction's _pre value (0x{decoded_instr._pre:02X if decoded_instr._pre is not None else 'None'}) "
        f"does not match expected PRE byte (0x{tc.expected_pre_val_in_instr:02X})"
    )

    actual_asm_string = asm_str(decoded_instr.render())
    assert actual_asm_string == tc.expected_asm_str, (
        f"Test '{tc.test_id}': Assembly string mismatch.\n"
        f"  Expected: '{tc.expected_asm_str}'\n"
        f"  Actual  : '{actual_asm_string}'"
    )

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    if tc.expected_A_val_after is not None:
        # This is a "MV A, (mem_src)" type test
        loaded_value_in_a = cpu.regs.get(RegisterName.A)
        assert loaded_value_in_a == tc.expected_A_val_after, (
            f"Test '{tc.test_id}': Expected Register A to be 0x{tc.expected_A_val_after:02X}, "
            f"but got 0x{loaded_value_in_a:02X}"
        )

    if tc.expected_mem_writes_after is not None:
        # This is a "MV (mem_dest), A" type test
        assert logged_writes == tc.expected_mem_writes_after, (
            f"Test '{tc.test_id}': Memory writes mismatch.\n"
            f"  Expected: {tc.expected_mem_writes_after}\n"
            f"  Actual  : {logged_writes}"
        )
        # Also verify the content in the raw_memory_array for writes
        for addr, val in tc.expected_mem_writes_after:
            assert raw_memory_array[addr] == val, (
                f"Test '{tc.test_id}': Memory content at 0x{addr:04X} is 0x{raw_memory_array[addr]:02X}, "
                f"expected 0x{val:02X}"
            )


class AdclDadlTestCase(NamedTuple):
    test_id: str
    instr_bytes: bytes
    init_memory_state: Dict[int, int]  # Includes internal mem values for operands
    init_register_state: Dict[RegisterName, int]  # Includes A, I, FC
    expected_asm_str: str
    # For (m) which is the destination
    expected_m_addr_start: int
    expected_m_values_after: List[int]  # Byte values written to (m)
    expected_I_after: int
    expected_FC_after: int
    expected_FZ_after: int


# ADCL Tests
# Opcode 0x54: ADCL (m), (n)
# Opcode 0x55: ADCL (m), A
adcl_test_cases: List[AdclDadlTestCase] = [
    # --- ADCL (m), (n) ---
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I1_NoCarryIn_NoCarryOut",
        instr_bytes=bytes([0x54, 0x10, 0x20]),  # ADCL (10), (20)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x12,
            INTERNAL_MEMORY_START + 0x20: 0x34,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x46],  # 0x12 + 0x34 = 0x46
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I1_WithCarryIn_NoCarryOut",
        instr_bytes=bytes([0x54, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x12,
            INTERNAL_MEMORY_START + 0x20: 0x34,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 1},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x47],  # 0x12 + 0x34 + 1 = 0x47
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I1_NoCarryIn_CarryOut",
        instr_bytes=bytes([0x54, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0xF0,
            INTERNAL_MEMORY_START + 0x20: 0x20,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x10],  # 0xF0 + 0x20 = 0x110 -> 0x10
        expected_I_after=0,
        expected_FC_after=1,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I1_NoCarryIn_ZeroResult_CarryOut",
        instr_bytes=bytes([0x54, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0xAA,
            INTERNAL_MEMORY_START + 0x20: 0x56,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x00],  # 0xAA + 0x56 = 0x100 -> 0x00
        expected_I_after=0,
        expected_FC_after=1,
        expected_FZ_after=1,
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I2_CarryPropagate_OverallNonZero",
        instr_bytes=bytes([0x54, 0x10, 0x20]),  # ADCL (10), (20)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0xFF,
            INTERNAL_MEMORY_START + 0x11: 0x01,  # (m)
            INTERNAL_MEMORY_START + 0x20: 0x01,
            INTERNAL_MEMORY_START + 0x21: 0x02,  # (n)
        },
        init_register_state={RegisterName.I: 2, RegisterName.FC: 0},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # Byte 0: 0xFF + 0x01 + 0 = 0x100 -> mem[0x10]=0x00, FC=1
        # Byte 1: 0x01 + 0x02 + 1 = 0x04  -> mem[0x11]=0x04, FC=0
        expected_m_values_after=[0x00, 0x04],
        expected_I_after=0,
        expected_FC_after=0,  # From last byte op
        expected_FZ_after=0,  # Overall: (0x00 | 0x04) != 0
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_(n)_I2_OverallZero",
        instr_bytes=bytes([0x54, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0xFF,
            INTERNAL_MEMORY_START + 0x11: 0xFF,  # (m)
            INTERNAL_MEMORY_START + 0x20: 0x01,
            INTERNAL_MEMORY_START + 0x21: 0x00,  # (n)
        },
        init_register_state={RegisterName.I: 2, RegisterName.FC: 0},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # Byte 0: 0xFF + 0x01 + 0 = 0x100 -> mem[0x10]=0x00, FC=1
        # Byte 1: 0xFF + 0x00 + 1 = 0x100 -> mem[0x11]=0x00, FC=1
        expected_m_values_after=[0x00, 0x00],
        expected_I_after=0,
        expected_FC_after=1,
        expected_FZ_after=1,  # Overall: (0x00 | 0x00) == 0
    ),
    # --- ADCL (m), A ---
    AdclDadlTestCase(
        test_id="ADCL_(m)_A_I1_NoCarryIn_NoCarryOut",
        instr_bytes=bytes([0x55, 0x10]),  # ADCL (10), A
        init_memory_state={INTERNAL_MEMORY_START + 0x10: 0x12},
        init_register_state={
            RegisterName.A: 0x34,
            RegisterName.I: 1,
            RegisterName.FC: 0,
        },
        expected_asm_str="ADCL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x46],  # 0x12 + 0x34 = 0x46
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="ADCL_(m)_A_I2_CarryPropagate",
        instr_bytes=bytes([0x55, 0x10]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0xFF,
            INTERNAL_MEMORY_START + 0x11: 0x01,
        },
        init_register_state={
            RegisterName.A: 0x01,
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },
        expected_asm_str="ADCL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # Byte 0: mem[0x10]=0xFF, A=0x01. 0xFF + 0x01 + 0 = 0x100 -> mem[0x10]=0x00, FC=1
        # Byte 1: mem[0x11]=0x01, A=0x01. 0x01 + 0x01 + 1 = 0x03  -> mem[0x11]=0x03, FC=0
        expected_m_values_after=[0x00, 0x03],
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,  # Overall: (0x00 | 0x03) != 0
    ),
]


@pytest.mark.parametrize(
    "tc", adcl_test_cases, ids=[case.test_id for case in adcl_test_cases]
)
def test_adcl_instruction(tc: AdclDadlTestCase) -> None:
    cpu, raw_memory_array, _, logged_writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, tc.init_memory_state, tc.instr_bytes
    )

    for reg, val in tc.init_register_state.items():
        cpu.regs.set(reg, val)

    decoded_instr = cpu.decode_instruction(0x00)
    assert decoded_instr is not None, f"Test '{tc.test_id}': Failed to decode"
    actual_asm_str = asm_str(decoded_instr.render())
    assert (
        actual_asm_str == tc.expected_asm_str
    ), f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert (
            actual_val == expected_val
        ), f"Test '{tc.test_id}': Memory mismatch at offset {i}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"

    # Verify logged writes if needed, though direct memory check is more robust here
    # For ADCL, (m) is destination, so writes should match expected_m_values_after
    expected_writes_to_m = [
        (tc.expected_m_addr_start + i, val)
        for i, val in enumerate(tc.expected_m_values_after)
    ]
    # Filter logged_writes to only include those to the (m) area
    actual_writes_to_m = sorted(
        [
            w
            for w in logged_writes
            if tc.expected_m_addr_start
            <= w[0]
            < tc.expected_m_addr_start + len(tc.expected_m_values_after)
        ]
    )
    assert actual_writes_to_m == sorted(
        expected_writes_to_m
    ), f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"

    assert (
        cpu.regs.get(RegisterName.I) == tc.expected_I_after
    ), f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    assert (
        cpu.regs.get(RegisterName.FC) == tc.expected_FC_after
    ), f"Test '{tc.test_id}': Flag C. Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"

    assert (
        cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after
    ), f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"


# DADL Tests
# Opcode 0xC4: DADL (m), (n)
# Opcode 0xC5: DADL (m), A
# Addresses for DADL are decremented, so m_addr_start is effectively the end address for comparison.
dadl_test_cases: List[AdclDadlTestCase] = [
    # --- DADL (m), (n) ---
    AdclDadlTestCase(
        test_id="DADL_(m)_(n)_I1_NoCarryIn_SimpleBCD",
        instr_bytes=bytes([0xC4, 0x10, 0x20]),  # DADL (10), (20)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x12,
            INTERNAL_MEMORY_START + 0x20: 0x34,
        },  # BCD 12, BCD 34
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DADL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START
        + 0x10,  # Addr (10) is used as is (LSB)
        expected_m_values_after=[0x46],  # BCD 12 + BCD 34 = BCD 46
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="DADL_(m)_(n)_I1_NoCarryIn_BCDHalfCarry",
        instr_bytes=bytes([0xC4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x05,
            INTERNAL_MEMORY_START + 0x20: 0x05,
        },  # BCD 05, BCD 05
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DADL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x10],  # BCD 05 + BCD 05 = BCD 10
        expected_I_after=0,
        expected_FC_after=0,  # No BCD carry-out from byte
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="DADL_(m)_(n)_I1_NoCarryIn_BCDCarryOut",
        instr_bytes=bytes([0xC4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x50,
            INTERNAL_MEMORY_START + 0x20: 0x50,
        },  # BCD 50, BCD 50
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DADL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x00],  # BCD 50 + BCD 50 = BCD 100 -> 00, C=1
        expected_I_after=0,
        expected_FC_after=1,
        expected_FZ_after=1,  # Overall Z
    ),
    AdclDadlTestCase(
        test_id="DADL_(m)_(n)_I2_BCDCarryPropagate_OverallNonZero",
        # (m) from 0x10, (n) from 0x20. I=2. Addrs decrement.
        # (m) values at 0x11 (LSB), 0x10 (MSB)
        # (n) values at 0x21 (LSB), 0x20 (MSB)
        instr_bytes=bytes(
            [0xC4, 0x11, 0x21]
        ),  # DADL (0x11), (0x21) -> refers to end addresses
        init_memory_state={
            INTERNAL_MEMORY_START + 0x11: 0x50,
            INTERNAL_MEMORY_START + 0x10: 0x01,  # (m) = BCD 0150
            INTERNAL_MEMORY_START + 0x21: 0x50,
            INTERNAL_MEMORY_START + 0x20: 0x02,  # (n) = BCD 0250
        },
        init_register_state={RegisterName.I: 2, RegisterName.FC: 0},
        expected_asm_str="DADL  (BP+11), (BP+21)",
        expected_m_addr_start=INTERNAL_MEMORY_START
        + 0x10,  # Start address for verification of written (m)
        # Byte 0 (LSB, addrs 0x11, 0x21): 0x50 + 0x50 + 0 = BCD 100 -> mem[0x11]=0x00, FC=1
        # Byte 1 (MSB, addrs 0x10, 0x20): 0x01 + 0x02 + 1 = BCD 04  -> mem[0x10]=0x04, FC=0
        expected_m_values_after=[0x04, 0x00],  # MSB then LSB for (m) area
        expected_I_after=0,
        expected_FC_after=0,  # From last byte op
        expected_FZ_after=0,  # Overall: (0x04 | 0x00) != 0
    ),
    # --- DADL (m), A ---
    AdclDadlTestCase(
        test_id="DADL_(m)_A_I1_NoCarryIn_SimpleBCD",
        instr_bytes=bytes([0xC5, 0x10]),  # DADL (10), A
        init_memory_state={INTERNAL_MEMORY_START + 0x10: 0x12},  # BCD 12
        init_register_state={
            RegisterName.A: 0x34,
            RegisterName.I: 1,
            RegisterName.FC: 0,
        },  # A = BCD 34
        expected_asm_str="DADL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x46],  # BCD 12 + BCD 34 = BCD 46
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    AdclDadlTestCase(
        test_id="DADL_(m)_A_I2_BCDCarryPropagate",
        instr_bytes=bytes([0xC5, 0x11]),  # DADL (0x11), A (0x11 is end addr for m)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x11: 0x99,
            INTERNAL_MEMORY_START + 0x10: 0x01,  # (m) = BCD 0199
        },
        init_register_state={
            RegisterName.A: 0x01,
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },  # A = BCD 01
        expected_asm_str="DADL  (BP+11), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # Start for verification
        # Byte 0 (LSB, addr 0x11): mem[0x11]=0x99, A=0x01. BCD 99 + BCD 01 + 0 = BCD 100 -> mem[0x11]=0x00, FC=1
        # Byte 1 (MSB, addr 0x10): mem[0x10]=0x01, A=0x01. BCD 01 + BCD 01 + 1 = BCD 03  -> mem[0x10]=0x03, FC=0
        expected_m_values_after=[0x03, 0x00],  # MSB then LSB
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,  # Overall: (0x03 | 0x00) != 0
    ),
]


@pytest.mark.parametrize(
    "tc", dadl_test_cases, ids=[case.test_id for case in dadl_test_cases]
)
def test_dadl_instruction(tc: AdclDadlTestCase) -> None:
    cpu, raw_memory_array, _, logged_writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, tc.init_memory_state, tc.instr_bytes
    )

    for reg, val in tc.init_register_state.items():
        cpu.regs.set(reg, val)

    decoded_instr = cpu.decode_instruction(0x00)
    assert decoded_instr is not None, f"Test '{tc.test_id}': Failed to decode"
    actual_asm_str = asm_str(decoded_instr.render())
    assert (
        actual_asm_str == tc.expected_asm_str
    ), f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert (
            actual_val == expected_val
        ), f"Test '{tc.test_id}': Memory mismatch at offset {i} from MSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"

    expected_writes_to_m = []
    num_bytes = len(tc.expected_m_values_after)
    lsb_address_m = tc.expected_m_addr_start + num_bytes - 1
    for i in range(num_bytes):
        addr = lsb_address_m - i  # X, X-1, ...
        val_idx_in_expected = num_bytes - 1 - i  # LSB_val, ..., MSB_val
        expected_writes_to_m.append(
            (addr, tc.expected_m_values_after[val_idx_in_expected])
        )

    actual_writes_to_m = sorted(
        [
            w
            for w in logged_writes
            if tc.expected_m_addr_start <= w[0] < tc.expected_m_addr_start + num_bytes
        ]
    )
    assert actual_writes_to_m == sorted(
        expected_writes_to_m
    ), f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"

    assert (
        cpu.regs.get(RegisterName.I) == tc.expected_I_after
    ), f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    assert (
        cpu.regs.get(RegisterName.FC) == tc.expected_FC_after
    ), f"Test '{tc.test_id}': Flag C. Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    assert (
        cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after
    ), f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"


# Add this to test_emulator.py


class SbclDsblTestCase(NamedTuple):
    test_id: str
    instr_bytes: bytes
    init_memory_state: Dict[int, int]  # Includes internal mem values for operands
    init_register_state: Dict[RegisterName, int]  # Includes A, I, FC
    expected_asm_str: str
    # For (m) which is the destination
    # For SBCL (forward): LSB address of (m). Values are [LSB, MSB, ...]
    # For DSBL (reverse): MSB address of (m). Values are [MSB, LSB, ...]
    expected_m_addr_start: int
    expected_m_values_after: List[int]  # Byte values written to (m)
    expected_I_after: int
    expected_FC_after: int  # FC=1 if borrow occurred, 0 otherwise for SUB/SBC based
    expected_FZ_after: int


# SBCL Tests
# Opcode 0x5C: SBCL (m), (n)
# Opcode 0x5D: SBCL (m), A
# SBCL is a forward operation (addresses for (m) and (n) increment)
sbcl_test_cases: List[SbclDsblTestCase] = [
    # --- SBCL (m), (n) ---
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I1_NoBorrowIn_NoBorrowOut",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),  # SBCL (10), (20)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x55,
            INTERNAL_MEMORY_START + 0x20: 0x22,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # LSB address
        expected_m_values_after=[0x33],  # 0x55 - 0x22 - 0 = 0x33
        expected_I_after=0,
        expected_FC_after=0,  # No borrow
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I1_WithBorrowIn_NoBorrowOut",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x55,
            INTERNAL_MEMORY_START + 0x20: 0x22,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 1},  # Borrow In
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x32],  # 0x55 - 0x22 - 1 = 0x32
        expected_I_after=0,
        expected_FC_after=0,  # No borrow
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I1_NoBorrowIn_BorrowOut",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x10,
            INTERNAL_MEMORY_START + 0x20: 0x20,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0xF0],  # 0x10 - 0x20 - 0 = 0xF0 (borrow)
        expected_I_after=0,
        expected_FC_after=1,  # Borrow occurred
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I1_NoBorrowIn_ZeroResult",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x20,
            INTERNAL_MEMORY_START + 0x20: 0x20,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x00],  # 0x20 - 0x20 - 0 = 0x00
        expected_I_after=0,
        expected_FC_after=0,  # No borrow
        expected_FZ_after=1,  # Zero result
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I2_BorrowPropagate_OverallNonZero",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),  # SBCL (10), (20)
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x00,  # LSB of (m)
            INTERNAL_MEMORY_START + 0x11: 0x50,  # MSB of (m) -> (m) = 0x5000
            INTERNAL_MEMORY_START + 0x20: 0x01,  # LSB of (n)
            INTERNAL_MEMORY_START + 0x21: 0x20,  # MSB of (n) -> (n) = 0x2001
        },
        init_register_state={
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },  # No initial borrow
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # LSB addr of (m)
        # Byte 0 (LSB, addrs 0x10, 0x20): m[0x10]=0x00, n[0x20]=0x01. 0x00 - 0x01 - 0 = 0xFF. mem[0x10]=0xFF, FC=1 (borrow)
        # Byte 1 (MSB, addrs 0x11, 0x21): m[0x11]=0x50, n[0x21]=0x20. 0x50 - 0x20 - 1(borrow_in) = 0x2F. mem[0x11]=0x2F, FC=0
        expected_m_values_after=[0xFF, 0x2F],  # LSB, MSB for (m) area
        expected_I_after=0,
        expected_FC_after=0,  # From last byte op
        expected_FZ_after=0,  # Overall: (0xFF | 0x2F) != 0
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_(n)_I2_OverallZero_WithBorrowOut",
        instr_bytes=bytes([0x5C, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x00,  # LSB of (m)
            INTERNAL_MEMORY_START + 0x11: 0x00,  # MSB of (m) -> (m) = 0x0000
            INTERNAL_MEMORY_START + 0x20: 0x00,  # LSB of (n)
            INTERNAL_MEMORY_START + 0x21: 0x00,  # MSB of (n) -> (n) = 0x0000
        },
        init_register_state={
            RegisterName.I: 2,
            RegisterName.FC: 1,
        },  # Initial Borrow In (e.g. from 0 - 0 - 1)
        expected_asm_str="SBCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # Byte 0: 0x00 - 0x00 - 1 = 0xFF. mem[0x10]=0xFF, FC=1
        # Byte 1: 0x00 - 0x00 - 1 = 0xFF. mem[0x11]=0xFF, FC=1
        expected_m_values_after=[0xFF, 0xFF],
        expected_I_after=0,
        expected_FC_after=1,  # Borrow from last op
        expected_FZ_after=0,  # Overall is 0xFFFF, not zero
    ),
    # --- SBCL (m), A ---
    SbclDsblTestCase(
        test_id="SBCL_(m)_A_I1_NoBorrowIn_NoBorrowOut",
        instr_bytes=bytes([0x5D, 0x10]),  # SBCL (10), A
        init_memory_state={INTERNAL_MEMORY_START + 0x10: 0xAA},
        init_register_state={
            RegisterName.A: 0x55,
            RegisterName.I: 1,
            RegisterName.FC: 0,
        },
        expected_asm_str="SBCL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x55],  # 0xAA - 0x55 - 0 = 0x55
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="SBCL_(m)_A_I2_BorrowPropagate",
        instr_bytes=bytes([0x5D, 0x10]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x00,  # LSB m
            INTERNAL_MEMORY_START + 0x11: 0x30,  # MSB m -> m=0x3000
        },
        init_register_state={
            RegisterName.A: 0x01,  # A will be source for each byte
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },
        expected_asm_str="SBCL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # Byte 0 (LSB, m_addr 0x10): m[0x10]=0x00, A=0x01. 0x00 - 0x01 - 0 = 0xFF. mem[0x10]=0xFF, FC=1
        # Byte 1 (MSB, m_addr 0x11): m[0x11]=0x30, A=0x01. 0x30 - 0x01 - 1 = 0x2E. mem[0x11]=0x2E, FC=0
        expected_m_values_after=[0xFF, 0x2E],
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
]


@pytest.mark.parametrize(
    "tc", sbcl_test_cases, ids=[case.test_id for case in sbcl_test_cases]
)
def test_sbcl_instruction(tc: SbclDsblTestCase) -> None:
    cpu, raw_memory_array, _, logged_writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, tc.init_memory_state, tc.instr_bytes
    )

    for reg, val in tc.init_register_state.items():
        cpu.regs.set(reg, val)

    decoded_instr = cpu.decode_instruction(0x00)
    assert decoded_instr is not None, f"Test '{tc.test_id}': Failed to decode"
    actual_asm_str = asm_str(decoded_instr.render())
    assert (
        actual_asm_str == tc.expected_asm_str
    ), f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert (
            actual_val == expected_val
        ), f"Test '{tc.test_id}': Memory mismatch at offset {i} from LSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"

    expected_writes_to_m = [
        (tc.expected_m_addr_start + i, val)
        for i, val in enumerate(tc.expected_m_values_after)
    ]
    actual_writes_to_m = sorted(
        [
            w
            for w in logged_writes
            if tc.expected_m_addr_start
            <= w[0]
            < tc.expected_m_addr_start + len(tc.expected_m_values_after)
        ]
    )
    assert actual_writes_to_m == sorted(
        expected_writes_to_m
    ), f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"

    assert (
        cpu.regs.get(RegisterName.I) == tc.expected_I_after
    ), f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    assert (
        cpu.regs.get(RegisterName.FC) == tc.expected_FC_after
    ), f"Test '{tc.test_id}': Flag C (Borrow). Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    assert (
        cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after
    ), f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"


# DSBL Tests
# Opcode 0xD4: DSBL (m), (n)
# Opcode 0xD5: DSBL (m), A  -- Note: Readme has (n),A but consistent with DADL, (m) is first operand, thus dest
# DSBL is a reverse operation (addresses for (m) and (n) decrement)
dsbl_test_cases: List[SbclDsblTestCase] = [
    # --- DSBL (m), (n) ---
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I1_NoBorrowIn_SimpleBCD",
        instr_bytes=bytes(
            [0xD4, 0x10, 0x20]
        ),  # DSBL (10), (20) -> (m) ends at 0x10, (n) ends at 0x20
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x55,  # BCD 55 for (m)
            INTERNAL_MEMORY_START + 0x20: 0x22,  # BCD 22 for (n)
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},  # No borrow in
        expected_asm_str="DSBL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START
        + 0x10,  # MSB (and LSB in this case) address of (m)
        expected_m_values_after=[0x33],  # BCD 55 - BCD 22 = BCD 33
        expected_I_after=0,
        expected_FC_after=0,  # No borrow out
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I1_WithBorrowIn_SimpleBCD",
        instr_bytes=bytes([0xD4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x55,
            INTERNAL_MEMORY_START + 0x20: 0x22,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 1},  # Borrow In
        expected_asm_str="DSBL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x32],  # BCD 55 - BCD 22 - 1 = BCD 32
        expected_I_after=0,
        expected_FC_after=0,  # No borrow out
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I1_NoBorrowIn_LowNibbleBorrow",
        instr_bytes=bytes([0xD4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x23,  # BCD 23
            INTERNAL_MEMORY_START + 0x20: 0x05,  # BCD 05
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DSBL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x18],  # BCD 23 - BCD 05 = BCD 18
        expected_I_after=0,
        expected_FC_after=0,  # No overall borrow
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I1_NoBorrowIn_HighNibbleBorrow_OverallBorrow",
        instr_bytes=bytes([0xD4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x10,  # BCD 10
            INTERNAL_MEMORY_START + 0x20: 0x20,  # BCD 20
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DSBL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[
            0x90
        ],  # BCD 10 - BCD 20 = BCD -10 -> BCD 90 with borrow
        expected_I_after=0,
        expected_FC_after=1,  # Overall borrow
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I1_NoBorrowIn_ZeroResult",
        instr_bytes=bytes([0xD4, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x25,
            INTERNAL_MEMORY_START + 0x20: 0x25,
        },
        init_register_state={RegisterName.I: 1, RegisterName.FC: 0},
        expected_asm_str="DSBL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x00],  # BCD 25 - BCD 25 = BCD 00
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=1,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_(n)_I2_BCDBorrowPropagate",
        instr_bytes=bytes(
            [0xD4, 0x11, 0x21]
        ),  # DSBL (11), (21) -> m_end=0x11, n_end=0x21
        init_memory_state={
            INTERNAL_MEMORY_START + 0x11: 0x00,  # LSB of (m)
            INTERNAL_MEMORY_START + 0x10: 0x20,  # MSB of (m) -> (m) = BCD 2000
            INTERNAL_MEMORY_START + 0x21: 0x01,  # LSB of (n)
            INTERNAL_MEMORY_START + 0x20: 0x00,  # MSB of (n) -> (n) = BCD 0001
        },
        init_register_state={RegisterName.I: 2, RegisterName.FC: 0},
        expected_asm_str="DSBL  (BP+11), (BP+21)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # MSB addr of (m)
        # Byte 0 (LSB, m_addr 0x11, n_addr 0x21): m[0x11]=0x00, n[0x21]=0x01. BCD 00 - BCD 01 - 0 = BCD 99. mem[0x11]=0x99, FC_out=1 (borrow)
        # Byte 1 (MSB, m_addr 0x10, n_addr 0x20): m[0x10]=0x20, n[0x20]=0x00. BCD 20 - BCD 00 - 1(borrow_in) = BCD 19. mem[0x10]=0x19, FC_out=0
        expected_m_values_after=[0x19, 0x99],  # MSB, LSB for (m) area
        expected_I_after=0,
        expected_FC_after=0,  # From last byte op
        expected_FZ_after=0,  # Overall: (0x1999) != 0
    ),
    # --- DSBL (m), A ---
    SbclDsblTestCase(
        test_id="DSBL_(m)_A_I1_SimpleBCD",
        instr_bytes=bytes([0xD5, 0x10]),  # DSBL (10), A -> m_end=0x10
        init_memory_state={INTERNAL_MEMORY_START + 0x10: 0x78},  # BCD 78 for (m)
        init_register_state={
            RegisterName.A: 0x12,  # BCD 12 for A
            RegisterName.I: 1,
            RegisterName.FC: 0,
        },
        expected_asm_str="DSBL  (BP+10), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        expected_m_values_after=[0x66],  # BCD 78 - BCD 12 = BCD 66
        expected_I_after=0,
        expected_FC_after=0,
        expected_FZ_after=0,
    ),
    SbclDsblTestCase(
        test_id="DSBL_(m)_A_I2_BCDBorrowPropagate",
        instr_bytes=bytes([0xD5, 0x11]),  # DSBL (11), A -> m_end=0x11
        init_memory_state={
            INTERNAL_MEMORY_START + 0x11: 0x00,  # LSB of (m)
            INTERNAL_MEMORY_START + 0x10: 0x01,  # MSB of (m) -> (m) = BCD 0100
        },
        init_register_state={
            RegisterName.A: 0x01,  # BCD 01 for A (used for each byte)
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },
        expected_asm_str="DSBL  (BP+11), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # MSB addr of (m)
        # Byte 0 (LSB, m_addr 0x11): m[0x11]=0x00, A=0x01. BCD 00 - BCD 01 - 0 = BCD 99. mem[0x11]=0x99, FC_out=1
        # Byte 1 (MSB, m_addr 0x10): m[0x10]=0x01, A=0x01. BCD 01 - BCD 01 - 1 = BCD 99. mem[0x10]=0x99, FC_out=1
        expected_m_values_after=[0x99, 0x99],  # MSB, LSB
        expected_I_after=0,
        expected_FC_after=1,  # Borrow from last op
        expected_FZ_after=0,
    ),
]


@pytest.mark.parametrize(
    "tc", dsbl_test_cases, ids=[case.test_id for case in dsbl_test_cases]
)
def test_dsbl_instruction(tc: SbclDsblTestCase) -> None:
    cpu, raw_memory_array, _, logged_writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, tc.init_memory_state, tc.instr_bytes
    )

    for reg, val in tc.init_register_state.items():
        cpu.regs.set(reg, val)

    decoded_instr = cpu.decode_instruction(0x00)
    assert decoded_instr is not None, f"Test '{tc.test_id}': Failed to decode"
    actual_asm_str = asm_str(decoded_instr.render())
    assert (
        actual_asm_str == tc.expected_asm_str
    ), f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    # DSBL processes memory in reverse, (m) operand in instruction is the LSB/end address.
    # tc.expected_m_addr_start is the MSB address for verification.
    # tc.expected_m_values_after is [MSB_val, ..., LSB_val].
    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[
            tc.expected_m_addr_start + i
        ]  # Iterates from MSB_addr
        assert (
            actual_val == expected_val
        ), f"Test '{tc.test_id}': Memory mismatch at offset {i} from MSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"

    # Verify logged writes. Addresses are written from LSB_addr_m downwards.
    num_bytes = len(tc.expected_m_values_after)
    # The (m) operand in the instruction is the LSB address.
    # For DSBL (11), (21), (m) operand is 0x11.
    # expected_m_addr_start is the MSB start for verification (e.g. 0x10 if I=2, m_operand=0x11)
    # So, LSB addr of (m) = expected_m_addr_start + num_bytes - 1
    lsb_address_m = tc.expected_m_addr_start + num_bytes - 1

    expected_writes_to_m = []
    for i in range(num_bytes):
        # Address iterates from lsb_address_m down to msb_address_m
        addr = lsb_address_m - i
        # Values in tc.expected_m_values_after are [MSB, ..., LSB]
        # So, for addr = lsb_address_m, we need value at index num_bytes - 1 - i
        val_idx_in_expected = num_bytes - 1 - i
        expected_writes_to_m.append(
            (addr, tc.expected_m_values_after[val_idx_in_expected])
        )

    actual_writes_to_m = sorted(
        [
            w
            for w in logged_writes
            if tc.expected_m_addr_start <= w[0] < tc.expected_m_addr_start + num_bytes
        ]
    )  # Filter and sort actual writes

    # Sort expected_writes_to_m because the order of processing is LSB->MSB for writes,
    # but comparison should be order-agnostic if elements are correct or sorted if order matters.
    # Here, we sort both to compare contents.
    assert actual_writes_to_m == sorted(
        expected_writes_to_m
    ), f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected (sorted): {sorted(expected_writes_to_m)}\nGot (sorted): {actual_writes_to_m}"

    assert (
        cpu.regs.get(RegisterName.I) == tc.expected_I_after
    ), f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    assert (
        cpu.regs.get(RegisterName.FC) == tc.expected_FC_after
    ), f"Test '{tc.test_id}': Flag C (Borrow). Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    assert (
        cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after
    ), f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"


# Add new NamedTuple for DSLL/DSRL test cases
class DsrlDsllTestCase(NamedTuple):
    test_id: str
    is_dsll: bool  # True for DSLL, False for DSRL
    instr_operand_n_val: int  # The 8-bit value for (n) in the instruction
    loop_count_I: int
    # For DSLL: [MSB_val, MSB-1_val, ..., LSB_val] e.g. BCD 1234 -> [0x12, 0x34]
    # For DSRL: [LSB_val, LSB+1_val, ..., MSB_val] e.g. BCD 1234 -> [0x34, 0x12]
    initial_bcd_logical_bytes: List[int]
    expected_final_bcd_logical_bytes: List[
        int
    ]  # Same order as initial_bcd_logical_bytes
    expected_FZ_after: int
    # FC is not affected by these instructions according to the book


# Helper functions to compute expected results for DSLL/DSRL
def compute_expected_dsll(logical_bcd_bytes: List[int]) -> List[int]:
    """
    Computes the result of DSLL operation on BCD bytes.
    logical_bcd_bytes is [MSB_val, MSB-1_val, ..., LSB_val].
    e.g., for BCD 123456, input is [0x12, 0x34, 0x56].
    Result for 123456 -> 234560 is [0x23, 0x45, 0x60].
    """
    if not logical_bcd_bytes:
        return []

    count = len(logical_bcd_bytes)
    shifted_bytes = [0] * count

    # u carries the LOW nibble of the previous (more significant) byte
    # to become the low nibble of the current (less significant) byte.
    # For the most significant byte, there's no "previous" byte, so u starts as 0.
    u_carry_from_prev_low_nibble = 0

    # Iterate from MSB to LSB (index 0 to count-1)
    for i in range(count):
        old_current_byte_val = logical_bcd_bytes[i]
        old_current_low_nibble = old_current_byte_val & 0x0F

        # New byte's high nibble is the old_current_byte's low nibble.
        # New byte's low nibble is u (which was the LOW nibble of the previous byte).
        shifted_bytes[i] = (old_current_low_nibble << 4) | u_carry_from_prev_low_nibble

        # Update u for the next iteration using the low nibble of the current byte
        u_carry_from_prev_low_nibble = old_current_low_nibble

    return shifted_bytes


def compute_expected_dsrl(logical_bcd_bytes: List[int]) -> List[int]:
    """
    Computes the result of DSRL operation on BCD bytes.
    logical_bcd_bytes is [LSB_val, LSB+1_val, ..., MSB_val].
    e.g., for BCD 123456, input is [0x56, 0x34, 0x12].
    Result for 123456 -> 012345 is [0x45, 0x23, 0x01].
    """
    if not logical_bcd_bytes:
        return []

    count = len(logical_bcd_bytes)
    shifted_bytes = [0] * count

    # u carries the HIGH nibble of the previous (less significant) byte
    # to become the high nibble of the current (more significant) byte.
    # For the least significant byte, there's no "previous" byte, so u starts as 0.
    u_carry_from_prev_high_nibble = 0

    # Iterate from LSB to MSB (index 0 to count-1)
    for i in range(count):
        old_current_byte_val = logical_bcd_bytes[i]
        old_current_high_nibble = (old_current_byte_val >> 4) & 0x0F

        # New byte's low nibble is the old_current_byte's high nibble.
        # New byte's high nibble is u (which was the HIGH nibble of the previous byte).
        shifted_bytes[i] = old_current_high_nibble | (
            u_carry_from_prev_high_nibble << 4
        )

        # Update u for the next iteration using the high nibble of the current byte
        u_carry_from_prev_high_nibble = old_current_high_nibble

    return shifted_bytes


dsrl_dsll_test_cases: List[DsrlDsllTestCase] = [
    # --- DSLL Test Cases ---
    DsrlDsllTestCase(
        test_id="DSLL_I1_Simple",  # BCD 12 -> 20
        is_dsll=True,
        instr_operand_n_val=0x10,
        loop_count_I=1,
        initial_bcd_logical_bytes=[0x12],  # [MSB]
        expected_final_bcd_logical_bytes=compute_expected_dsll([0x12]),  # [0x20]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSLL_I2_1234_to_2340",  # BCD 1234 -> 2340
        is_dsll=True,
        instr_operand_n_val=0x11,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x12, 0x34],  # [MSB, LSB]
        expected_final_bcd_logical_bytes=compute_expected_dsll(
            [0x12, 0x34]
        ),  # [0x23, 0x40]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSLL_I3_123456_to_234560",
        is_dsll=True,
        instr_operand_n_val=0x12,
        loop_count_I=3,
        initial_bcd_logical_bytes=[0x12, 0x34, 0x56],  # [MSB, Mid, LSB]
        expected_final_bcd_logical_bytes=compute_expected_dsll(
            [0x12, 0x34, 0x56]
        ),  # [0x23, 0x45, 0x60]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSLL_I2_0009_to_0090",
        is_dsll=True,
        instr_operand_n_val=0x11,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x00, 0x09],
        expected_final_bcd_logical_bytes=compute_expected_dsll(
            [0x00, 0x09]
        ),  # [0x00, 0x90]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSLL_I2_0000_to_0000_FZ1",
        is_dsll=True,
        instr_operand_n_val=0x11,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x00, 0x00],
        expected_final_bcd_logical_bytes=compute_expected_dsll(
            [0x00, 0x00]
        ),  # [0x00, 0x00]
        expected_FZ_after=1,
    ),
    # --- DSRL Test Cases ---
    DsrlDsllTestCase(
        test_id="DSRL_I1_Simple",  # BCD 12 -> 01
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=1,
        initial_bcd_logical_bytes=[0x12],  # [LSB] (also MSB here)
        expected_final_bcd_logical_bytes=compute_expected_dsrl([0x12]),  # [0x01]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I2_1234_to_0123",  # BCD 1234 -> 0123
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x34, 0x12],  # [LSB, MSB]
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x34, 0x12]
        ),  # [0x23, 0x01] (means LSB=23, MSB=01)
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I3_123456_to_012345",
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=3,
        initial_bcd_logical_bytes=[0x56, 0x34, 0x12],  # [LSB, Mid, MSB]
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x56, 0x34, 0x12]
        ),  # [0x45, 0x23, 0x01]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I2_9000_to_0900",
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x00, 0x90],  # LSB=0x00, MSB=0x90
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x00, 0x90]
        ),  # [0x00, 0x09] (LSB=00, MSB=09)
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I2_0000_to_0000_FZ1",
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x00, 0x00],
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x00, 0x00]
        ),  # [0x00, 0x00]
        expected_FZ_after=1,
    ),
]


@pytest.mark.parametrize(
    "tc", dsrl_dsll_test_cases, ids=[case.test_id for case in dsrl_dsll_test_cases]
)
def test_dsrl_dsll_instruction(tc: DsrlDsllTestCase) -> None:
    opcode = 0xEC if tc.is_dsll else 0xFC
    instr_bytes = bytes([opcode, tc.instr_operand_n_val])

    init_memory_state: Dict[int, int] = {}
    # Determine memory addresses for initial setup
    # For DSLL, n is MSB addr, mem is populated from n downwards (n, n-1, ..., n-I+1)
    # initial_bcd_logical_bytes is [MSB_val, MSB-1_val, ..., LSB_val]
    # So, initial_bcd_logical_bytes[i] goes to mem[INTERNAL_MEMORY_START + tc.instr_operand_n_val - i]
    if tc.is_dsll:
        for i in range(tc.loop_count_I):
            addr = INTERNAL_MEMORY_START + tc.instr_operand_n_val - i
            init_memory_state[addr] = tc.initial_bcd_logical_bytes[i]
    else:  # DSRL
        # For DSRL, n is LSB addr, mem is populated from n upwards (n, n+1, ..., n+I-1)
        # initial_bcd_logical_bytes is [LSB_val, LSB+1_val, ..., MSB_val]
        # So, initial_bcd_logical_bytes[i] goes to mem[INTERNAL_MEMORY_START + tc.instr_operand_n_val + i]
        for i in range(tc.loop_count_I):
            addr = INTERNAL_MEMORY_START + tc.instr_operand_n_val + i
            init_memory_state[addr] = tc.initial_bcd_logical_bytes[i]

    init_register_state = {RegisterName.I: tc.loop_count_I}

    cpu, raw_memory_array, _, _ = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, init_memory_state, instr_bytes
    )

    for reg, val in init_register_state.items():
        cpu.regs.set(reg, val)

    # Preserve initial FC for verification as it should not change
    initial_fc = cpu.regs.get(RegisterName.FC)

    # --- Decode and Verify Assembly ---
    # Note: This relies on OPCODES dict in instr.py having IMem8 for DSLL/DSRL.
    # If it's IMem20, this part of test might fail or look weird, but execution test is main goal.
    decoded_instr = cpu.decode_instruction(0x00)
    assert (
        decoded_instr is not None
    ), f"Test '{tc.test_id}': Failed to decode instruction"

    expected_mnemonic = "DSLL " if tc.is_dsll else "DSRL "
    # Assuming IMem8 is rendered as (BP+XX)
    expected_asm_str = f"{expected_mnemonic:6s}(BP+{tc.instr_operand_n_val:02X})"
    actual_asm_str = asm_str(decoded_instr.render())

    assert (
        actual_asm_str == expected_asm_str
    ), f"Test '{tc.test_id}': ASM string mismatch.\n  Expected: '{expected_asm_str}'\n  Actual  : '{actual_asm_str}'"

    # --- Execute ---
    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    # --- Verify Registers ---
    assert (
        cpu.regs.get(RegisterName.I) == 0
    ), f"Test '{tc.test_id}': Reg I. Expected 0, Got {cpu.regs.get(RegisterName.I)}"
    assert (
        cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after
    ), f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    assert (
        cpu.regs.get(RegisterName.FC) == initial_fc
    ), f"Test '{tc.test_id}': Flag C should not change. Initial {initial_fc}, Got {cpu.regs.get(RegisterName.FC)}"

    # --- Verify Memory ---
    # For DSLL, tc.expected_final_bcd_logical_bytes is [MSB_val, MSB-1_val, ..., LSB_val]
    # Check mem[n], mem[n-1], ...
    if tc.is_dsll:
        for i in range(tc.loop_count_I):
            addr_in_mem = INTERNAL_MEMORY_START + tc.instr_operand_n_val - i
            actual_val = raw_memory_array[addr_in_mem]
            expected_val = tc.expected_final_bcd_logical_bytes[i]
            assert (
                actual_val == expected_val
            ), f"Test '{tc.test_id}': Memory mismatch at addr 0x{addr_in_mem:X} (logical byte {i}). Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
    else:  # DSRL
        # For DSRL, tc.expected_final_bcd_logical_bytes is [LSB_val, LSB+1_val, ..., MSB_val]
        # Check mem[n], mem[n+1], ...
        for i in range(tc.loop_count_I):
            addr_in_mem = INTERNAL_MEMORY_START + tc.instr_operand_n_val + i
            actual_val = raw_memory_array[addr_in_mem]
            expected_val = tc.expected_final_bcd_logical_bytes[i]
            assert (
                actual_val == expected_val
            ), f"Test '{tc.test_id}': Memory mismatch at addr 0x{addr_in_mem:X} (logical byte {i}). Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"


def test_decode_all_opcodes() -> None:
    raw_memory = bytearray([0x00] * ADDRESS_SPACE_SIZE)

    # enumerate all opcodes, want index for each opcode
    for i, (b, s) in enumerate(opcode_generator()):
        if b is None:
            continue

        for j, byte in enumerate(b):
            raw_memory[j] = byte

        def read_mem(addr: int) -> int:
            # if addr < 0 or addr >= len(raw_memory):
            #     raise IndexError(f"Address out of bounds: {addr:04x}")
            return raw_memory[addr]

        def write_mem(addr: int, value: int) -> None:
            # if addr < 0 or addr >= len(raw_memory):
            #     raise IndexError(f"Address out of bounds: {addr:04x}")
            raw_memory[addr] = value

        skip = False
        # FIXME: need to ensure they're covered by specific tests that set up
        # the memory and registers properly.
        # MVL: done
        # ADCL, DADL: done
        # SBCL, DSBL: done
        ignore_instructions = [
            "???",
            "MVL",
            "ADCL",
            "DADL",
            "SBCL",
            "DSBL",
            "DSRL",
            "DSLL",
        ]
        for ignore in ignore_instructions:
            if s and s.startswith(ignore):
                skip = True
                break
        if skip:
            continue

        memory = Memory(read_mem, write_mem)
        cpu = Emulator(memory)

        address = 0x00
        cpu.regs.set(RegisterName.S, 0x1000)  # Set stack pointer to a valid location
        cpu.regs.set(RegisterName.U, 0x2000)  # Set stack pointer to a valid location

        cpu.regs.set(RegisterName.X, 0x10)

        try:
            _ = cpu.execute_instruction(address)
        except Exception as e:
            debug_instruction(cpu, address)
            raise ValueError(f"Failed to evaluate {s} at line {i+1}") from e
