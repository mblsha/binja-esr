from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from .emulator import (
    Registers,
    RegisterName,
    Emulator,
)
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START, PC_MASK
from .instr.opcodes import IMEMRegisters
from .instr import InvalidInstruction
from binja_test_mocks.eval_llil import Memory
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
            addr = imem((addr - INTERNAL_MEMORY_START) & 0xFF)
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:04x} out of bounds")
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        # print(f"Writing {value:02x} to address {addr:04x}") # Uncomment for debugging
        if addr >= INTERNAL_MEMORY_START:
            addr = imem((addr - INTERNAL_MEMORY_START) & 0xFF)
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:04x} out of bounds")
        raw[addr] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda addr, _pc=None: raw[addr],
    )
    cpu = Emulator(memory, reset_on_init=False)
    return cpu, raw, reads, writes


def imem(offset: int) -> int:
    """Return absolute address for an internal memory offset."""
    return INTERNAL_MEMORY_START + offset


def debug_instruction(cpu: Emulator, address: int) -> None:
    il = MockLowLevelILFunction()
    instr = cpu.decode_instruction(address)
    assert instr is not None, f"Failed to decode instruction at {address:04x}"
    instr.lift(il, address)

    rendered = asm_str(instr.render())

    print(f"Decoded instruction at {address:04x}: {rendered}")
    for llil in il.ils:
        print(f"  {llil}")


@pytest.mark.parametrize("opcode", [0x20, 0xBF])
def test_reserved_opcodes_fail_closed(opcode: int) -> None:
    cpu, _raw, reads, writes = _make_cpu_and_mem(0x100, {}, bytes([opcode]))
    cpu.regs.set(RegisterName.PC, 0)

    with pytest.raises(InvalidInstruction, match=f"0x{opcode:02X}"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert reads == [0]
    assert writes == []


def test_instruction_opcode_is_fetched_once() -> None:
    cpu, _raw, reads, writes = _make_cpu_and_mem(0x100, {}, bytes([0x00]))

    cpu.execute_instruction(0)

    assert reads == [0]
    assert writes == []


def test_tcl_fails_closed_until_timer_clear_is_modeled() -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(0x100, {}, bytes([0xCE]))
    cpu.regs.set(RegisterName.PC, 0)

    with pytest.raises(NotImplementedError, match="timer-phase-clear memory hook"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert writes == []


@dataclass
class InstructionTestCase:
    """A structured container for a single instruction test case.

    Unless a test explicitly specifies an expected program counter in
    ``expected_regs``, the runner will verify that the PC advances by the
    length of the instruction bytes. This helps catch cases where the emulator
    fails to move to the next instruction.
    """

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
        test_id="MV_BA_to_emem_reg_X_40000",
        instr_bytes=bytes.fromhex("B204"),
        init_regs={
            RegisterName.X: 0x40000,  # X points to address 0x40000
            RegisterName.BA: 0xAA55,  # BA contains 0xAA55
        },
        expected_mem_writes=[
            (0x40000, 0x55),  # Low byte of BA (little-endian)
            (0x40001, 0xAA),  # High byte of BA
        ],
        expected_mem_state={
            0x40000: 0x55,  # Low byte written first
            0x40001: 0xAA,  # High byte written second
        },
        expected_regs={
            RegisterName.X: 0x40000,  # X should remain unchanged (SIMPLE mode)
            RegisterName.BA: 0xAA55,  # BA should remain unchanged
        },
        expected_asm_str="MV    [X], BA",
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
        expected_mem_writes=[
            (0x2000, 0x34),
            (0x2001, 0x12),
        ],  # Write BA at Y, then increment
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
        init_regs={RegisterName.U: 0x8000, RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_mem_writes=[(0x7FFF, 0x03)],
        expected_mem_state={0x7FFF: 0x03},
        expected_regs={
            RegisterName.U: 0x7FFF,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        expected_asm_str="PUSHU F",
    ),
    InstructionTestCase(
        test_id="IR_saves_opcode_address_for_ROM_software_dispatch",
        instr_bytes=bytes.fromhex("FE"),
        init_regs={
            RegisterName.S: 0x3000,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        init_mem={
            imem(IMEMRegisters.IMR): 0xA5,
            0xFFFFA: 0x78,
            0xFFFFB: 0x56,
            0xFFFFC: 0x04,
        },
        expected_regs={
            RegisterName.S: 0x2FFB,
            RegisterName.PC: 0x45678,
            RegisterName.FC: 1,
            RegisterName.FZ: 1,
        },
        expected_mem_state={
            0x2FFB: 0xA5,  # saved IMR
            0x2FFC: 0x03,  # saved F (C | Z)
            0x2FFD: 0x45,  # saved IR opcode address, little-endian
            0x2FFE: 0x23,
            0x2FFF: 0x01,
            imem(IMEMRegisters.IMR): 0x25,  # IRM cleared on entry
        },
        initial_pc=0x12345,
        expected_asm_str="IR",
    ),
    # Test MV IL instruction - verify high byte is cleared
    InstructionTestCase(
        test_id="MV_IL_clears_high_byte",
        instr_bytes=bytes([0x09, 0x09]),  # MV IL, 09
        init_regs={RegisterName.I: 0xAABB},  # Initialize I to 0xAABB
        expected_regs={
            RegisterName.IL: 0x09,  # IL should be 0x09
            RegisterName.IH: 0x00,  # IH should be cleared to 0x00
            RegisterName.I: 0x0009,  # Full I register should be 0x0009
        },
        expected_asm_str="MV    IL, 09",
    ),
    # Test MV I instruction - verify all bytes are set
    InstructionTestCase(
        test_id="MV_I_preserves_all_bytes",
        instr_bytes=bytes([0x0B, 0x34, 0x12]),  # MV I, 1234 (0x0B is opcode for MV I)
        init_regs={RegisterName.I: 0xAABB},  # Initialize I to 0xAABB
        expected_regs={
            RegisterName.I: 0x1234,  # Full I register set to 0x1234
            RegisterName.IL: 0x34,  # IL is low byte
            RegisterName.IH: 0x12,  # IH is high byte
        },
        expected_asm_str="MV    I, 1234",
    ),
    InstructionTestCase(
        test_id="MV_external_negative_offset_wraps_on_20bit_bus",
        instr_bytes=bytes.fromhex("90C4AB"),  # MV A,[X-AB]
        init_regs={RegisterName.X: 0},
        init_mem={0xFFF55: 0x5A},
        expected_regs={RegisterName.A: 0x5A, RegisterName.X: 0},
        expected_asm_str="MV    A, [X-AB]",
    ),
    InstructionTestCase(
        test_id="MV_indirect_pointer_ignores_loaded_upper_nibble",
        instr_bytes=bytes.fromhex("30980000"),  # PRE (n), MV A,[(00)]
        init_mem={
            imem(0x00): 0x23,
            imem(0x01): 0x01,
            imem(0x02): 0xF8,  # raw pointer F80123 -> 20-bit 80123
            0x80123: 0x6B,
        },
        expected_regs={RegisterName.A: 0x6B},
        expected_asm_str="MV    A, [(00)]",
    ),
    InstructionTestCase(
        test_id="MVL_external_postincrement_wraps_at_20bit_boundary",
        instr_bytes=bytes.fromhex("E32420"),  # MVL (20),[X++]
        init_regs={RegisterName.I: 2, RegisterName.X: 0xFFFFF},
        init_mem={
            0xFFFFF: 0x11,
            0x00000: 0x22,
            imem(0x00): 0x99,  # catches an unmasked spill into synthetic IMEM
        },
        expected_regs={RegisterName.I: 0, RegisterName.X: 0x00001},
        expected_mem_state={imem(0x20): 0x11, imem(0x21): 0x22},
        initial_pc=0x100,
        expected_asm_str="MVL   (BP+20), [X++]",
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
    InstructionTestCase(
        test_id="ADD_A_imm_advances_pc",
        instr_bytes=bytes.fromhex("4001"),
        init_regs={RegisterName.A: 0x00},
        expected_regs={RegisterName.A: 0x01, RegisterName.FZ: 0, RegisterName.FC: 0},
        initial_pc=0x1000,
        expected_asm_str="ADD   A, 01",
    ),
    InstructionTestCase(
        test_id="ADD_I_A_zero_extends_ROM_mixed_width_source",
        instr_bytes=bytes.fromhex("4430"),
        init_regs={RegisterName.I: 0xFFFE, RegisterName.A: 0x03},
        expected_regs={RegisterName.I: 0x0001, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="ADD   I, A",
    ),
    InstructionTestCase(
        test_id="ADD_Y_BA_zero_extends_ROM_mixed_width_source",
        instr_bytes=bytes.fromhex("4552"),
        init_regs={RegisterName.Y: 0xFFFF0, RegisterName.BA: 0x0020},
        expected_regs={RegisterName.Y: 0x00010, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="ADD   Y, BA",
    ),
    InstructionTestCase(
        test_id="ADD_X_A_zero_extends_byte_source_and_sets_zero",
        instr_bytes=bytes.fromhex("4540"),
        init_regs={RegisterName.X: 0xFFFFF, RegisterName.A: 0x01},
        expected_regs={RegisterName.X: 0x00000, RegisterName.FZ: 1, RegisterName.FC: 1},
        expected_asm_str="ADD   X, A",
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
    InstructionTestCase(
        test_id="SUB_I_A_zero_extends_ROM_mixed_width_source",
        instr_bytes=bytes.fromhex("4C30"),
        init_regs={RegisterName.I: 0x0100, RegisterName.A: 0x01},
        expected_regs={RegisterName.I: 0x00FF, RegisterName.FZ: 0, RegisterName.FC: 0},
        expected_asm_str="SUB   I, A",
    ),
    InstructionTestCase(
        test_id="SUB_U_A_zero_extends_byte_source",
        instr_bytes=bytes.fromhex("4D60"),
        init_regs={RegisterName.U: 0x00000, RegisterName.A: 0x01},
        expected_regs={RegisterName.U: 0xFFFFF, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SUB   U, A",
    ),
    InstructionTestCase(
        test_id="SUB_A_IL_same_width_register_pair",
        instr_bytes=bytes.fromhex("4E01"),
        init_regs={RegisterName.A: 0x00, RegisterName.IL: 0x01},
        expected_regs={RegisterName.A: 0xFF, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SUB   A, IL",
    ),
    InstructionTestCase(
        test_id="ADC_rhs_plus_carry_wrap_preserves_carry_out",
        instr_bytes=bytes.fromhex("50FF"),
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x00, RegisterName.FZ: 1, RegisterName.FC: 1},
        expected_asm_str="ADC   A, FF",
    ),
    InstructionTestCase(
        test_id="SBC_rhs_plus_borrow_wrap_preserves_borrow_out",
        instr_bytes=bytes.fromhex("58FF"),
        init_regs={RegisterName.A: 0x80, RegisterName.FC: 1},
        expected_regs={RegisterName.A: 0x80, RegisterName.FZ: 0, RegisterName.FC: 1},
        expected_asm_str="SBC   A, FF",
    ),
    InstructionTestCase(
        test_id="PMDF_ROM_style_binary_adjust_with_model_flag_preservation",
        instr_bytes=bytes.fromhex("3047ECF5"),  # PMDF (BP), -0x0B
        init_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        init_mem={imem(IMEMRegisters.BP): 0x20},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_mem_state={imem(IMEMRegisters.BP): 0x15},
        expected_asm_str="PMDF  (BP), F5",
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
        test_id="EXL_exchanges_entire_I_byte_blocks",
        instr_bytes=bytes.fromhex("C31020"),  # EXL (10), (20)
        init_regs={RegisterName.I: 3},
        init_mem={
            imem(0x10): 0x11,
            imem(0x11): 0x22,
            imem(0x12): 0x33,
            imem(0x20): 0xAA,
            imem(0x21): 0xBB,
            imem(0x22): 0xCC,
        },
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            imem(0x10): 0xAA,
            imem(0x11): 0xBB,
            imem(0x12): 0xCC,
            imem(0x20): 0x11,
            imem(0x21): 0x22,
            imem(0x22): 0x33,
        },
        expected_asm_str="EXL   (BP+10), (BP+20)",
    ),
    InstructionTestCase(
        test_id="EXL_internal_addresses_wrap_at_FF",
        instr_bytes=bytes.fromhex("C3FE20"),  # EXL (FE), (20)
        init_regs={RegisterName.I: 3},
        init_mem={
            imem(0xFE): 0x11,
            imem(0xFF): 0x22,
            imem(0x00): 0x33,
            imem(0x20): 0xAA,
            imem(0x21): 0xBB,
            imem(0x22): 0xCC,
        },
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            imem(0xFE): 0xAA,
            imem(0xFF): 0xBB,
            imem(0x00): 0xCC,
            imem(0x20): 0x11,
            imem(0x21): 0x22,
            imem(0x22): 0x33,
        },
        expected_asm_str="EXL   (BP+FE), (BP+20)",
    ),
    InstructionTestCase(
        test_id="MVL_imem_overlap_fwd_clobber",
        instr_bytes=bytes.fromhex("CB5150"),  # MVL (51), (50)
        init_regs={RegisterName.I: 3},
        init_mem={
            imem(0x50): 0xAA,
            imem(0x51): 0xBB,
            imem(0x52): 0xCC,
        },
        # A naive forward copy clobbers the source.
        # Expected: mem[51]=mem[50]=AA; mem[52]=mem[51]=AA; mem[53]=mem[52]=AA
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            imem(0x50): 0xAA,
            imem(0x51): 0xAA,
            imem(0x52): 0xAA,
            imem(0x53): 0xAA,
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
            imem(0x50): 0xAA,
            imem(0x4F): 0xBB,
            imem(0x4E): 0xCC,
        },
        # A backward copy handles this overlap correctly.
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            imem(0x51): 0xAA,
            imem(0x50): 0xBB,
            imem(0x4F): 0xCC,
        },
        expected_asm_str="MVLD  (BP+51), (BP+50)",
    ),
    InstructionTestCase(
        test_id="MVL_imem_to_imem_wrap_around",
        instr_bytes=bytes.fromhex("CBFEF0"),  # MVL (FE), (F0)
        init_regs={RegisterName.I: 4},
        init_mem={
            imem(0xF0): 0x11,
            imem(0xF1): 0x22,
            imem(0xF2): 0x33,
            imem(0xF3): 0x44,
        },
        expected_regs={RegisterName.I: 0},
        expected_mem_state={
            imem(0xFE): 0x11,
            imem(0xFF): 0x22,
            imem(0x00): 0x33,
            imem(0x01): 0x44,
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
            # The source pre-decrements, but the internal destination advances.
            imem(0x52): 0xEF,
            imem(0x53): 0xBE,
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
            # Only the external source pre-decrements; the internal destination
            # still advances as documented for MVL (n),[--r3].
            imem(0x02): 0x44,  # First byte from 0x1FFF
            imem(0x03): 0x33,  # Second byte from 0x1FFE
            imem(0x04): 0x22,  # Third byte from 0x1FFD
            imem(0x05): 0x11,  # Fourth byte from 0x1FFC
            imem(0x06): 0x55,  # Fifth byte from 0x1FFB
        },
        expected_asm_str="MVL   (BP+02), [--X]",
    ),
    InstructionTestCase(
        test_id="MVL_(FE)_(50)_I5_wrap",
        instr_bytes=bytes.fromhex(
            "CBFE50"
        ),  # MVL (FE), (50) - no PRE, both use BP_N by default
        init_regs={
            RegisterName.I: 5,
        },
        init_mem={
            # Source data at internal memory starting at BP+50
            imem(0x50): 0xAA,
            imem(0x51): 0xBB,
            imem(0x52): 0xCC,
            imem(0x53): 0xDD,
            imem(0x54): 0xEE,
        },
        expected_regs={
            RegisterName.I: 0,
        },
        expected_mem_state={
            # MVL copies from (BP+50) to (BP+FE) with incrementing addresses
            # Destination addresses: 0xFE, 0xFF, 0x00 (wrapped), 0x01, 0x02
            imem(0xFE): 0xAA,  # From BP+50
            imem(0xFF): 0xBB,  # From BP+51
            imem(0x00): 0xCC,  # From BP+52 (wrapped from 0x100)
            imem(0x01): 0xDD,  # From BP+53 (wrapped from 0x101)
            imem(0x02): 0xEE,  # From BP+54 (wrapped from 0x102)
            # Source data remains unchanged
            imem(0x50): 0xAA,
            imem(0x51): 0xBB,
            imem(0x52): 0xCC,
            imem(0x53): 0xDD,
            imem(0x54): 0xEE,
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
            # Only the external source pre-decrements; the internal destination
            # advances from BP+00.
            imem(0x02): 0x11,  # From 0x1FFF
            imem(0x03): 0x22,  # From 0x1FFE
            imem(0x04): 0x33,  # From 0x1FFD
            imem(0x05): 0x44,  # From 0x1FFC
            imem(0x06): 0x55,  # From 0x1FFB
            # BP remains unchanged
            imem(IMEMRegisters.BP): 0x02,
        },
        expected_asm_str="MVL   (BP+00), [--X]",
    ),
    InstructionTestCase(
        test_id="MVL_predec_destination_moves_downward_only",
        instr_bytes=bytes.fromhex("EB3720"),  # MVL [--S], (20)
        init_regs={RegisterName.I: 3, RegisterName.S: 0x1003},
        init_mem={
            imem(0x20): 0x11,
            imem(0x21): 0x22,
            imem(0x22): 0x33,
        },
        expected_regs={RegisterName.I: 0, RegisterName.S: 0x1000},
        expected_mem_writes=[
            (0x1002, 0x11),
            (0x1001, 0x22),
            (0x1000, 0x33),
        ],
        expected_mem_state={
            0x1002: 0x11,
            0x1001: 0x22,
            0x1000: 0x33,
        },
        expected_asm_str="MVL   [--S], (BP+20)",
    ),
    InstructionTestCase(
        test_id="MVL_(00)_(50)_BP_FE_I3",
        instr_bytes=bytes.fromhex("CB0050"),  # MVL (00), (50) with BP=0xFE
        init_regs={
            RegisterName.I: 3,
        },
        init_mem={
            # BP register at internal memory
            imem(IMEMRegisters.BP): 0xFE,  # BP = 0xFE
            # Source data at internal memory (BP+50)
            # With BP=0xFE, (BP+50) = 0xFE + 0x50 = 0x14E, wrapped to 0x4E
            imem(0x4E): 0xAA,
            imem(0x4F): 0xBB,
            imem(0x50): 0xCC,
        },
        expected_regs={
            RegisterName.I: 0,
        },
        expected_mem_state={
            # BP=0xFE, so (BP+00) = address 0xFE
            # MVL copies from (BP+50) to (BP+00) with incrementing addresses
            # Source: 0x4E, 0x4F, 0x50
            # Destination: 0xFE, 0xFF, 0x00 (wrapped)
            imem(0xFE): 0xAA,  # From BP+50 (0x4E)
            imem(0xFF): 0xBB,  # From BP+51 (0x4F)
            imem(0x00): 0xCC,  # From BP+52 (0x50), wrapped from 0x100
            # Source and BP remain unchanged
            imem(0x4E): 0xAA,
            imem(0x4F): 0xBB,
            imem(0x50): 0xCC,
            imem(IMEMRegisters.BP): 0xFE,
        },
        expected_asm_str="MVL   (BP+00), (BP+50)",
    ),
    InstructionTestCase(
        test_id="ADCL_internal_addresses_wrap_at_FF",
        instr_bytes=bytes.fromhex("54FF10"),
        init_regs={RegisterName.I: 2, RegisterName.FC: 0},
        init_mem={
            imem(0xFF): 0x01,
            imem(0x00): 0x02,
            imem(0x10): 0x10,
            imem(0x11): 0x20,
        },
        expected_regs={RegisterName.I: 0, RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_mem_writes=[
            (imem(0xFF), 0x11),
            (imem(0x00), 0x22),
        ],
        expected_mem_state={imem(0xFF): 0x11, imem(0x00): 0x22},
        expected_asm_str="ADCL  (BP+FF), (BP+10)",
    ),
    InstructionTestCase(
        test_id="DADL_internal_addresses_wrap_below_00",
        instr_bytes=bytes.fromhex("C40010"),
        init_regs={RegisterName.I: 2, RegisterName.FC: 0},
        init_mem={
            imem(0x00): 0x99,
            imem(0xFF): 0x00,
            imem(0x10): 0x01,
            imem(0x0F): 0x00,
        },
        expected_regs={RegisterName.I: 0, RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_mem_writes=[
            (imem(0x00), 0x00),
            (imem(0xFF), 0x01),
        ],
        expected_mem_state={imem(0x00): 0x00, imem(0xFF): 0x01},
        expected_asm_str="DADL  (BP+00), (BP+10)",
    ),
    InstructionTestCase(
        test_id="DSLL_internal_address_wraps_below_00",
        instr_bytes=bytes.fromhex("EC00"),
        init_regs={RegisterName.I: 2, RegisterName.FC: 1},
        # DSLL starts at the least-significant byte and walks downward.
        # The wrapped field is BCD 1234: MSB at FF, LSB at 00.
        init_mem={imem(0x00): 0x34, imem(0xFF): 0x12},
        expected_regs={RegisterName.I: 0, RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_mem_writes=[
            (imem(0x00), 0x40),
            (imem(0xFF), 0x23),
        ],
        expected_mem_state={imem(0x00): 0x40, imem(0xFF): 0x23},
        expected_asm_str="DSLL  (BP+00)",
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
        init_mem={imem(0x10): 0x55},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(imem(0x10), 0xAA)],
        expected_mem_state={imem(0x10): 0xAA},
        expected_regs={RegisterName.FC: 0, RegisterName.FZ: 0},
        expected_asm_str="SHL   (BP+10)",
    ),
    InstructionTestCase(
        test_id="SHL_mem_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F710"),
        init_mem={imem(0x10): 0x80},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(imem(0x10), 0x00)],
        expected_mem_state={imem(0x10): 0x00},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHL   (BP+10)",
    ),
    # SHR (n) (0xF5)
    InstructionTestCase(
        test_id="SHR_mem_simple",
        instr_bytes=bytes.fromhex("F510"),
        init_mem={imem(0x10): 0x55},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(imem(0x10), 0x2A)],
        expected_mem_state={imem(0x10): 0x2A},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 0},
        expected_asm_str="SHR   (BP+10)",
    ),
    InstructionTestCase(
        test_id="SHR_mem_carry_out_and_zero",
        instr_bytes=bytes.fromhex("F510"),
        init_mem={imem(0x10): 0x01},
        init_regs={RegisterName.FC: 0},
        expected_mem_writes=[(imem(0x10), 0x00)],
        expected_mem_state={imem(0x10): 0x00},
        expected_regs={RegisterName.FC: 1, RegisterName.FZ: 1},
        expected_asm_str="SHR   (BP+10)",
    ),
    # --- MV Instruction for Internal Memory Register ---
    InstructionTestCase(
        test_id="MV_BP_register_immediate",
        instr_bytes=bytes.fromhex("30ccecc2"),
        init_mem={INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x00},  # Initial BP = 0x00
        expected_mem_state={
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0xC2
        },  # BP = 0xC2
        expected_asm_str="MV    (BP), C2",
    ),
    # --- AND Instructions ---
    # 0x70: AND A, imm8
    InstructionTestCase(
        test_id="AND_A_imm_zero_result",
        instr_bytes=bytes.fromhex("7000"),  # AND A, 00
        init_regs={RegisterName.A: 0xFF, RegisterName.FC: 1},
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 1,
        },  # FC unchanged
        expected_asm_str="AND   A, 00",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("700F"),  # AND A, 0F
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={
            RegisterName.A: 0x05,
            RegisterName.FZ: 0,
            RegisterName.FC: 0,
        },  # FC unchanged
        expected_asm_str="AND   A, 0F",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_all_ones",
        instr_bytes=bytes.fromhex("70FF"),  # AND A, FF
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={
            RegisterName.A: 0xAA,
            RegisterName.FZ: 0,
            RegisterName.FC: 1,
        },  # A unchanged, FC unchanged
        expected_asm_str="AND   A, FF",
    ),
    InstructionTestCase(
        test_id="AND_A_imm_zero_operand",
        instr_bytes=bytes.fromhex("700F"),  # AND A, 0F
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 0},
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 0,
        },  # Zero result
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
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 1,
        },  # FC unchanged
        expected_asm_str="OR    A, 00",
    ),
    InstructionTestCase(
        test_id="OR_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("780F"),  # OR A, 0F
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 0},
        expected_regs={
            RegisterName.A: 0x0F,
            RegisterName.FZ: 0,
            RegisterName.FC: 0,
        },  # FC unchanged
        expected_asm_str="OR    A, 0F",
    ),
    InstructionTestCase(
        test_id="OR_A_imm_all_ones",
        instr_bytes=bytes.fromhex("78FF"),  # OR A, FF
        init_regs={RegisterName.A: 0x00, RegisterName.FC: 1},
        expected_regs={
            RegisterName.A: 0xFF,
            RegisterName.FZ: 0,
            RegisterName.FC: 1,
        },  # FC unchanged
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
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 1,
        },  # FC unchanged
        expected_asm_str="XOR   A, FF",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_zero_result_pattern",
        instr_bytes=bytes.fromhex("6855"),  # XOR A, 55
        init_regs={RegisterName.A: 0x55, RegisterName.FC: 0},
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 0,
        },  # FC unchanged
        expected_asm_str="XOR   A, 55",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_non_zero_result",
        instr_bytes=bytes.fromhex("6855"),  # XOR A, 55
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        expected_regs={
            RegisterName.A: 0xFF,
            RegisterName.FZ: 0,
            RegisterName.FC: 1,
        },  # FC unchanged
        expected_asm_str="XOR   A, 55",
    ),
    InstructionTestCase(
        test_id="XOR_A_imm_with_zero",
        instr_bytes=bytes.fromhex("6800"),  # XOR A, 00
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 0},
        expected_regs={
            RegisterName.A: 0xAA,
            RegisterName.FZ: 0,
            RegisterName.FC: 0,
        },  # A unchanged
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
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 1,
        },  # FC unchanged
        expected_asm_str="AND   A, (BP+10)",
    ),
    # 0x7F: OR A, (n)
    InstructionTestCase(
        test_id="OR_A_mem_non_zero_result",
        instr_bytes=bytes.fromhex("7F10"),  # OR A, (BP+10)
        init_regs={RegisterName.A: 0xF0, RegisterName.FC: 0},
        init_mem={INTERNAL_MEMORY_START + 0x10: 0x0F},
        expected_regs={
            RegisterName.A: 0xFF,
            RegisterName.FZ: 0,
            RegisterName.FC: 0,
        },  # FC unchanged
        expected_asm_str="OR    A, (BP+10)",
    ),
    # 0x6F: XOR A, (n)
    InstructionTestCase(
        test_id="XOR_A_mem_zero_result",
        instr_bytes=bytes.fromhex("6F10"),  # XOR A, (BP+10)
        init_regs={RegisterName.A: 0xAA, RegisterName.FC: 1},
        init_mem={INTERNAL_MEMORY_START + 0x10: 0xAA},
        expected_regs={
            RegisterName.A: 0x00,
            RegisterName.FZ: 1,
            RegisterName.FC: 1,
        },  # FC unchanged
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
            RegisterName.FZ: 0,
        },
        init_mem={
            0xBE000: 0x00,
            0xBE001: 0x00,
        },  # Clear external memory at 0xBE000-0xBE001
        expected_regs={
            RegisterName.BA: 0x5AA5,
            RegisterName.X: 0xBE000,  # X unchanged
            RegisterName.FC: 0,
            RegisterName.FZ: 0,
        },
        expected_mem_writes=[
            (0xBE000, 0xA5),
            (0xBE001, 0x5A),
        ],  # Little-endian: LSB first
        expected_mem_state={0xBE000: 0xA5, 0xBE001: 0x5A},  # BA=0x5AA5 stored as A5 5A
        expected_asm_str="MV    [X], BA",
    ),
    # Test case for 30e904d4 instruction - MVW [X], (BL)
    InstructionTestCase(
        test_id="MVW_X_indirect_from_BP_30e904d4",
        instr_bytes=bytes.fromhex("30E904D4"),  # MVW [X], (BL)
        init_regs={
            RegisterName.X: 0x080000,  # X points to external memory address 0x080000
            RegisterName.FC: 0,
            RegisterName.FZ: 0,
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
            RegisterName.FZ: 0,
        },
        expected_mem_writes=[
            (0x080000, 0x34),  # Low byte written first
            (0x080001, 0x12),  # High byte written second
        ],
        expected_mem_state={
            0x080000: 0x34,  # Word 0x1234 stored little-endian
            0x080001: 0x12,
        },
        expected_asm_str="MVW   [X], (BL)",
    ),
    InstructionTestCase(
        test_id="MVW_abs_ext_from_direct_imem_PRE30_D9",
        instr_bytes=bytes.fromhex("30D9000200D4"),  # PRE30; MVW [00200], (BL)
        init_mem={
            INTERNAL_MEMORY_START + IMEMRegisters.BL: 0x01,
            INTERNAL_MEMORY_START + IMEMRegisters.BH: 0x01,
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0xCB,
            INTERNAL_MEMORY_START + 0x9F: 0x9F,
            INTERNAL_MEMORY_START + 0xA0: 0x9F,
            0x000200: 0x00,
            0x000201: 0x00,
        },
        expected_mem_writes=[
            (0x000200, 0x01),
            (0x000201, 0x01),
        ],
        expected_mem_state={
            0x000200: 0x01,
            0x000201: 0x01,
        },
        expected_asm_str="MVW   [00200], (BL)",
    ),
    InstructionTestCase(
        test_id="MV_Y_from_E6_PRE30",
        instr_bytes=bytes.fromhex("3085E6"),
        init_mem={
            # BP register at internal memory (not used with corrected PRE mode)
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,
            # Place test data at internal memory 0xE6-0xE8 (direct addressing)
            INTERNAL_MEMORY_START + 0xE6: 0x11,
            INTERNAL_MEMORY_START + 0xE7: 0x22,
            INTERNAL_MEMORY_START + 0xE8: 0x33,
        },
        expected_regs={
            RegisterName.Y: 0x032211,  # Little-endian: low byte first (20-bit)
        },
        expected_asm_str="MV    Y, (IOCS_WS)",
    ),
    InstructionTestCase(
        test_id="MVP_to_E6_internal_mem",
        instr_bytes=bytes.fromhex("30dce6defc0b"),
        init_mem={
            # Initialize destination memory to verify it gets overwritten
            INTERNAL_MEMORY_START + 0xE6: 0xFF,
            INTERNAL_MEMORY_START + 0xE7: 0xFF,
            INTERNAL_MEMORY_START + 0xE8: 0xFF,
        },
        expected_mem_state={
            # MVP writes raw 24-bit value 0x0BFCDE in little-endian order to internal memory offset 0xE6
            INTERNAL_MEMORY_START + 0xE6: 0xDE,  # Low byte
            INTERNAL_MEMORY_START + 0xE7: 0xFC,  # Mid byte
            INTERNAL_MEMORY_START
            + 0xE8: 0x0B,  # High byte (only low 4 bits used for 20-bit)
        },
        expected_asm_str="MVP   (IOCS_WS), 0BFCDE",
    ),
    InstructionTestCase(
        test_id="MV_A_from_indirect_ext_mem_with_offset",
        instr_bytes=bytes.fromhex("309880e608"),
        init_regs={
            RegisterName.B: 0xAB,  # Set B to ensure it stays intact
        },
        init_mem={
            # Set BP to 0x34 to ensure it's not being used
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x34,  # BP = 0x34
            # Internal memory at 0xE6 contains 20-bit external address (little-endian)
            INTERNAL_MEMORY_START + 0xE6: 0x00,  # Low byte
            INTERNAL_MEMORY_START + 0xE7: 0x10,  # Mid byte
            INTERNAL_MEMORY_START + 0xE8: 0x02,  # High byte (20-bit address = 0x021000)
            # External memory at 0x021000 + 0x08 = 0x021008 contains test value
            0x021008: 0x42,  # Test value to be loaded into A
        },
        expected_regs={
            RegisterName.A: 0x42,  # A should contain value from external memory
            RegisterName.B: 0xAB,  # B should remain unchanged
        },
        expected_asm_str="MV    A, [(IOCS_WS)+08]",
    ),
    InstructionTestCase(
        test_id="MV_Y_from_X_plus_offset",
        instr_bytes=bytes.fromhex("958412"),
        init_regs={
            RegisterName.X: 0x3000,  # Base address for external memory access
            RegisterName.Y: 0x999999,  # Initial value to verify it gets changed
        },
        init_mem={
            # Place 3 bytes at external memory address X+0x12 (0x3000+0x12=0x3012)
            0x3012: 0x11,  # Low byte
            0x3013: 0x22,  # Mid byte
            0x3014: 0x33,  # High byte
        },
        expected_regs={
            RegisterName.Y: 0x032211,  # Y should contain the 3 bytes from external memory (little-endian, 20-bit)
            RegisterName.X: 0x3000,  # X should remain unchanged (positive offset, not inc/dec)
        },
        expected_asm_str="MV    Y, [X+12]",
    ),
    InstructionTestCase(
        test_id="POPU_IMR",
        instr_bytes=bytes.fromhex("3f"),
        init_regs={
            RegisterName.U: 0x8000,  # User stack pointer in higher external memory
        },
        init_mem={
            # Set BP register to non-zero value
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,  # BP = 0x10
            # Place test value on stack where U points
            0x8000: 0xA5,  # Value to be popped to IMR
            # Initialize IMR to different value to verify it changes
            INTERNAL_MEMORY_START + IMEMRegisters.IMR: 0x00,
        },
        expected_regs={
            RegisterName.U: 0x8001,  # U incremented by 1 after pop
        },
        expected_mem_writes=[
            # POPU IMR writes the popped value to IMR register
            (INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0xA5),
        ],
        expected_mem_state={
            # IMR should contain the popped value (now using direct addressing)
            INTERNAL_MEMORY_START + IMEMRegisters.IMR: 0xA5,
            # BP should remain unchanged
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,
        },
        expected_asm_str="POPU  IMR",
    ),
    InstructionTestCase(
        test_id="PUSHU_IMR_with_BP",
        instr_bytes=bytes.fromhex("2f"),
        init_regs={
            RegisterName.U: 0x8000,  # User stack pointer in higher external memory
        },
        init_mem={
            # Set BP register to non-zero value
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,  # BP = 0x10
            # Set IMR to test value with high bit set
            INTERNAL_MEMORY_START + IMEMRegisters.IMR: 0xFF,  # Test masking behavior
        },
        expected_regs={
            RegisterName.U: 0x7FFF,  # U decremented by 1 after push
        },
        expected_mem_writes=[
            # PUSHU IMR has special behavior per documentation:
            # 1. Pushes original IMR value to stack
            (0x7FFF, 0xFF),  # Original IMR value pushed to stack
            # 2. Then clears bit 7 of IMR (IMR₇ ← 0) only if bit 7 was set
            (INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x7F),  # IMR with bit 7 cleared
        ],
        expected_asm_str="PUSHU IMR",
    ),
    InstructionTestCase(
        test_id="PUSHU_IMR_bit7_clear",
        instr_bytes=bytes.fromhex("2f"),
        init_regs={
            RegisterName.U: 0x8000,  # User stack pointer in higher external memory
        },
        init_mem={
            # Set BP register to non-zero value
            INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x10,  # BP = 0x10
            # Set IMR with bit 7 already clear
            INTERNAL_MEMORY_START + IMEMRegisters.IMR: 0x7F,  # Bit 7 is already 0
        },
        expected_regs={
            RegisterName.U: 0x7FFF,  # U decremented by 1 after push
        },
        expected_mem_writes=[
            # PUSHU IMR pushes the value to stack and always clears bit 7
            (0x7FFF, 0x7F),  # Original IMR value pushed to stack
            # Even though bit 7 is already 0, the instruction still writes
            (
                INTERNAL_MEMORY_START + IMEMRegisters.IMR,
                0x7F,
            ),  # IMR written (unchanged)
        ],
        expected_asm_str="PUSHU IMR",
    ),
    InstructionTestCase(
        test_id="MV_BA_to_indirect_ext_mem_with_offset",
        instr_bytes=bytes.fromhex("30ba80e604"),
        init_regs={
            RegisterName.BA: 0x1234,  # 16-bit test value
        },
        init_mem={
            # Internal memory at 0xE6 contains 20-bit external address (little-endian)
            INTERNAL_MEMORY_START + 0xE6: 0x00,  # Low byte
            INTERNAL_MEMORY_START + 0xE7: 0x20,  # Mid byte
            INTERNAL_MEMORY_START + 0xE8: 0x04,  # High byte (20-bit address = 0x042000)
            # Expected: External memory at 0x042000 + 0x04 = 0x042004 should receive BA value
        },
        expected_mem_writes=[
            # BUG: Currently only writes 1 byte, but should write 2 bytes for BA register
            (0x042004, 0x34),  # Low byte of BA
            (0x042005, 0x12),  # High byte of BA (this write is missing due to bug)
        ],
        expected_mem_state={
            # BUG: Currently only the first byte is written
            0x042004: 0x34,  # Low byte should be written
            0x042005: 0x12,  # High byte should be written (but currently isn't)
        },
        expected_asm_str="MV    [(IOCS_WS)+04], BA",
    ),
    # Test INC X instruction
    InstructionTestCase(
        test_id="INC_X_with_initial_0DF820",
        instr_bytes=bytes.fromhex("6c04"),
        init_regs={RegisterName.X: 0x0DF820},
        expected_regs={
            RegisterName.X: 0x0DF821,
            RegisterName.FZ: 0,  # Z flag cleared since result is non-zero
        },
        expected_asm_str="INC   X",
    ),
    # Test MV [--S], X instruction
    InstructionTestCase(
        test_id="MV_pre_dec_S_X",
        instr_bytes=bytes.fromhex("b437"),
        init_regs={RegisterName.S: 0x0BFC87, RegisterName.X: 0x0F28F9},
        expected_regs={
            RegisterName.S: 0x0BFC84  # S decremented by 3 (size of X)
        },
        expected_mem_writes=[
            (0x0BFC84, 0xF9),  # Low byte of X
            (0x0BFC85, 0x28),  # Middle byte of X
            (0x0BFC86, 0x0F),  # High byte of X
        ],
        expected_mem_state={0x0BFC84: 0xF9, 0x0BFC85: 0x28, 0x0BFC86: 0x0F},
        expected_asm_str="MV    [--S], X",
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

    # Ensure the program counter advances if not explicitly verified above
    if RegisterName.PC not in case.expected_regs:
        expected_pc = case.initial_pc + len(case.instr_bytes)
        actual_pc = cpu.regs.get(RegisterName.PC)
        assert actual_pc == expected_pc, (
            f"[{case.test_id}] Program counter mismatch: "
            f"Expected 0x{expected_pc:X}, Got 0x{actual_pc:X}"
        )


# HW-002 intentionally drives a complete 65,536-iteration register ring. Under
# coverage, slower CPython runners can legitimately exceed the global 60s guard.
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("mnemonic", "instr_bytes", "expected_fz", "write_multiplier"),
    [
        ("ADCL", bytes.fromhex("541020"), 0, 1),
        ("SBCL", bytes.fromhex("5C1020"), 1, 1),
        ("DADL", bytes.fromhex("C41020"), 0, 1),
        ("DSBL", bytes.fromhex("D41020"), 1, 1),
        ("MVL", bytes.fromhex("CB1020"), 1, 1),
        ("MVLD", bytes.fromhex("CF1020"), 1, 1),
        ("EXL", bytes.fromhex("C31020"), 1, 2),
        ("DSLL", bytes.fromhex("EC10"), 0, 1),
        ("DSRL", bytes.fromhex("FC10"), 0, 1),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_hw002_zero_counted_instruction_executes_full_16bit_ring(
    mnemonic: str,
    instr_bytes: bytes,
    expected_fz: int,
    write_multiplier: int,
) -> None:
    """Hardware says I=0 is 65,536 do-while iterations for all families."""
    cpu, raw, _reads, writes = _make_cpu_and_mem(ADDRESS_SPACE_SIZE, {}, instr_bytes)
    cpu.regs.set(RegisterName.I, 0)
    cpu.regs.set(RegisterName.FC, 0)
    cpu.regs.set(RegisterName.FZ, 1)

    decoded = cpu.decode_instruction(0)
    assert decoded is not None
    assert asm_str(decoded.render()).split()[0] == mnemonic

    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == expected_fz
    assert cpu.regs.get(RegisterName.PC) == len(instr_bytes)
    assert len(writes) == 0x10000 * write_multiplier
    assert raw[imem(0x10)] == 0
    assert raw[imem(0x20)] == 0


@pytest.mark.parametrize(("initial_i", "expected_cycles"), [(3, 3), (0, 0x10000)])
def test_hw002_wait_dispatch_uses_16bit_do_while_count(
    initial_i: int, expected_cycles: int
) -> None:
    instr_bytes = bytes.fromhex("EF")
    cpu, _raw, _reads, writes = _make_cpu_and_mem(ADDRESS_SPACE_SIZE, {}, instr_bytes)
    cpu.regs.set(RegisterName.I, initial_i)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 0)
    wait_calls: List[int] = []
    setattr(cpu.memory, "wait_cycles", wait_calls.append)

    info = cpu.execute_instruction(0)

    assert asm_str(info.instruction.render()) == "WAIT"
    assert wait_calls == [expected_cycles]
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.PC) == len(instr_bytes)
    assert writes == []


@pytest.mark.parametrize(("initial_i", "expected_cycles"), [(3, 3), (0, 0x10000)])
def test_hw002_wait_direct_llil_intrinsic_uses_16bit_do_while_count(
    initial_i: int, expected_cycles: int
) -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, {}, bytes.fromhex("EF")
    )
    cpu.regs.set(RegisterName.I, initial_i)
    cpu.regs.set(RegisterName.FC, 0)
    cpu.regs.set(RegisterName.FZ, 1)
    wait_calls: List[int] = []
    setattr(cpu.memory, "wait_cycles", wait_calls.append)
    instr = cpu.decode_instruction(0)
    il = MockLowLevelILFunction()
    instr.lift(il, 0)

    assert [getattr(node, "name", None) for node in il.ils] == ["WAIT"]
    for node in il.ils:
        cpu.evaluate(node)

    assert wait_calls == [expected_cycles]
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert writes == []


def test_wait_without_timing_hook_fails_before_state_mutation() -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, {}, bytes.fromhex("EF")
    )
    cpu.regs.set(RegisterName.I, 0x1234)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 0)

    with pytest.raises(NotImplementedError, match=r"memory\.wait_cycles"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert cpu.regs.get(RegisterName.I) == 0x1234
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert writes == []


def test_wait_direct_intrinsic_without_timing_hook_preserves_i_and_flags() -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, {}, bytes.fromhex("EF")
    )
    cpu.regs.set(RegisterName.I, 7)
    cpu.regs.set(RegisterName.FC, 0)
    cpu.regs.set(RegisterName.FZ, 1)
    instr = cpu.decode_instruction(0)
    il = MockLowLevelILFunction()
    instr.lift(il, 0)

    with pytest.raises(NotImplementedError, match=r"memory\.wait_cycles"):
        cpu.evaluate(il.ils[0])

    assert cpu.regs.get(RegisterName.I) == 7
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert writes == []


@pytest.mark.parametrize("instr_bytes", [bytes.fromhex("CE")])
def test_tcl_quarantine_uses_decoded_instruction(instr_bytes: bytes) -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(ADDRESS_SPACE_SIZE, {}, instr_bytes)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 0)

    with pytest.raises(NotImplementedError, match="timer-phase-clear memory hook"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert writes == []


@pytest.mark.parametrize(
    "instr_bytes",
    [
        bytes.fromhex("3000"),  # PRE + NOP
        bytes.fromhex("30EF"),  # PRE + WAIT
        bytes.fromhex("30CE"),  # PRE + TCL
        bytes.fromhex("30DE"),  # PRE + HALT
        bytes.fromhex("30DF"),  # PRE + OFF
        bytes.fromhex("30FF"),  # PRE + RESET
        bytes.fromhex("303132A005"),  # only two consecutive PRE bytes are proven
        bytes.fromhex("211100"),  # PRE followed by reserved JP selector
        bytes.fromhex("21E31000"),  # PRE followed by malformed E3 mode
        bytes.fromhex("257C01"),  # EFE2B is mid-instruction, not a code entry
        bytes.fromhex("23483F"),  # F0002 is the operand byte of FD 23, not an entry
    ],
)
def test_noncanonical_and_malformed_pre_prefixes_fail_closed(
    instr_bytes: bytes,
) -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(ADDRESS_SPACE_SIZE, {}, instr_bytes)

    with pytest.raises(InvalidInstruction, match="PRE"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert writes == []


def test_two_consecutive_pre_prefixes_use_the_second_latch() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE, {}, bytes.fromhex("3031A005")
    )
    cpu.regs.set(RegisterName.A, 0xE0)

    result = cpu.execute_instruction(0)

    assert result.instruction._pre == 0x31
    assert result.instruction.length() == 4
    assert cpu.regs.get(RegisterName.PC) == 4
    assert raw[INTERNAL_MEMORY_START + 0x05] == 0xE0
    assert writes[-1] == (INTERNAL_MEMORY_START + 0x05, 0xE0)


def test_host_write_failure_rolls_back_registers_and_poison_requires_reset() -> None:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes.fromhex("CC1042")  # MV (BP+10), 42
    fail_target_write = True

    def read_mem(address: int) -> int:
        return raw[address]

    def write_mem(address: int, value: int) -> None:
        nonlocal fail_target_write
        if fail_target_write and address == imem(0x10):
            raise RuntimeError("host write failed")
        raw[address] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda addr, _pc=None: raw[addr],
    )
    cpu = Emulator(memory, reset_on_init=False)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.A, 0x5A)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.call_sub_level = 7

    with pytest.raises(RuntimeError, match="host write failed"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert cpu.regs.get(RegisterName.A) == 0x5A
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.call_sub_level == 7
    assert raw[imem(0x10)] == 0

    fail_target_write = False
    with pytest.raises(RuntimeError, match="poisoned.*reset required"):
        cpu.execute_instruction(0)

    cpu.power_on_reset()
    cpu.execute_instruction(0)
    assert raw[imem(0x10)] == 0x42
    assert cpu.regs.get(RegisterName.PC) == 3


def test_failed_recovery_reset_preserves_first_poison_reason() -> None:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes.fromhex("CC1042")  # MV (BP+10), 42
    raw[0xFFFFD:0x100000] = bytes.fromhex("452301")
    failure = "instruction"

    def read_mem(address: int) -> int:
        return raw[address]

    def write_mem(address: int, value: int) -> None:
        if failure == "instruction" and address == imem(0x10):
            raise RuntimeError("first host-write fault")
        if failure == "reset" and address == imem(IMEMRegisters.UCR):
            raise RuntimeError("later reset fault")
        raw[address] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda addr, _pc=None: raw[addr],
    )
    cpu = Emulator(memory, reset_on_init=False)
    with pytest.raises(RuntimeError, match="first host-write fault"):
        cpu.execute_instruction(0)

    failure = "reset"
    with pytest.raises(RuntimeError, match="later reset fault"):
        cpu.power_on_reset()

    with pytest.raises(RuntimeError, match="first host-write fault") as poisoned:
        cpu.execute_instruction(0)
    assert "later reset fault" not in str(poisoned.value)


def test_wide_imem_access_wraps_each_byte_at_ff() -> None:
    initial_memory = {
        imem(0xFE): 0x56,
        imem(0xFF): 0x34,
        imem(0x00): 0x02,
    }
    cpu, _raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        initial_memory,
        bytes.fromhex("3084FE"),  # PRE (n); MV X,(LCC)
    )

    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.X) == 0x23456
    assert writes == []
    assert imem(0xFE) in reads
    assert imem(0xFF) in reads
    assert imem(0x00) in reads
    assert INTERNAL_MEMORY_START + 0x100 not in reads

    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {},
        bytes.fromhex("30A4FE"),  # PRE (n); MV (LCC),X
    )
    cpu.regs.set(RegisterName.X, 0x23456)

    cpu.execute_instruction(0)

    assert writes == [
        (imem(0xFE), 0x56),
        (imem(0xFF), 0x34),
        (imem(0x00), 0x02),
    ]
    assert raw[imem(0xFE)] == 0x56
    assert raw[imem(0xFF)] == 0x34
    assert raw[imem(0x00)] == 0x02


def test_wide_emem_access_wraps_each_byte_at_fffff() -> None:
    instr_addr = 0x100
    initial_memory = {
        0xFFFFE: 0x56,
        0xFFFFF: 0x34,
        0x00000: 0x02,
        imem(0x00): 0x99,  # catches a spill into the synthetic IMEM window
    }
    cpu, _raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        initial_memory,
        bytes.fromhex("8CFEFF0F"),  # MV X,[FFFFE]
        instr_addr,
    )

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.X) == 0x23456
    assert writes == []
    for address in (0xFFFFE, 0xFFFFF, 0x00000):
        assert address in reads
    assert imem(0x00) not in reads

    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {imem(0x00): 0x99},
        bytes.fromhex("ACFEFF0F"),  # MV [FFFFE],X
        instr_addr,
    )
    cpu.regs.set(RegisterName.X, 0x23456)

    cpu.execute_instruction(instr_addr)

    assert writes == [(0xFFFFE, 0x56), (0xFFFFF, 0x34), (0x00000, 0x02)]
    assert raw[0xFFFFE] == 0x56
    assert raw[0xFFFFF] == 0x34
    assert raw[0x00000] == 0x02
    assert raw[imem(0x00)] == 0x99


def test_indirect_external_pointer_load_wraps_inside_imem_at_ff() -> None:
    instr_addr = 0x100
    cpu, _raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {
            imem(0xFF): 0x23,
            imem(0x00): 0x01,
            imem(0x01): 0xF8,  # raw F80123; upper nibble is ignored
            0x80123: 0x6B,
        },
        bytes.fromhex("309800FF"),  # PRE (n); MV A,[(SSR)]
        instr_addr,
    )

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.A) == 0x6B
    assert writes == []
    for address in (imem(0xFF), imem(0x00), imem(0x01)):
        assert address in reads
    assert INTERNAL_MEMORY_START + 0x100 not in reads
    assert INTERNAL_MEMORY_START + 0x101 not in reads


def test_instruction_fetch_wraps_at_20bit_pc_boundary() -> None:
    cpu, _raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {
            0xFFFFF: 0x0A,  # MV BA,1234
            0x00000: 0x34,
            0x00001: 0x12,
            imem(0x00): 0x99,
            imem(0x01): 0x88,
        },
        b"",
    )

    info = cpu.execute_instruction(0xFFFFF)

    assert asm_str(info.instruction.render()) == "MV    BA, 1234"
    assert cpu.regs.get(RegisterName.BA) == 0x1234
    assert cpu.regs.get(RegisterName.PC) == 0x00002
    assert 0xFFFFF in reads
    assert 0x00000 in reads
    assert 0x00001 in reads
    assert imem(0x00) not in reads
    assert imem(0x01) not in reads
    assert writes == []


def test_wrapped_wide_store_snapshots_overlapping_source() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {imem(0xFE): 0xAA, imem(0xFF): 0xBB, imem(0x00): 0xCC},
        bytes.fromhex("32C9FFFE"),  # PRE (n),(n); MVW (SSR),(LCC)
    )

    cpu.execute_instruction(0)

    assert writes == [(imem(0xFF), 0xAA), (imem(0x00), 0xBB)]
    assert raw[imem(0xFE)] == 0xAA
    assert raw[imem(0xFF)] == 0xAA
    assert raw[imem(0x00)] == 0xBB


def test_wrapped_wide_store_snapshots_bp_destination_before_self_overlap() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {
            imem(IMEMRegisters.BP): 0xEC,
            imem(IMEMRegisters.PX): 0xA5,
            imem(0xFC): 0x34,
            imem(0xFD): 0x12,
        },
        bytes.fromhex("C90010"),  # MVW (BP+00),(BP+10)
    )

    cpu.execute_instruction(0)

    # The first byte overwrites BP itself.  The second destination must still
    # use the effective address captured before that write, so it lands on PX.
    assert writes == [
        (imem(IMEMRegisters.BP), 0x34),
        (imem(IMEMRegisters.PX), 0x12),
    ]
    assert raw[imem(IMEMRegisters.BP)] == 0x34
    assert raw[imem(IMEMRegisters.PX)] == 0x12
    assert raw[imem(0x35)] == 0x00


def test_wrapped_pointer_store_snapshots_bp_destination_across_bp_px_py() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {
            imem(IMEMRegisters.BP): 0xEC,
            imem(IMEMRegisters.PX): 0xA5,
            imem(IMEMRegisters.PY): 0x5A,
            imem(0xFC): 0x56,
            imem(0xFD): 0x34,
            imem(0xFE): 0x02,
        },
        bytes.fromhex("CA0010"),  # MVP (BP+00),(BP+10)
    )

    cpu.execute_instruction(0)

    assert writes == [
        (imem(IMEMRegisters.BP), 0x56),
        (imem(IMEMRegisters.PX), 0x34),
        (imem(IMEMRegisters.PY), 0x02),
    ]
    assert raw[imem(IMEMRegisters.BP)] == 0x56
    assert raw[imem(IMEMRegisters.PX)] == 0x34
    assert raw[imem(IMEMRegisters.PY)] == 0x02
    assert raw[imem(0x57)] == 0x00
    assert raw[imem(0x58)] == 0x00


def test_wide_register_indirect_store_snapshots_address_across_20bit_wrap() -> None:
    instr_addr = 0x100
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {imem(0x00): 0x99},
        bytes.fromhex("B404"),  # MV [X],X
        instr_addr,
    )
    cpu.regs.set(RegisterName.X, 0xFFFFF)

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.X) == 0xFFFFF
    assert writes == [(0xFFFFF, 0xFF), (0x00000, 0xFF), (0x00001, 0x0F)]
    assert raw[0xFFFFF] == 0xFF
    assert raw[0x00000] == 0xFF
    assert raw[0x00001] == 0x0F
    assert raw[imem(0x00)] == 0x99


def test_pushs_pops() -> None:
    # Test PUSHS F and POPS F instructions
    # Note: PUSHS IMR and POPS IMR do not exist in the SC62015 instruction set
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x10000, {}, bytes.fromhex("4F"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHS F"

    cpu.regs.set(RegisterName.F, 0x0)
    cpu.regs.set(RegisterName.S, 0x7000)  # System stack in higher external memory
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFF
    assert writes == [(0x6FFF, 0x0)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFE
    assert writes == [(0x6FFE, 0x2)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 0)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFD
    assert writes == [(0x6FFD, 0x1)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFC
    assert writes == [(0x6FFC, 0x3)]
    writes.clear()

    cpu.regs.set(RegisterName.F, 0)
    raw[0] = 0x5F  # Change to POPS instruction
    assert asm_str(cpu.decode_instruction(0x00).render()) == "POPS  F"
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFD
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFE
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x6FFF
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.S) == 0x7000
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 0


def test_pops_f_direct_llil_normalizes_before_advancing_stack() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {0x100: 0x80},
        bytes.fromhex("5F"),
    )
    cpu.regs.set(RegisterName.S, 0x100)
    cpu.regs.set(RegisterName.F, 0x01)
    cpu.regs.set(RegisterName.PC, 0)
    instr = cpu.decode_instruction(0)
    il = MockLowLevelILFunction()
    instr.lift(il, 0)

    raw_f_index = next(
        index
        for index, node in enumerate(il.ils)
        if node.op == "SET_REG.b" and getattr(node.ops[0], "name", None) == "TEMP0"
    )
    s_write_index = next(
        index
        for index, node in enumerate(il.ils)
        if node.op == "SET_REG.l" and getattr(node.ops[0], "name", None) == "S"
    )
    assert raw_f_index < s_write_index

    for node in il.ils:
        cpu.evaluate(node)

    assert cpu.regs.get(RegisterName.S) == 0x101
    assert cpu.regs.get(RegisterName.F) == 0x00
    assert cpu.regs.get(RegisterName.PC) == 0
    assert raw[0x100] == 0x80
    assert writes == []


def test_exp_high_nibble_direct_llil_exchanges_raw24_and_preserves_flags() -> None:
    initial = {
        imem(0x20): 0x11,
        imem(0x21): 0x22,
        imem(0x22): 0xA8,
        imem(0x30): 0x33,
        imem(0x31): 0x44,
        imem(0x32): 0xB9,
    }
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        initial,
        bytes.fromhex("32C22030"),
    )
    instr = cpu.decode_instruction(0)
    il = MockLowLevelILFunction()
    instr.lift(il, 0)
    cpu.regs.set(RegisterName.F, 0x03)

    for node in il.ils:
        cpu.evaluate(node)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert raw[imem(0x20) : imem(0x23)] == bytes.fromhex("3344B9")
    assert raw[imem(0x30) : imem(0x33)] == bytes.fromhex("1122A8")
    assert cpu.regs.get(RegisterName.F) == 0x03
    assert writes == [
        (imem(0x20), 0x33),
        (imem(0x30), 0x11),
        (imem(0x21), 0x44),
        (imem(0x31), 0x22),
        (imem(0x22), 0xB9),
        (imem(0x32), 0xA8),
    ]


def test_pushu_popu() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x10000, {}, bytes.fromhex("2E"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHU F"

    cpu.regs.set(RegisterName.F, 0x0)
    cpu.regs.set(RegisterName.U, 0x8000)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFF
    assert writes == [(0x7FFF, 0x0)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFE
    assert writes == [(0x7FFE, 0x2)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 0)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFD
    assert writes == [(0x7FFD, 0x1)]
    writes.clear()

    cpu.regs.set(RegisterName.FZ, 1)
    cpu.regs.set(RegisterName.FC, 1)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFC
    assert writes == [(0x7FFC, 0x3)]
    writes.clear()

    cpu.regs.set(RegisterName.F, 0)
    raw[0] = 0x3E  # POPU instruction
    assert asm_str(cpu.decode_instruction(0x00).render()) == "POPU  F"
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFD
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFE
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.FC) == 1

    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFF
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.FC) == 0


def test_pushu_popu_r2() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x10000, {}, bytes.fromhex("2A"))
    assert asm_str(cpu.decode_instruction(0x00).render()) == "PUSHU BA"

    cpu.regs.set(RegisterName.BA, 0x1234)
    cpu.regs.set(RegisterName.U, 0x8000)
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x7FFE
    assert writes == [(0x7FFF, 0x12), (0x7FFE, 0x34)]
    writes.clear()

    raw[0] = 0x3A  # POPU BA
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.U) == 0x8000
    assert cpu.regs.get(RegisterName.BA) == 0x1234


def test_pushu_popu_wide_access_wraps_at_20bit_boundary() -> None:
    instr_addr = 0x100
    cpu, raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {imem(0x00): 0x99},
        bytes.fromhex("2A"),  # PUSHU BA
        instr_addr,
    )
    cpu.regs.set(RegisterName.BA, 0x1234)
    cpu.regs.set(RegisterName.U, 0x00001)

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.U) == 0xFFFFF
    assert writes == [(0x00000, 0x12), (0xFFFFF, 0x34)]
    assert raw[0xFFFFF] == 0x34
    assert raw[0x00000] == 0x12
    assert raw[imem(0x00)] == 0x99

    cpu, _raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {0xFFFFF: 0x78, 0x00000: 0x56, imem(0x00): 0x99},
        bytes.fromhex("3A"),  # POPU BA
        instr_addr,
    )
    cpu.regs.set(RegisterName.U, 0xFFFFF)

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.BA) == 0x5678
    assert cpu.regs.get(RegisterName.U) == 0x00001
    assert 0xFFFFF in reads
    assert 0x00000 in reads
    assert imem(0x00) not in reads
    assert writes == []


@pytest.mark.parametrize("upper_nibble", range(16))
@pytest.mark.parametrize(("opcode", "mnemonic"), [(0x03, "JPF"), (0x05, "CALLF")])
def test_hw009_far_control_masks_every_encoded_upper_nibble(
    upper_nibble: int, opcode: int, mnemonic: str
) -> None:
    encoded_high = upper_nibble << 4
    cpu, _raw, _reads, writes = _make_cpu_and_mem(
        0x40, {}, bytes((opcode, 0x20, 0x00, encoded_high))
    )
    cpu.regs.set(RegisterName.S, 0x30)

    assert asm_str(cpu.decode_instruction(0).render()) == f"{mnemonic:<6}00020"
    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0x20
    if opcode == 0x03:
        assert cpu.regs.get(RegisterName.S) == 0x30
        assert writes == []
    else:
        assert cpu.regs.get(RegisterName.S) == 0x2D
        assert writes == [(0x2F, 0x00), (0x2E, 0x00), (0x2D, 0x04)]


def test_call_ret() -> None:
    cpu, raw, _reads, writes = _make_cpu_and_mem(0x10000, {}, bytes.fromhex("042000"))
    raw[0x20] = 0x06
    assert asm_str(cpu.decode_instruction(0x00).render()) == "CALL  0020"
    assert asm_str(cpu.decode_instruction(0x20).render()) == "RET"

    cpu.regs.set(RegisterName.S, 0x7000)  # Set system stack pointer to a valid location
    _ = cpu.execute_instruction(0x00)
    assert cpu.regs.get(RegisterName.PC) == 0x20
    assert cpu.regs.get(RegisterName.S) == 0x6FFE
    assert writes == [(0x6FFF, 0x00), (0x6FFE, 0x03)]
    writes.clear()

    _ = cpu.execute_instruction(cpu.regs.get(RegisterName.PC))
    assert cpu.regs.get(RegisterName.PC) == 0x03
    assert cpu.regs.get(RegisterName.S) == 0x7000
    assert writes == []


@pytest.mark.parametrize(
    ("program", "instruction_address", "target", "frame"),
    [
        (bytes.fromhex("042000"), 0x30000, 0x30020, bytes.fromhex("0300")),
        (bytes.fromhex("05200004"), 0x30000, 0x40020, bytes.fromhex("040003")),
    ],
)
def test_call_llil_call_writes_exactly_one_architectural_frame(
    program: bytes, instruction_address: int, target: int, frame: bytes
) -> None:
    cpu, _raw, _reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {},
        program,
        instr_addr=instruction_address,
    )
    initial_s = 0x0100
    cpu.regs.set(RegisterName.S, initial_s)

    cpu.execute_instruction(instruction_address)

    expected_s = initial_s - len(frame)
    assert cpu.regs.get(RegisterName.PC) == target
    assert cpu.regs.get(RegisterName.S) == expected_s
    assert writes == [
        (expected_s + offset, frame[offset]) for offset in reversed(range(len(frame)))
    ]


def test_callf_retf_stack_wraps_at_20bit_boundary() -> None:
    instr_addr = 0x100
    target_addr = 0x200
    cpu, raw, reads, writes = _make_cpu_and_mem(
        ADDRESS_SPACE_SIZE,
        {target_addr: 0x07, imem(0x00): 0x99},  # RETF at target
        bytes.fromhex("05000200"),  # CALLF 00200
        instr_addr,
    )
    cpu.regs.set(RegisterName.S, 0x00002)

    cpu.execute_instruction(instr_addr)

    assert cpu.regs.get(RegisterName.PC) == target_addr
    assert cpu.regs.get(RegisterName.S) == 0xFFFFF
    assert writes == [(0x00001, 0x00), (0x00000, 0x01), (0xFFFFF, 0x04)]
    assert raw[imem(0x00)] == 0x99
    writes.clear()

    cpu.execute_instruction(target_addr)

    assert cpu.regs.get(RegisterName.PC) == instr_addr + 4
    assert cpu.regs.get(RegisterName.S) == 0x00002
    assert 0xFFFFF in reads
    assert 0x00000 in reads
    assert 0x00001 in reads
    assert imem(0x00) not in reads
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
    assert writes == [(0x2F, 0x00), (0x2E, 0x03)]
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
    assert writes == [(0x2F, 0x00), (0x2E, 0x00), (0x2D, 0x04)]
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
    expected_pre_val_in_instr: (
        int  # The PRE byte value itself, as stored in the decoded instr
    )

    # For tests like MV A, (mem_source)
    expected_A_val_after: Optional[int] = None

    # For tests like MV (mem_dest), A
    expected_mem_writes_after: Optional[List[Tuple[int, int]]] = (
        None  # List of (address, value)
    )


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
            test_id="PRE_0x30_Single_Addressable_MV_A_(n)",  # 0x30 with single addressable operand uses PRE1
            instr_bytes=bytes([0x30, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
                # Place data at direct address since single addressable operands use PRE1
                INTERNAL_MEMORY_START + N_OPERAND_VAL: OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str=f"MV    A, ({N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x30,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
        PreTestCase(
            test_id="PRE_0x33_Single_Addressable_MV_A_(n)",  # 0x33 with single addressable operand uses PRE1
            instr_bytes=bytes([0x33, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.PY: PY_REG_VAL,
                # Place data at direct address since single addressable operands use PRE1
                INTERNAL_MEMORY_START + N_OPERAND_VAL: OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str=f"MV    A, ({N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x33,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
        PreTestCase(
            test_id="PRE_0x31_Single_Addressable_MV_A_(n)",  # 0x31 with single addressable operand uses PRE1
            instr_bytes=bytes([0x31, MV_A_DEST_MEM_SRC_OPCODE, N_OPERAND_VAL]),
            init_memory_state={
                INTERNAL_MEMORY_START + IMEMRegisters.BP: BP_REG_VAL,
                INTERNAL_MEMORY_START + IMEMRegisters.PY: PY_REG_VAL,
                # Place data at direct address since single addressable operands use PRE1
                INTERNAL_MEMORY_START + N_OPERAND_VAL: OPERAND_MEM_VAL,
            },
            init_register_state={RegisterName.A: 0x00},
            expected_asm_str=f"MV    A, ({N_OPERAND_VAL:02X})",
            expected_pre_val_in_instr=0x31,
            expected_A_val_after=OPERAND_MEM_VAL,
        ),
    ]
    # These are exactly the one-selector aliases established by the PC-E500
    # matrix. Source assembly still emits only canonical prefixes.
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
    assert decoded_instr is not None, (
        f"Test '{tc.test_id}': Failed to decode instruction bytes: {tc.instr_bytes.hex()}"
    )

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
        test_id="ADCL_(m)_(n)_I2_RhsPlusCarryWrapPropagates",
        instr_bytes=bytes([0x54, 0x10, 0x20]),
        init_memory_state={
            INTERNAL_MEMORY_START + 0x10: 0x00,
            INTERNAL_MEMORY_START + 0x11: 0x00,
            INTERNAL_MEMORY_START + 0x20: 0xFF,
            INTERNAL_MEMORY_START + 0x21: 0x00,
        },
        init_register_state={RegisterName.I: 2, RegisterName.FC: 1},
        expected_asm_str="ADCL  (BP+10), (BP+20)",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,
        # 00 + FF + 1 -> 00/C=1, then 00 + 00 + 1 -> 01/C=0.
        expected_m_values_after=[0x00, 0x01],
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
    assert actual_asm_str == tc.expected_asm_str, (
        f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"
    )

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert actual_val == expected_val, (
            f"Test '{tc.test_id}': Memory mismatch at offset {i}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
        )

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
    assert actual_writes_to_m == sorted(expected_writes_to_m), (
        f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"
    )

    assert cpu.regs.get(RegisterName.I) == tc.expected_I_after, (
        f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    )
    assert cpu.regs.get(RegisterName.FC) == tc.expected_FC_after, (
        f"Test '{tc.test_id}': Flag C. Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    )

    assert cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after, (
        f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    )


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
        # Byte 1 (MSB, addr 0x10): mem[0x10]=0x01, src=0x00 (register source only on first byte), carry-in=1.
        #                            BCD 01 + BCD 00 + 1 = BCD 02 -> mem[0x10]=0x02, FC=0
        expected_m_values_after=[0x02, 0x00],  # MSB then LSB
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
    assert actual_asm_str == tc.expected_asm_str, (
        f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"
    )

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert actual_val == expected_val, (
            f"Test '{tc.test_id}': Memory mismatch at offset {i} from MSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
        )

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
    assert actual_writes_to_m == sorted(expected_writes_to_m), (
        f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"
    )

    assert cpu.regs.get(RegisterName.I) == tc.expected_I_after, (
        f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    )
    assert cpu.regs.get(RegisterName.FC) == tc.expected_FC_after, (
        f"Test '{tc.test_id}': Flag C. Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    )
    assert cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after, (
        f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    )


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
    assert actual_asm_str == tc.expected_asm_str, (
        f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"
    )

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[tc.expected_m_addr_start + i]
        assert actual_val == expected_val, (
            f"Test '{tc.test_id}': Memory mismatch at offset {i} from LSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
        )

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
    assert actual_writes_to_m == sorted(expected_writes_to_m), (
        f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected: {sorted(expected_writes_to_m)}\nGot: {actual_writes_to_m}"
    )

    assert cpu.regs.get(RegisterName.I) == tc.expected_I_after, (
        f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    )
    assert cpu.regs.get(RegisterName.FC) == tc.expected_FC_after, (
        f"Test '{tc.test_id}': Flag C (Borrow). Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    )
    assert cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after, (
        f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    )


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
            RegisterName.A: 0x01,  # BCD 01 for A (consumed on first byte only)
            RegisterName.I: 2,
            RegisterName.FC: 0,
        },
        expected_asm_str="DSBL  (BP+11), A",
        expected_m_addr_start=INTERNAL_MEMORY_START + 0x10,  # MSB addr of (m)
        # Byte 0 (LSB, m_addr 0x11): m[0x11]=0x00, A=0x01. BCD 00 - BCD 01 - 0 = BCD 99. mem[0x11]=0x99, FC_out=1
        # Byte 1 (MSB, m_addr 0x10): m[0x10]=0x01, src=0x00 (register source only on first byte), borrow-in=1.
        #                            BCD 01 - BCD 00 - 1 = BCD 00. mem[0x10]=0x00, FC_out=0
        expected_m_values_after=[0x00, 0x99],  # MSB, LSB
        expected_I_after=0,
        expected_FC_after=0,  # Borrow from last op
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
    assert actual_asm_str == tc.expected_asm_str, (
        f"Test '{tc.test_id}': ASM string mismatch. Expected '{tc.expected_asm_str}', Got '{actual_asm_str}'"
    )

    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    # DSBL processes memory in reverse, (m) operand in instruction is the LSB/end address.
    # tc.expected_m_addr_start is the MSB address for verification.
    # tc.expected_m_values_after is [MSB_val, ..., LSB_val].
    for i, expected_val in enumerate(tc.expected_m_values_after):
        actual_val = raw_memory_array[
            tc.expected_m_addr_start + i
        ]  # Iterates from MSB_addr
        assert actual_val == expected_val, (
            f"Test '{tc.test_id}': Memory mismatch at offset {i} from MSB_addr 0x{tc.expected_m_addr_start:04X}. Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
        )

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
    assert actual_writes_to_m == sorted(expected_writes_to_m), (
        f"Test '{tc.test_id}': Logged memory writes to (m) mismatch.\nExpected (sorted): {sorted(expected_writes_to_m)}\nGot (sorted): {actual_writes_to_m}"
    )

    assert cpu.regs.get(RegisterName.I) == tc.expected_I_after, (
        f"Test '{tc.test_id}': Reg I. Expected {tc.expected_I_after}, Got {cpu.regs.get(RegisterName.I)}"
    )
    assert cpu.regs.get(RegisterName.FC) == tc.expected_FC_after, (
        f"Test '{tc.test_id}': Flag C (Borrow). Expected {tc.expected_FC_after}, Got {cpu.regs.get(RegisterName.FC)}"
    )
    assert cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after, (
        f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    )


# Add new NamedTuple for DSLL/DSRL test cases
class DsrlDsllTestCase(NamedTuple):
    test_id: str
    is_dsll: bool  # True for DSLL, False for DSRL
    instr_operand_n_val: int  # The 8-bit value for (n) in the instruction
    loop_count_I: int
    # Logical byte order is always most-significant to least-significant.
    # DSLL's operand points at the LSB and walks down; DSRL's operand points
    # at the MSB and walks up.
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

    digits = [
        nibble
        for byte in logical_bcd_bytes
        for nibble in ((byte >> 4) & 0x0F, byte & 0x0F)
    ]
    shifted_digits = digits[1:] + [0]
    return [
        (shifted_digits[i] << 4) | shifted_digits[i + 1]
        for i in range(0, len(shifted_digits), 2)
    ]


def compute_expected_dsrl(logical_bcd_bytes: List[int]) -> List[int]:
    """
    Computes the result of DSRL operation on BCD bytes.
    logical_bcd_bytes is [MSB_val, ..., LSB_val].
    e.g., for BCD 123456, input is [0x12, 0x34, 0x56].
    Result for 123456 -> 012345 is [0x01, 0x23, 0x45].
    """
    if not logical_bcd_bytes:
        return []

    digits = [
        nibble
        for byte in logical_bcd_bytes
        for nibble in ((byte >> 4) & 0x0F, byte & 0x0F)
    ]
    shifted_digits = [0] + digits[:-1]
    return [
        (shifted_digits[i] << 4) | shifted_digits[i + 1]
        for i in range(0, len(shifted_digits), 2)
    ]


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
        initial_bcd_logical_bytes=[0x12, 0x34],  # [MSB, LSB]
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x12, 0x34]
        ),  # [0x01, 0x23]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I3_123456_to_012345",
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=3,
        initial_bcd_logical_bytes=[0x12, 0x34, 0x56],  # [MSB, Mid, LSB]
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x12, 0x34, 0x56]
        ),  # [0x01, 0x23, 0x45]
        expected_FZ_after=0,
    ),
    DsrlDsllTestCase(
        test_id="DSRL_I2_9000_to_0900",
        is_dsll=False,
        instr_operand_n_val=0x10,
        loop_count_I=2,
        initial_bcd_logical_bytes=[0x90, 0x00],  # [MSB, LSB]
        expected_final_bcd_logical_bytes=compute_expected_dsrl(
            [0x90, 0x00]
        ),  # [0x09, 0x00]
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
    # DSLL starts at the LSB address and walks toward the MSB by decrementing.
    if tc.is_dsll:
        for i in range(tc.loop_count_I):
            addr = INTERNAL_MEMORY_START + tc.instr_operand_n_val - i
            init_memory_state[addr] = tc.initial_bcd_logical_bytes[-1 - i]
    else:  # DSRL
        # DSRL starts at the MSB address and walks toward the LSB by incrementing.
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
    assert decoded_instr is not None, (
        f"Test '{tc.test_id}': Failed to decode instruction"
    )

    expected_mnemonic = "DSLL " if tc.is_dsll else "DSRL "
    # Assuming IMem8 is rendered as (BP+XX)
    expected_asm_str = f"{expected_mnemonic:6s}(BP+{tc.instr_operand_n_val:02X})"
    actual_asm_str = asm_str(decoded_instr.render())

    assert actual_asm_str == expected_asm_str, (
        f"Test '{tc.test_id}': ASM string mismatch.\n  Expected: '{expected_asm_str}'\n  Actual  : '{actual_asm_str}'"
    )

    # --- Execute ---
    # debug_instruction(cpu, 0x00)
    _ = cpu.execute_instruction(0x00)

    # --- Verify Registers ---
    assert cpu.regs.get(RegisterName.I) == 0, (
        f"Test '{tc.test_id}': Reg I. Expected 0, Got {cpu.regs.get(RegisterName.I)}"
    )
    assert cpu.regs.get(RegisterName.FZ) == tc.expected_FZ_after, (
        f"Test '{tc.test_id}': Flag Z. Expected {tc.expected_FZ_after}, Got {cpu.regs.get(RegisterName.FZ)}"
    )
    assert cpu.regs.get(RegisterName.FC) == initial_fc, (
        f"Test '{tc.test_id}': Flag C should not change. Initial {initial_fc}, Got {cpu.regs.get(RegisterName.FC)}"
    )

    # --- Verify Memory ---
    if tc.is_dsll:
        for i in range(tc.loop_count_I):
            addr_in_mem = INTERNAL_MEMORY_START + tc.instr_operand_n_val - i
            actual_val = raw_memory_array[addr_in_mem]
            expected_val = tc.expected_final_bcd_logical_bytes[-1 - i]
            assert actual_val == expected_val, (
                f"Test '{tc.test_id}': Memory mismatch at addr 0x{addr_in_mem:X} (logical byte {i}). Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
            )
    else:  # DSRL
        for i in range(tc.loop_count_I):
            addr_in_mem = INTERNAL_MEMORY_START + tc.instr_operand_n_val + i
            actual_val = raw_memory_array[addr_in_mem]
            expected_val = tc.expected_final_bcd_logical_bytes[i]
            assert actual_val == expected_val, (
                f"Test '{tc.test_id}': Memory mismatch at addr 0x{addr_in_mem:X} (logical byte {i}). Expected 0x{expected_val:02X}, Got 0x{actual_val:02X}"
            )


def test_decode_all_opcodes() -> None:
    raw_memory = bytearray([0x00] * ADDRESS_SPACE_SIZE)

    # enumerate all opcodes, want index for each opcode
    for i, (b, s) in enumerate(opcode_generator()):
        if b is None:
            continue

        # Do not let operand bytes from the prior sample become an accidental
        # suffix for a standalone PRE byte (historically PRE21 fused with the
        # stale RETF opcode here).
        raw_memory[:8] = bytes(8)

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
            "EXL",
            "DSRL",
            "DSLL",
            "WAIT",
            "TCL",  # requires the timer-phase hook supplied by focused tests
            "PRE",  # standalone prefixes are deliberately not executable
            # Skip indirect addressing instructions that require proper memory setup
            "[(",  # Indirect addressing through internal memory
        ]
        for ignore in ignore_instructions:
            if s and s.startswith(ignore):
                skip = True
                break
        # Also skip indirect addressing patterns
        if s and "[(" in s:
            skip = True
        if skip:
            continue

        memory = Memory(read_mem, write_mem)
        setattr(memory, "wait_cycles", lambda _cycles: None)
        setattr(
            memory,
            "peek_byte_for_preflight",
            lambda addr, _pc=None: raw_memory[addr],
        )
        # Opcode enumeration is decoder/executor coverage, not a power-on
        # model test; avoid applying RESET to its synthetic memory image.
        cpu = Emulator(memory, reset_on_init=False)

        address = 0x00
        cpu.regs.set(RegisterName.S, 0x1000)  # Set stack pointer to a valid location
        cpu.regs.set(RegisterName.U, 0x2000)  # Set stack pointer to a valid location

        cpu.regs.set(
            RegisterName.X, 0x1000
        )  # Set X to larger value to avoid negative addresses

        try:
            _ = cpu.execute_instruction(address)
        except Exception as e:
            debug_instruction(cpu, address)
            raise ValueError(f"Failed to evaluate {s} at line {i + 1}") from e
