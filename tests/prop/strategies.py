from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from hypothesis import strategies as st

from sc62015.decoding.decode_map import PRE_LATCHES
from sc62015.pysc62015.constants import PC_MASK
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from .models import PropState, Scenario

REG_LIMITS: Dict[str, int] = {
    "A": 0xFF,
    "B": 0xFF,
    "IL": 0xFF,
    "IH": 0xFF,
    "BA": 0xFFFF,
    "I": 0xFFFF,
    "X": 0xFFFFFF,
    "Y": 0xFFFFFF,
    "U": 0xFFFFFF,
    "S": 0xFFFFFF,
    "PC": PC_MASK,
    "F": 0xFF,
}

FLAG_NAMES = ["Z", "C"]

PTR_REG_CODES = {"X": 0x4, "Y": 0x5, "U": 0x6, "S": 0x7}

EXT_LOAD_SPECS = [
    (0x90, "A", 1),
    (0x91, "IL", 1),
    (0x92, "BA", 2),
    (0x93, "I", 2),
    (0x94, "X", 3),
    (0x95, "Y", 3),
    (0x96, "U", 3),
]

EXT_STORE_SPECS = [
    (0xB0, "A", 1),
    (0xB1, "IL", 1),
    (0xB2, "BA", 2),
    (0xB3, "I", 2),
    (0xB4, "X", 3),
    (0xB5, "Y", 3),
    (0xB6, "U", 3),
]

EXT_PTR_LOAD_SPECS = [
    (0x98, "A", 1),
    (0x99, "IL", 1),
    (0x9A, "BA", 2),
    (0x9B, "I", 2),
    (0x9C, "X", 3),
    (0x9D, "Y", 3),
    (0x9E, "U", 3),
]

EXT_PTR_STORE_SPECS = [
    (0xB8, "A", 1),
    (0xB9, "IL", 1),
    (0xBA, "BA", 2),
    (0xBB, "I", 2),
    (0xBC, "X", 3),
    (0xBD, "Y", 3),
    (0xBE, "U", 3),
]

IMEM_FROM_EXT_SPECS = [
    (0xE0, 1),
    (0xE1, 2),
    (0xE2, 3),
]

EXT_FROM_IMEM_SPECS = [
    (0xE8, 1),
    (0xE9, 2),
    (0xEA, 3),
]

JR_SPEC = {
    "Z": (0x18, 0x19),
    "NZ": (0x1A, 0x1B),
    "C": (0x1C, 0x1D),
    "NC": (0x1E, 0x1F),
}

JP_SPEC = {
    None: 0x02,
    "Z": 0x14,
    "NZ": 0x15,
    "C": 0x16,
    "NC": 0x17,
}


def cpu_states() -> st.SearchStrategy[PropState]:
    reg_strategy = st.fixed_dictionaries(
        {name: st.integers(0, limit) for name, limit in REG_LIMITS.items()}
    )
    flag_strategy = st.fixed_dictionaries({flag: st.integers(0, 1) for flag in FLAG_NAMES})
    internal_bytes = st.binary(min_size=256, max_size=256).map(
        lambda blob: {idx: blob[idx] for idx in range(256)}
    )
    external_mem = st.dictionaries(
        st.integers(0, (1 << 20) - 1),
        st.integers(0, 0xFF),
        max_size=32,
    )
    return st.builds(
        PropState,
        regs=reg_strategy,
        flags=flag_strategy,
        internal=internal_bytes,
        external=external_mem,
    )


def _scenario(
    seq: Sequence[bytes],
    family: str,
    info: Dict[str, int],
    *,
    state_overrides: Dict[str, int] | None = None,
    description: str | None = None,
) -> Scenario:
    return Scenario(
        bytes_seq=list(seq),
        family=family,
        info=info,
        description=description or family,
        state_overrides=state_overrides,
    )


@st.composite
def _imm8_scenarios(draw):
    opcode = draw(
        st.sampled_from(
            [0x08, 0x40, 0x48, 0x50, 0x58, 0x64, 0x68, 0x70, 0x78]
        )
    )
    imm = draw(st.integers(0, 0xFF))
    return _scenario([bytes([opcode, imm])], "imm8", {"opcode": opcode})


@st.composite
def _jr_scenarios(draw):
    cond = draw(st.sampled_from(list(JR_SPEC.keys())))
    disp = draw(st.integers(-120, 120))
    opcode_pos, opcode_neg = JR_SPEC[cond]
    if disp >= 0:
        opcode = opcode_pos
        mag = disp & 0xFF
    else:
        opcode = opcode_neg
        mag = (-disp) & 0xFF
    info = {"disp": disp, "length": 2}
    return _scenario([bytes([opcode, mag])], "jr_rel", info)


@st.composite
def _jp_scenarios(draw):
    cond = draw(st.sampled_from(list(JP_SPEC.keys())))
    opcode = JP_SPEC[cond]
    value = draw(st.integers(0, 0xFFFF))
    lo = value & 0xFF
    hi = (value >> 8) & 0xFF
    info = {"lo16": value}
    if cond:
        flag = "Z" if "Z" in cond else "C"
        info["cond"] = flag
        info["expect"] = 0 if cond.startswith("N") else 1
    return _scenario([bytes([opcode, lo, hi])], "jp_paged", info)


@st.composite
def _jp_reg_scenarios(draw):
    reg = draw(st.sampled_from(list(PTR_REG_CODES.keys())))
    opcode = 0x11
    reg_byte = PTR_REG_CODES[reg]
    info = {"reg": reg}
    return _scenario([bytes([opcode, reg_byte])], "jp_reg", info)


@st.composite
def _ext_abs_scenarios(draw):
    opcode = draw(st.sampled_from([0x88, 0xA8]))
    addr = draw(st.integers(0, 0xFFFFFF))
    lo = addr & 0xFF
    mid = (addr >> 8) & 0xFF
    hi = (addr >> 16) & 0xFF
    family = "ext_load" if opcode == 0x88 else "ext_store"
    return _scenario([bytes([opcode, lo, mid, hi])], family, {"addr": addr})


def _encode_ext_ptr(ptr_name: str, mode: str, disp: int | None = None) -> Tuple[int, List[int]]:
    mode_codes = {"simple": 0x0, "post": 0x2, "pre": 0x3, "pos": 0x8, "neg": 0xC}
    reg_byte = (mode_codes[mode] << 4) | PTR_REG_CODES[ptr_name]
    bytes_out = [reg_byte]
    if mode in {"pos", "neg"} and disp is not None:
        bytes_out.append(disp & 0xFF)
    return reg_byte, bytes_out


@st.composite
def _ext_reg_scenarios(draw):
    spec = draw(st.sampled_from(EXT_LOAD_SPECS + EXT_STORE_SPECS))
    opcode, reg_name, width = spec
    ptr = draw(st.sampled_from(list(PTR_REG_CODES.keys())))
    mode = draw(st.sampled_from(["simple", "post", "pre", "pos", "neg"]))
    disp = draw(st.integers(1, 8)) if mode in {"pos", "neg"} else None
    _, operand_bytes = _encode_ext_ptr(ptr, mode, disp)
    info = {
        "ptr": ptr,
        "dest": reg_name,
        "width": width,
        "mode": {"simple": 0, "post": 1, "pre": 2, "pos": 3, "neg": 3}[mode],
    }
    if disp is not None:
        signed = disp if mode == "pos" else -disp
        info["disp"] = signed
    return _scenario([bytes([opcode] + operand_bytes)], "r3_simple" if mode == "simple" else f"r3_{mode}", info)


@st.composite
def _ext_ptr_scenarios(draw):
    spec = draw(st.sampled_from(EXT_PTR_LOAD_SPECS + EXT_PTR_STORE_SPECS))
    opcode, _, _ = spec
    base = draw(st.integers(0, 0xFF))
    mode = draw(st.sampled_from(["simple", "pos", "neg"]))
    disp = draw(st.integers(1, 8)) if mode != "simple" else None
    mode_byte = {"simple": 0x00, "pos": 0x80, "neg": 0xC0}[mode]
    seq = [opcode, mode_byte, base]
    info = {"base": base, "mode": {"simple": 0, "pos": 1, "neg": 2}[mode]}
    if disp is not None:
        seq.append(disp & 0xFF)
        info["disp"] = disp if mode == "pos" else -disp
    return _scenario([bytes(seq)], "imem_ptr", info)


@st.composite
def _imem_move_scenarios(draw):
    opcode = draw(st.sampled_from([0xC8, 0xC9, 0xCA]))
    dst = draw(st.integers(0, 0xFF))
    src = draw(st.integers(0, 0xFF))
    return _scenario([bytes([opcode, dst, src])], "imem_move", {"dst": dst, "src": src})


@st.composite
def _imem_swap_scenarios(draw):
    opcode = draw(st.sampled_from([0xC0, 0xC1, 0xC2]))
    left = draw(st.integers(0, 0xFF))
    right = draw(st.integers(0, 0xFF))
    return _scenario([bytes([opcode, left, right])], "imem_swap", {"left": left, "right": right})


@st.composite
def _imem_from_ext_scenarios(draw):
    opcode, width = draw(st.sampled_from(IMEM_FROM_EXT_SPECS))
    ptr = draw(st.sampled_from(list(PTR_REG_CODES.keys())))
    mode = draw(st.sampled_from(["simple", "post", "pre"]))
    reg_byte = ({"simple": 0x0, "post": 0x2, "pre": 0x3}[mode] << 4) | PTR_REG_CODES[ptr]
    seq = [opcode, reg_byte]
    offset = draw(st.integers(0, 0xFF))
    seq.append(offset)
    info = {"ptr": ptr, "width": width, "mode": {"simple": 0, "post": 1, "pre": 2}[mode], "imem": offset}
    return _scenario([bytes(seq)], "imem_from_ext", info)


@st.composite
def _ext_from_imem_scenarios(draw):
    opcode, width = draw(st.sampled_from(EXT_FROM_IMEM_SPECS))
    ptr = draw(st.sampled_from(list(PTR_REG_CODES.keys())))
    mode = draw(st.sampled_from(["simple", "post", "pre"]))
    reg_byte = ({"simple": 0x0, "post": 0x2, "pre": 0x3}[mode] << 4) | PTR_REG_CODES[ptr]
    seq = [opcode, reg_byte]
    offset = draw(st.integers(0, 0xFF))
    seq.append(offset)
    info = {"ptr": ptr, "width": width, "mode": {"simple": 0, "post": 1, "pre": 2}[mode], "imem": offset}
    return _scenario([bytes(seq)], "ext_from_imem", info)


@st.composite
def _pre_scenarios(draw):
    pre_opcode = draw(st.sampled_from(list(PRE_LATCHES.keys())))
    op1 = draw(st.integers(0, 0xFF))
    op2 = draw(st.integers(0, 0xFF))
    seq = [bytes([pre_opcode]), bytes([0x80, op1]), bytes([0x80, op2])]
    return Scenario(bytes_seq=seq, family="pre_single", info={}, description="pre_single", expect_pre_sequence=True)


@st.composite
def _loop_move_scenarios(draw):
    opcode = draw(st.sampled_from([0xCB, 0xCF]))
    dst = draw(st.integers(0, 0xFF))
    src = draw(st.integers(0, 0xFF))
    count = draw(st.integers(0, 4))
    info = {
        "dst": dst,
        "src": src,
        "loop_count": count,
        "direction": 1 if opcode == 0xCB else -1,
    }
    overrides = {"I": count}
    return _scenario([bytes([opcode, dst, src])], "loop_move", info, state_overrides=overrides)


LOOP_ARITH_MEM_SPECS: Sequence[Tuple[int, str]] = (
    (0x54, "loop_add"),
    (0x5C, "loop_sub"),
)


@st.composite
def _loop_arith_mem_scenarios(draw):
    opcode, family = draw(st.sampled_from(LOOP_ARITH_MEM_SPECS))
    dst = draw(st.integers(0, 0xFF))
    src = draw(st.integers(0, 0xFF))
    count = draw(st.integers(0, 4))
    info = {"dst": dst, "src": src, "loop_count": count}
    overrides = {"I": count}
    return _scenario([bytes([opcode, dst, src])], family, info, state_overrides=overrides)


BCD_LOOP_SPECS: Sequence[Tuple[int, str]] = (
    (0xC4, "loop_bcd_add"),
    (0xD4, "loop_bcd_sub"),
)


@st.composite
def _loop_bcd_scenarios(draw):
    opcode, family = draw(st.sampled_from(BCD_LOOP_SPECS))
    dst = draw(st.integers(0, 0xFF))
    src = draw(st.integers(0, 0xFF))
    count = draw(st.integers(0, 4))
    info = {"dst": dst, "src": src, "loop_count": count}
    overrides = {"I": count}
    return _scenario([bytes([opcode, dst, src])], family, info, state_overrides=overrides)


DECIMAL_SHIFT_SPECS: Sequence[Tuple[int, str]] = (
    (0xEC, "left"),
    (0xFC, "right"),
)


@st.composite
def _decimal_shift_scenarios(draw):
    opcode, direction = draw(st.sampled_from(DECIMAL_SHIFT_SPECS))
    base = draw(st.integers(0, 0xFF))
    count = draw(st.integers(0, 4))
    info = {"base": base, "direction": direction, "loop_count": count}
    overrides = {"I": count}
    return _scenario([bytes([opcode, base])], "decimal_shift", info, state_overrides=overrides)


@st.composite
def _pmdf_scenarios(draw):
    addr = draw(st.integers(0, 0xFF))
    imm = draw(st.integers(0, 0xFF))
    return _scenario([bytes([0x47, addr, imm])], "pmdf", {"addr": addr, "imm": imm})


SYSTEM_SPECS: Sequence[Tuple[int, str]] = (
    (0xDE, "system_intrinsic"),
    (0xDF, "system_intrinsic"),
    (0xEF, "system_wait"),
    (0xFE, "system_intrinsic"),
    (0xFF, "system_intrinsic"),
)


@st.composite
def _system_scenarios(draw):
    opcode, family = draw(st.sampled_from(SYSTEM_SPECS))
    info = {"length": 1}
    return _scenario([bytes([opcode])], family, info)


def instruction_scenarios() -> st.SearchStrategy[Scenario]:
    return st.one_of(
        _imm8_scenarios(),
        _jr_scenarios(),
        _jp_scenarios(),
        _jp_reg_scenarios(),
        _ext_abs_scenarios(),
        _ext_reg_scenarios(),
        _ext_ptr_scenarios(),
        _imem_move_scenarios(),
        _imem_swap_scenarios(),
        _imem_from_ext_scenarios(),
        _ext_from_imem_scenarios(),
        _pre_scenarios(),
        _loop_move_scenarios(),
        _loop_arith_mem_scenarios(),
        _loop_bcd_scenarios(),
        _decimal_shift_scenarios(),
        _pmdf_scenarios(),
        _system_scenarios(),
    )
