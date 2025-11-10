from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from binja_test_mocks.eval_llil import Memory, State, evaluate_llil
from binja_test_mocks.mock_llil import (
    MockGoto,
    MockIfExpr,
    MockLabel,
    MockLLIL,
    MockLowLevelILFunction,
)

from sc62015.decoding.compat_il import emit_instruction as emit_legacy_il
from sc62015.decoding.dispatcher import CompatDispatcher
from sc62015.decoding.bind import DecodedInstr
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START, PC_MASK
from sc62015.pysc62015.emulator import Registers
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.scil import from_decoded
from sc62015.scil.backend_llil import emit_llil
from sc62015.scil.compat_builder import CompatLLILBuilder

from .models import ExecutionResult, PropState, Scenario


class PropMemory:
    """Memory adapter that tracks reads and writes for invariants."""

    def __init__(self, internal: Dict[int, int], external: Dict[int, int]) -> None:
        self._internal = dict(internal)
        self._external = dict(external)
        self.log: List[Dict[str, int]] = []
        self.memory = Memory(self._read_byte, self._write_byte)

    @property
    def internal_view(self) -> Dict[int, int]:
        return dict(self._internal)

    @property
    def external_view(self) -> Dict[int, int]:
        return dict(self._external)

    def _space_for(self, addr: int) -> Tuple[str, Dict[int, int], int]:
        if addr >= INTERNAL_MEMORY_START:
            offset = addr - INTERNAL_MEMORY_START
            return "int", self._internal, offset & 0xFFFF
        return "ext", self._external, addr & 0xFFFFF

    def _read_byte(self, addr: int) -> int:
        space, backing, offset = self._space_for(addr)
        value = backing.get(offset, 0)
        self.log.append({"op": 0, "space": 0 if space == "int" else 1, "addr": offset, "value": value})
        return value

    def _write_byte(self, addr: int, value: int) -> None:
        space, backing, offset = self._space_for(addr)
        backing[offset] = value & 0xFF
        self.log.append({"op": 1, "space": 0 if space == "int" else 1, "addr": offset, "value": value & 0xFF})


def _canonicalize(il_func: MockLowLevelILFunction) -> List[str]:
    out: List[str] = []
    for node in il_func.ils:
        if isinstance(node, MockLabel):
            continue
        text = repr(node)
        text = re.sub(r"0x[0-9a-fA-F]+", "0x?", text)
        out.append(text)
    return out


def _build_registers(state: PropState) -> Registers:
    regs = Registers()
    for name, value in state.regs.items():
        regs.set_by_name(name, value)
    for flag, value in state.flags.items():
        regs.set_flag(flag, value & 1)
    return regs


def _snapshot_registers(regs: Registers) -> Tuple[Dict[str, int], Dict[str, int]]:
    reg_names = ["A", "B", "IL", "IH", "BA", "I", "X", "Y", "U", "S", "PC", "F"]
    snapshot = {name: regs.get_by_name(name) for name in reg_names}
    flags = {"Z": regs.get_flag("Z"), "C": regs.get_flag("C")}
    return snapshot, flags


def _execute_il(
    il_func: MockLowLevelILFunction,
    regs: Registers,
    memory: PropMemory,
) -> ExecutionResult:
    state = State()
    label_to_index: Dict[int, int] = {}
    for idx, node in enumerate(il_func.ils):
        if isinstance(node, MockLabel):
            label_to_index[id(node)] = idx
            label_to_index[id(node.label)] = idx
        elif node.__class__.__name__ == "LowLevelILLabel":
            label_to_index[id(node)] = idx

    pc = 0
    while pc < len(il_func.ils):
        node = il_func.ils[pc]
        if isinstance(node, MockLabel):
            pc += 1
            continue
        if isinstance(node, MockIfExpr):
            cond, _ = evaluate_llil(
                node.cond, regs, memory.memory, state, regs.get_flag, regs.set_flag
            )
            assert cond is not None, "IF condition evaluated to None"
            target = node.t if cond else node.f
            if isinstance(target, int):
                pc = target
            else:
                key = id(target)
                assert key in label_to_index, f"Unknown label {target}"
                pc = label_to_index[key]
            continue
        if isinstance(node, MockGoto):
            if isinstance(node.label, int):
                pc = node.label
            else:
                key = id(node.label)
                assert key in label_to_index, f"Unknown goto label {node.label}"
                pc = label_to_index[key]
            continue
        evaluate_llil(node, regs, memory.memory, state, regs.get_flag, regs.set_flag)
        pc += 1

    regs_snapshot, flag_snapshot = _snapshot_registers(regs)
    return ExecutionResult(
        regs=regs_snapshot,
        flags=flag_snapshot,
        internal=memory.internal_view,
        external=memory.external_view,
        mem_log=list(memory.log),
    )


@dataclass
class InstructionRecord:
    addr: int
    data: bytes
    decoded: DecodedInstr


def _decode_sequence(bytes_seq: Sequence[bytes], start_pc: int) -> List[InstructionRecord]:
    dispatcher = CompatDispatcher()
    addr = start_pc
    records: List[InstructionRecord] = []
    for raw in bytes_seq:
        result = dispatcher.try_decode(raw, addr)
        assert result is not None, "Decoder returned None for instruction"
        length, decoded = result
        if decoded is not None:
            records.append(InstructionRecord(addr=addr, data=raw, decoded=decoded))
        addr = (addr + length) & PC_MASK
    return records


def _emit_pair(record: InstructionRecord) -> Tuple[MockLowLevelILFunction, MockLowLevelILFunction]:
    legacy_il = MockLowLevelILFunction()
    emit_legacy_il(record.decoded, legacy_il, record.addr)

    scil_result = from_decoded.build(record.decoded)
    scil_il = MockLowLevelILFunction()
    emit_llil(
        scil_il,
        scil_result.instr,
        scil_result.binder,
        CompatLLILBuilder(scil_il),
        record.addr,
        scil_result.pre_applied,
    )
    return legacy_il, scil_il


def _compare_il_shapes(record: InstructionRecord, legacy_il, scil_il) -> None:
    lhs = _canonicalize(legacy_il)
    rhs = _canonicalize(scil_il)
    assert lhs == rhs, f"IL shape mismatch for {record.decoded.mnemonic} @ {record.addr:#x}"


def _compare_results(left: ExecutionResult, right: ExecutionResult) -> None:
    assert left.regs == right.regs, f"Register mismatch: {left.regs} vs {right.regs}"
    assert left.flags == right.flags, f"Flag mismatch: {left.flags} vs {right.flags}"
    assert left.internal == right.internal, "Internal memory mismatch"
    assert left.external == right.external, "External memory mismatch"


def _check_pc_width(result: ExecutionResult) -> None:
    assert result.regs["PC"] & ~PC_MASK == 0, "PC exceeded 20-bit mask"


def _check_external_width(log: List[Dict[str, int]]) -> None:
    for entry in log:
        if entry["space"] == 1:
            assert 0 <= entry["addr"] < (1 << 20), "External address exceeded 24-bit space"


def _check_branch_invariants(info: Dict[str, int], state: PropState, result: ExecutionResult) -> None:
    disp = info["disp"]
    length = info.get("length", 2)
    fall = (state.regs["PC"] + length) & PC_MASK
    taken = (fall + disp) & PC_MASK
    assert result.regs["PC"] in {fall, taken}, "JR invariant violated"


def _check_paged_jp(info: Dict[str, int], state: PropState, result: ExecutionResult) -> None:
    cond_flag = info.get("cond")
    expect_one = info.get("expect", 1)
    lo16 = info["lo16"]
    fall = (state.regs["PC"] + 3) & PC_MASK
    page = state.regs["PC"] & 0xF0000
    target = (page | lo16) & PC_MASK
    should_take = True
    if cond_flag:
        should_take = state.flags.get(cond_flag, 0) == expect_one
    assert result.regs["PC"] == (target if should_take else fall), "JP invariant violated"


def _check_pointer_update(info: Dict[str, int], state: PropState, result: ExecutionResult) -> None:
    if info.get("dest") == info["ptr"]:
        return
    reg = info["ptr"]
    width = info["width"]
    mode = info["mode"]
    start_val = state.regs[reg]
    end_val = result.regs[reg]
    if mode == 1:  # post-inc
        assert end_val == (start_val + width) & 0xFFFFFF, "Post-inc failed"
    elif mode == 2:  # pre-dec
        assert end_val == (start_val - width) & 0xFFFFFF, "Pre-dec failed"
    else:
        assert end_val == start_val, "Pointer should remain unchanged"


def _check_ptr_from_imem(info: Dict[str, int], state: PropState, result: ExecutionResult, log: List[Dict[str, int]]) -> None:
    base = info["base"]
    bp_val = state.internal.get(IMEMRegisters["BP"], 0) & 0xFF
    start = (bp_val + base) & 0xFF
    mode = info["mode"]
    disp = info.get("disp", 0)
    lo = state.internal.get(start & 0xFF, 0)
    mid = state.internal.get((start + 1) & 0xFF, 0)
    hi = state.internal.get((start + 2) & 0xFF, 0)
    ptr = (hi << 16) | (mid << 8) | lo
    if mode == 0:
        return
    if mode == 1:
        ptr = (ptr + disp) & 0xFFFFFF
    elif mode == 2:
        ptr = (ptr - disp) & 0xFFFFFF
    accessed = [entry for entry in log if entry["space"] == 1]
    if not accessed:
        return
    assert any(entry["addr"] == (ptr & 0xFFFFF) for entry in accessed), "Pointer target not accessed"


def _check_imem_ext_transfer(info: Dict[str, int], state: PropState, log: List[Dict[str, int]], expect_internal_write: bool) -> None:
    offset = info["imem"]
    width = info.get("width", 1)
    bp_val = state.internal.get(IMEMRegisters["BP"], 0) & 0xFF
    base = (bp_val + offset) & 0xFF
    addresses = {((base + delta) & 0xFF) for delta in range(max(1, width))}
    op_kind = 1 if expect_internal_write else 0
    touched = {
        entry["addr"]
        for entry in log
        if entry["space"] == 0 and entry["op"] == op_kind
    }
    debug = [
        entry for entry in log if entry["space"] == 0
    ]
    if expect_internal_write:
        assert addresses & touched, f"Expected internal write missing (need {addresses}, saw {touched}, log={debug})"
    else:
        assert addresses & touched, f"Expected internal read missing (need {addresses}, saw {touched}, log={debug})"


def _check_jp_reg(info: Dict[str, int], state: PropState, result: ExecutionResult) -> None:
    reg = info["reg"]
    start = state.regs[reg]
    expected = start & PC_MASK
    assert result.regs["PC"] == expected, "JP r3 invariant violated"


def _check_jp_imem(info: Dict[str, int], state: PropState, result: ExecutionResult) -> None:
    offset = info["offset"]
    bp_val = state.internal.get(IMEMRegisters["BP"], 0) & 0xFF
    start = (bp_val + offset) & 0xFF
    lo = state.internal.get(start & 0xFF, 0)
    mid = state.internal.get((start + 1) & 0xFF, 0)
    hi = state.internal.get((start + 2) & 0xFF, 0)
    pointer = ((hi << 16) | (mid << 8) | lo) & PC_MASK
    assert result.regs["PC"] == pointer, "JP (n) invariant violated"


def _check_invariants(
    scenario: Scenario,
    initial_state: PropState,
    result: ExecutionResult,
    mem_log: List[Dict[str, int]],
    records: List[InstructionRecord],
) -> None:
    _check_pc_width(result)
    _check_external_width(mem_log)
    info = scenario.info
    family = scenario.family
    if family == "jr_rel":
        _check_branch_invariants(info, initial_state, result)
    elif family == "jp_paged":
        _check_paged_jp(info, initial_state, result)
    elif family.startswith("r3_"):
        _check_pointer_update(info, initial_state, result)
    elif family == "imem_ptr":
        _check_ptr_from_imem(info, initial_state, result, mem_log)
    elif family == "imem_from_ext":
        _check_imem_ext_transfer(info, initial_state, mem_log, expect_internal_write=True)
    elif family == "ext_from_imem":
        _check_imem_ext_transfer(info, initial_state, mem_log, expect_internal_write=False)
    elif family == "jp_reg":
        _check_jp_reg(info, initial_state, result)
    elif family == "jp_imem":
        _check_jp_imem(info, initial_state, result)
    elif family == "pre_single":
        assert len(records) >= 2, "PRE scenario requires at least two records"
        assert records[0].decoded.pre_applied is not None, "First consumer must see PRE"
        assert records[1].decoded.pre_applied is None, "PRE should apply only once"


def _state_from_result(result: ExecutionResult) -> PropState:
    return PropState(
        regs=dict(result.regs),
        flags=dict(result.flags),
        internal=dict(result.internal),
        external=dict(result.external),
    )


def run_scenario(scenario: Scenario, state: PropState) -> None:
    initial_state = state.clone()
    records = _decode_sequence(scenario.bytes_seq, initial_state.regs["PC"])
    assert records, "Scenario produced no executable instruction"

    current_state = initial_state.clone()
    last_result: ExecutionResult | None = None
    mem_log: List[Dict[str, int]] = []

    for record in records:
        legacy_il, scil_il = _emit_pair(record)
        _compare_il_shapes(record, legacy_il, scil_il)

        regs_left = _build_registers(current_state)
        regs_right = _build_registers(current_state)
        fallthrough = (record.addr + record.decoded.length) & PC_MASK
        regs_left.set_by_name("PC", fallthrough)
        regs_right.set_by_name("PC", fallthrough)
        mem_left = PropMemory(current_state.internal, current_state.external)
        mem_right = PropMemory(current_state.internal, current_state.external)

        left_result = _execute_il(legacy_il, regs_left, mem_left)
        right_result = _execute_il(scil_il, regs_right, mem_right)
        _compare_results(left_result, right_result)

        current_state = _state_from_result(left_result)
        last_result = left_result
        mem_log = left_result.mem_log

    assert last_result is not None
    _check_invariants(scenario, initial_state, last_result, mem_log, records)

    corpus_dir = os.getenv("BN_PROP_SAVE_FAIL")
    if corpus_dir:
        os.makedirs(corpus_dir, exist_ok=True)
        fname = os.path.join(
            corpus_dir,
            f"{scenario.family}_{records[0].decoded.mnemonic}_{initial_state.regs['PC']:05X}.seed",
        )
        with open(fname, "wb") as fp:
            for chunk in scenario.bytes_seq:
                fp.write(chunk)
