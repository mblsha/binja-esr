from __future__ import annotations

from .. import backend_pyemu, from_decoded
from ...decoding.bind import DecodedInstr
from ...pysc62015.constants import PC_MASK


def execute_build(state, bus, build, *, advance_pc: bool = True) -> None:
    """Execute a prepared SCIL instruction against the provided state/bus."""

    start_pc = state.pc & PC_MASK
    backend_pyemu.step(
        state,
        bus,
        build.instr,
        binder=build.binder,
        pre_latch=build.pre_applied,
    )
    if advance_pc and state.pc == start_pc:
        state.pc = (start_pc + build.instr.length) & PC_MASK


def execute_decoded(state, bus, decoded: DecodedInstr, *, advance_pc: bool = True):
    """Build SCIL payload from a decoded instruction and execute it."""

    build = from_decoded.build(decoded)
    execute_build(state, bus, build, advance_pc=advance_pc)
