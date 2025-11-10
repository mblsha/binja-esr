from __future__ import annotations

from binaryninja.lowlevelil import LowLevelILFunction  # type: ignore

from ..scil.compat_builder import CompatLLILBuilder
from ..scil import from_decoded
from ..scil.backend_llil import emit_llil as emit_scil_llil
from .bind import DecodedInstr


def emit_instruction(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    """Legacy helper retained for tests; now always routes through SCIL."""
    payload = from_decoded.build(di)
    emit_scil_llil(
        il,
        payload.instr,
        payload.binder,
        CompatLLILBuilder(il),
        addr,
        pre_applied=payload.pre_applied,
    )
