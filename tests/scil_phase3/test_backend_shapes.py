from binja_test_mocks.mock_llil import MockLowLevelILFunction

import re

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.pysc62015.instr import OPCODES, decode as legacy_decode
from sc62015.scil import from_decoded
from sc62015.scil.backend_llil import emit_llil
from sc62015.scil.compat_builder import CompatLLILBuilder


def _decode_di(hex_bytes: str, addr: int):
    data = bytes.fromhex(hex_bytes)
    ctx = StreamCtx(pc=addr, data=data[1:], base_len=1)
    return decode_map.decode_opcode(data[0], ctx)


def _legacy_il(hex_bytes: str, addr: int) -> MockLowLevelILFunction:
    instr = legacy_decode(bytes.fromhex(hex_bytes), addr, OPCODES)
    il = MockLowLevelILFunction()
    assert instr is not None
    instr.lift(il, addr)
    return il


def _scil_il(hex_bytes: str, addr: int) -> MockLowLevelILFunction:
    decoded = _decode_di(hex_bytes, addr)
    scil_instr, binder = from_decoded.build(decoded)
    il = MockLowLevelILFunction()
    emit_llil(il, scil_instr, binder, CompatLLILBuilder(il), addr)
    return il


def _canonical(nodes):
    out = []
    for node in nodes.ils if hasattr(nodes, "ils") else nodes:
        if getattr(node, "op", "") == "LABEL":
            continue
        text = repr(node)
        text = re.sub(r"0x[0-9a-fA-F]+", "0x?", text)
        out.append(text)
    return out


def _assert_same(hex_bytes: str, addr: int) -> None:
    legacy = _legacy_il(hex_bytes, addr)
    scil = _scil_il(hex_bytes, addr)
    assert _canonical(legacy) == _canonical(scil)


def test_mv_a_n_shape_matches() -> None:
    _assert_same("085A", 0x1000)


def test_jrz_forward_shape_matches() -> None:
    _assert_same("1805", 0x2000)


def test_jp_shape_matches() -> None:
    _assert_same("023412", 0x34567)


def test_mv_a_abs_shape_matches() -> None:
    _assert_same("88341200", 0x1000)
