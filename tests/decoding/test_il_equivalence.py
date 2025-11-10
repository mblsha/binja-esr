import re

from binja_test_mocks.mock_llil import MockLowLevelILFunction

from sc62015.decoding import decode_map
from sc62015.decoding.compat_il import emit_instruction
from sc62015.decoding.reader import StreamCtx
from sc62015.pysc62015.instr import OPCODES, decode as legacy_decode


def _canonical(nodes):
    result = []
    for node in nodes:
        text = repr(node)
        text = re.sub(r"object at 0x[0-9a-fA-F]+", "object at 0x??", text)
        result.append(text)
    return result


def _legacy_il(hex_bytes: str, addr: int) -> MockLowLevelILFunction:
    data = bytes.fromhex(hex_bytes)
    instr = legacy_decode(data, addr, OPCODES)
    assert instr is not None
    il = MockLowLevelILFunction()
    instr.lift(il, addr)
    return il


def _compat_il(hex_bytes: str, addr: int) -> MockLowLevelILFunction:
    data = bytes.fromhex(hex_bytes)
    opcode = data[0]
    ctx = StreamCtx(pc=addr, data=data[1:], base_len=1)
    di = decode_map.decode_opcode(opcode, ctx)
    il = MockLowLevelILFunction()
    emit_instruction(di, il, addr)
    return il


def _assert_same_il(hex_bytes: str, addr: int = 0x1000) -> None:
    legacy = _legacy_il(hex_bytes, addr)
    compat = _compat_il(hex_bytes, addr)
    assert _canonical(legacy.ils) == _canonical(compat.ils)


def test_mv_a_n_il_matches() -> None:
    _assert_same_il("085A")


def test_jrz_forward_il_matches() -> None:
    _assert_same_il("1805", 0x2000)


def test_jrz_backward_il_matches() -> None:
    _assert_same_il("1903", 0x2000)


def test_jp_mn_il_matches() -> None:
    _assert_same_il("023412", 0x34567)


def test_mv_a_abs24_il_matches() -> None:
    _assert_same_il("88341200")
