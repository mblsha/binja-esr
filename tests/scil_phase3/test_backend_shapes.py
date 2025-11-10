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
    result = from_decoded.build(decoded)
    il = MockLowLevelILFunction()
    emit_llil(
        il,
        result.instr,
        result.binder,
        CompatLLILBuilder(il),
        addr,
        result.pre_applied,
    )
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


def test_add_shape_matches() -> None:
    _assert_same("4055", 0x1000)


def test_adc_shape_matches() -> None:
    _assert_same("5055", 0x1000)


def test_sbc_shape_matches() -> None:
    _assert_same("5855", 0x1000)


def test_and_shape_matches() -> None:
    _assert_same("7055", 0x1000)


def test_or_shape_matches() -> None:
    _assert_same("7855", 0x1000)


def test_xor_shape_matches() -> None:
    _assert_same("6855", 0x1000)


def test_test_shape_matches() -> None:
    _assert_same("6455", 0x1000)


def test_jrnz_shape_matches() -> None:
    _assert_same("1AF0", 0x2000)


def test_jrnc_shape_matches() -> None:
    _assert_same("1EF0", 0x2000)


def test_jpz_shape_matches() -> None:
    _assert_same("1455AA", 0x34567)


def test_ext_store_shape_matches() -> None:
    _assert_same("A8123400", 0x1000)


def test_imem_load_shape_matches() -> None:
    _assert_same("8034", 0x2000)


def test_imem_store_shape_matches() -> None:
    _assert_same("A034", 0x2000)


def test_inc_shape_matches() -> None:
    _assert_same("6C00", 0x3000)


def test_dec_shape_matches() -> None:
    _assert_same("7C00", 0x3000)


def test_jp_imem_shape_matches() -> None:
    _assert_same("1020", 0x5000)


def test_jp_r3_shape_matches() -> None:
    _assert_same("1104", 0x6000)


def test_call_shape_matches() -> None:
    _assert_same("041234", 0x34567)


def test_callf_shape_matches() -> None:
    _assert_same("05123400", 0x12345)


def test_ret_shape_matches() -> None:
    _assert_same("06", 0x2000)


def test_retf_shape_matches() -> None:
    _assert_same("07", 0x2000)


def test_reti_shape_matches() -> None:
    _assert_same("01", 0x2000)


def test_mvl_imem_shape_matches() -> None:
    _assert_same("CBFEF0", 0x1000)


def test_mvld_imem_shape_matches() -> None:
    _assert_same("CF5150", 0x1200)


def test_adcl_shape_matches() -> None:
    _assert_same("541020", 0x1800)


def test_sbcl_shape_matches() -> None:
    _assert_same("5C1020", 0x1900)


def test_dadl_shape_matches() -> None:
    _assert_same("C41020", 0x1A00)


def test_dsbl_shape_matches() -> None:
    _assert_same("D41020", 0x1B00)


def test_dsll_shape_matches() -> None:
    _assert_same("EC11", 0x1C00)


def test_dsrl_shape_matches() -> None:
    _assert_same("FC10", 0x1D00)


def test_pmdf_imm_shape_matches() -> None:
    _assert_same("471107", 0x1E00)


def test_pmdf_reg_shape_matches() -> None:
    _assert_same("5710", 0x1F00)


def test_halt_shape_matches() -> None:
    _assert_same("DE", 0x2000)


def test_off_shape_matches() -> None:
    _assert_same("DF", 0x2010)


def test_reset_shape_matches() -> None:
    _assert_same("FF", 0x2020)


def test_ir_shape_matches() -> None:
    _assert_same("FE", 0x2030)


def test_wait_shape_matches() -> None:
    _assert_same("EF", 0x2040)


def test_ext_ptr_load_shape_matches() -> None:
    _assert_same("980012", 0x7000)


def test_pushs_f_shape_matches() -> None:
    _assert_same("4F", 0x2000)


def test_pops_f_shape_matches() -> None:
    _assert_same("5F", 0x2000)


def test_pushu_a_shape_matches() -> None:
    _assert_same("28", 0x2000)


def test_popu_a_shape_matches() -> None:
    _assert_same("38", 0x2000)


def test_pushu_ba_shape_matches() -> None:
    _assert_same("2A", 0x2000)


def test_popu_ba_shape_matches() -> None:
    _assert_same("3A", 0x2000)


def test_pushu_f_shape_matches() -> None:
    _assert_same("2E", 0x2000)


def test_popu_f_shape_matches() -> None:
    _assert_same("3E", 0x2000)


def test_pushu_imr_shape_matches() -> None:
    _assert_same("2F", 0x2000)


def test_popu_imr_shape_matches() -> None:
    _assert_same("3F", 0x2000)


def test_ext_ptr_store_shape_matches() -> None:
    _assert_same("B80012", 0x7100)


def test_imem_from_ext_shape_matches() -> None:
    _assert_same("E00412", 0x7200)


def test_ext_from_imem_shape_matches() -> None:
    _assert_same("E80412", 0x7300)


def test_imem_move_shape_matches() -> None:
    _assert_same("C81020", 0x7200)


def test_imem_swap_shape_matches() -> None:
    _assert_same("C00020", 0x7300)
