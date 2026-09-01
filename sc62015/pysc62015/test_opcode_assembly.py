import re

import pytest
from binja_test_mocks.coding import Decoder, Encoder
from binja_test_mocks.tokens import asm_str

from .instr import decode as decode_instr
from .instr import (
    ADD,
    EMemReg,
    EMemRegMode,
    IMem8,
    Imm20,
    InvalidInstruction,
    MVL,
    Reg,
    Reg3,
    RegIMemOffset,
    RegIMemOffsetOrder,
    RegPair,
    encode,
)
from .instr.opcode_table import OPCODES
from .sc_asm import Assembler, AssemblerError
from .test_instr import decode, opcode_generator


REGS = {
    "A",
    "B",
    "IL",
    "IH",
    "I",
    "BA",
    "X",
    "Y",
    "U",
    "S",
    "PC",
    "IMR",
    "F",
    "FC",
    "FZ",
}
PATTERN = re.compile(r"(?<![A-Za-z0-9])([+-]?)([A-F0-9]+)(?![A-Za-z0-9])")


def _transform(instr: str) -> str:
    instr = instr.replace("(FB)", "IMR")
    parts = instr.split(maxsplit=1)
    mnemonic = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    def repl(match: re.Match[str]) -> str:
        sign, token = match.groups()
        if token.isalpha():
            # A hexadecimal value can look like a register name inside an
            # addressing expression. Preserve only genuine register tokens.
            start = match.start()
            token_start = start + len(sign)
            prev = rest[token_start - 1] if token_start > 0 else ""
            if token in REGS and prev not in {"(", "+", "-"}:
                return match.group(0)
        return f"{sign}0x{token}"

    rest = PATTERN.sub(repl, rest)
    return f"{mnemonic} {rest}".strip()


def test_opcode_table_semantic_roundtrip_is_canonical() -> None:
    """Every printable sample must round-trip to one stable canonical encoding.

    Raw opcode samples intentionally include redundant PRE encodings and RegPair
    byte aliases, so text is not injective over the original bytes. The honest
    contract is semantic equality followed by exact idempotence of the
    assembler's canonical result.
    """

    assembler = Assembler()
    mismatches: list[str] = []

    for idx, (sample_bytes, asm_text) in enumerate(opcode_generator()):
        if sample_bytes is None or asm_text is None:
            continue
        if asm_text.startswith("PRE") or asm_text.startswith("???"):
            continue

        try:
            sample_instr = decode(sample_bytes, 0)
            source = _transform(asm_str(sample_instr.render()))
            canonical = assembler.assemble(source).as_binary()
            canonical_instr = decode(canonical, 0)
            canonical_source = _transform(asm_str(canonical_instr.render()))
            canonical_again = assembler.assemble(canonical_source).as_binary()
        except Exception as exc:
            mismatches.append(
                f"Line {idx + 1}: {asm_text} ({sample_bytes.hex()}) -> "
                f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"
            )
            continue

        if sample_instr.render() != canonical_instr.render():
            mismatches.append(
                f"Line {idx + 1}: semantic change: {sample_bytes.hex()} -> "
                f"{canonical.hex()} ({source})"
            )
        elif canonical_again != canonical:
            mismatches.append(
                f"Line {idx + 1}: unstable canonical bytes: {canonical.hex()} -> "
                f"{canonical_again.hex()} ({canonical_source})"
            )

    if mismatches:
        pytest.fail("Opcode table round-trip divergence:\n" + "\n".join(mismatches))


def test_hw009_far_control_alias_decodes_but_assembles_canonically() -> None:
    canonical = Assembler().assemble("JPF 0xFCDAB").as_binary()
    assert canonical == bytearray.fromhex("03ABCD0F")
    assert asm_str(decode(canonical, 0).render()) == "JPF   FCDAB"

    # PC-E500 HW-009 executed all 16 encoded upper nibbles for both JPF and
    # CALLF. Every variant reached the same low-20-bit target.
    for opcode, mnemonic in ((0x03, "JPF"), (0x05, "CALLF")):
        for upper_nibble in range(16):
            raw_alias = bytearray((opcode, 0xE0, 0x01, (upper_nibble << 4) | 0x01))
            decoded = decode_instr(Decoder(raw_alias), 0, OPCODES)
            assert decoded is not None
            assert asm_str(decoded.render()) == f"{mnemonic:<6}101E0"

            encoder = Encoder()
            decoded.encode(encoder, 0)
            assert encoder.buf == bytearray((opcode, 0xE0, 0x01, 0x01))

    with pytest.raises(AssemblerError, match="20-bit immediate out of range"):
        Assembler().assemble("CALLF 0xF101E0")

    # Generic/unmeasured Imm20 users remain strict.
    raw_alias = Imm20()
    raw_alias.value = 0xC073A
    raw_alias.extra_hi = 0x7C
    with pytest.raises(ValueError, match="high byte out of range"):
        raw_alias.encode(Encoder(), 0)


@pytest.mark.parametrize(
    ("opcode", "register"),
    [
        (0x88, "A"),
        (0x89, "IL"),
        (0x8A, "BA"),
        (0x8B, "I"),
        (0x8C, "X"),
        (0x8D, "Y"),
        (0x8E, "U"),
        (0x8F, "S"),
    ],
)
def test_hw009_direct_read_alias_decodes_but_assembles_canonically(
    opcode: int, register: str
) -> None:
    raw_alias = bytearray((opcode, 0xF0, 0x01, 0x81))
    decoded = decode_instr(Decoder(raw_alias), 0, OPCODES)
    assert decoded is not None
    assert asm_str(decoded.render()) == f"MV    {register}, [101F0]"

    encoder = Encoder()
    decoded.encode(encoder, 0)
    assert encoder.buf == bytearray((opcode, 0xF0, 0x01, 0x01))
    assert Assembler().assemble(f"MV {register},[0x101F0]").as_binary() == bytearray(
        (opcode, 0xF0, 0x01, 0x01)
    )
    with pytest.raises(AssemblerError, match="20-bit immediate out of range"):
        Assembler().assemble(f"MV {register},[0x8101F0]")


@pytest.mark.parametrize(
    ("opcode", "mnemonic", "immediate"),
    [
        (0x62, "CMP", 0x00),
        (0x66, "TEST", 0xFF),
        (0x6A, "XOR", 0x00),
        (0x72, "AND", 0xFF),
        (0x7A, "OR", 0x00),
    ],
)
def test_hw009_absolute_byte_alias_decodes_but_assembles_canonically(
    opcode: int, mnemonic: str, immediate: int
) -> None:
    raw_alias = bytearray((opcode, 0xD0, 0x06, 0x84, immediate))
    decoded = decode_instr(Decoder(raw_alias), 0, OPCODES)
    assert decoded is not None
    assert asm_str(decoded.render()) == f"{mnemonic:<6}[406D0], {immediate:02X}"

    encoder = Encoder()
    decoded.encode(encoder, 0)
    assert encoder.buf == bytearray((opcode, 0xD0, 0x06, 0x04, immediate))
    assert Assembler().assemble(
        f"{mnemonic} [0x406D0],0x{immediate:02X}"
    ).as_binary() == bytearray((opcode, 0xD0, 0x06, 0x04, immediate))
    with pytest.raises(AssemblerError, match="20-bit immediate out of range"):
        Assembler().assemble(f"{mnemonic} [0x8406D0],0x{immediate:02X}")


@pytest.mark.parametrize(
    ("opcode", "register"),
    [
        (0xA8, "A"),
        (0xA9, "IL"),
        (0xAA, "BA"),
        (0xAB, "I"),
        (0xAC, "X"),
        (0xAD, "Y"),
        (0xAE, "U"),
        (0xAF, "S"),
    ],
)
def test_hw009_direct_write_alias_decodes_but_assembles_canonically(
    opcode: int, register: str
) -> None:
    decoded = decode_instr(Decoder(bytearray((opcode, 0xD0, 0x06, 0x84))), 0, OPCODES)
    assert decoded is not None
    assert asm_str(decoded.render()) == f"MV    [406D0], {register}"

    encoder = Encoder()
    decoded.encode(encoder, 0)
    assert encoder.buf == bytearray((opcode, 0xD0, 0x06, 0x04))


@pytest.mark.parametrize(
    ("opcode", "mnemonic", "to_external"),
    [
        (0xD0, "MV", False),
        (0xD1, "MVW", False),
        (0xD2, "MVP", False),
        (0xD3, "MVL", False),
        (0xD8, "MV", True),
        (0xD9, "MVW", True),
        (0xDA, "MVP", True),
        (0xDB, "MVL", True),
    ],
)
def test_hw009_absolute_transfer_alias_decodes_but_assembles_canonically(
    opcode: int, mnemonic: str, to_external: bool
) -> None:
    if to_external:
        raw_alias = bytearray((opcode, 0xD0, 0x06, 0x84, 0x60))
        canonical = bytearray((opcode, 0xD0, 0x06, 0x04, 0x60))
        rendered = f"{mnemonic:<6}[406D0], (BP+60)"
        source = f"{mnemonic} [0x406D0],(BP+0x60)"
        invalid_source = f"{mnemonic} [0x8406D0],(BP+0x60)"
    else:
        raw_alias = bytearray((opcode, 0x60, 0xF0, 0x01, 0x81))
        canonical = bytearray((opcode, 0x60, 0xF0, 0x01, 0x01))
        rendered = f"{mnemonic:<6}(BP+60), [101F0]"
        source = f"{mnemonic} (BP+0x60),[0x101F0]"
        invalid_source = f"{mnemonic} (BP+0x60),[0x8101F0]"

    decoded = decode_instr(Decoder(raw_alias), 0, OPCODES)
    assert decoded is not None
    assert asm_str(decoded.render()) == rendered

    encoder = Encoder()
    decoded.encode(encoder, 0)
    assert encoder.buf == canonical
    assert Assembler().assemble(source).as_binary() == canonical
    with pytest.raises(AssemblerError, match="20-bit immediate out of range"):
        Assembler().assemble(invalid_source)


@pytest.mark.parametrize(
    ("opcode", "register"),
    [(0x0C, "X"), (0x0D, "Y"), (0x0E, "U"), (0x0F, "S")],
)
def test_register_immediate_decoder_normalizes_to_architectural_20_bits(
    opcode: int, register: str
) -> None:
    raw = bytearray((opcode, 0xA5, 0x5A, 0x3C))
    decoded = decode_instr(Decoder(raw), 0, OPCODES)
    assert decoded is not None
    assert asm_str(decoded.render()) == f"MV    {register}, C5AA5"

    # Text assembly expresses the effective register value and emits the
    # canonical high byte. It must not manufacture ignored upper bits.
    assert Assembler().assemble(f"MV {register}, 0xC5AA5").as_binary() == bytearray(
        (opcode, 0xA5, 0x5A, 0x0C)
    )


@pytest.mark.parametrize(
    ("source", "expected_hex"),
    [
        ("MVP (0x20), 0x112233", "30dc20332211"),
        ("MVP (SI), 0x505954", "30dcda545950"),  # IQ-7000 ROM: "TYP"
        ("MVP (SI), 0x4D414E", "30dcda4e414d"),  # IQ-7000 ROM: "NAM"
        ("MVP (SI), 0x505344", "30dcda445350"),  # IQ-7000 ROM: "DSP"
    ],
)
def test_mvp_raw24_immediate_preserves_every_encoded_bit(
    source: str, expected_hex: str
) -> None:
    encoded = Assembler().assemble(source).as_binary()
    assert encoded.hex() == expected_hex
    assert (
        Assembler()
        .assemble(_transform(asm_str(decode(encoded, 0).render())))
        .as_binary()
        == encoded
    )


@pytest.mark.parametrize(
    "source",
    [
        "JPF 0x100000",
        "MV X, 0x100000",
        "MV A, [0x100000]",
    ],
)
def test_imm20_source_values_outside_architectural_range_are_rejected(
    source: str,
) -> None:
    with pytest.raises(AssemblerError, match="20-bit immediate out of range"):
        Assembler().assemble(source)


def test_mvp_raw24_source_outside_range_is_rejected() -> None:
    with pytest.raises(AssemblerError, match="24-bit immediate out of range"):
        Assembler().assemble("MVP (0x20), 0x1000000")


def test_add_sub_register_pair_assembler_matrix_round_trips_or_fails_closed() -> None:
    """The assembler and decoder must agree on every register-pair selector.

    The legal matrices include mixed-width ROM forms such as ``ADD Y, BA`` and
    ``SUB I, A``; invalid wider sources must be rejected instead of emitting an
    instruction that immediately becomes undecodable.
    """

    registers = ("A", "IL", "BA", "I", "X", "Y", "U", "S")
    assembler = Assembler()

    for mnemonic, opcode_by_dest in (
        ("ADD", (0x46, 0x46, 0x44, 0x44, 0x45, 0x45, 0x45, 0x45)),
        ("SUB", (0x4E, 0x4E, 0x4C, 0x4C, 0x4D, 0x4D, 0x4D, 0x4D)),
    ):
        for dest_code, dest in enumerate(registers):
            for src_code, src in enumerate(registers):
                source = f"{mnemonic} {dest}, {src}"
                legal = (
                    src_code <= 1
                    if dest_code <= 1
                    else src_code <= 3
                    if dest_code <= 3
                    else True
                )
                if not legal:
                    with pytest.raises(
                        AssemblerError, match="Invalid arithmetic register pair"
                    ):
                        assembler.assemble(source)
                    continue

                encoded = assembler.assemble(source).as_binary()
                expected_opcode = opcode_by_dest[dest_code]
                assert encoded == bytearray(
                    (expected_opcode, (dest_code << 4) | src_code)
                )

                decoded = decode_instr(Decoder(encoded), 0, OPCODES)
                assert decoded is not None
                decoded_dest, decoded_src = decoded.operands()
                assert isinstance(decoded_dest, Reg)
                assert isinstance(decoded_src, Reg)
                assert decoded_dest.reg == dest
                assert decoded_src.reg == src
                assert (
                    assembler.assemble(
                        _transform(asm_str(decoded.render()))
                    ).as_binary()
                    == encoded
                )

    # Pin the coherent stock-ROM mixed-width examples explicitly so a future
    # simplification cannot accidentally turn the matrix back into equal-width
    # pairs only.
    assert assembler.assemble("ADD Y, BA").as_binary().hex() == "4552"
    assert assembler.assemble("SUB I, A").as_binary().hex() == "4c30"


def test_direct_instruction_encoder_rejects_invalid_arithmetic_reg_pair() -> None:
    """The lower-level encoder used outside ``sc_asm.Assembler`` is strict too."""

    pair = RegPair(size=2)
    pair.reg1 = Reg("BA")
    pair.reg2 = Reg("X")
    pair.reg_raw = 0x24
    instr = ADD("ADD", operands=[pair], cond=None, ops_reversed=None)
    instr.opcode = 0x44
    with pytest.raises(InvalidInstruction, match="Invalid arithmetic register pair"):
        encode(instr, 0)


def test_mv_ex_register_pair_assembler_preserves_meaning_or_fails_closed() -> None:
    """ED/FD r2 selector aliases must not silently rename A/IL source text."""

    registers = ("A", "IL", "BA", "I", "X", "Y", "U", "S")
    assembler = Assembler()
    for mnemonic, opcode in (("MV", 0xFD), ("EX", 0xED)):
        for dest_code, dest in enumerate(registers):
            for src_code, src in enumerate(registers):
                source = f"{mnemonic} {dest}, {src}"
                if dest_code < 2 or src_code < 2:
                    with pytest.raises(
                        AssemblerError, match="A/IL would change meaning"
                    ):
                        assembler.assemble(source)
                    continue

                encoded = assembler.assemble(source).as_binary()
                assert encoded == bytearray((opcode, (dest_code << 4) | src_code))
                decoded = decode_instr(Decoder(encoded), 0, OPCODES)
                assert decoded is not None
                decoded_dest, decoded_src = decoded.operands()
                assert isinstance(decoded_dest, Reg)
                assert isinstance(decoded_src, Reg)
                assert decoded_dest.reg == dest
                assert decoded_src.reg == src
                assert (
                    assembler.assemble(
                        _transform(asm_str(decoded.render()))
                    ).as_binary()
                    == encoded
                )

    # Dedicated single-byte A/B encodings remain available and unambiguous.
    assert assembler.assemble("MV A, B").as_binary().hex() == "74"
    assert assembler.assemble("MV B, A").as_binary().hex() == "75"
    assert assembler.assemble("EX A, B").as_binary().hex() == "dd"


def test_cmpw_cmpp_register_class_assembler_matrix() -> None:
    """CMPW accepts r2 and CMPP accepts r3, exactly as their decoders do."""

    registers = ("A", "IL", "BA", "I", "X", "Y", "U", "S")
    assembler = Assembler()
    for mnemonic, opcode, valid_codes, error in (
        ("CMPW", 0xD6, {2, 3}, "CMPW register selector must be BA or I"),
        ("CMPP", 0xD7, {4, 5, 6, 7}, "CMPP register selector must be X, Y, U, or S"),
    ):
        for reg_code, reg in enumerate(registers):
            source = f"{mnemonic} (BP+0x20), {reg}"
            if reg_code not in valid_codes:
                with pytest.raises(AssemblerError, match=error):
                    assembler.assemble(source)
                continue

            encoded = assembler.assemble(source).as_binary()
            assert encoded == bytearray((opcode, reg_code, 0x20))
            decoded = decode_instr(Decoder(encoded), 0, OPCODES)
            assert decoded is not None
            assert (
                assembler.assemble(_transform(asm_str(decoded.render()))).as_binary()
                == encoded
            )


def test_jp_register_selector_matrix_matches_rom_proven_register_class() -> None:
    """JP r3 is a 20-bit register jump, not a generic low-three-bit alias."""

    registers = ("A", "IL", "BA", "I", "X", "Y", "U", "S")
    assembler = Assembler()
    for reg_code, reg in enumerate(registers):
        source = f"JP {reg}"
        if reg_code < 4:
            with pytest.raises(
                AssemblerError, match="JP register selector must be X, Y, U, or S"
            ):
                assembler.assemble(source)
            continue

        encoded = assembler.assemble(source).as_binary()
        assert encoded == bytearray((0x11, reg_code))
        assert decode_instr(Decoder(encoded), 0, OPCODES) is not None

    # Coherent stock-ROM dispatch paths use the exact selector bytes: PC-E500
    # E2F5F and IQ-7000 E4766 use 11 04 (JP X); PC-E500 F2053 and IQ-7000
    # F5230 use 11 05 (JP Y). Apparent narrow/high-bit forms occur only in
    # table/misaligned regions, so they are not accepted as architecture proof.
    assert assembler.assemble("JP X").as_binary().hex() == "1104"
    assert assembler.assemble("JP Y").as_binary().hex() == "1105"


def test_external_register_pointer_assembler_matrix() -> None:
    """Every external register-indirect form requires X/Y/U/S at encode time."""

    registers = ("A", "IL", "BA", "I", "X", "Y", "U", "S")
    patterns = (
        "MV A, [{reg}]",
        "MV [{reg}], A",
        "MV A, [{reg}++]",
        "MV [--{reg}], A",
        "MV (BP+0x20), [{reg}]",
        "MV [{reg}], (BP+0x20)",
    )
    assembler = Assembler()
    for pattern in patterns:
        for reg_code, reg in enumerate(registers):
            source = pattern.format(reg=reg)
            if reg_code < 4:
                with pytest.raises(
                    AssemblerError, match="Invalid register for r3 instruction"
                ):
                    assembler.assemble(source)
                continue

            encoded = assembler.assemble(source).as_binary()
            decoded = decode_instr(Decoder(encoded), 0, OPCODES)
            assert decoded is not None
            assert (
                assembler.assemble(_transform(asm_str(decoded.render()))).as_binary()
                == encoded
            )


@pytest.mark.parametrize(
    ("source", "expected_hex"),
    [
        ("AND (ISR), A", "3073fc"),
        ("MV A, (BP+0x20)", "8020"),
        ("MV (BP+0x10), (BP+0x20)", "c81020"),
        ("MV (BP+PX), (BP+PY)", "25c80000"),
        ("MV A, [(0x40)]", "30980040"),
        ("MV A, [(BP+0x40)]", "980040"),
        ("MV A, [(PX+0x40)]", "34980040"),
        ("MV (PX+0x20), [(PY+0x40)]", "37f0002040"),
        ("MV (0x20), [X]", "30e00420"),
        ("MV (BP+0x20), [X]", "e00420"),
        ("MV (PX+0x20), [X]", "34e00420"),
    ],
)
def test_canonical_pre_encodings(source: str, expected_hex: str) -> None:
    assert Assembler().assemble(source).as_binary().hex() == expected_hex


@pytest.mark.parametrize(
    ("first", "second", "prefix"),
    [
        ("0x10", "0x20", "32"),
        ("0x10", "BP+0x20", "30"),
        ("0x10", "PY+0x20", "33"),
        ("0x10", "BP+PY", "31"),
        ("BP+0x10", "0x20", "22"),
        ("BP+0x10", "BP+0x20", ""),
        ("BP+0x10", "PY+0x20", "23"),
        ("BP+0x10", "BP+PY", "21"),
        ("PX+0x10", "0x20", "36"),
        ("PX+0x10", "BP+0x20", "34"),
        ("PX+0x10", "PY+0x20", "37"),
        ("PX+0x10", "BP+PY", "35"),
        ("BP+PX", "0x20", "26"),
        ("BP+PX", "BP+0x20", "24"),
        ("BP+PX", "PY+0x20", "27"),
        ("BP+PX", "BP+PY", "25"),
    ],
)
def test_two_selector_pre_matrix(first: str, second: str, prefix: str) -> None:
    first_byte = "00" if first == "BP+PX" else "10"
    second_byte = "00" if second == "BP+PY" else "20"
    expected = f"{prefix}c8{first_byte}{second_byte}"
    output = Assembler().assemble(f"MV ({first}), ({second})").as_binary()
    assert output.hex() == expected


@pytest.mark.parametrize("mode", ["PY+0x10", "BP+PY"])
def test_single_selector_rejects_pre2_only_modes(mode: str) -> None:
    with pytest.raises(AssemblerError, match="cannot be encoded as the single IMEM"):
        Assembler().assemble(f"ROR ({mode})")


@pytest.mark.parametrize("raw", ["e30400", "e3840010", "eb0400", "eb840010"])
def test_mvl_rejects_undocumented_external_memory_modes(raw: str) -> None:
    # Invalid encodings are ordinary decoder misses, not leaked assertions.
    assert decode_instr(Decoder(bytearray.fromhex(raw)), 0, OPCODES) is None


@pytest.mark.parametrize("raw", ["e32400", "e33400", "eb2400", "eb3400"])
def test_mvl_accepts_documented_increment_decrement_modes(raw: str) -> None:
    assert decode_instr(Decoder(bytearray.fromhex(raw)), 0, OPCODES) is not None


def _direct_reg3(mode: EMemRegMode) -> Reg3:
    reg = Reg3()
    reg.reg = Reg3.reg_name(4)  # X
    reg.reg_raw = 0x04 | (mode.value << 4)
    reg.high4 = mode.value
    return reg


@pytest.mark.parametrize(
    ("opcode", "order"),
    [
        (0x56, RegIMemOffsetOrder.DEST_IMEM),
        (0x5E, RegIMemOffsetOrder.DEST_REG_OFFSET),
    ],
)
def test_direct_reg_imem_offset_encoder_cannot_bypass_opcode_mode_policy(
    opcode: int, order: RegIMemOffsetOrder
) -> None:
    operand = RegIMemOffset(order=order, allowed_modes=None)
    operand.reg = _direct_reg3(EMemRegMode.SIMPLE)
    operand.imem = IMem8()
    operand.imem.value = 0x20
    operand.mode = EMemRegMode.SIMPLE
    operand.offset = None
    instr = MVL("MVL", operands=[operand], cond=None, ops_reversed=None)
    instr.opcode = opcode

    with pytest.raises(InvalidInstruction, match="Invalid external-memory mode SIMPLE"):
        encode(instr, 0)


@pytest.mark.parametrize("opcode", [0xE3, 0xEB])
def test_direct_emem_reg_encoder_cannot_bypass_opcode_mode_policy(opcode: int) -> None:
    pointer = EMemReg(width=1, allowed_modes=None)
    pointer.reg = _direct_reg3(EMemRegMode.SIMPLE)
    pointer.mode = EMemRegMode.SIMPLE
    pointer.offset = None
    imem = IMem8()
    imem.value = 0x20
    instr = MVL(
        "MVL",
        operands=[imem, pointer],
        cond=None,
        ops_reversed=opcode == 0xE3,
    )
    instr.opcode = opcode

    with pytest.raises(InvalidInstruction, match="Invalid external-memory mode SIMPLE"):
        encode(instr, 0)


def test_direct_emem_reg_encoder_rejects_mode_metadata_byte_disagreement() -> None:
    pointer = EMemReg(width=1, allowed_modes=None)
    pointer.reg = _direct_reg3(EMemRegMode.SIMPLE)
    pointer.mode = EMemRegMode.POST_INC
    pointer.offset = None
    imem = IMem8()
    imem.value = 0x20
    instr = MVL("MVL", operands=[imem, pointer], cond=None, ops_reversed=True)
    instr.opcode = 0xE3

    with pytest.raises(InvalidInstruction, match="does not match encoded mode SIMPLE"):
        encode(instr, 0)
