from sc62015.scil import serde, specs


def test_round_trip_json_preserves_instruction() -> None:
    instr = specs.mv_a_abs_ext()
    payload = serde.to_json(instr, indent=0)
    restored = serde.from_json(payload)
    assert serde.to_json(restored, indent=0) == payload


def test_multiple_specs_round_trip() -> None:
    payloads = [
        serde.to_json(factory(), indent=0)
        for factory in (specs.mv_a_imm, specs.jrz_rel)
    ]
    for payload in payloads:
        restored = serde.from_json(payload)
        assert serde.to_json(restored, indent=0) == payload
