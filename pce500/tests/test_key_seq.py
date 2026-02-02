import pytest

from pce500.run_pce500 import parse_key_seq, resolve_key_seq_key


def test_key_seq_parses_waiters_and_hold() -> None:
    actions = parse_key_seq(
        "pf1:20,wait-op:5,wait-text:MAIN MENU,wait-power:off,wait-screen-change,wait-screen-empty,wait-screen-draw",
        10,
    )
    kinds = [action.kind.value for action in actions]
    assert kinds == [
        "press",
        "wait_op",
        "wait_text",
        "wait_power",
        "wait_screen_change",
        "wait_screen_empty",
        "wait_screen_draw",
    ]
    assert actions[0].hold == 20
    assert actions[1].op_target == 5
    assert actions[2].text == "MAIN MENU"
    assert actions[3].power_on is False


def test_key_seq_accepts_space_alias() -> None:
    actions = parse_key_seq("space", 10)
    assert len(actions) == 1
    assert actions[0].key == resolve_key_seq_key("space")


def test_key_seq_wait_op_is_relative() -> None:
    actions = parse_key_seq("wait-op:5,pf1", 10)
    assert actions[0].op_target == 5
    assert actions[1].label == "pf1"


def test_key_seq_rejects_empty_wait_text() -> None:
    with pytest.raises(ValueError):
        parse_key_seq("wait-text:", 10)
