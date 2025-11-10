from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, given, settings

from .harness import run_scenario
from .strategies import cpu_states, instruction_scenarios


FAST_MAX_EXAMPLES = int(os.getenv("BN_PROP_EXAMPLES", "300"))
NIGHTLY_MAX_EXAMPLES = int(os.getenv("BN_PROP_NIGHTLY_EXAMPLES", "5000"))


@given(scenario=instruction_scenarios(), state=cpu_states())
@settings(
    max_examples=FAST_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_prop_diff_small(scenario, state) -> None:
    run_scenario(scenario, state)


@pytest.mark.nightly
@given(scenario=instruction_scenarios(), state=cpu_states())
@settings(
    max_examples=NIGHTLY_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_prop_diff_nightly(scenario, state) -> None:
    if not os.getenv("BN_PROP_RUN_NIGHTLY"):
        pytest.skip("Nightly fuzzing disabled (set BN_PROP_RUN_NIGHTLY=1 to enable)")
    run_scenario(scenario, state)
