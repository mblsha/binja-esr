"""Shared pytest fixtures for SC62015 core tests."""

from __future__ import annotations

from typing import List

import pytest

from sc62015.pysc62015 import available_backends

_VALID_BACKENDS = {"python", "llama"}


def _parse_backend_option(raw: str | None) -> List[str]:
    if not raw:
        return ["python"]
    names: List[str] = []
    for chunk in raw.split(","):
        name = chunk.strip().lower()
        if not name:
            continue
        if name not in _VALID_BACKENDS:
            raise pytest.UsageError(
                f"Unknown CPU backend '{name}' (expected one of: {sorted(_VALID_BACKENDS)})"
            )
        names.append(name)
    return names or ["python"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cpu-backend",
        action="store",
        metavar="LIST",
        default="python,llama",
        help="Comma-separated list of CPU backends to exercise (default: python,llama)",
    )


@pytest.fixture(scope="session")
def available_cpu_backends() -> tuple[str, ...]:
    return available_backends()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "cpu_backend" in metafunc.fixturenames:
        selected = _parse_backend_option(metafunc.config.getoption("--cpu-backend"))
        metafunc.parametrize("cpu_backend", selected, indirect=True)


@pytest.fixture
def cpu_backend(
    request: pytest.FixtureRequest, available_cpu_backends: tuple[str, ...]
) -> str:
    backend = request.param
    if backend not in available_cpu_backends:
        pytest.skip(f"CPU backend '{backend}' not available in this runtime")
    return backend
