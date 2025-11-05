"""Shared pytest fixtures for SC62015 core tests."""

from __future__ import annotations

from typing import List

import pytest

from sc62015.pysc62015 import available_backends

_VALID_BACKENDS = {"python", "rust"}


def _parse_backend_option(raw: str | None) -> List[str]:
    if not raw:
        return ["python"]
    names: List[str] = []
    for chunk in raw.split(","):
        name = chunk.strip().lower()
        if not name:
            continue
        if name not in _VALID_BACKENDS:
            raise pytest.UsageError(f"Unknown CPU backend '{name}' (expected one of: {sorted(_VALID_BACKENDS)})")
        names.append(name)
    return names or ["python"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cpu-backends",
        dest="cpu_backends",
        action="store",
        default="python",
        help="Comma-separated list of CPU backends to exercise (python,rust).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "cpu_backend" not in metafunc.fixturenames:
        return

    config_backends = _parse_backend_option(metafunc.config.getoption("cpu_backends"))
    available = set(available_backends())

    params: List[object] = []
    for backend in config_backends:
        if backend not in available:
            params.append(
                pytest.param(
                    backend,
                    marks=pytest.mark.skip(reason=f"SC62015 backend '{backend}' unavailable in this environment"),
                )
            )
        else:
            params.append(pytest.param(backend, id=backend))

    metafunc.parametrize("cpu_backend", params, scope="session")


@pytest.fixture(scope="session")
def cpu_backend(request: pytest.FixtureRequest) -> str:
    """Name of the backend selected for a parameterized test."""

    return str(request.param)
