"""Shared pytest fixtures for SC62015 core tests."""

from __future__ import annotations

import os
from importlib import import_module
from typing import List

import pytest

try:
    from sc62015.pysc62015 import available_backends

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - handles optional deps (binja_test_mocks)

    def available_backends() -> tuple[str, ...]:
        return ()

    _IMPORT_ERROR = exc

_VALID_BACKENDS = {"python", "llama"}


def _native_module_is_absent() -> bool:
    """Distinguish an optional missing extension from a broken present one."""

    try:
        import_module("_sc62015_rustcore")
    except ModuleNotFoundError as exc:
        if exc.name == "_sc62015_rustcore":
            return True
        raise
    return False


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
    if _IMPORT_ERROR is not None:
        pytest.fail(f"SC62015 test package failed to import: {_IMPORT_ERROR}")
    return available_backends()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "cpu_backend" in metafunc.fixturenames:
        selected = _parse_backend_option(metafunc.config.getoption("--cpu-backend"))
        metafunc.parametrize("cpu_backend", selected, indirect=True)


@pytest.fixture
def cpu_backend(
    request: pytest.FixtureRequest, available_cpu_backends: tuple[str, ...]
) -> str:
    if _IMPORT_ERROR is not None:
        pytest.fail(f"SC62015 test package failed to import: {_IMPORT_ERROR}")
    backend = request.param
    if backend not in available_cpu_backends:
        if backend != "llama":
            pytest.fail(f"Selected CPU backend {backend!r} was not advertised")
        if not _native_module_is_absent():
            pytest.fail(
                "LLAMA extension is present but does not advertise a usable backend"
            )
        if os.environ.get("CI"):
            pytest.fail(
                "CPU backend 'llama' is required for CI parity tests but is unavailable"
            )
        pytest.skip(f"CPU backend '{backend}' not available in this runtime")
    return backend
