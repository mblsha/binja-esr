"""Backend selector for the SC62015 CPU emulation core.

The existing Python implementation is still the default, but this module
provides a single entry point (`CPU`) that can later dispatch to the Rust
backend once it is feature-complete. A lightweight proxy exposes the same
attribute surface as the underlying implementation so existing callers can
interact with the returned object as if it were the original `Emulator`.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Callable, Iterable, Literal, Optional, Tuple

from .emulator import Emulator

CPUBackendName = Literal["python", "rust"]

_ENV_VAR = "SC62015_CPU_BACKEND"
_DEFAULT_BACKEND: CPUBackendName = "python"


def _load_rust_backend() -> Optional[object]:
    """Attempt to import the optional Rust backend module.

    Returns the imported module when available *and* when it advertises a usable
    CPU implementation. The scaffolding crate currently exposes the constant
    `HAS_CPU_IMPLEMENTATION` so the selector can decide whether it is ready for
    traffic.
    """
    try:
        rust_module = import_module("_sc62015_rustcore")
    except ModuleNotFoundError:
        return None

    if not getattr(rust_module, "HAS_CPU_IMPLEMENTATION", False):
        return None

    return rust_module


def available_backends(
    rust_loader: Callable[[], Optional[object]] | None = None,
) -> Tuple[CPUBackendName, ...]:
    """Return the list of CPU backends that can be used in the current runtime."""

    loader = rust_loader or _load_rust_backend
    backends: list[CPUBackendName] = ["python"]
    if loader() is not None:
        backends.append("rust")
    return tuple(backends)


def _normalise_backend_name(name: str) -> CPUBackendName:
    lowered = name.strip().lower()
    if lowered in {"py", "python"}:
        return "python"
    if lowered in {"rs", "rust", "rustcore"}:
        return "rust"
    raise ValueError(f"Unknown SC62015 backend '{name}'")


def select_backend(
    preferred: Optional[str] = None,
    *,
    rust_loader: Callable[[], Optional[object]] | None = None,
) -> Tuple[CPUBackendName, Optional[object]]:
    """Resolve the backend that should power new CPU instances.

    Order of precedence:
    1. Explicit `preferred` argument.
    2. `SC62015_CPU_BACKEND` environment variable.
    3. Default (`python`).

    If the Rust backend is requested but unavailable, a descriptive RuntimeError
    is raised so callers know how to proceed.
    """

    loader = rust_loader or _load_rust_backend

    requested: Optional[str] = preferred or os.environ.get(_ENV_VAR)
    backend_name: CPUBackendName
    rust_module: Optional[object] = None
    if requested:
        backend_name = _normalise_backend_name(requested)
    else:
        backend_name = _DEFAULT_BACKEND

    if backend_name == "rust":
        rust_module = loader()
        if rust_module is None:
            raise RuntimeError(
                "SC62015 Rust backend requested but not available. "
                "Run `uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml` "
                "to build the optional extension."
            )

    return backend_name, rust_module


class CPU:
    """Facade that delegates to either the Python Emulator or the Rust core."""

    def __init__(
        self,
        memory,
        *,
        reset_on_init: bool = True,
        backend: Optional[str] = None,
    ) -> None:
        backend_name, rust_module = select_backend(backend)

        if backend_name == "python":
            self._impl = Emulator(memory, reset_on_init=reset_on_init)
        else:
            assert rust_module is not None
            rust_cpu_cls = getattr(rust_module, "CPU")
            self._impl = rust_cpu_cls(memory=memory, reset_on_init=reset_on_init)

        self.backend: CPUBackendName = backend_name

    def __getattr__(self, name: str):
        return getattr(self._impl, name)

    def __dir__(self) -> Iterable[str]:
        return sorted(set(dir(self.__class__)) | set(dir(self._impl)))

    def unwrap(self) -> object:
        """Expose the underlying backend instance (useful for testing)."""

        return self._impl


__all__ = [
    "CPU",
    "CPUBackendName",
    "available_backends",
    "select_backend",
]
