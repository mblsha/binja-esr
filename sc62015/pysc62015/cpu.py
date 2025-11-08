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
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, cast

from .constants import ADDRESS_SPACE_SIZE
from .emulator import (
    Emulator,
    InstructionEvalInfo,
    RegisterName,
    USE_CACHED_DECODER,
)
from .instr import Instruction, decode
from .stepper import CPURegistersSnapshot
from .instr.opcode_table import OPCODES

try:
    from binaryninja import InstructionInfo  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Binary Ninja not installed
    from binja_test_mocks.binja_api import InstructionInfo  # type: ignore

try:
    from .cached_decoder import CachedFetchDecoder  # type: ignore
except ImportError:  # pragma: no cover - cached decoder optional
    CachedFetchDecoder = None  # type: ignore
from binja_test_mocks.coding import FetchDecoder

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
            self.regs = self._impl.regs
            self.state = self._impl.state
        else:
            assert rust_module is not None
            rust_cpu_cls = getattr(rust_module, "CPU")
            self._impl = rust_cpu_cls(memory=memory, reset_on_init=reset_on_init)
            self.regs = _RustRegisterProxy(self._impl)
            self.state = _RustStateProxy(self._impl)

        self.memory = memory
        self.backend: CPUBackendName = backend_name

    def __getattr__(self, name: str):
        return getattr(self._impl, name)

    def __dir__(self) -> Iterable[str]:
        return sorted(set(dir(self.__class__)) | set(dir(self._impl)))

    def unwrap(self) -> object:
        """Expose the underlying backend instance (useful for testing)."""

        return self._impl

    def decode_instruction(self, address: int) -> Instruction:
        if self.backend == "python":
            return self._impl.decode_instruction(address)
        return _decode_instruction(self.memory, address)

    def execute_instruction(self, address: int) -> InstructionEvalInfo:
        if self.backend == "python":
            return self._impl.execute_instruction(address)

        instr = self.decode_instruction(address)
        info = InstructionInfo()
        instr.analyze(info, address)

        opcode, length = cast(Tuple[int, int], self._impl.execute_instruction(address))
        declared_length = int(info.length) if info.length is not None else None
        if declared_length is not None and declared_length != length:
            raise RuntimeError(
                f"Decoded length ({declared_length}) disagrees with runtime ({length}) "
                f"for opcode 0x{opcode:02X} at {address:#06X}"
            )

        return InstructionEvalInfo(instruction_info=info, instruction=instr)

    def power_on_reset(self) -> None:
        if self.backend == "python":
            self._impl.power_on_reset()
        else:
            self._impl.power_on_reset()

    def snapshot_registers(self) -> CPURegistersSnapshot:
        if self.backend == "python":
            return CPURegistersSnapshot.from_registers(self.regs)
        rust_impl = cast(Any, self._impl)
        snapshot = rust_impl.snapshot_cpu_registers()
        assert isinstance(snapshot, CPURegistersSnapshot)
        return snapshot

    def apply_snapshot(self, snapshot: CPURegistersSnapshot) -> None:
        if self.backend == "python":
            snapshot.apply_to(self.regs)
        else:
            rust_impl = cast(Any, self._impl)
            rust_impl.load_cpu_snapshot(snapshot)


def _decode_instruction(memory, address: int) -> Instruction:
    """Decode an instruction using the shared Python decoder."""

    def _fetch(offset: int) -> int:
        return memory.read_byte(address + offset)

    if USE_CACHED_DECODER and CachedFetchDecoder is not None:
        decoder = CachedFetchDecoder(_fetch, ADDRESS_SPACE_SIZE)  # type: ignore[arg-type]
    else:
        decoder = FetchDecoder(_fetch, ADDRESS_SPACE_SIZE)
    return decode(decoder, address, OPCODES)  # type: ignore[arg-type]


class _RustRegisterProxy:
    """Adapter exposing the Emulator.Registers API for the Rust backend."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @staticmethod
    def _reg_name(reg: object) -> str:
        if isinstance(reg, RegisterName):
            return reg.name
        if isinstance(reg, str):
            return reg
        candidate = getattr(reg, "name", None)
        if candidate is None:
            raise TypeError(f"Unsupported register identifier: {reg!r}")
        return str(candidate)

    def get(self, reg) -> int:
        return self._backend.read_register(self._reg_name(reg))

    def set(self, reg, value: int) -> None:
        self._backend.write_register(self._reg_name(reg), int(value))

    def get_by_name(self, name: str) -> int:
        return self._backend.read_register(name)

    def set_by_name(self, name: str, value: int) -> None:
        self._backend.write_register(name, int(value))

    def get_flag(self, name: str) -> int:
        return int(self._backend.read_flag(name))

    def set_flag(self, name: str, value: int) -> None:
        self._backend.write_flag(name, int(value))

    @property
    def call_sub_level(self) -> int:
        return int(self._backend.call_sub_level)

    @call_sub_level.setter
    def call_sub_level(self, level: int) -> None:
        self._backend.call_sub_level = int(level)


class _RustStateProxy:
    """Adapter exposing the Emulator.State API for the Rust backend."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @property
    def halted(self) -> bool:
        return bool(self._backend.halted)

    @halted.setter
    def halted(self, value: bool) -> None:
        self._backend.halted = bool(value)


__all__ = [
    "CPU",
    "CPUBackendName",
    "available_backends",
    "select_backend",
]
