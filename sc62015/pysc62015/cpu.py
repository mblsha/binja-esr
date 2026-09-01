"""Backend selector for the SC62015 CPU emulation core."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Tuple, cast

from .emulator import (
    Emulator,
    InstructionEvalInfo,
    RegisterName,
    _read_byte_with_pc,
    validate_vector_transfer,
    validate_vector_transfer_stability,
)
from .instr import (
    IR,
    RESET,
    Instruction,
    InvalidInstruction,
    PRE,
    TCL,
)
from .instr.opcodes import (
    ENTRY_POINT_ADDR,
    INTERRUPT_VECTOR_ADDR,
)
from .stepper import CPURegistersSnapshot, CPUStepResult, CPUStepper

try:
    from binaryninja import InstructionInfo  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from binja_test_mocks.binja_api import InstructionInfo  # type: ignore

CPUBackendName = Literal["python", "llama"]

_ENV_VAR = "SC62015_CPU_BACKEND"
_DEFAULT_BACKEND: CPUBackendName = "python"


def _load_native_backend() -> Optional[object]:
    """Attempt to import the optional native (LLAMA) backend module."""

    try:
        rust_module = import_module("_sc62015_rustcore")
    except ModuleNotFoundError:
        return None

    return rust_module


def available_backends(
    native_loader: Callable[[], Optional[object]] | None = None,
) -> Tuple[CPUBackendName, ...]:
    """Return the list of CPU backends available in this runtime."""

    loader = native_loader or _load_native_backend
    backends: list[CPUBackendName] = ["python"]
    rust_module = loader()
    if rust_module is not None:
        if getattr(rust_module, "HAS_LLAMA_IMPLEMENTATION", False):
            backends.append("llama")
    return tuple(backends)


def _normalise_backend_name(name: str) -> CPUBackendName:
    lowered = name.strip().lower()
    if lowered in {"py", "python"}:
        return "python"
    if lowered in {"llama", "rust", "native"}:
        return "llama"
    raise ValueError(f"Unknown SC62015 backend '{name}'")


def select_backend(
    preferred: Optional[str] = None,
    *,
    native_loader: Callable[[], Optional[object]] | None = None,
) -> Tuple[CPUBackendName, Optional[object]]:
    """Resolve the backend that should power CPU instances."""

    loader = native_loader or _load_native_backend

    requested: Optional[str] = preferred or os.environ.get(_ENV_VAR)
    backend_name: CPUBackendName
    rust_module: Optional[object] = None
    if requested:
        backend_name = _normalise_backend_name(requested)
    else:
        backend_name = _DEFAULT_BACKEND

    if backend_name == "llama":
        rust_module = loader()
        if rust_module is None:
            raise RuntimeError(
                "SC62015 LLAMA backend requested but not available. "
                "Run `uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml` "
                "to build the optional extension."
            )
        if not getattr(rust_module, "HAS_LLAMA_IMPLEMENTATION", False):
            raise RuntimeError(
                "LLAMA backend requested but not available in rustcore module."
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
        timer_scale: float | None = None,
    ) -> None:
        backend_name, rust_module = select_backend(backend)

        # LLAMA still uses the Python object as a decoder/rendering facade, but
        # it must not execute RESET against the shared host memory.  The native
        # backend owns initialization in that mode; running both used to apply
        # reset callbacks twice before the first instruction.
        legacy = Emulator(
            memory,
            reset_on_init=reset_on_init if backend_name == "python" else False,
        )
        self._impl: Any
        if backend_name == "python":
            self._impl = legacy
            self.regs = self._impl.regs
            self.state = self._impl.state
            self._legacy_decoder = legacy
        else:
            assert rust_module is not None
            rust_cpu_cls = getattr(rust_module, "LlamaCPU")
            scale = 1.0 if timer_scale is None else float(timer_scale)
            self._impl = rust_cpu_cls(
                memory=memory, reset_on_init=reset_on_init, timer_scale=scale
            )
            self.regs = _RustRegisterProxy(self._impl)
            self.state = _RustStateProxy(self._impl)
            self._legacy_decoder = legacy

        self.memory = memory
        self.backend: CPUBackendName = backend_name
        # A backend contract mismatch is detected only after the native core
        # has executed.  Retrying could therefore apply the same instruction's
        # side effects twice.  Preserve the first mismatch until a complete
        # reset succeeds.
        self._contract_poisoned: str | None = None

    def __getattr__(self, name: str):
        return getattr(self._impl, name)

    def __dir__(self) -> Iterable[str]:
        return sorted(set(dir(self.__class__)) | set(dir(self._impl)))

    def set_perfetto_trace(self, path: Optional[str]) -> None:
        if hasattr(self._impl, "set_perfetto_trace"):
            self._impl.set_perfetto_trace(path)

    def flush_perfetto(self) -> None:
        if hasattr(self._impl, "flush_perfetto"):
            self._impl.flush_perfetto()

    def unwrap(self) -> object:
        """Expose the underlying backend instance (useful for testing)."""

        return self._impl

    def backend_stats(self) -> dict[str, int | str]:
        """Expose backend-specific counters (e.g., Rust bridge stats)."""

        if self.backend == "python":
            return {"backend": "python"}
        rust_impl = cast(Any, self._impl)
        getter = getattr(rust_impl, "get_stats", None)
        if not callable(getter):
            return {"backend": self.backend}
        stats = getter()
        if not isinstance(stats, dict):
            return {"backend": self.backend}
        return {"backend": self.backend, **stats}

    def runtime_profile_stats(self) -> dict[str, object]:
        return {}

    def set_runtime_profile_enabled(self, enabled: bool) -> None:
        _ = enabled

    def reset_runtime_profile_stats(self) -> None:
        return

    def export_lcd_snapshot(self):
        return None, None

    def decode_instruction(
        self, address: int, read_fn: Callable[[int], int] | None = None
    ) -> Instruction:
        if self.backend == "python":
            if read_fn is None:
                return cast(Instruction, self._impl.decode_instruction(address))
            return cast(
                Instruction, self._impl.decode_instruction(address, read_fn=read_fn)
            )

        prev_cpu = getattr(self.memory, "cpu", None)
        can_switch = hasattr(self.memory, "set_cpu")
        if can_switch and prev_cpu is not self._legacy_decoder:
            self.memory.set_cpu(self._legacy_decoder)
        try:
            if read_fn is None:
                instr = self._legacy_decoder.decode_instruction(address)
            else:
                instr = self._legacy_decoder.decode_instruction(
                    address, read_fn=read_fn
                )
        finally:
            if can_switch and prev_cpu is not self._legacy_decoder:
                self.memory.set_cpu(prev_cpu)
        return cast(Instruction, instr)

    def validate_before_scheduling(
        self, address: int, *, validate_data_dependent: bool = True
    ) -> Instruction:
        """Reject quarantined opcodes before per-instruction time advances.

        The backend dispatchers repeat these checks at execution time.  This
        earlier facade check exists for machine wrappers which account for
        peripheral time before calling :meth:`execute_instruction`.  Interrupt
        delivery may still replace ``address`` before this preflight is called.
        """

        if self._contract_poisoned is not None:
            raise RuntimeError(
                "SC62015 CPU facade is poisoned after a backend contract "
                f"violation; reset required: {self._contract_poisoned}"
            )
        peek_method = getattr(self.memory, "peek_byte_for_preflight", None)
        if not callable(peek_method):
            raise RuntimeError(
                "SC62015 scheduling preflight requires a side-effect-free "
                "memory peek implementation"
            )

        def peek(byte_address: int) -> int:
            return int(peek_method(byte_address, address & 0xFFFFF)) & 0xFF

        instr = self.decode_instruction(address, read_fn=peek)
        if isinstance(instr, PRE):
            raise InvalidInstruction(
                f"Unfused or malformed PRE instruction at 0x{address & 0xFFFFF:05X}"
            )
        if isinstance(instr, TCL):
            if not callable(getattr(self.memory, "clear_timer_phases", None)):
                raise NotImplementedError(
                    "TCL requires a timer-phase-clear memory hook"
                )

        if isinstance(instr, IR):
            self.preflight_vector_transfer(INTERRUPT_VECTOR_ADDR, source_pc=address)
        elif isinstance(instr, RESET):
            self.preflight_vector_transfer(ENTRY_POINT_ADDR, source_pc=address)

        if not validate_data_dependent:
            return instr

        return instr

    def preflight_vector_transfer(
        self, vector_address: int, *, source_pc: int | None = None
    ) -> int:
        """Resolve and statically validate a vector without observable reads.

        This helper deliberately remains usable while a backend is poisoned:
        a complete RESET is the recovery path, but it must still prove that its
        own destination is safe before clearing any machine state.
        """

        return validate_vector_transfer(
            self.memory,
            self.regs,
            vector_address,
            source_pc=source_pc,
        )

    def preflight_vector_transfer_for_scheduling(
        self, vector_address: int, *, source_pc: int
    ) -> int:
        target = self.preflight_vector_transfer(
            vector_address,
            source_pc=source_pc,
        )
        validate_vector_transfer_stability(
            self.memory,
            vector_address,
            target,
            require_metadata=True,
        )
        return target

    def prepare_vector_transfer(
        self,
        vector_address: int,
        *,
        source_pc: int,
        require_immutable: bool = False,
        scope: str = "instruction",
    ) -> int:
        """Fetch and retain one unforgeable transfer in the active backend."""

        if self.backend == "python":
            return int(
                self._impl.prepare_vector_transfer(
                    vector_address,
                    source_pc=source_pc,
                    require_immutable=require_immutable,
                    scope=scope,
                )
            )
        return int(
            self._impl.prepare_vector_transfer(
                vector_address,
                source_pc,
                require_immutable=require_immutable,
                scope=scope,
            )
        )

    def cancel_prepared_vector_transfer(self) -> None:
        cancel = getattr(self._impl, "cancel_prepared_vector_transfer", None)
        if callable(cancel):
            cancel()

    def prepare_power_on_reset(self) -> int:
        """Prepare the sole reset-vector fetch across machine reset mutation."""

        self.cancel_prepared_vector_transfer()
        source_pc = self.regs.get(RegisterName.PC)
        target = self.preflight_vector_transfer(
            ENTRY_POINT_ADDR,
            source_pc=source_pc,
        )
        validate_vector_transfer_stability(
            self.memory,
            ENTRY_POINT_ADDR,
            target,
            require_immutable=True,
            require_metadata=True,
        )
        try:
            return self.prepare_vector_transfer(
                ENTRY_POINT_ADDR,
                source_pc=source_pc,
                require_immutable=True,
                scope="machine_reset",
            )
        except Exception as exc:
            self._contract_poisoned = (
                "machine RESET architectural vector preparation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.cancel_prepared_vector_transfer()
            raise

    def prepare_instruction_before_scheduling(self, address: int) -> Instruction:
        """Stage a callback-free instruction and any vector before timer tick."""

        self.cancel_prepared_vector_transfer()
        instr = self.validate_before_scheduling(address)
        info = InstructionInfo()
        instr.analyze(info, address)
        length = int(info.length or instr.length())
        stability_check = getattr(
            self.memory, "instruction_byte_is_callback_free", None
        )
        if not callable(stability_check):
            raise RuntimeError(
                "SC62015 scheduling requires executable-memory stability metadata"
            )
        for offset in range(length):
            byte_address = (address + offset) & 0xFFFFF
            if not bool(stability_check(byte_address)):
                raise RuntimeError(
                    "SC62015 scheduling refuses callback-backed instruction "
                    f"byte 0x{byte_address:05X}"
                )
        try:
            actual_opcode = _read_byte_with_pc(self.memory, address & 0xFFFFF, address)
        except Exception as exc:
            self._contract_poisoned = (
                "scheduled architectural opcode fetch failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        prefixed_opcode = getattr(instr, "_pre", None)
        expected_opcode = (
            int(prefixed_opcode)
            if prefixed_opcode is not None
            else int(instr.opcode)
            if instr.opcode is not None
            else -1
        )
        if actual_opcode != expected_opcode:
            self._contract_poisoned = (
                "scheduled architectural opcode disagrees with preflight"
            )
            raise RuntimeError(
                "SC62015 architectural opcode fetch disagrees with safe preflight "
                f"at 0x{address & 0xFFFFF:05X}: fetched 0x{actual_opcode:02X}, "
                f"preflight 0x{expected_opcode & 0xFF:02X}"
            )
        prepare_opcode = getattr(self._impl, "prepare_scheduled_opcode", None)
        if not callable(prepare_opcode):
            self._contract_poisoned = "backend cannot retain prepared opcode"
            raise RuntimeError(self._contract_poisoned)
        try:
            prepare_opcode(address, actual_opcode)
            if isinstance(instr, IR):
                self.prepare_vector_transfer(
                    INTERRUPT_VECTOR_ADDR,
                    source_pc=address,
                )
            elif isinstance(instr, RESET):
                self.prepare_vector_transfer(
                    ENTRY_POINT_ADDR,
                    source_pc=address,
                )
        except Exception as exc:
            self._contract_poisoned = (
                "scheduled architectural preparation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.cancel_prepared_vector_transfer()
            raise
        return instr

    def execute_instruction(self, address: int) -> InstructionEvalInfo:
        if self._contract_poisoned is not None:
            raise RuntimeError(
                "SC62015 CPU facade is poisoned after a backend contract "
                f"violation; reset required: {self._contract_poisoned}"
            )
        if self.backend == "python":
            return self._impl.execute_instruction(address)

        # Native execution owns the architectural instruction fetch. The
        # Python decoder is used only to render/analyze the result, so feed it
        # the same side-effect-free peek contract as scheduling preflight
        # instead of reading callback-backed code a second time.
        peek_method = getattr(self.memory, "peek_byte_for_preflight", None)
        if not callable(peek_method):
            raise RuntimeError(
                "SC62015 LLAMA facade decode requires memory.peek_byte_for_preflight"
            )

        def peek(byte_address: int) -> int:
            return int(peek_method(byte_address, address & 0xFFFFF)) & 0xFF

        instr = self.decode_instruction(address, read_fn=peek)
        info = InstructionInfo()
        instr.analyze(info, address)

        opcode, length = cast(Tuple[int, int], self._impl.execute_instruction(address))
        declared_length = int(info.length) if info.length is not None else None
        if declared_length is not None and declared_length != length:
            mismatch = (
                f"Decoded length ({declared_length}) disagrees with runtime ({length}) "
                f"for opcode 0x{opcode:02X} at {address:#06X}"
            )
            if self._contract_poisoned is None:
                self._contract_poisoned = mismatch
            raise RuntimeError(mismatch)

        return InstructionEvalInfo(instruction_info=info, instruction=instr)

    def power_on_reset(self) -> None:
        self._impl.power_on_reset()
        self._contract_poisoned = None

    def snapshot_registers(self) -> CPURegistersSnapshot:
        if self.backend == "python":
            return CPURegistersSnapshot.from_registers(self.regs)
        rust_impl = cast(Any, self._impl)
        snapshot = rust_impl.snapshot_cpu_registers()
        assert isinstance(snapshot, CPURegistersSnapshot)
        return snapshot

    def notify_host_write(self, address: int, value: int) -> None:
        """Propagate host-initiated memory writes into the active backend."""

        if self.backend != "llama":
            return
        rust_impl = cast(Any, self._impl)
        notifier = getattr(rust_impl, "notify_host_write", None)
        if not callable(notifier):
            raise RuntimeError(
                "LLAMA backend does not expose the required notify_host_write hook"
            )
        notifier(int(address) & 0xFFFFFF, int(value) & 0xFF)

    def apply_snapshot(self, snapshot: CPURegistersSnapshot) -> None:
        if self.backend == "python":
            snapshot.apply_to(self.regs)
        else:
            rust_impl = cast(Any, self._impl)
            rust_impl.load_cpu_snapshot(snapshot)

    def step_snapshot(
        self,
        registers: CPURegistersSnapshot,
        memory_image: Mapping[int, int],
        *,
        default_memory_value: int = 0,
    ) -> CPUStepResult:
        """Execute a single instruction from a snapshot/memory image."""

        stepper = CPUStepper(
            default_memory_value=default_memory_value,
            backend=self.backend,
        )
        return stepper.step(registers, memory_image)


class _RustRegisterProxy:
    """Adapter exposing the Emulator.Registers API for the LLAMA (native) backend."""

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
    """Adapter exposing the Emulator.State API for the LLAMA (native) backend."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @property
    def halted(self) -> bool:
        return bool(self._backend.halted)

    @halted.setter
    def halted(self, value: bool) -> None:
        self._backend.halted = bool(value)

    @property
    def power_state(self) -> str:
        return str(self._backend.power_state)

    @power_state.setter
    def power_state(self, value: str) -> None:
        self._backend.power_state = str(value)


__all__ = [
    "CPU",
    "CPUBackendName",
    "available_backends",
    "select_backend",
]
