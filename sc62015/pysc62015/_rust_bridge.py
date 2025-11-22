from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Tuple, Callable, Iterable

import _sc62015_rustcore as rustcore
from sc62015.pysc62015 import emulator as _emulator
from sc62015.pysc62015.constants import (
    INTERNAL_MEMORY_START,
    PC_MASK,
    ADDRESS_SPACE_SIZE,
    ISRFlag,
)
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.pysc62015.stepper import CPURegistersSnapshot

InstructionInfo = _emulator.InstructionInfo  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)

_LCD_LOOP_TRACE_ENABLED = os.getenv("LCD_LOOP_TRACE") == "1"
_LCD_LOOP_RANGE: tuple[int, int] | None = None
_LCD_LOOP_RANGE_DEFAULT: tuple[int, int] = (0x0F29A0, 0x0F2B00)
_LCD_LOOP_REGISTERS = ("PC", "A", "B", "BA", "I", "X", "Y", "U", "S")
_LCD_LOOP_FLAGS = ("C", "Z")
_LCD_TRACE_BP_ENABLED = os.getenv("LCD_TRACE_BP") == "1"

_STACK_SNAPSHOT_RANGE: tuple[int, int] | None = None
_STACK_SNAPSHOT_LEN: int | None = None
_IMEM_WATCH: tuple[int, ...] | None = None


def _imem_watch_list() -> tuple[int, ...]:
    global _IMEM_WATCH
    if _IMEM_WATCH is not None:
        return _IMEM_WATCH
    raw = os.getenv("RUST_IMEM_WATCH")
    if not raw:
        _IMEM_WATCH = ()
        return _IMEM_WATCH
    addrs: list[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo = int(lo_str, 0)
                hi = int(hi_str, 0)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            addrs.extend(range(lo, hi + 1))
            continue
        try:
            addrs.append(int(part, 0))
        except ValueError:
            continue
    _IMEM_WATCH = tuple(addrs)
    return _IMEM_WATCH


def _imem_ignore_list() -> tuple[int, ...]:
    raw = os.getenv("RUST_IGNORE_IMEM_WRITES")
    if not raw:
        addrs: list[int] = []
    else:
        addrs = []
        for part in raw.replace(",", " ").split():
            if not part:
                continue
            if "-" in part:
                lo_str, hi_str = part.split("-", 1)
                try:
                    lo = int(lo_str, 0)
                    hi = int(hi_str, 0)
                except ValueError:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                addrs.extend(range(lo, hi + 1))
            else:
                try:
                    addrs.append(int(part, 0))
                except ValueError:
                    continue

    if os.getenv("RUST_SYNC_IMR_ISR", "1") == "0":
        base = INTERNAL_MEMORY_START
        addrs.extend(
            [
                base + int(IMEMRegisters.IMR),
                base + int(IMEMRegisters.ISR),
            ]
        )
    addrs: list[int] = []
    return tuple(addrs)


def _clamp_imr_isr(value: int, addr: int, memory) -> int:
    """Optionally clamp IMR/ISR writes to the current Python view."""

    clamp_env = os.getenv("RUST_CLAMP_IMR")
    if not clamp_env or clamp_env == "0":
        return value
    try:
        current = memory.read_byte(addr) & 0xFF
    except Exception:
        return value
    # If RUST_CLAMP_IMR is set, prefer the current Python value entirely.
    return current


def _lcd_loop_range() -> tuple[int, int]:
    global _LCD_LOOP_RANGE
    if _LCD_LOOP_RANGE is not None:
        return _LCD_LOOP_RANGE
    env = os.getenv("LCD_LOOP_RANGE")
    if env:
        parts = env.strip().split("-")
        try:
            start = int(parts[0], 0)
        except ValueError:
            start = _LCD_LOOP_RANGE_DEFAULT[0]
        if len(parts) > 1:
            try:
                end = int(parts[1], 0)
            except ValueError:
                end = _LCD_LOOP_RANGE_DEFAULT[1]
        else:
            end = start
        if start > end:
            start, end = end, start
        _LCD_LOOP_RANGE = (start, end)
    else:
        _LCD_LOOP_RANGE = _LCD_LOOP_RANGE_DEFAULT
    return _LCD_LOOP_RANGE


def _should_trace_lcd(address: int) -> bool:
    if not _LCD_LOOP_TRACE_ENABLED:
        return False
    start, end = _lcd_loop_range()
    return start <= address <= end


def _stack_snapshot_range() -> tuple[int, int] | None:
    global _STACK_SNAPSHOT_RANGE
    if _STACK_SNAPSHOT_RANGE is not None:
        return _STACK_SNAPSHOT_RANGE
    env = os.getenv("STACK_SNAPSHOT_RANGE")
    if not env:
        return None
    parts = env.strip().split("-")
    if not parts:
        return None
    try:
        start = int(parts[0], 0)
        end = int(parts[1], 0) if len(parts) > 1 else start
    except ValueError:
        return None
    if start > end:
        start, end = end, start
    _STACK_SNAPSHOT_RANGE = (start, end)
    return _STACK_SNAPSHOT_RANGE


def _stack_snapshot_len() -> int:
    global _STACK_SNAPSHOT_LEN
    if _STACK_SNAPSHOT_LEN is not None:
        return _STACK_SNAPSHOT_LEN
    env = os.getenv("STACK_SNAPSHOT_LEN")
    length = 8
    if env:
        try:
            candidate = int(env, 0)
            if candidate > 0:
                length = candidate
        except ValueError:
            pass
    _STACK_SNAPSHOT_LEN = length
    return length


def _should_snapshot_stack(address: int) -> bool:
    rng = _stack_snapshot_range()
    if not rng:
        return False
    start, end = rng
    return start <= address <= end


def _log_lcd_loop_state(
    prefix: str,
    address: int,
    read_register: Callable[[str], int],
    read_flag: Callable[[str], int],
) -> None:
    reg_vals = " ".join(
        f"{name}={read_register(name):06X}" for name in _LCD_LOOP_REGISTERS
    )
    flag_vals = " ".join(f"{name}={read_flag(name):01X}" for name in _LCD_LOOP_FLAGS)
    print(f"[lcd-loop] {prefix} pc=0x{address:06X} {reg_vals} flags={flag_vals}")


def _log_bp_bytes(bridge: "BridgeCPU", address: int) -> None:
    if not _LCD_TRACE_BP_ENABLED:
        return
    memory = bridge.memory
    try:
        bp = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.BP) & 0xFF
    except Exception:
        return
    window = []
    for offset in (3, 4, 5):
        addr = INTERNAL_MEMORY_START + ((bp + offset) & 0xFF)
        try:
            window.append(memory.read_byte(addr) & 0xFF)
        except Exception:
            window.append(0)
    print(
        "[lcd-loop-bp] rust pc=0x{pc:06X} BP=0x{bp:02X} "
        "bp+3=0x{bp3:02X} bp+4=0x{bp4:02X} bp+5=0x{bp5:02X}".format(
            pc=address & PC_MASK,
            bp=bp,
            bp3=window[0],
            bp4=window[1],
            bp5=window[2],
        )
    )


def _log_stack_snapshot(bridge: "BridgeCPU", address: int) -> None:
    rng = _stack_snapshot_range()
    if not rng:
        return
    start, end = rng
    if not (start <= address <= end):
        return
    stack_reg = bridge.read_register("S")
    length = _stack_snapshot_len()
    bytes_ = [
        bridge.memory.read_byte((stack_reg + offset) & 0xFFFFFF)
        for offset in range(length)
    ]
    regs = {
        name: bridge.read_register(name)
        for name in ("A", "B", "BA", "X", "Y", "U", "S")
    }
    byte_str = " ".join(f"{b:02X}" for b in bytes_)
    reg_str = " ".join(f"{name}={value:06X}" for name, value in regs.items())
    print(
        f"[stack-snapshot] backend=rust pc=0x{address:06X} S=0x{stack_reg:06X} bytes={byte_str} {reg_str}"
    )


def _mask(bits: int) -> int:
    return (1 << bits) - 1


@dataclass
class _Snapshot:
    registers: CPURegistersSnapshot
    temps: Dict[int, int]


class MemoryAdapter:
    """Bridge SCIL space-aware loads/stores to the emulator memory object."""

    def __init__(self, memory) -> None:
        self._memory = memory

    def load(self, space: str, addr: int, size: int) -> int:
        base = self._resolve(space, addr)
        value = 0
        width = max(1, size // 8)
        for offset in range(width):
            byte = self._memory.read_byte((base + offset) & 0xFFFFFF) & 0xFF
            value |= byte << (offset * 8)
        return value & _mask(size or (width * 8))

    def store(self, space: str, addr: int, size: int, value: int) -> None:
        base = self._resolve(space, addr)
        width = max(1, size // 8)
        for offset in range(width):
            byte = (value >> (offset * 8)) & 0xFF
            self._memory.write_byte((base + offset) & 0xFFFFFF, byte)

    def _resolve(self, space: str, addr: int) -> int:
        if space == "int":
            return INTERNAL_MEMORY_START + (addr & 0xFF)
        return addr & 0xFFFFFF


class BridgeCPU:
    """Python helper that executes instructions via the Rust SCIL backend."""

    def __init__(self, memory, reset_on_init: bool = True) -> None:
        self.memory = memory
        self._runtime = rustcore.Runtime(memory=memory, reset_on_init=reset_on_init)
        self.call_sub_level = 0
        self.halted = False
        # Hint to harness that timers are handled inside Rust when enabled.
        self._timer_in_rust = os.getenv("RUST_TIMER_IN_RUST") == "1"
        self._temps: Dict[int, int] = {}
        self._fallback_cpu = None  # Lazily instantiated façade (python backend)
        self.stats_steps_rust = 0
        self.stats_decode_miss = 0
        self.stats_fallback_steps = 0
        self.stats_rust_errors = 0
        self._memory_synced = False
        if reset_on_init:
            self.power_on_reset()

    def power_on_reset(self) -> None:
        self._runtime.power_on_reset()
        self.call_sub_level = 0
        self.halted = False
        self._temps.clear()
        self._memory_synced = False

    def set_perfetto_trace(self, path: str | None) -> None:
        if hasattr(self._runtime, "set_perfetto_trace"):
            self._runtime.set_perfetto_trace(path)

    def flush_perfetto(self) -> None:
        if hasattr(self._runtime, "flush_perfetto"):
            self._runtime.flush_perfetto()

    # ------------------------------------------------------------------ #
    # Register / flag access

    def read_register(self, name: str) -> int:
        name = name.upper()
        if name.startswith("TEMP"):
            index = int(name[4:])
            return self._temps.get(index, 0)
        return int(self._runtime.read_register(name))

    def write_register(self, name: str, value: int) -> None:
        name = name.upper()
        if name.startswith("TEMP"):
            index = int(name[4:])
            if value:
                self._temps[index] = value & 0xFFFFFF
            elif index in self._temps:
                del self._temps[index]
            return
        self._runtime.write_register(name, int(value))

    def read_flag(self, name: str) -> int:
        return int(self._runtime.read_flag(name.upper()))

    def write_flag(self, name: str, value: int) -> None:
        self._runtime.write_flag(name.upper(), int(value))

    # ------------------------------------------------------------------ #
    # Execution helpers

    def execute_instruction(self, address: int) -> Tuple[int, int]:
        if not self._memory_synced:
            self._initialise_rust_memory()
        address &= PC_MASK
        prev_bp = None
        if _LCD_TRACE_BP_ENABLED:
            try:
                prev_bp = (
                    self.memory.read_byte(
                        INTERNAL_MEMORY_START + IMEMRegisters.BP
                    )
                    & 0xFF
                )
            except Exception:  # pragma: no cover - diagnostics only
                prev_bp = None
        if _should_trace_lcd(address):
            _log_lcd_loop_state("rust", address, self.read_register, self.read_flag)
            _log_bp_bytes(self, address)
        if _should_snapshot_stack(address):
            _log_stack_snapshot(self, address)
        if address != self._read_pc():
            self._runtime.write_register("PC", address)
        snapshot = self.snapshot_registers()
        try:
            opcode, length = self._runtime.execute_instruction()
            self._flush_external_writes()
            self._drain_rust_irq()
        except Exception as exc:  # pragma: no cover - exercised via integration
            self.stats_rust_errors += 1
            logger.warning(
                "Rust SCIL execution failed at PC=%06X: %s",
                address & PC_MASK,
                exc,
            )
            self.restore_snapshot(snapshot)
            opcode, length = self._execute_via_fallback(address)
            return opcode, length

        self.halted = bool(getattr(self._runtime, "halted", False))
        self.stats_steps_rust += 1
        if _LCD_TRACE_BP_ENABLED and prev_bp is not None:
            try:
                new_bp = (
                    self.memory.read_byte(
                        INTERNAL_MEMORY_START + IMEMRegisters.BP
                    )
                    & 0xFF
                )
            except Exception:  # pragma: no cover - diagnostics only
                new_bp = prev_bp
            if new_bp != prev_bp:
                print(
                    "[bp-write] rust pc=0x{pc:06X} BP=0x{bp:02X}".format(
                        pc=address & PC_MASK,
                        bp=new_bp,
                    )
                )
        if os.getenv("RUST_PC_TRACE"):
            print(f"[rust-pc] PC=0x{address:06X}")
        return opcode, length

    # ------------------------------------------------------------------ #
    # Snapshots

    def _read_pc(self) -> int:
        return int(self._runtime.read_register("PC")) & PC_MASK

    def snapshot_cpu_registers(self) -> CPURegistersSnapshot:
        temps = dict(self._temps)
        snapshot = CPURegistersSnapshot(
            pc=self._read_pc(),
            ba=self.read_register("BA"),
            i=self.read_register("I"),
            x=self.read_register("X"),
            y=self.read_register("Y"),
            u=self.read_register("U"),
            s=self.read_register("S"),
            f=self.read_register("F"),
            temps=temps,
            call_sub_level=self.call_sub_level,
        )
        return snapshot

    def load_cpu_snapshot(self, snapshot: CPURegistersSnapshot) -> None:
        self._runtime.write_register("PC", snapshot.pc & PC_MASK)
        self._runtime.write_register("BA", snapshot.ba)
        self._runtime.write_register("I", snapshot.i)
        self._runtime.write_register("X", snapshot.x)
        self._runtime.write_register("Y", snapshot.y)
        self._runtime.write_register("U", snapshot.u)
        self._runtime.write_register("S", snapshot.s)
        self._runtime.write_register("F", snapshot.f)
        self._temps = dict(snapshot.temps)
        self.call_sub_level = snapshot.call_sub_level

    # ------------------------------------------------------------------ #
    # Keyboard helpers (pure-Rust bridge)

    def keyboard_press_matrix_code(self, matrix_code: int) -> bool:
        try:
            # Pass the matrix code straight through; the Rust keyboard handles debouncing.
            self._runtime.keyboard_press(int(matrix_code) & 0x7F)
            return True
        except AttributeError:
            return False

    def keyboard_release_matrix_code(self, matrix_code: int) -> bool:
        try:
            self._runtime.keyboard_release(int(matrix_code) & 0x7F)
            return True
        except AttributeError:
            return False

    def keyboard_scan_tick(self) -> int:
        try:
            return int(self._runtime.keyboard_scan_tick())
        except AttributeError:
            return 0

    def keyboard_fifo_snapshot(self) -> bytes:
        try:
            return bytes(self._runtime.keyboard_fifo_snapshot())
        except AttributeError:
            return b""

    def keyboard_irq_count(self) -> int:
        try:
            return int(self._runtime.keyboard_irq_count())
        except AttributeError:
            return 0

    # ------------------------------------------------------------------ #
    # Misc helpers (used by CPU facade proxies)

    def snapshot_registers(self) -> _Snapshot:
        snapshot = self.snapshot_cpu_registers()
        return _Snapshot(registers=snapshot, temps=dict(self._temps))

    def restore_snapshot(self, snapshot: _Snapshot) -> None:
        self.load_cpu_snapshot(snapshot.registers)
        self._temps = dict(snapshot.temps)

    # ------------------------------------------------------------------ #
    # Fallback helpers

    def _get_fallback_cpu(self):
        if self._fallback_cpu is None:
            from sc62015.pysc62015.cpu import CPU as _FacadeCPU

            self._fallback_cpu = _FacadeCPU(
                self.memory, reset_on_init=False, backend="python"
            )
        return self._fallback_cpu

    def _execute_via_fallback(self, address: int) -> Tuple[int, int]:
        fallback = self._get_fallback_cpu()
        snapshot = self.snapshot_registers()
        fallback.apply_snapshot(snapshot.registers)

        previous_cpu = getattr(self.memory, "cpu", None)
        self.memory.set_cpu(fallback)
        eval_info = None
        try:
            eval_info = fallback.execute_instruction(address)
            new_regs = fallback.snapshot_registers()
            self.stats_fallback_steps += 1
        finally:
            if previous_cpu is not None:
                self.memory.set_cpu(previous_cpu)
            else:
                self.memory.set_cpu(self)

        self.load_cpu_snapshot(new_regs)
        self._runtime.write_register("PC", new_regs.pc & PC_MASK)
        self.halted = False  # Legacy path doesn't provide halted info; assume false
        instr_info = getattr(eval_info, "instruction_info", None) if eval_info else None
        length = 0
        if instr_info is not None and getattr(instr_info, "length", None) is not None:
            length = int(instr_info.length)  # type: ignore[attr-defined]
        opcode = self.memory.read_byte(address & PC_MASK) & 0xFF
        self._memory_synced = False
        return opcode, length or 1

    # ------------------------------------------------------------------ #
    # Stats

    def get_stats(self) -> Dict[str, int | dict[int, int]]:
        stats = {
            "steps_rust": self.stats_steps_rust,
            "decode_miss": self.stats_decode_miss,
            "fallback_steps": self.stats_fallback_steps,
            "rust_errors": self.stats_rust_errors,
        }
        profile = self.get_runtime_profile_stats()
        if profile:
            stats["runtime_profile"] = profile
        return stats

    def set_runtime_profile_enabled(self, enabled: bool) -> None:
        setter = getattr(self._runtime, "set_profile_enabled", None)
        if callable(setter):
            setter(bool(enabled))

    def reset_runtime_profile_stats(self) -> None:
        reset = getattr(self._runtime, "reset_profile_stats", None)
        if callable(reset):
            reset()

    def get_runtime_profile_stats(self) -> Dict[str, object]:
        getter = getattr(self._runtime, "get_profile_stats", None)
        if not callable(getter):
            return {}
        stats = getter()
        if isinstance(stats, dict):
            return stats
        try:
            return dict(stats)
        except Exception:
            return {}

    def export_lcd_snapshot(self):
        exporter = getattr(self._runtime, "export_lcd_snapshot", None)
        if not callable(exporter):
            return None, None
        return exporter()

    def set_interrupt_state(
        self,
        pending: bool,
        imr: int,
        isr: int,
        next_mti: int,
        next_sti: int,
        source: str | None = None,
        in_interrupt: bool = False,
        interrupt_stack: Iterable[int] | None = None,
        next_interrupt_id: int = 0,
    ) -> None:
        sync = getattr(self._runtime, "set_interrupt_state", None)
        if not callable(sync):
            return
        stack = list(interrupt_stack) if interrupt_stack is not None else []
        sync(
            bool(pending),
            int(imr) & 0xFF,
            int(isr) & 0xFF,
            int(next_mti),
            int(next_sti),
            source,
            bool(in_interrupt),
            stack,
            int(next_interrupt_id),
        )

    def _initialise_rust_memory(self) -> None:
        exported = self.memory.export_flat_memory()
        if len(exported) == 3:
            snapshot, fallback_ranges, readonly_ranges = exported
        elif len(exported) == 2:
            snapshot, fallback_ranges = exported
            readonly_ranges = ()
        else:  # pragma: no cover - defensive path
            snapshot, fallback_ranges, readonly_ranges = bytes(), (), ()
        ranges = list(fallback_ranges)
        if os.getenv("RUST_FORCE_PYTHON_MEMORY") == "1":
            ranges.append((INTERNAL_MEMORY_START, INTERNAL_MEMORY_START + 0xFF))
            ranges.append((0, INTERNAL_MEMORY_START - 1))
        self._runtime.load_external_memory(snapshot)
        try:
            internal_bytes = self.memory.get_internal_memory_bytes()
            self._runtime.load_internal_memory(internal_bytes)
            if os.getenv("RUST_DEBUG_INIT") == "1":
                imr_index = int(IMEMRegisters.IMR) & 0xFF
                imr_value = internal_bytes[imr_index] if imr_index < len(internal_bytes) else 0
                print(f"[rust-init] Loaded IMR=0x{imr_value:02X} len={len(internal_bytes)}")
                regs = self.snapshot_cpu_registers()
                print(
                    "[rust-init] Registers: PC=0x{pc:06X} U=0x{u:06X} S=0x{s:06X} F=0x{f:02X}".format(
                        pc=regs.pc & PC_MASK,
                        u=regs.u & PC_MASK,
                        s=regs.s & PC_MASK,
                        f=regs.f & 0xFF,
                    )
                )
        except Exception:
            logger.exception("Failed to mirror internal memory into Rust runtime")
        self._runtime.set_python_ranges(ranges)
        try:
            self._runtime.set_readonly_ranges(list(readonly_ranges))
        except AttributeError:
            pass
        try:
            keyboard_bridge = bool(
                getattr(self.memory, "_keyboard_bridge_enabled", False)
            )
            self._runtime.set_keyboard_bridge(keyboard_bridge)
        except AttributeError:
            pass
        self._memory_synced = True

    def _flush_external_writes(self) -> None:
        writes = self._runtime.drain_external_writes()
        if writes:
            self.memory.apply_external_writes([(addr, value) for addr, value in writes])
        drain_internal = getattr(self._runtime, "drain_internal_writes", None)
        if callable(drain_internal):
            internal_writes = drain_internal()
            if internal_writes:
                if os.getenv("RUST_INTERNAL_WRITE_TRACE") == "1":
                    print(
                        f"[rust-internal-sync] count={len(internal_writes)} first=0x{internal_writes[0][0]:06X}"
                    )
                watch = _imem_watch_list()
                ignore = _imem_ignore_list()
                filtered = []
                for addr, value in internal_writes:
                    if addr in ignore:
                        continue
                    clamped_val = _clamp_imr_isr(value, addr, self.memory)
                    filtered.append((addr, clamped_val))
                if filtered:
                    self.memory.apply_internal_writes(
                        [(addr, value) for addr, value in filtered]
                    )
                else:
                    internal_writes = []
                if watch:
                    pc_val = self.read_register("PC")
                    for addr, value in internal_writes:
                        if addr in watch:
                            print(
                                "[rust-imem-watch] pc=0x{pc:06X} addr=0x{addr:06X} value=0x{val:02X}".format(
                                    pc=pc_val & PC_MASK,
                                    addr=addr & PC_MASK,
                                    val=value & 0xFF,
                                )
                            )

    def notify_host_write(self, address: int, value: int) -> None:
        """Mirror host-initiated writes into the Rust runtime snapshot."""

        if not self._memory_synced:
            return
        addr_mask = ADDRESS_SPACE_SIZE - 1
        addr24 = int(address) & addr_mask
        if os.getenv("RUST_HOST_WRITE_TRACE") == "1":
            print(
                f"[rust-host-write] addr=0x{addr24:06X} value=0x{int(value) & 0xFF:02X}"
            )
        try:
            self._runtime.apply_host_write(addr24, int(value) & 0xFF)
        except RuntimeError as exc:
            if "Already borrowed" not in str(exc):
                logger.exception("Failed to mirror host write to Rust backend")
        except Exception:
            logger.exception("Failed to mirror host write to Rust backend")

    def _drain_rust_irq(self) -> None:
        """Poll Rust-side timer IRQs when timers run inside Rust."""

        try:
            pending = getattr(self._runtime, "drain_pending_irq", None)
            if not callable(pending):
                return
            source = pending()
        except Exception:
            return
        if not source:
            return
        # Mirror ISR bits into Python memory so the existing irq machinery sees them
        try:
            isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            isr_val = self.memory.read_byte(isr_addr) & 0xFF
            if "MTI" in source:
                isr_val |= int(ISRFlag.MTI)
            if "STI" in source:
                isr_val |= int(ISRFlag.STI)
            self.memory.write_byte(isr_addr, isr_val & 0xFF)
            cpu = getattr(self.memory, "cpu", None)
            if cpu is not None:
                try:
                    cpu._irq_pending = True  # type: ignore[attr-defined]
                    cpu._irq_source = (
                        source if isinstance(source, str) else str(source)
                    )  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            return

    # Snapshot helpers (Rust-native snapshots). Note: these do not currently
    # propagate state back into the Python memory image; intended for pure-Rust
    # capture/replay paths.
    def save_snapshot(self, path: str) -> None:
        saver = getattr(self._runtime, "save_snapshot", None)
        if callable(saver):
            saver(path)

    def load_snapshot(self, path: str) -> None:
        loader = getattr(self._runtime, "load_snapshot", None)
        if callable(loader):
            loader(path)
            self._memory_synced = True
        # These helpers intentionally do not mirror Rust snapshot contents back
        # into the Python memory image; they are used for pure-Rust runs.


__all__ = ["BridgeCPU", "MemoryAdapter"]
