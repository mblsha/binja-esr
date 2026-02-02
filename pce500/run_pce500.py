#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from pce500.emulator import IRQSource
from pce500.display.text_decoder import decode_display_text
from pce500.keyboard_matrix import KEY_LOCATIONS, KEY_NAMES
from pce500.tracing.perfetto_tracing import trace_dispatcher
from pce500.tracing.perfetto_tracing import tracer as new_tracer
from sc62015.pysc62015.constants import ISRFlag
from sc62015.pysc62015.emulator import RegisterName

KEY_SEQ_DEFAULT_HOLD = 1000
KEY_SEQ_POLL_INTERVAL = 200


class KeySeqKind(str, Enum):
    PRESS = "press"
    WAIT_OP = "wait_op"
    WAIT_TEXT = "wait_text"
    WAIT_POWER = "wait_power"
    WAIT_SCREEN_CHANGE = "wait_screen_change"
    WAIT_SCREEN_EMPTY = "wait_screen_empty"
    WAIT_SCREEN_DRAW = "wait_screen_draw"


@dataclass
class KeySeqAction:
    kind: KeySeqKind
    key: str | None = None
    label: str = ""
    hold: int = 0
    op_target: int = 0
    text: str = ""
    power_on: bool | None = None
    op_target_set: bool = False
    screen_baseline_set: bool = False
    screen_baseline: int = 0


def _parse_u64_value(raw: str) -> int:
    raw = raw.strip()
    if not raw:
        raise ValueError("missing numeric value")
    lowered = raw.lower()
    if lowered.startswith("0x"):
        return int(lowered, 16)
    return int(raw)


def resolve_key_seq_key(raw: str) -> str:
    token = raw.strip()
    if not token:
        raise ValueError("empty key token")
    lowered = token.lower()
    if lowered in ("enter", "return", "ret"):
        key = KEY_NAMES.get("ENTER")
        if key:
            return key
        raise ValueError("enter key is not mapped in the keyboard matrix")
    if lowered == "space":
        key = KEY_NAMES.get("SPACE")
        if key:
            return key
        raise ValueError("space key is not mapped in the keyboard matrix")
    if lowered in ("pf1", "pf2", "pf3", "pf4", "pf5"):
        key = KEY_NAMES.get(lowered.upper())
        if key:
            return key
        raise ValueError(f"{token} key is not mapped in the keyboard matrix")
    if lowered in ("on", "key_on", "onk"):
        return "KEY_ON"
    if token.upper().startswith("KEY_") and token.upper() in KEY_LOCATIONS:
        return token.upper()
    if len(token) == 1:
        if token.isalpha():
            lookup = token.upper()
        else:
            lookup = token
        key = KEY_NAMES.get(lookup)
        if key:
            return key
    key = KEY_NAMES.get(token)
    if key:
        return key
    raise ValueError(f"unknown key token '{raw}'")


def parse_key_seq(raw: str, default_hold: int) -> list[KeySeqAction]:
    actions: list[KeySeqAction] = []
    for token_raw in raw.replace(";", ",").split(","):
        token = token_raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered.startswith("wait-op"):
            sep = token.find(":")
            if sep == -1:
                sep = token.find("=")
            if sep == -1:
                raise ValueError(f"wait-op missing value: '{token}'")
            count = _parse_u64_value(token[sep + 1 :])
            actions.append(KeySeqAction(kind=KeySeqKind.WAIT_OP, op_target=count))
            continue
        if lowered.startswith("wait-text"):
            sep = token.find(":")
            if sep == -1:
                sep = token.find("=")
            if sep == -1:
                raise ValueError(f"wait-text missing value: '{token}'")
            value = token[sep + 1 :].strip()
            if not value:
                raise ValueError(f"wait-text expects non-empty value: '{token}'")
            actions.append(KeySeqAction(kind=KeySeqKind.WAIT_TEXT, text=value))
            continue
        if lowered.startswith("wait-screen-change"):
            if ":" in token or "=" in token:
                raise ValueError(f"wait-screen-change does not take a value: '{token}'")
            actions.append(KeySeqAction(kind=KeySeqKind.WAIT_SCREEN_CHANGE))
            continue
        if lowered.startswith("wait-screen-empty"):
            if ":" in token or "=" in token:
                raise ValueError(f"wait-screen-empty does not take a value: '{token}'")
            actions.append(KeySeqAction(kind=KeySeqKind.WAIT_SCREEN_EMPTY))
            continue
        if lowered.startswith("wait-screen-draw"):
            if ":" in token or "=" in token:
                raise ValueError(f"wait-screen-draw does not take a value: '{token}'")
            actions.append(KeySeqAction(kind=KeySeqKind.WAIT_SCREEN_DRAW))
            continue
        if lowered.startswith("wait-power"):
            sep = token.find(":")
            if sep == -1:
                sep = token.find("=")
            if sep == -1:
                raise ValueError(f"wait-power missing value: '{token}'")
            value = token[sep + 1 :].strip().lower()
            if value not in ("on", "off"):
                raise ValueError(f"wait-power expects on/off, got '{value}'")
            actions.append(
                KeySeqAction(
                    kind=KeySeqKind.WAIT_POWER,
                    power_on=(value == "on"),
                )
            )
            continue

        key_part = token
        hold = default_hold
        if ":" in token:
            key_part, hold_raw = token.split(":", 1)
            key_part = key_part.strip()
            hold_raw = hold_raw.strip()
            if hold_raw:
                hold = _parse_u64_value(hold_raw)
        key_code = resolve_key_seq_key(key_part)
        actions.append(
            KeySeqAction(
                kind=KeySeqKind.PRESS,
                key=key_code,
                label=key_part,
                hold=int(hold),
            )
        )
    return actions


def _fnv1a_64(data: bytes) -> int:
    hash_value = 0xCBF29CE484222325
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def _sync_lcd_from_backend(emu: PCE500Emulator) -> None:
    try:
        if hasattr(emu, "_sync_lcd_from_backend"):
            emu._sync_lcd_from_backend()
    except Exception:
        pass


def _capture_screen_state(emu: PCE500Emulator) -> tuple[int, bool, str]:
    _sync_lcd_from_backend(emu)
    buffer = emu.get_display_buffer()
    payload = buffer.tobytes()
    signature = _fnv1a_64(payload)
    is_blank = not any(payload)
    lines = decode_display_text(emu.lcd, emu.memory)
    text = "\n".join(lines) if lines else ""
    return signature, is_blank, text


def _is_power_on(emu: PCE500Emulator) -> bool:
    try:
        return not bool(getattr(emu.cpu, "halted"))
    except Exception:
        return True


def _inject_keyboard_event(
    emu: PCE500Emulator, key_code: str, *, release: bool = False
) -> bool:
    """Inject a debounced FIFO event and raise KEYI immediately."""

    try:
        matrix = getattr(emu.keyboard, "_matrix", None)
        inject = getattr(matrix, "inject_event", None) if matrix else None
        if not callable(inject):
            return False
        if not inject(key_code, release=release):
            return False
        emu._set_isr_bits(int(ISRFlag.KEYI))
        emu._irq_pending = True
        emu._irq_source = IRQSource.KEY
        return True
    except Exception:
        return False


class KeySeqRunner:
    def __init__(self, actions: list[KeySeqAction], *, log: bool = False):
        self.actions = actions
        self.log = log
        self.index = 0
        self.active_key: str | None = None
        self.release_at: int | None = None
        self.last_poll: int = 0

    def done(self) -> bool:
        return self.index >= len(self.actions) and self.active_key is None

    def _log(self, message: str) -> None:
        if self.log:
            print(message)

    def step(self, emu: PCE500Emulator, op_index: int) -> None:
        if self.active_key and self.release_at is not None:
            if op_index >= self.release_at:
                try:
                    if self.active_key == "KEY_ON":
                        emu.release_key(self.active_key)
                    else:
                        if not _inject_keyboard_event(
                            emu, self.active_key, release=True
                        ):
                            emu.release_key(self.active_key)
                except Exception:
                    pass
                self._log(f"key-seq: release {self.active_key} at {op_index}")
                self.active_key = None
                self.release_at = None

        if self.active_key is not None:
            return
        if self.index >= len(self.actions):
            return

        action = self.actions[self.index]
        if action.kind == KeySeqKind.PRESS:
            key_code = action.key
            if not key_code:
                self.index += 1
                return
            try:
                if key_code == "KEY_ON":
                    emu.press_key(key_code)
                else:
                    if not _inject_keyboard_event(emu, key_code, release=False):
                        emu.press_key(key_code)
            except Exception:
                pass
            hold = max(1, int(action.hold))
            self.active_key = key_code
            self.release_at = op_index + hold
            label = action.label or key_code
            self._log(f"key-seq: press {label} at {op_index} hold {hold}")
            self.index += 1
            return

        if action.kind == KeySeqKind.WAIT_OP:
            if not action.op_target_set:
                action.op_target = action.op_target + op_index
                action.op_target_set = True
                self._log(f"key-seq: wait-op until {action.op_target}")
            if op_index >= action.op_target:
                self._log(f"key-seq: wait-op done at {op_index}")
                self.index += 1
            return

        if action.kind in (
            KeySeqKind.WAIT_TEXT,
            KeySeqKind.WAIT_SCREEN_CHANGE,
            KeySeqKind.WAIT_SCREEN_EMPTY,
            KeySeqKind.WAIT_SCREEN_DRAW,
        ):
            if op_index - self.last_poll < KEY_SEQ_POLL_INTERVAL:
                return
            self.last_poll = op_index
            signature, is_blank, text = _capture_screen_state(emu)
            if action.kind == KeySeqKind.WAIT_TEXT:
                if action.text and action.text in text:
                    self._log(f"key-seq: wait-text '{action.text}' at {op_index}")
                    self.index += 1
            elif action.kind == KeySeqKind.WAIT_SCREEN_CHANGE:
                if not action.screen_baseline_set:
                    action.screen_baseline = signature
                    action.screen_baseline_set = True
                    self._log(f"key-seq: wait-screen-change baseline {signature}")
                elif signature != action.screen_baseline:
                    self._log(f"key-seq: wait-screen-change detected at {op_index}")
                    self.index += 1
            elif action.kind == KeySeqKind.WAIT_SCREEN_EMPTY:
                if is_blank:
                    self._log(f"key-seq: wait-screen-empty at {op_index}")
                    self.index += 1
            elif action.kind == KeySeqKind.WAIT_SCREEN_DRAW:
                if not is_blank:
                    self._log(f"key-seq: wait-screen-draw at {op_index}")
                    self.index += 1
            return

        if action.kind == KeySeqKind.WAIT_POWER:
            power_on = _is_power_on(emu)
            if action.power_on is None or power_on == action.power_on:
                desired = "on" if action.power_on else "off"
                self._log(f"key-seq: wait-power {desired} at {op_index}")
                self.index += 1
            return


def run_emulator(
    num_steps=20000,
    dump_pc=None,
    no_dump=False,
    save_lcd=True,
    perfetto_trace=True,
    print_stats=True,
    timeout_secs: float | None = None,
    new_perfetto=False,
    trace_file="pc-e500.perfetto-trace",
    memory_card_present: bool = True,
    # Sweep controls: iterate rows on active column, one bit at a time
    sweep_rows: bool = False,
    sweep_hold_instr: int = 2000,
    debug_draw_on_key: bool = False,
    force_display_on: bool = False,
    fast_mode: bool | None = None,
    boot_skip_steps: int = 0,
    disasm_trace: bool = False,
    display_trace: bool = False,
    display_trace_log: str | None = None,
    lcd_trace_file: str | None = None,
    lcd_trace_limit: int = 50000,
    load_snapshot: str | Path | None = None,
    save_snapshot: str | Path | None = None,
    save_snapshots_at_steps: list[int] | None = None,
    save_snapshots_prefix: str | Path = "snapshot",
    key_seq: str | None = None,
    key_seq_log: bool = False,
):
    """Run PC-E500 emulator and return the instance.

    Args:
        num_steps: Number of instructions to execute
        dump_pc: PC address to trigger memory dump
        no_dump: Disable memory dumps
        save_lcd: Save LCD display on exit
        perfetto_trace: Enable Perfetto tracing
        print_stats: Print statistics to stdout
        new_perfetto: Enable new Perfetto tracing system
        trace_file: Path for new trace file

    Returns:
        PCE500Emulator: The emulator instance after running
    """
    backend_env = (os.getenv("SC62015_CPU_BACKEND") or "").lower()
    default_fast_mode = backend_env == "llama"
    resolved_fast_mode = default_fast_mode if fast_mode is None else bool(fast_mode)
    default_timeout = 0.0 if backend_env == "llama" else 10.0
    timeout_secs = default_timeout if timeout_secs is None else float(timeout_secs)

    # Create emulator
    defer_perfetto_start = bool(load_snapshot) and bool(perfetto_trace)
    defer_trace_start = bool(load_snapshot) and bool(new_perfetto)
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=bool(perfetto_trace) and not defer_perfetto_start,
        save_lcd_on_exit=save_lcd,
        memory_card_present=bool(memory_card_present),
        enable_new_tracing=bool(new_perfetto) and not defer_trace_start,
        trace_path=trace_file,
        disasm_trace=disasm_trace,
        enable_display_trace=display_trace,
        lcd_trace_file=lcd_trace_file,
        lcd_trace_event_limit=lcd_trace_limit,
    )
    # Enable debug draw-on-key if requested
    try:
        setattr(emu, "debug_draw_on_key", bool(debug_draw_on_key))
    except Exception:
        pass
    try:
        setattr(emu, "force_display_on", bool(force_display_on))
    except Exception:
        pass
    try:
        setattr(emu, "fast_mode", bool(resolved_fast_mode))
    except Exception:
        pass

    if print_stats:
        trace_msgs = []
        if perfetto_trace:
            trace_msgs.append("retrobus-perfetto tracing")
        if new_perfetto:
            trace_msgs.append(f"new Perfetto tracing to {trace_file}")

        if trace_msgs:
            print(f"Created emulator with {' and '.join(trace_msgs)} enabled")
        else:
            print("Created emulator")

        if perfetto_trace:
            print("Retrobus trace will be saved to pc-e500.trace")

    # Handle memory dump configuration
    if no_dump:
        emu.set_memory_dump_pc(0xFFFFFF)
        if print_stats:
            print("Internal memory dumps disabled")
    elif dump_pc is not None:
        emu.set_memory_dump_pc(dump_pc)
        if print_stats:
            print(f"Internal memory dump will trigger at PC=0x{dump_pc:06X}")

    # Load ROM
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
    if rom_path.exists():
        with open(rom_path, "rb") as f:
            rom_data = f.read()
            assert len(rom_data) >= 0x100000
            rom_portion = rom_data[0xC0000:0x100000]
            emu.load_rom(rom_portion)

    # Reset and/or load snapshot then run
    emu.reset()
    if load_snapshot:
        if print_stats:
            print(f"Loading snapshot from {load_snapshot} ...")
        emu.load_snapshot(load_snapshot, backend=backend_env or None)
        if print_stats:
            print(f"PC after snapshot: {emu.cpu.regs.get(RegisterName.PC):06X}")
        if defer_perfetto_start:
            try:
                # Snapshot loading can write megabytes of RAM/IMEM; keep retrobus-perfetto
                # tracing disabled during that phase, then enable it once the runtime state
                # is ready (so traces only contain the replay window).
                emu.perfetto_enabled = True
                trace_dispatcher.start_trace(trace_file)
                emu.lcd.set_perfetto_enabled(True)
                emu.memory.set_perfetto_enabled(True)
                rust_trace_path = trace_file
                if new_perfetto:
                    rust_trace_path = f"{trace_file}.rust"
                if getattr(emu.cpu, "backend", None) == "llama":
                    emu.cpu.set_perfetto_trace(rust_trace_path)
            except Exception:
                pass
        if defer_trace_start:
            # Snapshot loading can write megabytes of RAM/IMEM; keep tracing disabled during
            # that phase, then enable it once the runtime state is ready.
            new_tracer.start(trace_file)
            # Keep Perfetto timestamps aligned to instruction indices:
            # 1 instruction tick == 1us in the final Perfetto UI.
            new_tracer.set_manual_clock_mode(True, tick_ns=1_000)
            emu.memory.set_perf_tracer(new_tracer)
            try:
                # Enable instruction-level snapshots/slices in the emulator step loop.
                setattr(emu, "_new_trace_enabled", True)
            except Exception:
                pass
            try:
                if getattr(emu.cpu, "backend", None) == "llama":
                    emu.cpu.set_perfetto_trace(f"{trace_file}.rust")
                    # Align Perfetto op_index to the snapshot instruction_count. The Rust tracer
                    # resets its internal counter when tracing starts; seed it so comparisons
                    # against standalone Rust traces remain stable.
                    setter = getattr(emu.cpu, "set_perf_instr_counter", None)
                    if callable(setter):
                        setter(int(emu.instruction_count))
            except Exception:
                pass
        boot_skip_steps = 0
    elif print_stats:
        print(f"PC after reset: {emu.cpu.regs.get(RegisterName.PC):06X}")

    if print_stats:
        print(f"Running {num_steps} instructions...")

    # Optional: skip initial boot instructions without tracing to reach post-init state
    if boot_skip_steps and boot_skip_steps > 0:
        if print_stats:
            print(
                f"Boot-skip: executing {boot_skip_steps} instructions without tracing..."
            )
        # Stop new tracing if enabled
        try:
            if new_tracer.enabled:
                new_tracer.stop()
        except Exception:
            pass
        # Speed up skip phase
        try:
            setattr(emu, "fast_mode", True)
        except Exception:
            pass
        for _ in range(int(boot_skip_steps)):
            emu.step()
        # Restore requested fast_mode after skip
        try:
            setattr(emu, "fast_mode", bool(resolved_fast_mode))
        except Exception:
            pass
        # Restart new tracer if requested for this run
        try:
            if new_perfetto and not new_tracer.enabled:
                new_tracer.start(trace_file)
                emu.memory.set_perf_tracer(new_tracer)
        except Exception:
            pass

    # Abort run after timeout_secs to avoid long hangs
    start_time = time.perf_counter()
    timed_out = False
    current_sweep_row = 0
    sweep_hold_until = None
    last_active_cols = []
    start_instruction_count = emu.instruction_count
    target_instructions = start_instruction_count + int(num_steps)
    pending_snapshots: list[int] = []
    if save_snapshots_at_steps:
        pending_snapshots = sorted(
            {int(v) for v in save_snapshots_at_steps if int(v) >= 0}
        )
    key_seq_runner: KeySeqRunner | None = None
    if key_seq:
        try:
            actions = parse_key_seq(key_seq, KEY_SEQ_DEFAULT_HOLD)
        except Exception as exc:
            raise ValueError(f"--key-seq: {exc}")
        key_seq_runner = KeySeqRunner(actions, log=bool(key_seq_log))
    while emu.instruction_count < target_instructions:
        if key_seq_runner is not None:
            key_seq_runner.step(emu, emu.instruction_count)

        emu.step()

        if key_seq_runner is not None:
            key_seq_runner.step(emu, emu.instruction_count)

        if pending_snapshots and emu.instruction_count >= pending_snapshots[0]:
            snap_step = pending_snapshots.pop(0)
            try:
                prefix = Path(save_snapshots_prefix)
                out_path = prefix.with_name(f"{prefix.name}_{snap_step}.pcsnap")
                emu.save_snapshot(out_path)
                if print_stats:
                    print(f"[snapshot] saved {out_path} @instr={emu.instruction_count}")
            except Exception as exc:
                if print_stats:
                    print(f"[snapshot] failed @instr={emu.instruction_count}: {exc}")

        # If sweeping, find active columns from last KOL/KOH and press one row at a time
        if sweep_rows:
            pressed = False
            held_keycode: str | None = None
            koh = getattr(emu, "_last_koh", 0)
            kol = getattr(emu, "_last_kol", 0)
            # Derive active columns for active-high mapping (handler semantics):
            # KO0..KO7 from KOL bits 0..7, KO8..KO10 from KOH bits 0..2
            active_cols = []
            for col in range(8):
                if kol & (1 << col):
                    active_cols.append(col)
            for col in range(3):
                if koh & (1 << col):
                    active_cols.append(col + 8)

            # If an active column exists, pick the first and sweep rows
            if active_cols:
                active_col = active_cols[0]
                # If columns changed, reset sweep row
                if active_cols != last_active_cols:
                    current_sweep_row = 0
                    sweep_hold_until = None
            # Release any currently held key
            try:
                if pressed and held_keycode:
                    emu.release_key(held_keycode)
                    pressed = False
                    held_keycode = None
            except Exception:
                pass
                last_active_cols = active_cols

                # If not currently holding, press key at (active_col, current_sweep_row)
                if (
                    sweep_hold_until is None
                    or emu.instruction_count >= sweep_hold_until
                ):
                    # Release previous key
                    try:
                        if pressed and held_keycode:
                            emu.release_key(held_keycode)
                            pressed = False
                            held_keycode = None
                    except Exception:
                        pass
                    # Find a key at (active_col, current_sweep_row)
                    keycode = None
                    try:
                        locs = getattr(emu.keyboard, "key_locations", {})
                        for kc, loc in locs.items():
                            if (
                                loc.column == active_col
                                and loc.row == current_sweep_row
                            ):
                                keycode = kc
                                break
                    except Exception:
                        keycode = None
                    if keycode:
                        emu.press_key(keycode)
                        held_keycode = keycode
                        pressed = True
                        sweep_hold_until = emu.instruction_count + max(
                            1, int(sweep_hold_instr)
                        )
                    # Advance to next row for next time
                    current_sweep_row = (current_sweep_row + 1) % 8

        if timeout_secs > 0 and (time.perf_counter() - start_time) > timeout_secs:
            timed_out = True
            if print_stats:
                print(
                    f"ERROR: Aborting run after {timeout_secs:.0f}s timeout",
                    file=sys.stderr,
                )
            break

    if print_stats:
        print(f"Executed {emu.instruction_count} instructions")
        print(f"Memory reads: {emu.memory_read_count}")
        print(f"Memory writes: {emu.memory_write_count}")

        backend_stats = getattr(emu.cpu, "backend_stats", None)
        if callable(backend_stats):
            stats = backend_stats()
            backend_name = stats.get("backend")
            if backend_name == "llama":
                print("\nSC62015 LLAMA backend stats: no additional counters exposed")

        print(f"\nCPU State after {emu.cycle_count} cycles:")
        print(f"  PC: {emu.cpu.regs.get(RegisterName.PC):06X}")
        print(
            f"  A: {emu.cpu.regs.get(RegisterName.A):02X}  B: {emu.cpu.regs.get(RegisterName.B):02X}"
        )
        print(
            f"  X: {emu.cpu.regs.get(RegisterName.X):06X}  Y: {emu.cpu.regs.get(RegisterName.Y):06X}"
        )
        print(
            f"  S: {emu.cpu.regs.get(RegisterName.S):06X}  U: {emu.cpu.regs.get(RegisterName.U):06X}"
        )
        print(f"  Flags: Z={emu.cpu.regs.get_flag('Z')} C={emu.cpu.regs.get_flag('C')}")

        # Display LCD state
        print("\nLCD Controller State:")
        print(f"  Display on: {emu.lcd.display_on}")
        print(f"  Page: {emu.lcd.page}")
        print(f"  Column: {emu.lcd.column}")
        # Keyboard/IRQ debug stats
        try:
            if hasattr(emu, "_kb_irq_count"):
                print(f"\nKeyboard IRQs delivered: {getattr(emu, '_kb_irq_count', 0)}")
        except Exception:
            pass

        # Display detailed statistics for each chip
        print("\nDetailed LCD Chip Statistics:")
        print(
            f"  Chip select usage: BOTH={emu.lcd.cs_both_count}, LEFT={emu.lcd.cs_left_count}, RIGHT={emu.lcd.cs_right_count}"
        )
        stats = emu.lcd.get_chip_statistics()
        for stat in stats:
            chip_name = "Left" if stat["chip"] == 0 else "Right"
            print(f"\n  {chip_name} Chip (Chip {stat['chip']}):")
            print(f"    Display ON: {stat['on']}")
            print(f"    Instructions received: {stat['instructions']}")
            print(f"    ON/OFF commands: {stat['on_off_commands']}")
            print(f"    Data bytes written: {stat['data_written']}")
            print(f"    Data bytes read: {stat['data_read']}")
            print(f"    Current page: {stat['page']}")
            print(f"    Current column: {stat['column']}")

        # Flag the emulator if we timed out so callers can fail with non-zero exit
        setattr(emu, "_timed_out", timed_out)
        # If available, print IMEM register access tracking for KOL/KOH/KIL
        try:
            tracking = emu.memory.get_imem_access_tracking()
            for reg in ("KOL", "KOH", "KIL", "IMR", "ISR", "EIL", "EIH"):
                if reg in tracking:
                    reads = (
                        sum(c for _, c in tracking[reg]["reads"])
                        if tracking[reg]["reads"]
                        else 0
                    )
                    writes = (
                        sum(c for _, c in tracking[reg]["writes"])
                        if tracking[reg]["writes"]
                        else 0
                    )
                    print(f"IMEM {reg}: reads={reads} writes={writes}")
            # Print last observed keyboard scan state and strobe count
            if hasattr(emu, "_kil_read_count"):
                print(
                    f"Keyboard reads: {getattr(emu, '_kil_read_count', 0)} last_cols={getattr(emu, '_last_kil_columns', [])} last KOL=0x{getattr(emu, '_last_kol', 0):02X} KOH=0x{getattr(emu, '_last_koh', 0):02X} strobe_writes={getattr(emu, '_kb_strobe_count', 0)}"
                )
                try:
                    hist = list(getattr(emu, "_kb_col_hist", []))
                    if hist:
                        hist_str = ", ".join(f"KO{i}:{c}" for i, c in enumerate(hist))
                        print(f"Column strobe histogram: {hist_str}")
                except Exception:
                    pass
        except Exception:
            pass

    if display_trace:
        trace_payload = emu.get_display_trace_log()
        spans = trace_payload.get("spans", [])
        events = trace_payload.get("events", [])
        if print_stats:
            print(
                f"Display trace captured {len(spans)} span(s) and {len(events)} event(s)"
            )
            for span in spans[-5:]:
                name = span.get("name")
                writes = len(span.get("writes", []))
                duration = span.get("duration_instr")
                print(
                    f"  - {name} wrote {writes} event(s) over {duration} instruction(s)"
                )
        if display_trace_log:
            try:
                Path(display_trace_log).write_text(json.dumps(trace_payload, indent=2))
                if print_stats:
                    print(f"Display trace written to {display_trace_log}")
            except Exception as exc:
                print(f"WARNING: could not write display trace log: {exc}")

    # Save disassembly trace if enabled
    if disasm_trace:
        emu.save_disasm_trace()

    if lcd_trace_file:
        trace_path = emu.save_lcd_trace(lcd_trace_file)
        if trace_path and print_stats:
            print(f"LCD write trace saved to {trace_path}")
        elif lcd_trace_file and print_stats:
            print("LCD trace requested but no events were captured")

    if save_snapshot:
        snapshot_path = emu.save_snapshot(save_snapshot)
        if print_stats:
            print(f"Snapshot saved to {snapshot_path}")

    if hasattr(emu, "close"):
        emu.close()

    return emu


def main(
    dump_pc=None,
    no_dump=False,
    save_lcd=True,
    perfetto=True,
    new_perfetto=False,
    trace_file="pc-e500.perfetto-trace",
    profile_emulator=False,
    memory_card_present: bool = True,
    steps: int = 20000,
    timeout_secs: float | None = None,
    sweep_rows: bool = False,
    sweep_hold_instr: int = 2000,
    debug_draw_on_key: bool = False,
    force_display_on: bool = False,
    fast_mode: bool | None = None,
    boot_skip: int = 0,
    disasm_trace: bool = False,
    display_trace: bool = False,
    display_trace_log: str | None = None,
    lcd_trace: str | None = None,
    lcd_trace_limit: int = 50000,
    load_snapshot: str | Path | None = None,
    save_snapshot: str | Path | None = None,
    perfetto_on_snapshot: bool = False,
    save_snapshots_at_steps: list[int] | None = None,
    save_snapshots_prefix: str | Path = "snapshot",
    key_seq: str | None = None,
    key_seq_log: bool = False,
):
    """Example with Perfetto tracing enabled."""
    # Enable performance profiling if requested
    if profile_emulator:
        from pce500.tracing.perfetto_tracing import enable_profiling, tracer

        enable_profiling("emulator-profile.perfetto-trace")

    backend_env = (os.getenv("SC62015_CPU_BACKEND") or "").lower()
    resolved_fast_mode = fast_mode
    if resolved_fast_mode is None:
        resolved_fast_mode = backend_env == "llama"
    resolved_timeout = timeout_secs
    if resolved_timeout is None:
        resolved_timeout = 0.0 if backend_env == "llama" else 10.0
    perfetto_enabled = perfetto
    # When loading from a snapshot and the caller did not explicitly enable tracing,
    # default to no perfetto to avoid overhead on short replay windows.
    if (
        load_snapshot
        and not new_perfetto
        and perfetto_enabled
        and not perfetto_on_snapshot
    ):
        perfetto_enabled = False

    # Use context manager for automatic cleanup
    with run_emulator(
        num_steps=steps,
        dump_pc=dump_pc,
        no_dump=no_dump,
        save_lcd=save_lcd,
        perfetto_trace=perfetto_enabled,
        new_perfetto=new_perfetto,
        trace_file=trace_file,
        memory_card_present=bool(memory_card_present),
        timeout_secs=resolved_timeout,
        sweep_rows=sweep_rows,
        sweep_hold_instr=sweep_hold_instr,
        debug_draw_on_key=debug_draw_on_key,
        force_display_on=force_display_on,
        fast_mode=resolved_fast_mode,
        boot_skip_steps=boot_skip,
        disasm_trace=disasm_trace,
        display_trace=display_trace,
        display_trace_log=display_trace_log,
        lcd_trace_file=lcd_trace,
        lcd_trace_limit=lcd_trace_limit,
        load_snapshot=load_snapshot,
        save_snapshot=save_snapshot,
        save_snapshots_at_steps=save_snapshots_at_steps,
        save_snapshots_prefix=save_snapshots_prefix,
        key_seq=key_seq,
        key_seq_log=key_seq_log,
    ) as emu:
        if False:
            pass
        # Set performance tracer for SC62015 if profiling
        if profile_emulator:
            emu.memory.set_perf_tracer(tracer)
        pass  # Everything is done in run_emulator

    # Stop profiling if it was enabled
    if profile_emulator:
        from pce500.tracing.perfetto_tracing import disable_profiling

        disable_profiling()

    # Exit with error if we hit the timeout
    if getattr(emu, "_timed_out", False):
        sys.exit(1)

    # Emit LCD text derived from controller state
    _sync_lcd_from_backend(emu)
    lines = decode_display_text(emu.lcd, emu.memory)
    print("\nLCD TEXT:")
    if lines:
        for idx, line in enumerate(lines):
            print(f"ROW{idx}: {line}")
    else:
        print("<no text decoded>")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PC-E500 Emulator Example")
    parser.add_argument(
        "--dump-pc",
        type=lambda x: int(x, 0),
        help="PC address to trigger internal memory dump (hex or decimal, e.g., 0x0F119C)",
    )
    parser.add_argument(
        "--no-dump", action="store_true", help="Disable internal memory dumps entirely"
    )
    parser.add_argument(
        "--no-lcd",
        action="store_true",
        help="Don't save LCD displays as PNG files on exit",
    )
    parser.add_argument(
        "--no-perfetto",
        action="store_true",
        help="Disable Perfetto tracing to isolate performance effects",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Number of instructions to execute (default: 20000)",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=None,
        help="Abort run after this many seconds (default: 0 for LLAMA backend, 10.0 otherwise)",
    )
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Enable new Perfetto tracing (wall-clock time)",
    )
    parser.add_argument(
        "--trace-file",
        default="pc-e500.perfetto-trace",
        help="Path to write trace file (default: pc-e500.perfetto-trace)",
    )
    parser.add_argument(
        "--profile-emulator",
        action="store_true",
        help="Enable performance profiling of emulator execution (outputs emulator-profile.perfetto-trace)",
    )
    parser.add_argument(
        "--card",
        choices=("present", "absent"),
        default="present",
        help="Enable/disable memory card emulation (default: present)",
    )
    parser.add_argument(
        "--sweep-rows",
        action="store_true",
        help="After threshold, iterate one row at a time on the active column (one-bit KIL patterns)",
    )
    parser.add_argument(
        "--sweep-hold-instr",
        type=int,
        default=2000,
        help="Instructions to hold each row during sweep (default: 2000)",
    )
    parser.add_argument(
        "--debug-draw-on-key",
        action="store_true",
        help="Debug: draw a visible marker on the LCD when a key is pressed",
    )
    parser.add_argument(
        "--force-display-on",
        action="store_true",
        help="Debug: force both LCD chips on at reset",
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Minimize step() overhead to run more instructions (default: on for LLAMA backend)",
    )
    parser.add_argument(
        "--boot-skip",
        type=int,
        default=0,
        help="Execute this many instructions first without tracing (skip boot)",
    )
    parser.add_argument(
        "--disasm-trace",
        action="store_true",
        help="Generate disassembly trace of executed instructions with control flow annotations",
    )
    parser.add_argument(
        "--display-trace",
        action="store_true",
        help="Enable display controller tracing (aggregates ROM draw calls)",
    )
    parser.add_argument(
        "--display-trace-log",
        help="Write display trace JSON payload to this path",
    )
    parser.add_argument(
        "--lcd-trace",
        help="Write raw LCD controller write trace (JSON) to this path",
    )
    parser.add_argument(
        "--lcd-trace-limit",
        type=int,
        default=50000,
        help="Maximum LCD write events to capture (default: 50000)",
    )
    parser.add_argument(
        "--load-snapshot",
        type=str,
        help="Load a .pcsnap bundle before stepping (disables perfetto by default for speed)",
    )
    parser.add_argument(
        "--perfetto-on-snapshot",
        action="store_true",
        help="When --load-snapshot is used, keep retrobus-perfetto tracing enabled (default: off)",
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        help="Save a .pcsnap bundle after execution",
    )
    parser.add_argument(
        "--save-snapshots-at-steps",
        type=str,
        help="Comma-separated list of instruction counts at which to save snapshots "
        "(e.g., 900000,1100000).",
    )
    parser.add_argument(
        "--save-snapshots-prefix",
        type=str,
        default="snapshot",
        help="Prefix (path) for --save-snapshots-at-steps outputs (default: snapshot). "
        "Files will be written as <prefix>_<steps>.pcsnap",
    )
    parser.add_argument(
        "--key-seq",
        default=None,
        help="Key sequence script (e.g. 'wait-op:10;pf1:5;wait-text:MAIN MENU').",
    )
    parser.add_argument(
        "--key-seq-log",
        action="store_true",
        help="Emit key-seq press/release/wait events while running.",
    )
    args = parser.parse_args()
    main(
        steps=args.steps,
        timeout_secs=args.timeout_secs,
        dump_pc=args.dump_pc,
        no_dump=args.no_dump,
        save_lcd=not args.no_lcd,
        perfetto=not args.no_perfetto,
        new_perfetto=args.perfetto,
        trace_file=args.trace_file,
        profile_emulator=args.profile_emulator,
        memory_card_present=(args.card == "present"),
        sweep_rows=args.sweep_rows,
        sweep_hold_instr=args.sweep_hold_instr,
        debug_draw_on_key=args.debug_draw_on_key,
        force_display_on=args.force_display_on,
        fast_mode=args.fast_mode,
        boot_skip=args.boot_skip,
        disasm_trace=args.disasm_trace,
        display_trace=args.display_trace,
        display_trace_log=args.display_trace_log,
        lcd_trace=args.lcd_trace,
        lcd_trace_limit=args.lcd_trace_limit,
        load_snapshot=args.load_snapshot,
        save_snapshot=args.save_snapshot,
        perfetto_on_snapshot=args.perfetto_on_snapshot,
        save_snapshots_at_steps=(
            [int(v) for v in args.save_snapshots_at_steps.split(",") if v.strip()]
            if args.save_snapshots_at_steps
            else None
        ),
        save_snapshots_prefix=args.save_snapshots_prefix,
        key_seq=args.key_seq,
        key_seq_log=args.key_seq_log,
    )
