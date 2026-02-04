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
from pce500.display.text_decoder import decode_display_text
from pce500.keyboard_matrix import KEY_LOCATIONS, KEY_NAMES
from pce500.tracing.perfetto_tracing import trace_dispatcher
from pce500.tracing.perfetto_tracing import tracer as new_tracer
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
    start_instruction_count = emu.instruction_count
    target_instructions = start_instruction_count + int(num_steps)
    pending_snapshots: list[int] = []
    if save_snapshots_at_steps:
        pending_snapshots = sorted(
            {int(v) for v in save_snapshots_at_steps if int(v) >= 0}
        )
    while emu.instruction_count < target_instructions:
        emu.step()

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
    if hasattr(emu, "_sync_lcd_from_backend"):
        emu._sync_lcd_from_backend()
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
    )
