#!/usr/bin/env python3
"""Example script to run PC-E500 emulator."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pce500 import PCE500Emulator
from PIL import Image, ImageOps
from sc62015.pysc62015.emulator import RegisterName


def run_emulator(
    num_steps=20000,
    dump_pc=None,
    no_dump=False,
    save_lcd=True,
    perfetto_trace=True,
    print_stats=True,
    timeout_secs: float = 10.0,
    keyboard_impl: str = "compat",
    new_perfetto=False,
    trace_file="pc-e500.perfetto-trace",
    # Optional auto key-press controls
    auto_press_key: str | None = None,
    auto_press_after_pc: int | None = None,
    auto_release_after_instr: int | None = None,
    require_strobes: int | None = None,
    min_hold_instr: int | None = None,
    # Sweep controls: iterate rows on active column, one bit at a time
    sweep_rows: bool = False,
    sweep_hold_instr: int = 2000,
    debug_draw_on_key: bool = False,
    force_display_on: bool = False,
    fast_mode: bool = False,
    boot_skip_steps: int = 0,
    disasm_trace: bool = False,
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
    # Create emulator
    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=perfetto_trace,
        save_lcd_on_exit=save_lcd,
        keyboard_impl=keyboard_impl,
        enable_new_tracing=new_perfetto,
        trace_path=trace_file,
        disasm_trace=disasm_trace,
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
        setattr(emu, "fast_mode", bool(fast_mode))
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

    # Reset and run
    emu.reset()
    if print_stats:
        print(f"PC after reset: {emu.cpu.regs.get(RegisterName.PC):06X}")

    if print_stats:
        print(f"Running {num_steps} instructions...")

    # Optional: skip initial boot instructions without tracing to reach post-init state
    if boot_skip_steps and boot_skip_steps > 0:
        if print_stats:
            print(f"Boot-skip: executing {boot_skip_steps} instructions without tracing...")
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
            setattr(emu, "fast_mode", bool(fast_mode))
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
    pressed = False
    release_countdown = None
    latched_kil_seen = False
    strobe_target = None
    hold_until_instr = None
    current_sweep_row = 0
    sweep_hold_until = None
    last_active_cols = []
    for _ in range(num_steps):
        # Check PC before executing the next instruction
        pc_before = emu.cpu.regs.get(RegisterName.PC)

        # Auto-press once when we first reach or pass the threshold PC
        if (
            not pressed
            and auto_press_key
            and auto_press_after_pc is not None
            and pc_before >= auto_press_after_pc
        ):
            emu.press_key(auto_press_key)
            pressed = True
            if auto_release_after_instr and auto_release_after_instr > 0:
                release_countdown = auto_release_after_instr
            # Set strobe target if requested
            if require_strobes and require_strobes > 0:
                try:
                    strobe_target = getattr(emu, "_kb_strobe_count", 0) + int(
                        require_strobes
                    )
                except Exception:
                    strobe_target = None
            # Set minimum hold instructions if requested
            if min_hold_instr and min_hold_instr > 0:
                hold_until_instr = emu.instruction_count + int(min_hold_instr)

        emu.step()

        # If sweeping, find active columns from last KOL/KOH and press one row at a time
        if sweep_rows:
            koh = getattr(emu, "_last_koh", 0)
            kol = getattr(emu, "_last_kol", 0)
            # Derive active columns for active-high mapping
            active_cols = []
            for col in range(8):
                if koh & (1 << col):
                    active_cols.append(col)
            for col in range(3):
                if kol & (1 << col):
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
                        if pressed and auto_press_key:
                            emu.release_key(auto_press_key)
                            pressed = False
                    except Exception:
                        pass
                last_active_cols = active_cols

                # If not currently holding, press key at (active_col, current_sweep_row)
                if sweep_hold_until is None or emu.instruction_count >= sweep_hold_until:
                    # Release previous key
                    try:
                        if pressed and auto_press_key:
                            emu.release_key(auto_press_key)
                            pressed = False
                    except Exception:
                        pass
                    # Find a key at (active_col, current_sweep_row)
                    keycode = None
                    try:
                        locs = getattr(emu.keyboard, "key_locations", {})
                        for kc, loc in locs.items():
                            if loc.column == active_col and loc.row == current_sweep_row:
                                keycode = kc
                                break
                    except Exception:
                        keycode = None
                    if keycode:
                        emu.press_key(keycode)
                        auto_press_key = keycode
                        pressed = True
                        sweep_hold_until = emu.instruction_count + max(1, int(sweep_hold_instr))
                    # Advance to next row for next time
                    current_sweep_row = (current_sweep_row + 1) % 8

        # If we are holding the key, and we see a KIL read on PF1's column (KO10),
        # latch that we've been sampled and allow countdown-based release
        # PF1 is column 10 in the matrix
        if pressed and not latched_kil_seen:
            try:
                if 10 in getattr(emu, "_last_kil_columns", []):
                    latched_kil_seen = True
            except Exception:
                pass

        # Enforce strobe-run hold and/or min hold
        ready_by_strobes = True
        if pressed and strobe_target is not None:
            try:
                ready_by_strobes = getattr(emu, "_kb_strobe_count", 0) >= strobe_target
            except Exception:
                ready_by_strobes = True
        ready_by_min_hold = True
        if pressed and hold_until_instr is not None:
            ready_by_min_hold = emu.instruction_count >= hold_until_instr

        if (
            pressed
            and release_countdown is not None
            and latched_kil_seen
            and ready_by_strobes
            and ready_by_min_hold
        ):
            release_countdown -= 1
            if release_countdown <= 0:
                emu.release_key(auto_press_key)
                release_countdown = None
        if (time.perf_counter() - start_time) > timeout_secs:
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
        # Keyboard/IRQ debug stats (hardware keyboard only)
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
            # Print last observed keyboard scan state and strobe count (hardware keyboard)
            if hasattr(emu, "_kil_read_count"):
                print(
                    f"Keyboard reads: {getattr(emu, '_kil_read_count', 0)} last_cols={getattr(emu, '_last_kil_columns', [])} last KOL=0x{getattr(emu, '_last_kol', 0):02X} KOH=0x{getattr(emu, '_last_koh', 0):02X} strobe_writes={getattr(emu, '_kb_strobe_count', 0)}"
                )
                try:
                    hist = list(getattr(emu, '_kb_col_hist', []))
                    if hist:
                        hist_str = ", ".join(f"KO{i}:{c}" for i,c in enumerate(hist))
                        print(f"Column strobe histogram: {hist_str}")
                except Exception:
                    pass
        except Exception:
            pass
    
    # Save disassembly trace if enabled
    if disasm_trace:
        emu.save_disasm_trace()

    return emu


def main(
    dump_pc=None,
    no_dump=False,
    save_lcd=True,
    perfetto=True,
    keyboard_impl="compat",
    new_perfetto=False,
    trace_file="pc-e500.perfetto-trace",
    profile_emulator=False,
    steps: int = 20000,
    timeout_secs: float = 10.0,
    auto_press_key: str | None = None,
    auto_press_after_pc: int | None = None,
    auto_release_after_instr: int | None = None,
    require_strobes: int | None = None,
    min_hold_instr: int | None = None,
    sweep_rows: bool = False,
    sweep_hold_instr: int = 2000,
    debug_draw_on_key: bool = False,
    force_display_on: bool = False,
    fast_mode: bool = False,
    boot_skip: int = 0,
    disasm_trace: bool = False,
):
    """Example with Perfetto tracing enabled."""
    # Enable performance profiling if requested
    if profile_emulator:
        from pce500.tracing.perfetto_tracing import enable_profiling, tracer
        enable_profiling("emulator-profile.perfetto-trace")
    
    # Use context manager for automatic cleanup
    with run_emulator(
        num_steps=steps,
        dump_pc=dump_pc,
        no_dump=no_dump,
        save_lcd=save_lcd,
        perfetto_trace=perfetto,
        keyboard_impl=keyboard_impl,
        new_perfetto=new_perfetto,
        trace_file=trace_file,
        timeout_secs=timeout_secs,
        auto_press_key=auto_press_key,
        auto_press_after_pc=auto_press_after_pc,
        auto_release_after_instr=auto_release_after_instr,
        require_strobes=require_strobes,
        min_hold_instr=min_hold_instr,
        sweep_rows=sweep_rows,
        sweep_hold_instr=sweep_hold_instr,
        debug_draw_on_key=debug_draw_on_key,
        force_display_on=force_display_on,
        fast_mode=fast_mode,
        boot_skip_steps=boot_skip,
        disasm_trace=disasm_trace,
    ) as emu:
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

    # After emulator run completes, attempt OCR on saved LCD image
    try:
        ocr_path = Path("lcd_display.png")
        if ocr_path.exists():
            try:
                import pytesseract  # type: ignore
            except Exception as e:
                print(f"OCR: pytesseract not available: {e}")
            else:
                # Load and preprocess: add border, invert, upscale 4x, binarize
                # Based on optimization testing, 4x scaling with PSM 3 gives 98.6% accuracy
                im = Image.open(ocr_path).convert("L")
                # Add a 5px white border to help OCR segmentation
                im = ImageOps.expand(im, border=5, fill=255)
                # Scale 4x for better OCR accuracy (testing showed 98.6% vs 40% at 1x)
                im = im.resize((im.width * 4, im.height * 4), Image.LANCZOS)
                # Invert colors (dark text on light background)
                im = Image.eval(im, lambda v: 255 - v)
                # Binarize with threshold
                th = 128
                im = im.point(lambda v: 255 if v > th else 0, mode="1")
                try:
                    # Use PSM 3 (fully automatic segmentation) instead of PSM 6
                    # Testing showed PSM 3 at 4x scale gives best results
                    text = pytesseract.image_to_string(im, config="--psm 3")
                    print("\nOCR (lcd_display.png):")
                    print(text.strip() or "<no text recognized>")
                except Exception as e:
                    print(f"OCR failed: {e}")
        else:
            print("OCR: lcd_display.png not found (skipping)")
    except Exception as e:
        print(f"OCR error: {e}")


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
        "--keyboard",
        choices=["compat", "hardware"],
        default="compat",
        help="Select keyboard implementation (default: compat)",
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
        default=10.0,
        help="Abort run after this many seconds (default: 10.0)",
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
        "--auto-press-key",
        help="Automatically press a key (e.g., KEY_F1) once when PC threshold is reached",
    )
    parser.add_argument(
        "--auto-press-after-pc",
        type=lambda x: int(x, 0),
        help="PC threshold (hex or dec). First time PC >= this, the key is pressed",
    )
    parser.add_argument(
        "--auto-release-after-instr",
        type=int,
        default=500,
        help="Release the auto-pressed key after N instructions (default: 500)",
    )
    parser.add_argument(
        "--require-strobes",
        type=int,
        help="Hold the key until at least this many KOL/KOH writes have occurred after press",
    )
    parser.add_argument(
        "--min-hold-instr",
        type=int,
        help="Hold the key for at least this many instructions after press",
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
        action="store_true",
        help="Minimize step() overhead to run more instructions",
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
    args = parser.parse_args()
    main(
        steps=args.steps,
        timeout_secs=args.timeout_secs,
        dump_pc=args.dump_pc,
        no_dump=args.no_dump,
        save_lcd=not args.no_lcd,
        perfetto=not args.no_perfetto,
        keyboard_impl=args.keyboard,
        new_perfetto=args.perfetto,
        trace_file=args.trace_file,
        profile_emulator=args.profile_emulator,
        auto_press_key=args.auto_press_key,
        auto_press_after_pc=args.auto_press_after_pc,
        auto_release_after_instr=args.auto_release_after_instr,
        require_strobes=args.require_strobes,
        min_hold_instr=args.min_hold_instr,
        sweep_rows=args.sweep_rows,
        sweep_hold_instr=args.sweep_hold_instr,
        debug_draw_on_key=args.debug_draw_on_key,
        force_display_on=args.force_display_on,
        fast_mode=args.fast_mode,
        boot_skip=args.boot_skip,
        disasm_trace=args.disasm_trace,
    )
