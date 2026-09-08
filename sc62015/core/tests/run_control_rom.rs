// PY_SOURCE: pce500/emulator.py:PCE500Emulator
//! Opt-in Rust ROM comparison. No synthetic screen or FIFO injection.

use sc62015_core::pacing::{ExecutionMode, Pacer};
use sc62015_core::{collect_registers, CoreRuntime, DeviceModel};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires private PC-E500 ROM; run explicitly with --ignored"]
fn pc_initialization_header_precedes_firmware_key_release_readiness() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pc-e500-en.bin");
    let rom = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let model = DeviceModel::PcE500;
    let decoder = model.text_decoder(&rom).expect("ROM font decoder");
    let pf1 = sc62015_core::physical_keys::matrix_key(model, "PF1").unwrap();
    for wait_for_release in [false, true] {
        let mut rt = CoreRuntime::for_model(model, &rom).unwrap();
        rt.power_on_reset().unwrap();
        // Fixed failing ROM trace: first press at 7k, physical release at 47k,
        // S1 header visible at 548k while the firmware still remembers PF1.
        rt.step_scheduler_boundaries(7_000).unwrap();
        rt.keyboard
            .as_mut()
            .unwrap()
            .press_matrix_code(pf1, &mut rt.memory);
        rt.step_scheduler_boundaries(40_000).unwrap();
        rt.keyboard
            .as_mut()
            .unwrap()
            .release_matrix_code(pf1, &mut rt.memory);
        rt.step_scheduler_boundaries(501_000).unwrap();
        assert!(decoder
            .decode_display_text(rt.lcd.as_deref().unwrap())
            .join("\n")
            .contains("S1(MAIN):NEW CARD"));
        assert!(rt
            .keyboard
            .as_ref()
            .unwrap()
            .pressed_matrix_codes()
            .is_empty());
        assert_eq!(rt.memory.read_byte_for_preflight(0xbfcb5, None), Some(0x13));

        if wait_for_release {
            // Bound the diagnostic, but stop on state, not a chosen delay.
            for _ in 0..50_000 {
                if rt.state.pc() == 0xf175f && rt.state.is_halted() {
                    break;
                }
                rt.step_scheduler_boundaries(1).unwrap();
            }
            assert_eq!(rt.state.pc(), 0xf175f);
            assert!(rt.state.is_halted());
            assert_eq!(rt.memory.read_byte_for_preflight(0xbfcb5, None), Some(0));
            assert_eq!(rt.memory.read_byte_for_preflight(0xbfcb6, None), Some(0));
        }
        rt.keyboard
            .as_mut()
            .unwrap()
            .press_matrix_code(pf1, &mut rt.memory);
        rt.step_scheduler_boundaries(40_000).unwrap();
        rt.keyboard
            .as_mut()
            .unwrap()
            .release_matrix_code(pf1, &mut rt.memory);
        rt.step_scheduler_boundaries(600_000).unwrap();
        let text = decoder
            .decode_display_text(rt.lcd.as_deref().unwrap())
            .join("\n");
        if wait_for_release {
            assert!(text.contains("MAIN MENU"), "{text}");
        } else {
            assert!(text.contains("S1(MAIN):NEW CARD"), "{text}");
            assert!(!text.contains("MAIN MENU"));
        }
    }
}

#[test]
#[ignore = "requires both private ROMs; run explicitly with --ignored --nocapture"]
fn sliced_execution_matches_direct_boot_for_both_real_roms() {
    for (model, filename) in [
        (DeviceModel::PcE500, "pc-e500-en.bin"),
        (DeviceModel::Iq7000, "iq-7000.bin"),
    ] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(filename);
        // Missing evidence is an explicit failure of this opt-in test, not a
        // successful early return disguised as a ROM pass.
        let rom = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let machine = || {
            let mut runtime = CoreRuntime::for_model(model, &rom).unwrap();
            runtime.power_on_reset().unwrap();
            if model == DeviceModel::Iq7000 {
                runtime
                    .set_iq7000_clock_seed_yyyymmddhhmm("202609060000")
                    .unwrap();
            }
            runtime
        };
        let mut direct = machine();
        let mut sliced = machine();
        let started = Instant::now();
        direct.step_scheduler_boundaries(1_000_000).unwrap();
        let direct_elapsed = started.elapsed();
        let started = Instant::now();
        let mut remaining = 1_000_000;
        let mut slice_times = Vec::new();
        while remaining > 0 {
            let slice_start = Instant::now();
            let result = sliced
                .run_slice(remaining, |_| {
                    slice_start.elapsed() >= Duration::from_millis(4)
                })
                .unwrap();
            slice_times.push(slice_start.elapsed());
            remaining -= result.progress.boundary_budget_used;
        }
        let sliced_elapsed = started.elapsed();
        assert_eq!(
            collect_registers(&sliced.state),
            collect_registers(&direct.state)
        );
        assert_eq!(sliced.state.power_state(), direct.state.power_state());
        assert_eq!(sliced.instruction_count(), direct.instruction_count());
        assert_eq!(sliced.cycle_count(), direct.cycle_count());
        assert_eq!(
            sliced.memory.internal_slice(),
            direct.memory.internal_slice()
        );
        assert_eq!(
            sliced.memory.external_slice(),
            direct.memory.external_slice()
        );
        assert_eq!(sliced.timer.next_mti, direct.timer.next_mti);
        assert_eq!(sliced.timer.next_sti, direct.timer.next_sti);
        assert_eq!(sliced.iq7000_rtc_state(), direct.iq7000_rtc_state());
        let direct_lcd = direct.lcd.as_ref().unwrap().display_buffer();
        let sliced_lcd = sliced.lcd.as_ref().unwrap().display_buffer();
        assert_eq!(sliced_lcd, direct_lcd);
        assert!(
            direct_lcd.iter().flatten().any(|pixel| *pixel != 0),
            "ROM did not draw"
        );
        // Native rendering receives an immutable display copy, not the live
        // controller. It must decode the same text even after the guest resets.
        let decoder = model.text_decoder(&rom).expect("ROM font decoder");
        let lcd = sliced.lcd.as_deref().unwrap();
        let expected_text = decoder.decode_display_text(lcd);
        let text_frame = decoder.capture_text_frame(lcd);
        assert_eq!(
            decoder.decode_text_frame(&text_frame),
            Some(expected_text.clone())
        );
        sliced.lcd.as_deref_mut().unwrap().reset();
        assert_eq!(decoder.decode_text_frame(&text_frame), Some(expected_text));
        for mode in [ExecutionMode::Interactive, ExecutionMode::Turbo] {
            let mut paced = machine();
            let mut pacer = Pacer::for_model(model, mode);
            let mut remaining = 1_000_000;
            let mut now = 0;
            let mut calls = 0;
            while remaining > 0 {
                let slice_start = Instant::now();
                let result = paced
                    .run_automatic_slice(&mut pacer, now, remaining, |_| {
                        slice_start.elapsed() >= Duration::from_millis(4)
                    })
                    .unwrap();
                remaining -= result
                    .slice
                    .map_or(0, |slice| slice.progress.boundary_budget_used);
                // Inject scheduler jitter/background stalls without waiting in
                // real time. No clock/input/FIFO/memory patch is applied to ROM.
                now += [1_000, 1_000_000, 1_000_000_000][calls % 3];
                calls += 1;
            }
            assert_eq!(
                collect_registers(&paced.state),
                collect_registers(&direct.state)
            );
            assert_eq!(paced.state.power_state(), direct.state.power_state());
            assert_eq!(paced.instruction_count(), direct.instruction_count());
            assert_eq!(paced.cycle_count(), direct.cycle_count());
            assert_eq!(paced.elapsed_timing_units(), direct.elapsed_timing_units());
            assert_eq!(
                paced.memory.internal_slice(),
                direct.memory.internal_slice()
            );
            assert_eq!(
                paced.memory.external_slice(),
                direct.memory.external_slice()
            );
            assert_eq!(paced.timer.next_mti, direct.timer.next_mti);
            assert_eq!(paced.timer.next_sti, direct.timer.next_sti);
            assert_eq!(paced.iq7000_rtc_state(), direct.iq7000_rtc_state());
            assert_eq!(paced.lcd.as_ref().unwrap().display_buffer(), direct_lcd);
            eprintln!(
                "{}: {} pacing matched 1M explicit boundaries after {calls} host calls",
                model.label(),
                mode.label()
            );
        }
        slice_times.sort();
        let p99 = slice_times[(slice_times.len() - 1) * 99 / 100];
        eprintln!(
            "{}: direct={direct_elapsed:?}, sliced={sliced_elapsed:?}, slices={}, p99={p99:?}, max={:?}; native host-slice timing, NOT UI input latency or physical calibration",
            model.label(), slice_times.len(), slice_times.last().unwrap()
        );
    }
}
