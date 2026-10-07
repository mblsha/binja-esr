// PY_SOURCE: pce500/tests/test_oz9600_reference.py
// PY_SOURCE: pce500/tests/test_oz9600_audio.py
// PY_SOURCE: pce500/tests/test_uart.py
use super::*;
use crate::llama::state::{
    BlockTransferPolicy, ByteArithmeticSourcePolicy, IsrSoftwareWritePolicy,
};

fn synthetic_runtime() -> CoreRuntime {
    let mut fixed = vec![0; 0x20000];
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 14]);
    configure_hardware(&fixed, Hardware::default()).unwrap()
}

#[test]
fn workspace_and_top_ram_share_guest_byte_word_and_preflight_accesses() {
    let mut rt = synthetic_runtime();
    let program = [
        0x0a, 0x5a, 0xa5, // MV BA,A55A
        0xaa, 0x00, 0x00, 0x01, // MV [10000],BA
        0x8a, 0x00, 0x00, 0x0b, // MV BA,[B0000]
        0xa8, 0xff, 0xff, 0x0b, // MV [BFFFF],A
        0x88, 0xff, 0xff, 0x01, // MV A,[1FFFF]
    ];
    rt.load_rom(&program, 0xe0000);
    rt.step_scheduler_boundaries(3).unwrap();
    assert_eq!(rt.get_reg("BA"), 0xa55a);
    assert_eq!(rt.memory.read_byte_for_preflight(0x10000, None), Some(0x5a));
    assert_eq!(rt.memory.read_byte_for_preflight(0xb0001, None), Some(0xa5));
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.get_reg("A"), 0x5a);
    assert_eq!(rt.memory.read_byte_for_preflight(0x1ffff, None), Some(0x5a));
    assert_eq!(rt.oz9600_hardware().unwrap().borrow().ram.len(), 0x40000);
    // Neither view is backed by the old external shadow. A host fixture write
    // there must not change a guest/preflight read or the retained payload.
    let before = rt.oz9600_retained_state().unwrap();
    rt.memory.write_external_slice(0x10000, &[0x11, 0x22]);
    assert_eq!(rt.memory.read_byte_for_preflight(0x10000, None), Some(0x5a));
    assert_eq!(rt.oz9600_retained_state().unwrap(), before);
    assert_eq!(before.len(), retained::IMAGE_SIZE);
    assert_eq!(&before[..8], b"OZBAT02\0");
}

#[test]
fn opt_in_audio_uses_shared_boundary_timing_and_does_not_mutate_execution() {
    let mut program = vec![0x30, 0xcc, 0xfd, 0x90]; // MV (SCR), ISE | BZ0
    program.extend([0x00; 64]); // NOPs
    program.extend([0x30, 0xcc, 0xfd, 0x80]); // MV (SCR), ISE
    program.extend([0x00; 64]);
    let run = |sliced: bool, capture: bool| {
        let mut rt = synthetic_runtime();
        rt.load_rom(&program, 0xe0000);
        rt.set_oz9600_audio_enabled(capture).unwrap();
        if sliced {
            for _ in 0..130 {
                rt.step_scheduler_boundaries(1).unwrap();
            }
        } else {
            rt.step_scheduler_boundaries(130).unwrap();
        }
        let state = (
            rt.state.pc(),
            rt.cycle_count(),
            rt.instruction_count(),
            rt.memory.read_internal_byte_silent(0xfd),
            rt.oz9600_retained_state().unwrap(),
        );
        let pcm = rt.take_oz9600_audio().unwrap();
        (rt, state, pcm)
    };
    let (mut rt, state, batch) = run(false, true);
    let (_, sliced_state, sliced) = run(true, true);
    let (_, silent_state, silent) = run(false, false);
    assert_eq!(state, sliced_state);
    assert_eq!(state, silent_state);
    assert_eq!(batch.samples, sliced.samples);
    assert!(!batch.samples.is_empty());
    assert!(batch.samples.iter().any(|&s| s != 0));
    assert!(silent.samples.is_empty());
    assert_eq!(
        batch.total_samples,
        rt.elapsed_timing_units() * u64::from(audio::SAMPLE_RATE) / audio::TIMEBASE_HZ
    );
    rt.power_on_reset().unwrap();
    assert!(rt.oz9600_hardware().unwrap().borrow().audio.enabled());
    assert_eq!(rt.take_oz9600_audio().unwrap().total_samples, 0);
    let mut other = CoreRuntime::new();
    assert!(other.set_oz9600_audio_enabled(true).is_err());
    assert!(other.take_oz9600_audio().is_err());
}

#[test]
fn eport_input_backing_is_device_owned_for_guest_word_rmw_and_block_stores() {
    let mut rt = synthetic_runtime();
    // Two word stores, OR EIL, block copy F4..F7, external F5.
    // These are guest instructions; host updates below are a component fixture.
    let program = [
        0x30, 0xcd, 0xf4, 0xf0, 0xff, // MVW (EOH), FFF0
        0x30, 0xcd, 0xf5, 0x11, 0x22, // MVW (EIL), 2211
        0x30, 0x79, 0xf5, 0xff, // OR (EIL), FF
        0x0b, 0x04, 0x00, // I=4 bytes
        0x30, 0xcb, 0xf4, 0x20, // MVL (EOH),(20)
        0xa8, 0xf5, 0x00, 0x00, // MV [000F5],A
    ];
    rt.load_rom(&program, 0xe0000);
    for (offset, value) in (0x20..0x24).zip([0x10, 0x20, 0x30, 0x40]) {
        rt.memory.write_internal_byte(offset, value);
    }
    rt.memory.write_internal_byte(0xf5, 0x04);
    rt.memory.write_internal_byte(0xf6, 0xa5);
    rt.step_scheduler_boundaries(3).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xf4), Some(0xf0));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0x04));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf6), Some(0xa5));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf7), Some(0));
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xf4), Some(0x10));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0x04));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf6), Some(0xa5));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf7), Some(0x40));
    rt.set_reg("A", 0x5a);
    rt.step_scheduler_boundaries(1).unwrap();
    assert_eq!(rt.memory.read_byte_for_preflight(0xf5, None), Some(0x5a));
    rt.memory
        .apply_host_write_with_cycle(0x1000f5, 0x40, None, None);
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0x40));
    // The device rule must not silently change the shared PC-E500 default.
    let mut other = CoreRuntime::new();
    other.load_rom(&program[..5], 0xe0000);
    other.set_reg("PC", 0xe0000);
    other.step_scheduler_boundaries(1).unwrap();
    assert_eq!(other.memory.read_internal_byte_silent(0xf5), Some(0xff));
}

#[test]
fn guest_input_write_reference_only_claims_eil_and_eih() {
    for address in [0x000f5, 0x000f6, 0x1000f4, 0x1000f7, 0x1000ff] {
        assert!(guest_internal_write_is_allowed(address));
    }
    for address in [0x1000f5, 0x1000f6] {
        assert!(!guest_internal_write_is_allowed(address));
    }
}

#[test]
fn model_factory_requires_a_verified_bundle_and_preserves_target_on_error() {
    assert_eq!(DeviceModel::parse("oz9600"), Some(DeviceModel::Oz9600));
    assert_eq!(DeviceModel::Oz9600.label(), "oz-9600");
    let mut rt = CoreRuntime::new();
    rt.memory.write_external_byte(0x1234, 0xa5);
    assert!(DeviceModel::Oz9600
        .configure_runtime(&mut rt, &vec![0; 0x100000])
        .is_err());
    assert_eq!(rt.device_model(), DeviceModel::PcE500);
    assert_eq!(rt.memory.read_byte_for_preflight(0x1234, None), Some(0xa5));
    assert!(rt.oz9600_hardware().is_none());
}

#[test]
fn strict_device_defaults_do_not_enable_diagnostic_cpu_clock_or_pc_devices() {
    let mut rt = synthetic_runtime();
    assert_eq!(rt.device_model(), DeviceModel::Oz9600);
    assert!(rt.pce500_peripherals.is_none());
    let sio = rt.sio.as_ref().expect("OZ owns a register UART");
    assert!(sio.is_register_uart());
    assert_eq!(sio.uart().unwrap().control, 0);
    let serial = sio.snapshot(&rt.memory);
    assert!(serial.workspace.is_empty());
    assert!(serial.auto_response.is_none());
    assert!(!serial.rom_shortcuts_enabled);
    assert!(!rt.timer.enabled);
    assert_eq!(
        rt.state.block_transfer_policy(),
        BlockTransferPolicy::Independent
    );
    assert_eq!(
        rt.state.byte_arithmetic_source_policy(),
        ByteArithmeticSourcePolicy::Strict
    );
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    assert_eq!(
        rt.lcd.as_deref().unwrap().matrix_frame().geometry(),
        (336, 240)
    );
    assert_eq!(rt.memory.read_byte_for_preflight(0xc0000, None), None);
    rt.step(1).unwrap();
    assert_eq!(rt.instruction_count(), 1);
    assert!(rt
        .configure_oz9600_profile(ExecutionProfile::Experimental)
        .is_err());
}

#[test]
fn experimental_profile_is_explicit_and_retained_import_suppresses_diagnostic_seed() {
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::Experimental)
        .unwrap();
    assert!(rt.timer.enabled);
    assert_eq!(
        rt.state.block_transfer_policy(),
        BlockTransferPolicy::CoupledPredecrement
    );
    assert_eq!(
        rt.state.byte_arithmetic_source_policy(),
        ByteArithmeticSourcePolicy::LowByte
    );
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0xd0));
    let hw = rt.oz9600_hardware().unwrap().clone();
    assert_eq!(
        hw.borrow().execution.diagnostic_rtc_a2_period,
        Some(1_000_000)
    );
    assert!(rt.restore_oz9600_retained_state(&[0; 104]).is_err());
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0xd0));
    hw.borrow_mut().retained_loaded = true; // Unit fixture, not a restore proof.
    rt.configure_oz9600_profile(ExecutionProfile::Experimental)
        .unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    rt.configure_oz9600_profile(ExecutionProfile::Strict)
        .unwrap();
    assert!(!rt.timer.enabled);
    assert!(!hw.borrow().execution.diagnostic_lcc7_halt_main_timer);
}

#[test]
fn clear_only_experimental_profile_uses_zero_bp_without_host_register_corrections() {
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalIsrClearOnly)
        .unwrap();
    assert!(rt.timer.enabled);
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    assert_eq!(
        rt.state.isr_software_write_policy(),
        IsrSoftwareWritePolicy::ClearOnly
    );
    assert_eq!(
        rt.state.block_transfer_policy(),
        BlockTransferPolicy::CoupledPredecrement
    );
    assert_eq!(
        rt.state.byte_arithmetic_source_policy(),
        ByteArithmeticSourcePolicy::LowByte
    );
    rt.configure_oz9600_profile(ExecutionProfile::Strict)
        .unwrap();
    assert_eq!(
        rt.state.isr_software_write_policy(),
        IsrSoftwareWritePolicy::Replace
    );
    assert!(!rt.timer.enabled);
}

#[test]
fn mti_writable_isr_profile_uses_zero_bp_and_explicit_cpu_policy() {
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalIsrMtiWritable)
        .unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    assert_eq!(
        rt.state.isr_software_write_policy(),
        IsrSoftwareWritePolicy::ClearOnlyExceptMti
    );
    assert!(rt.timer.enabled);
}

#[test]
fn v1_profile_initializes_cpu_without_patching_empty_or_saved_battery_memory() {
    let mut rt = synthetic_runtime();
    let empty = rt.oz9600_retained_state().unwrap();
    assert!(rt
        .oz9600_hardware()
        .unwrap()
        .borrow()
        .ram
        .iter()
        .all(|b| *b == 0));
    rt.configure_oz9600_profile(ExecutionProfile::ProvisionalV1)
        .unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0xd0));
    assert_eq!(rt.oz9600_retained_state().unwrap(), empty);
    assert_eq!(rt.instruction_count(), 0);
    assert_eq!(rt.cycle_count(), 0);
    // Export/reload/reset before initialization must still use the cold policy.
    rt.oz9600_hardware().unwrap().borrow_mut().retained_loaded = true;
    rt.configure_oz9600_profile(ExecutionProfile::ProvisionalV1)
        .unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0xd0));
    assert_eq!(rt.oz9600_retained_state().unwrap(), empty);
    assert_eq!(
        rt.state.isr_software_write_policy(),
        IsrSoftwareWritePolicy::ClearOnlyExceptMti
    );
    {
        let hw = rt.oz9600_hardware().unwrap().borrow();
        assert!(hw.execution.diagnostic_on_irq_edge_only);
        assert!(hw.execution.diagnostic_irq_imr_only);
        assert_eq!(hw.execution.experimental_rtc_second_period, Some(1_024_000));
    }
    rt.oz9600_hardware().unwrap().borrow_mut().ram[0x2345] = 0xa5;
    // Synthetic ROM is deliberately ineligible for validated restoration.
    // Model a loaded backing only to test profile configuration; genuine ROM
    // restoration is exercised by the physical-input workflow suite.
    rt.oz9600_hardware().unwrap().borrow_mut().retained_loaded = true;
    let saved = rt.oz9600_retained_state().unwrap();
    rt.configure_oz9600_profile(ExecutionProfile::ProvisionalV1)
        .unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    assert_eq!(rt.oz9600_retained_state().unwrap(), saved);
    rt.step(1).unwrap();
    assert!(rt
        .configure_oz9600_profile(ExecutionProfile::ProvisionalV1)
        .is_err());
}

#[test]
fn experimental_calendar_runs_in_halt_freezes_in_hold_and_requires_profile_opt_in() {
    let mut rt = synthetic_runtime();
    let hw = rt.oz9600_hardware().unwrap().clone();
    assert!(hw
        .borrow()
        .execution
        .experimental_rtc_second_period
        .is_none());
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalRtc)
        .unwrap();
    assert_eq!(
        hw.borrow().execution.experimental_rtc_second_period,
        Some(1_024_000)
    );
    assert_eq!(
        hw.borrow().execution.diagnostic_rtc_a2_period,
        Some(512_000)
    );
    assert_eq!(rt.memory.read_internal_byte_silent(0xec), Some(0));
    assert_eq!(
        rt.state.isr_software_write_policy(),
        IsrSoftwareWritePolicy::ClearOnlyExceptMti
    );
    rt.configure_oz9600_profile(ExecutionProfile::Strict)
        .unwrap();
    assert!(hw
        .borrow()
        .execution
        .experimental_rtc_second_period
        .is_none());
    // Component fixture: shortened interval and masked IRQ. Execution still
    // traverses the real shared boundary hook, including HALT idle boundaries.
    let start = rtc::Clock {
        year: 1993,
        month: 1,
        day: 1,
        hour: 23,
        minute: 59,
        second: 59,
        weekday: 0,
    };
    {
        let mut h = hw.borrow_mut();
        h.execution.experimental_rtc_second_period = Some(10);
        for (offset, byte) in start.encode([0; 5]).unwrap().into_iter().enumerate() {
            h.rtc.write(offset, byte);
        }
    }
    rt.state.set_halted(true);
    rt.step_scheduler_boundaries(37).unwrap();
    let elapsed = rt.cycle_count();
    assert!(elapsed >= 10);
    assert_eq!(hw.borrow().execution.rtc_elapsed_seconds, elapsed / 10);
    assert_eq!(hw.borrow().execution.rtc_second_phase_units, elapsed % 10);
    assert_eq!(hw.borrow().rtc.clock().unwrap().day, 2);
    assert_eq!(hw.borrow().rtc.read(rtc::CAUSE_A), 0x78);
    assert_eq!(rt.instruction_count(), 0);
    let saved = hw.borrow().rtc.registers();
    let phase = hw.borrow().execution.rtc_second_phase_units;
    let seconds = hw.borrow().execution.rtc_elapsed_seconds;
    hw.borrow_mut().rtc.write(rtc::CONTROL, rtc::HOLD);
    rt.step_scheduler_boundaries(11).unwrap();
    assert_eq!(hw.borrow().execution.rtc_elapsed_seconds, seconds);
    assert_eq!(hw.borrow().execution.rtc_second_phase_units, phase);
    assert_eq!(
        hw.borrow().execution.rtc_held_units,
        rt.cycle_count() - elapsed
    );
    hw.borrow_mut().rtc.write(rtc::CONTROL, 0);
    assert_eq!(hw.borrow().rtc.registers(), saved);
    rt.step_scheduler_boundaries(11).unwrap();
    assert!(hw.borrow().execution.rtc_elapsed_seconds > seconds);
    assert!(rt.state.is_halted());
}

#[test]
fn unqualified_zero_target_is_reported_without_executing_empty_ram() {
    let mut rt = synthetic_runtime();
    rt.set_reg("PC", 0);
    let error = rt.step(1).unwrap_err().to_string();
    assert!(error.contains("unqualified zero code target"));
    assert_eq!(rt.instruction_count(), 0);
    assert_eq!(rt.cycle_count(), 0);
}

#[test]
fn rtc_advances_through_off_without_cpu_timers_or_automatic_power_wake() {
    use crate::llama::state::PowerState;
    for (profile, sliced) in [
        (ExecutionProfile::ExperimentalRtc, false),
        (ExecutionProfile::ExperimentalRtc, true),
        (ExecutionProfile::ExperimentalRtcIrqImr, false),
        (ExecutionProfile::ExperimentalRtcIrqImr, true),
    ] {
        let mut rt = synthetic_runtime();
        rt.configure_oz9600_profile(profile).unwrap();
        let hw = rt.oz9600_hardware().unwrap().clone();
        let start = rtc::Clock {
            year: 1993,
            month: 1,
            day: 1,
            hour: 23,
            minute: 59,
            second: 58,
            weekday: 0,
        };
        let deadline_clock = rtc::Clock {
            year: 1993,
            month: 1,
            day: 2,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 1,
        };
        {
            let mut h = hw.borrow_mut();
            h.execution.experimental_rtc_second_period = Some(10);
            h.execution.diagnostic_rtc_a2_period = Some(5);
            for (i, b) in start.encode([0; 5]).unwrap().into_iter().enumerate() {
                h.rtc.write(i, b);
            }
            for (i, b) in deadline_clock
                .encode([0; 5])
                .unwrap()
                .into_iter()
                .enumerate()
            {
                h.rtc.write(16 + i, b);
            }
            h.rtc.write(21, 7);
            h.rtc.write(rtc::MASK_A, 1);
            h.gate[0x10] = 1;
        }
        rt.state.set_power_state(PowerState::Off);
        let deadline = rt.timer.next_mti;
        if sliced {
            rt.run_slice(27, |_| false).unwrap();
        } else {
            rt.step(27).unwrap();
        }
        assert!(rt.state.is_off());
        assert_eq!(
            (
                rt.instruction_count(),
                rt.cycle_count(),
                rt.elapsed_timing_units()
            ),
            (0, 0, 27)
        );
        assert_eq!(rt.timer.next_mti, deadline);
        assert_eq!(rt.timer.irq_total, 0);
        {
            let h = hw.borrow();
            assert_eq!(h.rtc.clock().unwrap(), deadline_clock);
            assert_eq!(h.rtc.read(rtc::CAUSE_A), 0x7d);
            assert_eq!(h.execution.rtc_timing_units, 27);
            assert_eq!(h.execution.rtc_off_elapsed_units, 27);
            assert_eq!(h.execution.rtc_second_phase_units, 7);
            assert_eq!(h.execution.rtc_a2_ticks, 5);
        }
        hw.borrow_mut().rtc.write(rtc::CONTROL, rtc::HOLD);
        rt.step(11).unwrap();
        assert_eq!(hw.borrow().execution.rtc_held_units, 11);
        assert_eq!(hw.borrow().execution.rtc_second_phase_units, 7);
        hw.borrow_mut().rtc.write(rtc::CONTROL, 0);
        rt.press_on_key();
        rt.step(1).unwrap();
        assert!(!rt.state.is_off());
        let h = hw.borrow();
        assert_eq!(h.execution.rtc_off_elapsed_units, 38);
        assert_eq!(h.execution.rtc_timing_units, rt.elapsed_timing_units());
        assert_eq!(
            h.execution.rtc_elapsed_seconds * 10
                + h.execution.rtc_second_phase_units
                + h.execution.rtc_held_units,
            h.execution.rtc_timing_units
        );
    }
    // The historical profile still uses CPU cycles and keeps its old OFF policy.
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalIsrMtiWritable)
        .unwrap();
    rt.state.set_power_state(PowerState::Off);
    rt.step(30).unwrap();
    assert_eq!(
        rt.oz9600_hardware()
            .unwrap()
            .borrow()
            .execution
            .rtc_a2_phase_units,
        0
    );
}

#[test]
fn physical_replay_on_contact_is_separate_from_matrix_and_released_without_cpu_execution() {
    let mut rt = synthetic_runtime();
    let replay = input::PhysicalReplay::parse(
        br#"{"steps":[{"boundaries":0,"on_key":true},{"boundaries":0,"on_key":false}]}"#,
    )
    .unwrap();
    let mut levels = Vec::new();
    replay
        .run_with_observer(&mut rt, |_, rt| {
            levels.push(rt.physical_on_key_pressed());
            Ok(())
        })
        .unwrap();
    assert_eq!(levels, [true, false]);
    assert_eq!(rt.instruction_count(), 0);
    assert_eq!(rt.cycle_count(), 0);
}

#[test]
fn on_contact_is_visible_in_the_rom_polled_ssr_bit_without_changing_card_presence() {
    let mut rt = synthetic_runtime();
    // F0D6B/F0D6E: MV A,(SSR); AND A,08. Keep SSR.1 as a card fixture.
    rt.load_rom(&[0x30, 0x80, 0xff, 0x70, 0x08], 0xe0000);
    rt.memory.write_internal_byte(0xff, 2);
    rt.press_on_key();
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.get_reg("A"), 8);
    assert_eq!(rt.memory.read_internal_byte_silent(0xff), Some(2));
    rt.release_on_key();
    rt.set_reg("PC", 0xe0000);
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.get_reg("A"), 0);
    assert_eq!(rt.memory.read_internal_byte_silent(0xff), Some(2));
}

#[test]
fn opt_in_on_edge_keeps_live_ssr_and_allows_foreground_after_acknowledgement() {
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalOnEdge)
        .unwrap();
    // Poll the held input, acknowledge ONKI, poll again, enable IRM|ONKM,
    // then execute foreground NOPs. A held-level predictor would transfer
    // to the interrupt vector instead of retiring the first NOP.
    rt.load_rom(
        &[
            0x30, 0x80, 0xff, 0x30, 0x71, 0xfc, 0xf7, 0x30, 0x80, 0xff, 0x30, 0xcc, 0xfb, 0x88, 0,
            0, 0,
        ],
        0xe0000,
    );
    rt.load_rom(&[0, 0x10, 0xe], 0xffffa);
    rt.load_rom(&[0x30, 0x71, 0xfc, 0xf7, 0x01], 0xe1000);
    rt.set_reg("S", 0x1f000);
    rt.memory.write_internal_byte(0xff, 2); // Separate card-presence fixture.
    rt.press_on_key();
    rt.step_scheduler_boundaries(3).unwrap();
    assert!(rt.physical_on_key_pressed());
    assert_eq!(rt.get_reg("A"), 10);
    assert_eq!(rt.memory.read_internal_byte_silent(0xff), Some(2));
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
    rt.press_on_key(); // A duplicate host notification is not another edge.
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.state.pc(), 0xe000f);
    assert_eq!(rt.timer.irq_total, 0);
    rt.release_on_key();
    rt.press_on_key();
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 8);
    rt.release_on_key(); // Release cannot acknowledge a short press.
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 8);
    rt.step_scheduler_boundaries(2).unwrap();
    assert_eq!(rt.timer.irq_total, 1);
    assert!(!rt.timer.in_interrupt);
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
    rt.step_scheduler_boundaries(1).unwrap();
    assert_eq!(rt.state.pc(), 0xe0010);
    assert_eq!(rt.timer.irq_total, 1);
}

#[test]
fn on_edge_profile_is_explicit_and_historical_profiles_keep_level_reassertion() {
    for profile in [
        ExecutionProfile::Strict,
        ExecutionProfile::Experimental,
        ExecutionProfile::ExperimentalIsrClearOnly,
        ExecutionProfile::ExperimentalIsrMtiWritable,
        ExecutionProfile::ExperimentalRtc,
    ] {
        let mut rt = synthetic_runtime();
        rt.configure_oz9600_profile(ExecutionProfile::ExperimentalOnEdge)
            .unwrap();
        rt.configure_oz9600_profile(profile).unwrap();
        assert!(
            !rt.oz9600_hardware()
                .unwrap()
                .borrow()
                .execution
                .diagnostic_on_irq_edge_only
        );
        rt.load_rom(&[0x30, 0x71, 0xfc, 0xf7, 0], 0xe0000);
        rt.press_on_key();
        rt.step_scheduler_boundaries(1).unwrap();
        assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
        rt.step_scheduler_boundaries(1).unwrap();
        assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 8);
    }
}

#[test]
fn opt_in_irq_imr_policy_accepts_after_guest_reenables_master_despite_stale_handler_metadata() {
    for profile in [
        ExecutionProfile::ExperimentalOnEdge,
        ExecutionProfile::ExperimentalIrqImr,
        ExecutionProfile::ExperimentalRtcIrqImr,
    ] {
        let mut rt = synthetic_runtime();
        rt.configure_oz9600_profile(profile).unwrap();
        rt.timer.enabled = false;
        rt.set_reg("S", 0x1f000);
        // Fixture for an abandoned interrupt context. IMR starts masked;
        // only the guest instruction may reenable IRM | ONKM.
        rt.timer.in_interrupt = true;
        rt.load_rom(&[0x30, 0xcc, 0xfb, 0x88, 0, 0], 0xe0000);
        rt.load_rom(&[0, 0x10, 0xe], 0xffffa);
        rt.load_rom(&[0x30, 0x71, 0xfc, 0xf7, 0x01], 0xe1000);
        rt.press_on_key();
        rt.release_on_key();
        rt.step_scheduler_boundaries(1).unwrap();
        assert_eq!(rt.state.pc(), 0xe0004);
        assert_eq!(
            rt.timer.irq_total, 0,
            "Master-off must still inhibit delivery"
        );
        rt.step_scheduler_boundaries(1).unwrap();
        if matches!(
            profile,
            ExecutionProfile::ExperimentalIrqImr | ExecutionProfile::ExperimentalRtcIrqImr
        ) {
            assert_eq!(rt.state.pc(), 0xe1004);
            assert_eq!(rt.timer.irq_total, 1);
            assert_eq!(rt.get_reg("S"), 0x1effb);
            rt.step_scheduler_boundaries(1).unwrap();
            assert_eq!(rt.state.pc(), 0xe0004);
            assert_eq!(rt.get_reg("S"), 0x1f000);
            assert_eq!(rt.memory.read_internal_byte_silent(0xfb), Some(0x88));
            assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
        } else {
            assert_eq!(rt.state.pc(), 0xe0005);
            assert_eq!(rt.timer.irq_total, 0);
        }
    }
}

#[test]
fn combined_rtc_irq_policy_is_explicit_and_reconfiguration_removes_each_experiment() {
    let mut rt = synthetic_runtime();
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalRtcIrqImr)
        .unwrap();
    let hw = rt.oz9600_hardware().unwrap().clone();
    {
        let h = hw.borrow();
        assert!(h.execution.diagnostic_on_irq_edge_only);
        assert!(h.execution.diagnostic_irq_imr_only);
        assert_eq!(h.execution.experimental_rtc_second_period, Some(1_024_000));
        assert_eq!(h.execution.diagnostic_rtc_a2_period, Some(512_000));
    }
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalIrqImr)
        .unwrap();
    {
        let h = hw.borrow();
        assert!(h.execution.diagnostic_on_irq_edge_only && h.execution.diagnostic_irq_imr_only);
        assert!(h.execution.experimental_rtc_second_period.is_none());
        assert_eq!(h.execution.diagnostic_rtc_a2_period, Some(1_000_000));
    }
    rt.configure_oz9600_profile(ExecutionProfile::ExperimentalRtc)
        .unwrap();
    {
        let h = hw.borrow();
        assert!(!h.execution.diagnostic_on_irq_edge_only && !h.execution.diagnostic_irq_imr_only);
        assert_eq!(h.execution.experimental_rtc_second_period, Some(1_024_000));
    }
    rt.configure_oz9600_profile(ExecutionProfile::Strict)
        .unwrap();
    let h = hw.borrow();
    assert!(!h.execution.diagnostic_on_irq_edge_only && !h.execution.diagnostic_irq_imr_only);
    assert!(h.execution.experimental_rtc_second_period.is_none());
    assert!(h.execution.diagnostic_rtc_a2_period.is_none());
}

#[test]
fn irq_imr_experiment_is_never_inferred_and_profile_reconfiguration_removes_it() {
    for profile in [
        ExecutionProfile::Strict,
        ExecutionProfile::Experimental,
        ExecutionProfile::ExperimentalIsrClearOnly,
        ExecutionProfile::ExperimentalIsrMtiWritable,
        ExecutionProfile::ExperimentalOnEdge,
        ExecutionProfile::ExperimentalRtc,
    ] {
        let mut rt = synthetic_runtime();
        assert!(
            !rt.oz9600_hardware()
                .unwrap()
                .borrow()
                .execution
                .diagnostic_irq_imr_only
        );
        rt.configure_oz9600_profile(ExecutionProfile::ExperimentalIrqImr)
            .unwrap();
        let hw = rt.oz9600_hardware().unwrap().borrow();
        assert!(hw.execution.diagnostic_irq_imr_only && hw.execution.diagnostic_on_irq_edge_only);
        assert!(hw.execution.experimental_rtc_second_period.is_none());
        drop(hw);
        rt.configure_oz9600_profile(profile).unwrap();
        assert!(
            !rt.oz9600_hardware()
                .unwrap()
                .borrow()
                .execution
                .diagnostic_irq_imr_only
        );
    }
}
