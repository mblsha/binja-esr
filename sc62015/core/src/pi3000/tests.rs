// PY_SOURCE: pce500/pi3000/hardware.py
// PY_SOURCE: pce500/pi3000/test_retained.py
use super::*;
use sha2::Digest;

#[test]
fn retained_backing_round_trip_keeps_raw_rtc_and_fresh_devices() {
    let source = fixture(&[0; 8]);
    {
        let mut h = source.pi3000_hardware().unwrap().borrow();
        h.ram[0] = 0x12;
        h.ram[retained::MAIN_RAM_SIZE - 1] = 0x34;
        h.scratch[0] = 0x56;
        h.scratch[retained::SCRATCH_SIZE - 1] = 0x78;
        h.rtc.write(12, 0x80);
        h.rtc.write(13, 0);
        h.rtc.write(10, 4);
        h.gate[0x12] = 0x40;
        h.gpio[0] = 0x91;
    }
    let image = source.pi3000_retained_state().unwrap();
    assert_eq!(image.len(), retained::IMAGE_SIZE);
    assert_eq!(&image[..8], b"PIRAM01\0");
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(&image)),
        "7c2d76d7983487b93dfd076b2410a0e5561780f6eed09c688e23e57b25625118"
    );
    assert_eq!(source.instruction_count(), 0);
    assert_eq!(source.elapsed_timing_units(), 0);
    assert_eq!(source.pi3000_hardware().unwrap().borrow().gate[0x12], 0x40);
    let mut restored = fixture(&[0; 8]);
    restored.restore_pi3000_retained_state(&image).unwrap();
    assert_eq!(restored.pi3000_retained_state().unwrap(), image);
    let h = restored.pi3000_hardware().unwrap().borrow();
    assert_eq!((h.ram[0], h.ram[retained::MAIN_RAM_SIZE - 1]), (0x12, 0x34));
    assert_eq!(
        (h.scratch[0], h.scratch[retained::SCRATCH_SIZE - 1]),
        (0x56, 0x78)
    );
    assert_eq!(h.rtc.retained_registers()[13], 0);
    assert_eq!(h.rtc.read(13), 0x80);
    assert_eq!(h.gate[0x12], 0);
    assert_eq!(h.gpio[0], 0);
    assert_eq!(h.selector, 0xfd);
    assert_eq!(h.tablet.data_reads, 0);
}

#[test]
fn retained_backing_rejects_corruption_and_cross_rom_bank_identity_atomically() {
    let source = fixture(&[0; 8]);
    let image = source.pi3000_retained_state().unwrap();
    let mut target = fixture(&[0; 8]);
    target.pi3000_hardware().unwrap().borrow().ram[7] = 0x93;
    let before = target.pi3000_retained_state().unwrap();
    for offset in [0, 8, 40, 72, retained::IMAGE_SIZE - 1] {
        let mut bad = image.clone();
        bad[offset] ^= 1;
        assert!(target.restore_pi3000_retained_state(&bad).is_err());
        assert_eq!(target.pi3000_retained_state().unwrap(), before);
    }
    assert!(target
        .restore_pi3000_retained_state(&image[..image.len() - 1])
        .is_err());
    let mut extended = image.clone();
    extended.push(0);
    assert!(target.restore_pi3000_retained_state(&extended).is_err());
    assert_eq!(target.pi3000_retained_state().unwrap(), before);
    let mut different = fixture(&[1; 8]);
    assert!(different.restore_pi3000_retained_state(&image).is_err());
    target
        .pi3000_hardware()
        .unwrap()
        .borrow()
        .banks
        .insert(3, vec![0xa5; 0x20000]);
    let bank_before = target.pi3000_retained_state().unwrap();
    assert!(target.restore_pi3000_retained_state(&image).is_err());
    assert_eq!(target.pi3000_retained_state().unwrap(), bank_before);
}

#[test]
fn retained_backing_rejects_running_and_elapsed_halt_restore() {
    use crate::llama::state::PowerState;
    for halted in [false, true] {
        let mut rt = fixture(&[0; 8]);
        let image = rt.pi3000_retained_state().unwrap();
        if halted {
            rt.state.set_power_state(PowerState::Halted);
        }
        rt.step_scheduler_boundaries(1).unwrap();
        assert_eq!(rt.instruction_count(), if halted { 0 } else { 1 });
        let before = rt.pi3000_retained_state().unwrap();
        assert!(rt.restore_pi3000_retained_state(&image).is_err());
        assert_eq!(rt.pi3000_retained_state().unwrap(), before);
    }
    let mut other = CoreRuntime::new();
    assert!(other.pi3000_retained_state().is_err());
    assert!(other.restore_pi3000_retained_state(&[]).is_err());
}

#[test]
fn auxiliary_sample_irq_is_explicit_and_requires_active_capture() {
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::AuxiliaryControl);
    h.write(0x8a06, 0x80, None);
    assert_eq!(h.gate[0x12], 0);
    h.auxiliary_sample_irq_candidate = true;
    h.write(0x8a06, 0x80, None);
    assert_eq!(h.gate[0x12], 0);
    h.write(0x8a20, 0xfc, None);
    h.write(0x8a21, 3, None);
    h.write(0x8a02, 0x80, None);
    h.write(0x8a06, 0x80, None);
    assert_eq!(h.gate[0x12], 8);
    assert!(h.fault.is_none());
    assert!(h.ram.iter().all(|v| *v == 0));
    assert!(h.scratch.iter().all(|v| *v == 0));
    h.gate[0x12] = 0;
    h.advance_auxiliary_sample_irq(4095);
    assert_eq!(h.gate[0x12], 0);
    h.advance_auxiliary_sample_irq(4096);
    assert_eq!(h.gate[0x12], 8);
    h.gate[0x12] = 0;
    h.write(0x8a06, 0, None);
    h.advance_auxiliary_sample_irq(8192);
    assert_eq!(h.gate[0x12], 0);
}

fn fixture(code: &[u8]) -> CoreRuntime {
    let mut fixed = vec![0xff; 0x20000];
    fixed[0x10000..0x10000 + code.len()].copy_from_slice(code);
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 15]);
    configure(&fixed, BTreeMap::new(), BootstrapProfile::Strict).unwrap()
}

#[test]
fn periodic_rtc_ticks_coalesce_and_reassert_only_until_both_latches_are_acknowledged() {
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::Strict);
    h.rtc_tick_period = Some(10);
    h.advance_rtc_ticks(9);
    assert_eq!(h.rtc.read(10), 0);
    h.advance_rtc_ticks(21);
    assert_eq!((h.rtc_ticks, h.rtc_tick_phase), (3, 0));
    assert_eq!(h.rtc.read(10), 4);
    assert!(!h.external_irq()); // masked device cause remains latched
    assert_eq!(h.gate[0x12], 0);
    h.rtc.write(8, 4);
    assert!(!h.external_irq()); // gate mask is still zero
    assert_eq!(h.gate[0x12], 1);
    h.gate[0x10] = 1;
    assert!(h.external_irq()); // independent of the absent auxiliary device
    h.gate[0x12] = 0;
    assert!(h.external_irq());
    h.rtc.write(10, 0);
    assert!(h.external_irq()); // gate must also be acknowledged
    h.gate[0x12] = 0;
    assert!(!h.external_irq());
    assert!(h.ram.iter().all(|v| *v == 0));
    assert_eq!(&h.rtc.registers()[..5], &[0; 5]); // calendar policy is separate
}

#[test]
fn explicit_rtc_tick_clock_runs_in_halt_and_off_without_retiring_instructions() {
    use crate::llama::state::PowerState;
    for power in [PowerState::Running, PowerState::Halted, PowerState::Off] {
        let mut rt = fixture(&[0; 8]);
        assert!(rt.configure_pi3000_rtc_tick_candidate(0).is_err());
        rt.configure_pi3000_rtc_tick_candidate(3).unwrap();
        rt.state.set_power_state(power);
        rt.step_scheduler_boundaries(8).unwrap();
        let h = rt.pi3000_hardware().unwrap().borrow();
        assert_eq!((h.rtc_ticks, h.rtc_tick_phase), (2, 2));
        assert_eq!(h.rtc.read(10), 4);
        assert_eq!(
            rt.instruction_count(),
            if power == PowerState::Running { 8 } else { 0 }
        );
        drop(h);
        assert!(rt.configure_pi3000_rtc_tick_candidate(3).is_err());
    }
}

#[test]
fn explicit_calendar_uses_scheduler_time_in_running_halt_and_off() {
    use crate::llama::state::PowerState;
    for power in [PowerState::Running, PowerState::Halted, PowerState::Off] {
        let mut rt = fixture(&[0; 8]);
        assert!(rt.configure_pi3000_rtc_calendar_candidate(0).is_err());
        assert_eq!(
            rt.pi3000_hardware().unwrap().borrow().rtc_second_period,
            None
        );
        rt.configure_pi3000_rtc_calendar_candidate(3).unwrap();
        rt.pi3000_hardware().unwrap().borrow().rtc_event_candidate = true;
        {
            let mut h = rt.pi3000_hardware().unwrap().borrow();
            for (offset, value) in [59, 0, 8, 0xe1, 5].into_iter().enumerate() {
                h.rtc.write(offset, value); // explicit component date: 1994-01-01 00:00:59
            }
        }
        rt.state.set_power_state(power);
        rt.step_scheduler_boundaries(8).unwrap();
        let h = rt.pi3000_hardware().unwrap().borrow();
        assert_eq!((h.rtc_seconds, h.rtc_second_phase), (2, 2));
        let clock = h.rtc.clock().unwrap();
        assert_eq!((clock.year, clock.month, clock.day), (1994, 1, 1));
        assert_eq!((clock.hour, clock.minute, clock.second), (0, 1, 1));
        assert_eq!(h.rtc.read(10), 0x18); // A3 second and A4 minute latch while masked.
        assert!(h.ram.iter().all(|v| *v == 0));
        assert_eq!(
            rt.instruction_count(),
            if power == PowerState::Running { 8 } else { 0 }
        );
        drop(h);
        assert!(rt.configure_pi3000_rtc_calendar_candidate(3).is_err());
    }
}

#[test]
fn calendar_hold_preserves_phase_and_errors_preserve_clock_and_counters() {
    use crate::oz9600::rtc::{Clock, CONTROL, HOLD};
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::Strict);
    h.rtc_second_period = Some(10);
    h.rtc_event_candidate = true;
    h.rtc_second_phase = 7;
    assert!(h.advance_rtc_calendar(3).is_err()); // invalid fresh clock is never repaired
    assert_eq!((h.rtc_seconds, h.rtc_second_phase), (0, 7));
    assert_eq!(&h.rtc.registers()[..5], &[0; 5]);
    let before = Clock {
        year: 1994,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 59,
        weekday: 3,
    };
    for (offset, value) in before
        .encode([0xc0, 0, 0, 0, 0x80])
        .unwrap()
        .into_iter()
        .enumerate()
    {
        h.rtc.write(offset, value);
    }
    h.rtc.write(CONTROL, HOLD);
    h.advance_rtc_calendar(25).unwrap();
    for _ in 0..100 {
        assert_eq!(h.read(0x8000), Some(0xfb));
    }
    assert_eq!(h.rtc.clock().unwrap(), before);
    assert_eq!(
        (h.rtc_seconds, h.rtc_second_phase, h.rtc_held_units),
        (0, 7, 25)
    );
    h.rtc_tick_period = Some(5);
    h.advance_rtc_ticks(25);
    assert_eq!(h.rtc.read(10), 4); // separate A2 source continues during HOLD
    h.rtc.write(CONTROL, 0);
    h.advance_rtc_calendar(3).unwrap();
    assert_eq!(
        h.rtc.clock().unwrap(),
        Clock {
            year: 1995,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 4
        }
    );
    assert_eq!((h.rtc_seconds, h.rtc_second_phase), (1, 0));
    assert_eq!(h.rtc.read(0) & 0xc0, 0xc0);
    assert_eq!(h.rtc.read(4) & 0x80, 0x80);
    assert_eq!(h.rtc.read(10), 0x7c); // Existing A2 plus second/minute/hour/day.
}

#[test]
fn calendar_candidate_latches_rom_programmed_alarm_without_writing_guest_ram() {
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::Strict);
    h.rtc_second_period = Some(10);
    h.rtc_event_candidate = true;
    // FC:C196A/C1A09 getter/setter fixtures: 1994-01-01 03:59:58 -> 04:00:00.
    for (offset, value) in [0x3a, 0xfb, 0x08, 0xe1, 0x65].into_iter().enumerate() {
        h.rtc.write(offset, value);
    }
    for (i, value) in [0, 0, 0x09, 0xe1, 0x65, 7].into_iter().enumerate() {
        h.rtc.write(16 + i, value);
    }
    // C19A6's disabled second value 3C must not raise the second alarm.
    for (i, value) in [0x3c, 0, 0x09, 0xe1, 0x65, 7].into_iter().enumerate() {
        h.rtc.write(24 + i, value);
    }
    h.advance_rtc_calendar(19).unwrap();
    assert_eq!(h.rtc.read(10), 8);
    h.rtc.write(10, 0);
    h.advance_rtc_calendar(1).unwrap();
    assert_eq!(h.rtc.read(10), 0x39); // A0 alarm, A3 second, A4 minute, A5 hour.
    assert!(!h.external_irq());
    h.rtc.write(8, 1);
    h.gate[0x10] = 1;
    assert!(h.external_irq());
    h.rtc.write(10, 0);
    h.gate[0x12] = 0;
    assert!(!h.external_irq());
    h.advance_rtc_calendar(10).unwrap();
    assert_eq!(h.rtc.read(10), 8); // Acknowledged exact deadline does not reassert.
    assert!(h.ram.iter().all(|v| *v == 0));
    assert!(h.scratch.iter().all(|v| *v == 0));
}

#[test]
fn guest_cannot_write_device_owned_eport_inputs() {
    let mut rt = fixture(&[
        0x30, 0xcc, 0xf5, 0x44, 0x30, 0xcc, 0xf6, 0x55, 0x30, 0xcc, 0xf3, 0xa5,
    ]);
    rt.memory.write_internal_byte(0xf5, 0xa1);
    rt.memory.write_internal_byte(0xf6, 0xa2);
    rt.step(3).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0xa1));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf6), Some(0xa2));
    assert_eq!(rt.memory.read_internal_byte_silent(0xf3), Some(0xa5));
}

#[test]
fn unmapped_read_candidate_is_bounded_and_does_not_create_ram_or_allow_writes() {
    let code = [0x88, 0x44, 0x58, 0]; // MV A,[05844], the startup filename failure
    let mut strict = fixture(&code);
    assert!(strict
        .step(1)
        .unwrap_err()
        .to_string()
        .contains("unqualified PI read 05844"));
    for value in [0, 0xe0, 0xff] {
        let mut rt = fixture(&code);
        rt.configure_pi3000_unmapped_low_read_candidate(value)
            .unwrap();
        {
            let h = rt.pi3000_hardware().unwrap().borrow();
            for address in [0x5000, 0x5844, 0x7fff] {
                for _ in 0..10 {
                    assert_eq!(h.read(address), Some(value));
                }
            }
            assert_eq!(h.read(0x4fff), None);
            assert_eq!(h.read(0x8000), Some(0)); // adjacent qualified RTC
            assert!(h.ram.iter().chain(&h.scratch).all(|v| *v == 0));
            assert!(h.fault.is_none());
        }
        rt.step(1).unwrap();
        assert_eq!(rt.get_reg("A"), u32::from(value));
        assert!(rt
            .configure_pi3000_unmapped_low_read_candidate(value)
            .is_err());
        let mut h = rt.pi3000_hardware().unwrap().borrow();
        h.write(0x5844, 0x55, None);
        assert!(h
            .fault
            .as_ref()
            .unwrap()
            .contains("unqualified PI write 05844"));
        assert_eq!(h.read(0x5844), Some(value));
        assert!(h.ram.iter().chain(&h.scratch).all(|v| *v == 0));
    }
}

#[test]
fn unknown_bus_accesses_poison_the_committed_boundary() {
    let mut rt = fixture(&[0x88, 0, 0x8a, 0]);
    assert!(rt
        .step(1)
        .unwrap_err()
        .to_string()
        .contains("unqualified PI read 08A00"));
    assert_eq!(rt.instruction_count(), 1);
    assert!(rt.step(1).unwrap_err().to_string().contains("poisoned"));
    let mut rt = fixture(&[0x08, 0x7f, 0xa8, 1, 0x88, 0]);
    let error = rt.step(2).unwrap_err().to_string();
    assert!(error.contains("unqualified PI write 08801"));
    assert_eq!(rt.instruction_count(), 2);
    assert!(rt.step(1).unwrap_err().to_string().contains("poisoned"));
}

#[test]
fn absent_auxiliary_never_fabricates_completion_or_response() {
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::AuxiliaryAbsent);
    h.write(0x8a20, 0xfe, None);
    h.write(0x8a21, 1, None);
    h.write(0x8a02, 0x80, None);
    assert_eq!(&h.auxiliary_request[..2], &[0xfe, 1]);
    assert_eq!(h.read(0x8a00), Some(2));
    assert_eq!(h.read(0x8a40), None);
    h.write(0x8a03, 0xbf, None);
    assert_eq!(h.read(0x8a00), Some(2));
}

#[test]
fn unavailable_bank_never_fetches_old_image_or_generic_shadow() {
    let mut rt = fixture(&[0x08, 0x10, 0xa8, 0x21, 0x40, 0, 0x03, 0, 0, 12]);
    rt.step(3).unwrap();
    assert_eq!(rt.get_reg("PC"), 0xc0000);
    assert!(rt.step(1).is_err());
    assert_eq!(rt.instruction_count(), 3);
}

#[test]
fn tablet_route_is_explicit_and_samples_are_held_across_peeks() {
    for route in [TabletAdcRoute::Integrated, TabletAdcRoute::External] {
        let mut rt = fixture(&[0]);
        assert!(rt.set_pi3000_raw_contact(403, 601, true).is_err());
        rt.configure_pi3000_tablet_route(route).unwrap();
        rt.set_pi3000_raw_contact(403, 601, true).unwrap();
        let (control, data, other) = if route == TabletAdcRoute::External {
            (0x8800, 0x8c00, 0x8bc0)
        } else {
            (0x8b80, 0x8bc0, 0x8c00)
        };
        let mut h = rt.pi3000_hardware().unwrap().borrow();
        assert_eq!(
            h.read(0x8b1b),
            Some(u8::from(route == TabletAdcRoute::External))
        );
        assert_eq!(h.read(other), None);
        h.write(control, 6, None);
        h.write(0x8b04, 0xa8, None);
        for _ in 0..8 {
            assert_eq!(h.read(data), Some(100));
        }
        assert_eq!(h.tablet.data_reads, 0);
        assert_eq!(h.architectural_read(data, None), 100);
        h.tablet.set_contact(404, 602, true).unwrap();
        for _ in 0..8 {
            assert_eq!(h.read(data), Some(3));
        }
        assert_eq!(h.architectural_read(data, None), 3);
        h.write(control, 7, None);
        h.write(0x8b04, 0x8a, None);
        assert_eq!(h.architectural_read(data, None), 150);
        assert_eq!(h.architectural_read(data, None), 2);
        assert!(h.fault.is_none());
        drop(h);
        assert!(rt.set_pi3000_raw_contact(1024, 0, false).is_err());
        assert!(rt
            .configure_pi3000_tablet_route(if route == TabletAdcRoute::External {
                TabletAdcRoute::Integrated
            } else {
                TabletAdcRoute::External
            })
            .is_err());
    }
}

#[test]
fn raw_press_uses_only_the_candidate_gate_edge_and_invalid_inputs_are_atomic() {
    let mut rt = fixture(&[0]);
    rt.pi3000_hardware().unwrap().borrow().auxiliary = Some(auxiliary::Auxiliary::default());
    rt.configure_pi3000_tablet_route(TabletAdcRoute::Integrated)
        .unwrap();
    rt.set_pi3000_raw_contact(403, 601, true).unwrap();
    let mut h = rt.pi3000_hardware().unwrap().borrow();
    assert_eq!(h.gate[0x12], 8);
    assert!(!h.external_irq());
    h.gate[0x10] = 8;
    assert!(h.external_irq());
    h.gate[0x12] = 0;
    drop(h);
    rt.set_pi3000_raw_contact(404, 602, true).unwrap();
    assert_eq!(rt.pi3000_hardware().unwrap().borrow().gate[0x12], 0);
    assert!(rt.set_pi3000_raw_contact(1024, 2, false).is_err());
    assert!(rt.pi3000_hardware().unwrap().borrow().tablet.pressed);
    rt.set_pi3000_raw_contact(404, 602, false).unwrap();
    rt.set_pi3000_raw_contact(404, 602, true).unwrap();
    assert_eq!(rt.pi3000_hardware().unwrap().borrow().gate[0x12], 8);
}

#[test]
fn foreground_contact_candidate_uses_gate1_and_cannot_switch_after_sampling() {
    let mut rt = fixture(&[0]);
    rt.pi3000_hardware().unwrap().borrow().auxiliary = Some(auxiliary::Auxiliary::default());
    rt.configure_pi3000_tablet_route(TabletAdcRoute::Integrated)
        .unwrap();
    rt.configure_pi3000_tablet_interrupt_source(TabletInterruptSource::Gate1)
        .unwrap();
    rt.set_pi3000_raw_contact(66, 66, true).unwrap();
    assert_eq!(rt.pi3000_hardware().unwrap().borrow().gate[0x12], 2);
    assert!(rt
        .configure_pi3000_tablet_interrupt_source(TabletInterruptSource::Gate3)
        .is_err());
    rt.pi3000_hardware().unwrap().borrow().gate[0x12] = 0;
    rt.set_pi3000_raw_contact(66, 67, true).unwrap();
    assert_eq!(rt.pi3000_hardware().unwrap().borrow().gate[0x12], 0);
    rt.set_pi3000_raw_contact(66, 67, false).unwrap();
    {
        let mut h = rt.pi3000_hardware().unwrap().borrow();
        h.tablet.write_conversion_control(6).unwrap();
        h.tablet.read_data().unwrap();
    }
    assert!(rt
        .configure_pi3000_tablet_interrupt_source(TabletInterruptSource::Gate3)
        .is_err());
}

#[test]
fn halt_main_timer_candidate_requires_lcc7_and_preserves_default_freeze() {
    for (enabled, lcc, wakes) in [(false, 0x80, false), (true, 0, false), (true, 0x80, true)] {
        let mut rt = fixture(&[0xde, 0, 0, 0, 0, 0, 0, 0]);
        *rt.timer = TimerContext::new(true, 4, 0);
        rt.memory.write_internal_byte(0xfe, lcc);
        rt.configure_pi3000_halt_main_timer_candidate(enabled)
            .unwrap();
        rt.step_scheduler_boundaries(7).unwrap();
        assert_eq!(rt.instruction_count() > 1, wakes);
        assert_eq!(
            rt.pi3000_hardware().unwrap().borrow().halt_main_timer_units > 0,
            wakes
        );
    }
}

#[test]
fn mapper_readbacks_follow_the_saved_output_latches() {
    let mut h = Hardware::new(BTreeMap::new(), BootstrapProfile::Strict);
    for (i, value) in [0x12, 0x34, 0x56].into_iter().enumerate() {
        h.write(0xf00c + i as u32, value, None);
        assert_eq!(h.read(0xf01c + i as u32), Some(value));
    }
    assert_eq!(h.startup_latches, [0x12, 0x34, 0x56]);
}
