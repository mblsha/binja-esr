// PY_SOURCE: pce500/pi3000/hardware.py
use super::*;

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
        assert_eq!(h.rtc.read(10), 0); // no unqualified PI cyclic/alarm cause generation
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
