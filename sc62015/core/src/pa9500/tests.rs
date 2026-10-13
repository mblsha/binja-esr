// PY_SOURCE: pce500/pa9500/bootstrap.py
use super::*;

#[test]
fn power_input_pair_is_read_only_distinct_from_outputs_and_bounded() {
    let (mut rt, hw) = fixture();
    assert!(install_observed_power_pair(&mut rt, &hw, [0x12, 0x34]).is_err());
    install_media(&mut rt, &hw, None).unwrap();
    install_zero_data(&mut rt, &hw, 0xe0).unwrap();
    install_observed_low_pair(&mut rt, &hw, [0xe0; 2]).unwrap();
    install_observed_low_triad(&mut rt, &hw, [0xe0; 3]).unwrap();
    install_observed_field_words(&mut rt, &hw, [0xe0; 4]).unwrap();
    for address in [0xf01c, 0xf01d, 0xf01e, 0xf01f] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), None);
    }
    install_observed_power_pair(&mut rt, &hw, [0x12, 0x34]).unwrap();
    assert!(install_observed_power_pair(&mut rt, &hw, [0; 2]).is_err());
    hw.borrow().write(0xf00d, 0xaa, None);
    hw.borrow().write(0xf00e, 0xbb, None);
    assert_eq!(rt.memory.read_byte_for_preflight(0xf01d, None), Some(0x12));
    assert_eq!(rt.memory.read_byte_for_preflight(0xf01e, None), Some(0x34));
    assert_eq!(hw.borrow().startup_latches, [0xaa, 0xbb]);
    for address in [0xf01c, 0xf01f, 0xf00d, 0xf00e] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), None);
    }
    hw.borrow().write(0xf01d, 0xff, None);
    assert!(hw
        .borrow()
        .fault
        .as_ref()
        .unwrap()
        .contains("unqualified PA write"));
    assert_eq!(hw.borrow().observed_power_pair, Some([0x12, 0x34]));
}

#[test]
fn explicit_irq_imr_policy_accepts_after_guest_reenables_master_with_stale_metadata() {
    for candidate in [false, true] {
        let (mut rt, hw) = fixture();
        if candidate {
            install_irq_imr_candidate(&mut rt, &hw).unwrap();
            assert!(install_irq_imr_candidate(&mut rt, &hw).is_err());
        }
        rt.timer.enabled = false;
        rt.set_reg("PC", 0xe0000);
        rt.set_reg("S", 0x80020);
        rt.timer.in_interrupt = true;
        // Synthetic abandoned frame: the guest enables IRM | ONKM. The IRQ
        // handler clears ISR.ONKI, then RETI restores the exact five-byte frame.
        rt.load_rom(&[0x30, 0xcc, 0xfb, 0x88, 0, 0], 0xe0000);
        rt.load_rom(&[0, 0x10, 0xe], 0xffffa);
        rt.load_rom(&[0x30, 0x71, 0xfc, 0xf7, 0x01], 0xe1000);
        rt.press_on_key();
        rt.release_on_key();
        rt.step_scheduler_boundaries(1).unwrap();
        assert_eq!(rt.get_reg("PC"), 0xe0004);
        assert_eq!(rt.timer.irq_total, 0, "IRM-off must still inhibit IRQ");
        rt.step_scheduler_boundaries(1).unwrap();
        if candidate {
            assert_eq!(rt.get_reg("PC"), 0xe1004);
            assert_eq!(rt.timer.irq_total, 1);
            assert_eq!(rt.get_reg("S"), 0x8001b);
            rt.step_scheduler_boundaries(1).unwrap();
            assert_eq!(rt.get_reg("PC"), 0xe0004);
            assert_eq!(rt.get_reg("S"), 0x80020);
            assert_eq!(rt.memory.read_internal_byte_silent(0xfb), Some(0x88));
            assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
        } else {
            assert_eq!(rt.get_reg("PC"), 0xe0005);
            assert_eq!(rt.timer.irq_total, 0);
        }
        assert!(install_irq_imr_candidate(&mut rt, &hw).is_err());
    }
    assert!(install_irq_imr_candidate(&mut CoreRuntime::new(), &fixture().1).is_err());
}

#[test]
fn controller_view_observes_the_board_without_bus_routing_or_copying() {
    let (_, hw) = fixture();
    hw.borrow().lcd = Some(LcdController::with_geometry(195, 145));
    // Configure the full read/write window explicitly; reset descriptors clip.
    for (offset, value) in [
        (24, 0),
        (25, 0),
        (26, 0),
        (27, 0),
        (28, 194),
        (29, 0),
        (30, 144),
        (31, 0),
    ] {
        hw.borrow().write(0x8420 + offset, value, None);
    }
    let view = lcd_view::LcdView(hw.clone());
    let frame = crate::LcdHal::matrix_frame(&view);
    assert_eq!(frame.geometry(), (195, 145));
    assert!(frame.pixels().iter().all(|p| *p == 0));
    for (offset, value) in [(31, 2), (3, 255), (16, 24), (18, 48), (2, 0x80)] {
        hw.borrow().write(0x8420 + offset, value, None);
    }
    let before = hw.borrow().events.clone();
    let actual = crate::LcdHal::matrix_frame(&view);
    assert_eq!(actual, hw.borrow().lcd.as_ref().unwrap().matrix_frame());
    assert_eq!(actual.pixels().iter().filter(|p| **p != 0).count(), 1);
    assert_eq!(actual.pixels()[48 * 195 + 24], 1);
    assert_eq!(hw.borrow().events, before);
    for address in 0..0x10000 {
        assert!(!crate::LcdHal::handles(&view, address));
    }
    assert!(!crate::LcdHal::may_handle_span(&view, 0, 0xfffff));
}

#[test]
fn lcd_coordinate_restore_candidate_requires_a_fresh_pa_controller() {
    let (mut rt, hw) = fixture();
    hw.borrow().lcd = Some(LcdController::with_geometry(195, 145));
    assert!(!hw.borrow().lcd_coordinate_restore_candidate);
    install_lcd_coordinate_restore_candidate(&mut rt, &hw).unwrap();
    assert!(install_lcd_coordinate_restore_candidate(&mut rt, &hw).is_err());
    let (mut rt, hw) = fixture();
    hw.borrow().lcd = Some(LcdController::with_geometry(195, 145));
    rt.step(1).unwrap();
    assert!(install_lcd_coordinate_restore_candidate(&mut rt, &hw).is_err());
    let (mut rt, hw) = fixture();
    hw.borrow().lcd = None;
    assert!(install_lcd_coordinate_restore_candidate(&mut rt, &hw).is_err());
    assert!(install_lcd_coordinate_restore_candidate(&mut CoreRuntime::new(), &hw).is_err());
}

#[test]
fn uart_candidate_is_explicit_and_cannot_replace_or_attach_to_running_runtime() {
    let (mut rt, _) = fixture();
    assert!(rt.sio.is_none());
    install_uart_candidate(&mut rt).unwrap();
    let uart = rt.sio.as_ref().unwrap().uart().unwrap();
    assert_eq!(
        (uart.timebase_hz, uart.baud_divisor, uart.control),
        (1_024_000, 2, 0)
    );
    assert!(install_uart_candidate(&mut rt).is_err());
    let (mut running, _) = fixture();
    running.step(1).unwrap();
    assert!(install_uart_candidate(&mut running).is_err());
    assert!(running.sio.is_none());
    assert!(install_uart_candidate(&mut CoreRuntime::new()).is_err());
}

#[test]
fn on_contact_edge_candidate_keeps_physical_level_without_irq_reassertion() {
    for edge in [false, true] {
        let (mut rt, _) = fixture();
        if edge {
            install_on_contact_edge_candidate(&mut rt).unwrap();
        }
        rt.timer.enabled = false;
        // Ordinary guest acknowledgement, followed by another instruction.
        rt.load_rom(&[0x30, 0x71, 0xfc, 0xf7, 0, 0x30, 0x80, 0xff], 0xe0000);
        rt.set_reg("PC", 0xe0000);
        rt.press_on_key();
        rt.step(1).unwrap();
        assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 0);
        rt.press_on_key(); // A duplicate host notification is not a new edge.
        rt.step(1).unwrap();
        assert_eq!(
            rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8,
            if edge { 0 } else { 8 }
        );
        assert!(rt.physical_on_key_pressed());
        rt.step(1).unwrap(); // MV A,(SSR) samples the normal physical input.
        assert_eq!(rt.get_reg("A") & 8, 8);
        rt.release_on_key();
        rt.press_on_key();
        assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 8, 8);
        assert!(install_on_contact_edge_candidate(&mut rt).is_err());
    }
    assert!(install_on_contact_edge_candidate(&mut CoreRuntime::new()).is_err());
}

fn fixture() -> (CoreRuntime, SharedHardware) {
    let mut fixed = vec![0; 0x20000];
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 15]);
    // MV A,FA; MV [04021],A; JPF C0000. The next banked fetch must
    // see the newly selected byte view within batch and slice execution.
    fixed[0x10000..0x1000a].copy_from_slice(&[8, 250, 0xa8, 0x21, 0x40, 0, 3, 0, 0, 12]);
    let mut banks = BTreeMap::new();
    for selector in SELECTORS {
        let mut bank = vec![0; 0x20000];
        bank[..2].copy_from_slice(&[8, selector]);
        banks.insert(selector, bank);
    }
    // Exact byte aliases must retain the requested latch identity.
    banks.insert(250, banks[&248].clone());
    configure(&fixed, banks, BootstrapProfile::TrapPeripherals).unwrap()
}

#[test]
fn bank_write_changes_next_fetch_in_batch_and_slice_without_aliasing_latch() {
    for sliced in [false, true] {
        let (mut rt, hw) = fixture();
        if sliced {
            rt.run_slice(4, |_| false).unwrap();
        } else {
            rt.step(4).unwrap();
        }
        assert_eq!(rt.get_reg("PC"), 0xc0002);
        assert_eq!(rt.get_reg("A"), 248);
        assert_eq!(hw.borrow().selector, 250);
        assert_eq!(rt.memory.read_byte_for_preflight(0x4021, None), Some(250));
        assert_eq!(hw.borrow().ram, vec![0; 0x40000]);
    }
}

#[test]
fn unacquired_selector_stops_before_an_instruction_or_fallback_mapping() {
    let (mut rt, hw) = fixture();
    hw.borrow().write(0x4021, 0x08, None);
    assert!(rt.step(1).unwrap_err().to_string().contains("not acquired"));
    assert_eq!(rt.instruction_count(), 0);
    assert_eq!(rt.get_reg("PC"), 0xf0000);
    assert_eq!(hw.borrow().read(0xc0000), None);
}

#[test]
fn trap_profile_does_not_invent_lcd_registers_or_oz_workspace_alias() {
    let (rt, hw) = fixture();
    assert_eq!(rt.memory.read_byte_for_preflight(0x8401, None), None);
    assert_eq!(rt.memory.read_byte_for_preflight(0x1fda8, None), None);
    let hw = hw.borrow();
    assert_eq!(hw.read(0x8401), None);
    assert_eq!(hw.read(0x1fda8), None);
    assert_eq!(hw.read(0xbfda8), Some(0));
}

#[test]
fn raw_tablet_channels_preserve_unrelated_eil_bits_and_release_contact() {
    let (mut rt, hw) = fixture();
    hw.borrow().adc_control = Some(0);
    hw.borrow().tablet = Some(TabletInput {
        x: 37,
        y: 129,
        pressure: 224,
        contact: true,
    });
    rt.memory.write_internal_byte(0xf5, 0x81);
    rt.step(1).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0xa1));
    for (control, value) in [(6, 9), (7, 32), (22, 224), (14, 0), (30, 0)] {
        hw.borrow().write(0x8800, control, None);
        assert_eq!(hw.borrow().read(0x8c00), Some(value));
    }
    hw.borrow().tablet.as_mut().unwrap().contact = false;
    rt.step(1).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xf5), Some(0x81));
    hw.borrow().write(0x8800, 6, None);
    assert_eq!(hw.borrow().read(0x8c00), Some(0));
}

#[test]
fn tablet_conversion_holds_low_bits_and_peeks_do_not_advance() {
    let mut hw = Hardware::new(BTreeMap::new(), BootstrapProfile::ControllerHypothesis);
    hw.tablet = Some(TabletInput {
        x: 403,
        y: 601,
        pressure: 255,
        contact: true,
    });
    hw.write(0x8800, 6, None);
    for _ in 0..4 {
        assert_eq!(hw.read(0x8c00), Some(100));
    }
    assert_eq!(hw.adc_data_reads, 0);
    assert_eq!(hw.architectural_read(0x8c00, None), 100);
    hw.tablet = Some(TabletInput {
        x: 404,
        y: 602,
        pressure: 255,
        contact: true,
    });
    for _ in 0..4 {
        assert_eq!(hw.read(0x8c00), Some(3));
    }
    assert_eq!(hw.architectural_read(0x8c00, None), 3);
    hw.write(0x8800, 7, None);
    assert_eq!(hw.architectural_read(0x8c00, None), 150);
    assert_eq!(hw.architectural_read(0x8c00, None), 2);
    assert_eq!(hw.adc_data_reads, 4);
    hw.gate[16] = 2;
    hw.gate[18] = 0x81;
    assert_eq!(hw.read(0x4012), Some(0x83));
    hw.tablet.as_mut().unwrap().contact = false;
    assert_eq!(hw.read(0x4012), Some(0x81));
}

#[test]
fn explicit_absence_and_one_observed_byte_do_not_fill_unknown_bus_space() {
    let (mut rt, hw) = fixture();
    assert_eq!(rt.memory.read_byte_for_preflight(0x40000, None), None);
    install_media(&mut rt, &hw, None).unwrap();
    for address in [0x20000, 0x30000, 0x40000, 0x7ffff] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), Some(0xff));
        hw.borrow().write(address, 0x55, None);
        assert_eq!(hw.borrow().read(address), Some(0xff));
    }
    assert!(install_media(&mut rt, &hw, None).is_err());
    assert_eq!(rt.memory.read_byte_for_preflight(0, None), None);
    install_zero_data(&mut rt, &hw, 0xe0).unwrap();
    assert_eq!(rt.memory.read_byte_for_preflight(0, None), Some(0xe0));
    assert_eq!(rt.memory.read_byte_for_preflight(1, None), None);
    assert!(install_zero_data(&mut rt, &hw, 0).is_err());
    hw.borrow().write(0, 0, None);
    assert!(hw.borrow().fault.is_some());
}

#[test]
fn observed_low_pair_does_not_fill_neighbors_or_allow_writes() {
    let (mut rt, hw) = fixture();
    assert!(install_observed_low_pair(&mut rt, &hw, [0xe0; 2]).is_err());
    install_zero_data(&mut rt, &hw, 0xe0).unwrap();
    for address in [0xb, 0xc, 0xd, 0xe] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), None);
    }
    install_observed_low_pair(&mut rt, &hw, [0xe0; 2]).unwrap();
    assert!(install_observed_low_pair(&mut rt, &hw, [0; 2]).is_err());
    assert_eq!(rt.memory.read_byte_for_preflight(0, None), Some(0xe0));
    assert_eq!(rt.memory.read_byte_for_preflight(0xc, None), Some(0xe0));
    assert_eq!(rt.memory.read_byte_for_preflight(0xd, None), Some(0xe0));
    assert_eq!(rt.memory.read_byte_for_preflight(0xb, None), None);
    assert_eq!(rt.memory.read_byte_for_preflight(0xe, None), None);
    rt.memory.store(0xc, 8, 0x42).unwrap();
    assert!(hw.borrow().fault.is_some());
    assert_eq!(hw.borrow().read(0xc), Some(0xe0));
}

#[test]
fn observed_low_triad_requires_fresh_pair_and_preserves_other_low_reads() {
    let (mut rt, hw) = fixture();
    assert!(install_observed_low_triad(&mut rt, &hw, [0xe0; 3]).is_err());
    install_zero_data(&mut rt, &hw, 0xe0).unwrap();
    assert!(install_observed_low_triad(&mut rt, &hw, [0xe0; 3]).is_err());
    install_observed_low_pair(&mut rt, &hw, [0xa5, 0x5a]).unwrap();
    install_observed_low_triad(&mut rt, &hw, [0x12, 0x34, 0x05]).unwrap();
    assert!(install_observed_low_triad(&mut rt, &hw, [0; 3]).is_err());
    for (address, byte) in [
        (0, 0xe0),
        (1, 0x12),
        (2, 0x34),
        (3, 0x05),
        (0xc, 0xa5),
        (0xd, 0x5a),
    ] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), Some(byte));
    }
    for address in [4, 0xb, 0xe] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), None);
    }
    rt.memory.store(2, 8, 0xff).unwrap();
    assert!(hw.borrow().fault.is_some());
    assert_eq!(hw.borrow().observed_low_triad, Some([0x12, 0x34, 0x05]));
    let (mut running, running_hw) = fixture();
    install_zero_data(&mut running, &running_hw, 0xe0).unwrap();
    install_observed_low_pair(&mut running, &running_hw, [0xe0; 2]).unwrap();
    running.step(1).unwrap();
    assert!(install_observed_low_triad(&mut running, &running_hw, [0xe0; 3]).is_err());
}

#[test]
fn observed_field_words_keep_fresh_setup_neighbors_and_writes_strict() {
    let (mut rt, hw) = fixture();
    assert!(install_observed_field_words(&mut rt, &hw, [0xe0; 4]).is_err());
    install_zero_data(&mut rt, &hw, 0xe0).unwrap();
    install_observed_low_pair(&mut rt, &hw, [0xe0; 2]).unwrap();
    assert!(install_observed_field_words(&mut rt, &hw, [0xe0; 4]).is_err());
    install_observed_low_triad(&mut rt, &hw, [0xe0; 3]).unwrap();
    for a in 5..=8 {
        assert_eq!(rt.memory.read_byte_for_preflight(a, None), None);
    }
    install_observed_field_words(&mut rt, &hw, [1, 2, 3, 4]).unwrap();
    assert!(install_observed_field_words(&mut rt, &hw, [0; 4]).is_err());
    for (a, b) in [(5, 1), (6, 2), (7, 3), (8, 4)] {
        assert_eq!(rt.memory.read_byte_for_preflight(a, None), Some(b));
        assert_eq!(hw.borrow().read(a), Some(b));
    }
    for a in [4, 9, 10, 11, 14] {
        assert_eq!(rt.memory.read_byte_for_preflight(a, None), None);
    }
    for a in 5..=8 {
        hw.borrow().fault = None;
        rt.memory.store(a, 8, 0xff).unwrap();
        assert!(hw
            .borrow()
            .fault
            .as_ref()
            .unwrap()
            .contains("unqualified PA write"));
    }
    assert_eq!(hw.borrow().observed_field_words, Some([1, 2, 3, 4]));
    let (mut running, running_hw) = fixture();
    install_zero_data(&mut running, &running_hw, 0xe0).unwrap();
    install_observed_low_pair(&mut running, &running_hw, [0xe0; 2]).unwrap();
    install_observed_low_triad(&mut running, &running_hw, [0xe0; 3]).unwrap();
    running.step(1).unwrap();
    assert!(install_observed_field_words(&mut running, &running_hw, [0xe0; 4]).is_err());
}

#[test]
fn scratch_candidate_is_bounded_distinct_and_hidden_until_the_explicit_command() {
    let (mut strict, strict_hw) = fixture();
    assert!(install_scratch_candidate(&mut strict, &strict_hw).is_err());
    let mut fixed = vec![0; 0x20000];
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 15]);
    let banks = SELECTORS
        .into_iter()
        .map(|s| (s, vec![0; 0x20000]))
        .collect();
    let (mut rt, hw) = configure(&fixed, banks, BootstrapProfile::ControllerHypothesis).unwrap();
    assert_eq!(rt.memory.read_byte_for_preflight(0xe000, None), None);
    install_scratch_candidate(&mut rt, &hw).unwrap();
    assert_eq!(rt.memory.read_byte_for_preflight(0xe000, None), None);
    hw.borrow().compatibility_command = Some(1);
    rt.step(1).unwrap();
    for (address, byte) in [(0xe000, 0xA5), (0xefff, 0x5A)] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), Some(0));
        hw.borrow().write(address, byte, None);
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), Some(byte));
    }
    for address in [0xdfff, 0xf000] {
        assert_eq!(rt.memory.read_byte_for_preflight(address, None), None);
    }
    assert!(hw.borrow().ram.iter().all(|byte| *byte == 0));
    assert!(install_scratch_candidate(&mut rt, &hw).is_err());
    hw.borrow().compatibility_command = Some(0);
    rt.step(1).unwrap();
    assert_eq!(rt.memory.read_byte_for_preflight(0xefff, None), None);
}

#[test]
fn controller_hypothesis_uses_pa_bounds_and_independently_expected_pixels() {
    let mut hw = Hardware::new(BTreeMap::new(), BootstrapProfile::ControllerHypothesis);
    // Full window, mask FF, SET mode, X=190 Y=144; 10101010 clips at X=194.
    for (offset, value) in [
        (24, 0),
        (25, 0),
        (26, 0),
        (27, 0),
        (28, 194),
        (29, 0),
        (30, 144),
        (31, 0),
    ] {
        hw.write(0x8420 + offset, value, None);
    }
    for (offset, value) in [
        (31, 2),
        (3, 255),
        (8, 0),
        (16, 190),
        (17, 0),
        (18, 144),
        (19, 0),
        (2, 0xaa),
    ] {
        hw.write(0x8420 + offset, value, None);
    }
    assert!(hw.fault.is_none());
    let frame = hw.lcd.as_ref().unwrap().matrix_frame();
    assert_eq!(frame.geometry(), (195, 145));
    let lcd = hw.lcd.as_ref().unwrap();
    for y in 0..145 {
        for x in 0..195 {
            assert_eq!(lcd.pixel(x, y), y == 144 && [190, 192, 194].contains(&x));
        }
    }
}
