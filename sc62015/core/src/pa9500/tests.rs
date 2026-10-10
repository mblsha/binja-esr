// PY_SOURCE: pce500/pa9500/bootstrap.py
use super::*;

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
fn controller_hypothesis_uses_pa_bounds_and_independently_expected_pixels() {
    let mut hw = Hardware::new(BTreeMap::new(), BootstrapProfile::ControllerHypothesis);
    // Full window, mask FF, SET mode, X=190 Y=144; 10101010 clips at X=194.
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
