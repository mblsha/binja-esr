#[allow(dead_code)]
mod common;

use sc62015_core::device::DeviceModel;
use sc62015_core::iq7000::load_iq7000_rom_image;
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::memory::{IMEM_EOH_OFFSET, IMEM_ISR_OFFSET};
use sc62015_core::CoreRuntime;
use std::fs;
use std::path::PathBuf;

const EXTERNAL_HANDLER: u32 = 0x0F527F;
const ACTIVE_IOCS_LIST_POINTER: u32 = 0x01FDA8;
const ACTIVE_IOCS_LIST_START: u32 = 0x0F03F5;
const ISR_EXI: u8 = 0x40;
const STATE_RAW_ADDR: u32 = 0x01FDA3;
const SHADOW_RAW_ADDR: u32 = 0x006160;

fn iq7000_rom_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../roms/iq-7000.bin")
}

#[test]
fn held_external_level_reaches_device_zero_stdo_raw_bit() {
    let rom_path = iq7000_rom_path();
    if !rom_path.exists() {
        eprintln!("Skipping: ROM not present at {}", rom_path.display());
        return;
    }

    let rom = fs::read(&rom_path).expect("read IQ-7000 ROM");
    let mut rt = CoreRuntime::new();
    rt.set_device_model(DeviceModel::Iq7000)
        .expect("set IQ-7000 model");
    load_iq7000_rom_image(&mut rt, &rom).expect("load IQ-7000 ROM");

    // The boot initializer normally installs the built-in IOCS list head.
    common::write_u24(&mut rt, ACTIVE_IOCS_LIST_POINTER, ACTIVE_IOCS_LIST_START);
    rt.memory
        .store(STATE_RAW_ADDR, 8, 0)
        .expect("clear state byte");
    rt.memory
        .store(SHADOW_RAW_ADDR, 8, 0)
        .expect("clear shadow byte");
    rt.memory.write_internal_byte(IMEM_EOH_OFFSET, 0);
    rt.state.set_reg(RegName::S, 0x01FF00);
    rt.state.set_reg(RegName::U, 0x01FF80);

    // Execute the handler as an active interrupt so its asserted level can
    // re-latch ISR.EXI after the firmware's clear/NOP sequence without the
    // runtime recursively delivering another interrupt.
    rt.timer.in_interrupt = true;
    rt.set_external_interrupt_level(true);
    let result = common::call_with_sentinel(&mut rt, EXTERNAL_HANDLER, 10_000);
    assert!(
        result.returned,
        "F527F did not return (pc={:#07x}, steps={})",
        result.pc, result.steps
    );
    assert_ne!(
        rt.memory.load(STATE_RAW_ADDR, 8).unwrap_or(0) as u8 & 0x80,
        0,
        "device-0 STDO IL=0x46 should set raw state bit 0x80"
    );
    assert_ne!(
        rt.memory.load(SHADOW_RAW_ADDR, 8).unwrap_or(0) as u8 & 0x80,
        0,
        "device-0 STDO IL=0x46 should mirror raw bit 0x80"
    );
    assert_ne!(
        rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_EXI,
        0,
        "asserted external level should remain latched after F527F"
    );

    rt.set_external_interrupt_level(false);
    assert!(!rt.external_interrupt_level());
    assert_eq!(
        rt.memory.read_internal_byte(IMEM_ISR_OFFSET).unwrap_or(0) & ISR_EXI,
        0,
        "deasserting the neutral input should clear ISR.EXI"
    );
}
