use sc62015_core::lcd_text::Pce500FontMap;
use sc62015_core::llama::opcodes::RegName;
use sc62015_core::pce500::{
    load_pce500_rom_window, pce500_font_map_from_rom, DEFAULT_MTI_PERIOD, DEFAULT_STI_PERIOD,
};
use sc62015_core::CoreRuntime;
use std::fs;
use std::path::PathBuf;

pub const IOCS_PUBLIC_ENTRY_ADDR: u32 = 0x00FFFE8;

pub fn default_rom_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in [
        manifest.join("../../data/pc-e500-en.bin"),
        manifest.join("../../../roms/pc-e500-en.bin"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }
    manifest.join("../../data/pc-e500-en.bin")
}

pub fn boot_pce500() -> Option<CoreRuntime> {
    let rom_path = default_rom_path();
    if !rom_path.exists() {
        eprintln!("Skipping: ROM not present at {}", rom_path.display());
        return None;
    }

    let rom = fs::read(&rom_path).expect("read ROM");
    let mut rt = CoreRuntime::new();
    rt.timer.enabled = true;
    rt.timer.mti_period = DEFAULT_MTI_PERIOD;
    rt.timer.sti_period = DEFAULT_STI_PERIOD;
    rt.timer.reset(rt.cycle_count());
    load_pce500_rom_window(&mut rt, &rom).expect("load ROM window");
    rt.power_on_reset();
    rt.step(20_000).expect("boot");
    Some(rt)
}

pub fn load_pce500_font() -> Option<Pce500FontMap> {
    let rom_path = default_rom_path();
    if !rom_path.exists() {
        eprintln!("Skipping: ROM not present at {}", rom_path.display());
        return None;
    }
    let rom = fs::read(&rom_path).expect("read ROM");
    Some(pce500_font_map_from_rom(&rom).expect("load font map"))
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct IocsResult {
    pub returned: bool,
    pub steps: u32,
    pub a: u32,
    pub i: u32,
    pub x: u32,
    pub y: u32,
    pub pc: u32,
    pub carry: bool,
}

pub fn call_with_sentinel(rt: &mut CoreRuntime, addr: u32, max_instructions: u32) -> IocsResult {
    let before_pc = rt.state.pc();
    let before_sp = rt.state.get_reg(RegName::S);
    let sentinel_low16: u32 = 0xD00D;
    let sentinel_pc = ((addr & 0x0f_0000) | sentinel_low16) & 0x000f_ffff;

    let new_sp = before_sp.wrapping_sub(3) & 0x00ff_ffff;
    for i in 0..3u32 {
        let byte = (sentinel_pc >> (8 * i)) & 0xff;
        let _ = rt.memory.store(new_sp.wrapping_add(i), 8, byte);
    }
    rt.state.set_reg(RegName::S, new_sp);
    rt.state.set_pc(addr & 0x000f_ffff);

    let mut steps = 0;
    while steps < max_instructions {
        if rt.state.pc() == sentinel_pc {
            break;
        }
        if rt.step(1).is_err() {
            break;
        }
        steps += 1;
    }
    let returned = rt.state.pc() == sentinel_pc;
    let result = IocsResult {
        returned,
        steps,
        a: rt.state.get_reg(RegName::A),
        i: rt.state.get_reg(RegName::I),
        x: rt.state.get_reg(RegName::X),
        y: rt.state.get_reg(RegName::Y),
        pc: rt.state.pc(),
        carry: (rt.state.get_reg(RegName::FC) & 1) != 0,
    };

    rt.state.set_pc(before_pc);
    rt.state.set_reg(RegName::S, before_sp);
    result
}

pub fn call_iocs(
    rt: &mut CoreRuntime,
    device: u8,
    command: u16,
    max_instructions: u32,
) -> IocsResult {
    rt.memory.write_internal_byte(0xD6, device);
    rt.memory.write_internal_byte(0xD7, 0);
    rt.state.set_reg(RegName::I, u32::from(command));
    call_with_sentinel(rt, IOCS_PUBLIC_ENTRY_ADDR, max_instructions)
}

pub fn write_u24(rt: &mut CoreRuntime, addr: u32, value: u32) {
    for offset in 0..3u32 {
        let _ = rt
            .memory
            .store(addr.wrapping_add(offset), 8, (value >> (8 * offset)) & 0xff);
    }
}
