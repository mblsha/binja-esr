// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py

use crate::lcd_text::Pce500FontMap;
use crate::memory::MemoryImage;
use crate::{CoreRuntime, Result};

pub const ROM_WINDOW_START: usize = 0x0C0000;
pub const ROM_WINDOW_LEN: usize = 0x40000;
pub const ROM_RESET_VECTOR_ADDR: u32 = 0x0FFFFD;
pub const ROM_FONT_BASE_ADDR: u32 = 0x00F_2215;

pub const DEFAULT_MTI_PERIOD: u64 = 500;
pub const DEFAULT_STI_PERIOD: u64 = 5000;

pub const PF1_CODE: u8 = 0x56; // col=10, row=6
pub const PF2_CODE: u8 = 0x55; // col=10, row=5

pub fn load_pce500_rom_window(rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
    load_pce500_rom_window_into_memory(&mut rt.memory, rom);
    rt.memory.set_readonly_ranges(vec![(
        ROM_WINDOW_START as u32,
        (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32,
    )]);
    rt.memory.set_keyboard_bridge(false);
    Ok(())
}

pub fn load_pce500_rom_window_into_memory(memory: &mut MemoryImage, rom: &[u8]) {
    let src_start = rom.len().saturating_sub(ROM_WINDOW_LEN);
    let slice = &rom[src_start..];
    let copy_len = slice.len().min(ROM_WINDOW_LEN);
    let start_in_slice = slice.len().saturating_sub(copy_len);
    memory.write_external_slice(
        ROM_WINDOW_START,
        &slice[start_in_slice..start_in_slice + copy_len],
    );
}

pub fn pce500_font_map_from_rom(rom: &[u8]) -> Option<Pce500FontMap> {
    let font = Pce500FontMap::from_rom(rom, ROM_FONT_BASE_ADDR, ROM_WINDOW_START as u32);
    if font.is_empty() {
        None
    } else {
        Some(font)
    }
}
