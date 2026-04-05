// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py

use crate::lcd_text::Pce500FontMap;
use crate::memory::{MemoryImage, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET};
use crate::{CoreRuntime, Result};

pub const SYSTEM_IMAGE_LEN: usize = 0x100000;
pub const ROM_WINDOW_START: usize = 0x0C0000;
pub const ROM_WINDOW_LEN: usize = 0x40000;
pub const ROM_RESET_VECTOR_ADDR: u32 = 0x0FFFFD;
pub const ROM_ENGLISH_FONT_BASE_ADDR: u32 = 0x00F_2215;
pub const ROM_JP_FONT_ATLAS_BASE_ADDR: u32 = 0x00F_21A5;
pub const ROM_FONT_BASE_ADDR: u32 = ROM_ENGLISH_FONT_BASE_ADDR;
pub const NO_RAM_WINDOW_START: usize = 0x00000;
pub const NO_RAM_WINDOW_END: usize = 0x3FFFF;
pub const BOOTSTRAP_IMR_VALUE: u8 = 0x43;
pub const BOOTSTRAP_ISR_VALUE: u8 = 0x00;

// Best-guess clock for real hardware: 3.072 MHz / 3 = 1.024 MHz.
// INT_FAST (MTI) fires every ~2 ms and INT_SLOW (STI) every ~0.5 s at that rate.
pub const DEFAULT_CPU_HZ: u64 = 1_024_000;
pub const DEFAULT_MTI_PERIOD: u64 = DEFAULT_CPU_HZ / 1000 * 2;
pub const DEFAULT_STI_PERIOD: u64 = DEFAULT_CPU_HZ / 2;

pub const PF1_CODE: u8 = 0x56; // col=10, row=6
pub const PF2_CODE: u8 = 0x55; // col=10, row=5

pub fn load_pce500_rom_window(rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
    load_pce500_rom_window_into_memory(&mut rt.memory, rom);
    configure_pce500_memory_map(&mut rt.memory);
    rt.memory.set_keyboard_bridge(false);
    Ok(())
}

pub fn load_pce500_system_image(rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
    load_pce500_system_image_into_memory(&mut rt.memory, rom);
    configure_pce500_memory_map(&mut rt.memory);
    rt.memory.set_keyboard_bridge(false);
    seed_pce500_bootstrap_imem(&mut rt.memory);
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

pub fn load_pce500_system_image_into_memory(memory: &mut MemoryImage, rom: &[u8]) {
    if rom.len() >= SYSTEM_IMAGE_LEN {
        memory.load_external(&rom[..SYSTEM_IMAGE_LEN]);
    } else {
        load_pce500_rom_window_into_memory(memory, rom);
    }
}

pub fn configure_pce500_memory_map(memory: &mut MemoryImage) {
    memory.set_readonly_ranges(vec![
        (NO_RAM_WINDOW_START as u32, NO_RAM_WINDOW_END as u32),
        (
            ROM_WINDOW_START as u32,
            (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32,
        ),
    ]);
}

pub fn seed_pce500_bootstrap_imem(memory: &mut MemoryImage) {
    memory.write_internal_byte(IMEM_IMR_OFFSET, BOOTSTRAP_IMR_VALUE);
    memory.write_internal_byte(IMEM_ISR_OFFSET, BOOTSTRAP_ISR_VALUE);
}

pub fn pce500_font_map_from_rom(rom: &[u8]) -> Option<Pce500FontMap> {
    let font = Pce500FontMap::from_pce500_rom(rom, ROM_WINDOW_START as u32);
    if font.is_empty() {
        None
    } else {
        Some(font)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        load_pce500_system_image_into_memory, seed_pce500_bootstrap_imem, BOOTSTRAP_IMR_VALUE,
        BOOTSTRAP_ISR_VALUE, SYSTEM_IMAGE_LEN,
    };
    use crate::memory::{MemoryImage, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET};

    #[test]
    fn system_image_loads_low_and_high_windows() {
        let mut rom = vec![0u8; SYSTEM_IMAGE_LEN];
        rom[0x010012] = 0xE2;
        rom[0x0E02ED] = 0x0B;

        let mut memory = MemoryImage::new();
        load_pce500_system_image_into_memory(&mut memory, &rom);

        assert_eq!(memory.load(0x010012, 8), Some(0xE2));
        assert_eq!(memory.load(0x0E02ED, 8), Some(0x0B));
    }

    #[test]
    fn bootstrap_seeds_imr_isr_defaults() {
        let mut memory = MemoryImage::new();
        seed_pce500_bootstrap_imem(&mut memory);

        assert_eq!(
            memory.read_internal_byte(IMEM_IMR_OFFSET),
            Some(BOOTSTRAP_IMR_VALUE)
        );
        assert_eq!(
            memory.read_internal_byte(IMEM_ISR_OFFSET),
            Some(BOOTSTRAP_ISR_VALUE)
        );
    }
}
