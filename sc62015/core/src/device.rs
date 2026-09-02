// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py

use crate::iq7000;
use crate::keyboard::KeyboardMatrix;
use crate::lcd::create_lcd;
use crate::lcd::{Iq7000LcdController, LcdController, LcdHal, LcdKind};
use crate::lcd_text::{
    decode_display_text, decode_iq7000_display_text_auto, Iq7000FontMap, Iq7000LargeFontMap,
    LcdCharMatcher, Pce500FontMap,
};
use crate::pce500;
use crate::timer::TimerContext;
use crate::{CoreRuntime, MemoryImage, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct DeviceSpec {
    pub label: &'static str,
    pub rom_basename: &'static str,
    pub lcd_kind: LcdKind,
    pub rom_window_start: u32,
    pub rom_window_len: usize,
    pub font_base_addr: Option<u32>,
    pub text_decoder: Option<DeviceTextDecoderKind>,
    pub timer: DeviceTimerProfile,
    pub internal_ram_mirror: bool,
    pub keyboard: DeviceKeyboardProfile,
    pub sio_stub: bool,
    pub default_memory_card: DeviceMemoryCardProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceKeyboardProfile {
    pub columns_active_high: bool,
    pub fifo_mirroring: bool,
    pub keyi_on_any_press: bool,
    pub raw_kil: bool,
    pub press_threshold: u8,
}

impl DeviceKeyboardProfile {
    pub fn apply(self, keyboard: &mut KeyboardMatrix) {
        keyboard.set_columns_active_high(self.columns_active_high);
        keyboard.set_fifo_mirroring(self.fifo_mirroring);
        keyboard.set_keyi_on_any_press(self.keyi_on_any_press);
        keyboard.set_raw_kil(self.raw_kil);
        keyboard.set_press_threshold(self.press_threshold);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryCardProfile {
    Absent,
    BlankWritable64KiB,
}

impl DeviceMemoryCardProfile {
    pub fn is_present(self) -> bool {
        matches!(self, Self::BlankWritable64KiB)
    }

    pub fn apply(self, memory: &mut MemoryImage) -> Result<()> {
        match self {
            Self::Absent => memory.set_memory_card_slot_present(false),
            Self::BlankWritable64KiB => {
                memory.set_memory_card_slot_present(true);
                memory.load_memory_card(&vec![0; 65_536])?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerProfileProvenance {
    PcE500BoardEstimate,
    PcE500CompatibilityFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceTimerProfile {
    pub cpu_hz: u64,
    pub mti_period: u64,
    pub sti_period: u64,
    pub provenance: TimerProfileProvenance,
}

impl DeviceTimerProfile {
    pub fn new_context(self, enabled: bool) -> TimerContext {
        if enabled {
            TimerContext::new(true, self.mti_period as i32, self.sti_period as i32)
        } else {
            TimerContext::new(false, 0, 0)
        }
    }

    pub fn is_provisional(self) -> bool {
        matches!(
            self.provenance,
            TimerProfileProvenance::PcE500CompatibilityFallback
        )
    }

    pub fn provenance_label(self) -> &'static str {
        match self.provenance {
            TimerProfileProvenance::PcE500BoardEstimate => "PC-E500 board-clock estimate",
            TimerProfileProvenance::PcE500CompatibilityFallback => {
                "provisional PC-E500-compatible fallback; IQ-7000 timing is uncalibrated"
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceTextDecoderKind {
    Pce500,
    Iq7000,
}

impl DeviceTextDecoderKind {
    fn build(self, rom: &[u8]) -> Option<DeviceTextDecoder> {
        match self {
            Self::Pce500 => pce500::pce500_font_map_from_rom(rom).map(DeviceTextDecoder::Pce500),
            Self::Iq7000 => {
                let small_font = Iq7000FontMap::from_rom(rom, 0x00F_1B45);
                let large_font = Iq7000LargeFontMap::from_rom(rom, 0x00F_2145);
                if small_font.is_empty() && large_font.is_empty() {
                    return None;
                }
                Some(DeviceTextDecoder::Iq7000 {
                    small_font,
                    large_font,
                })
            }
        }
    }
}

/// Supported complete machine profiles around the shared SC62015 CPU core.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceModel {
    #[cfg_attr(feature = "cli", value(name = "iq-7000"))]
    #[serde(rename = "iq-7000")]
    Iq7000,
    #[cfg_attr(feature = "cli", value(name = "pc-e500"))]
    #[serde(rename = "pc-e500")]
    PcE500,
    #[cfg_attr(feature = "cli", value(name = "pc-e500-jp"))]
    #[serde(rename = "pc-e500-jp")]
    PcE500Jp,
}

/// Configure device-specific LCD character matching for Perfetto tracing.
///
/// This installs the trie-backed matcher used by the "LCD Characters" Perfetto track so that
/// write streams can be decoded into glyph slices. Callers must pass a ROM blob that includes the
/// device's font tables (either the raw ROM image or a snapshot's external memory dump).
pub fn configure_lcd_char_tracing(lcd: &mut dyn LcdHal, model: DeviceModel, rom: &[u8]) {
    match model {
        DeviceModel::PcE500 | DeviceModel::PcE500Jp => {
            let matcher = pce500::pce500_font_map_from_rom(rom)
                .and_then(|font| LcdCharMatcher::from_pce500_font_map(&font));
            if let Some(controller) = lcd.as_any_mut().downcast_mut::<LcdController>() {
                controller.set_char_matcher(matcher);
            }
        }
        DeviceModel::Iq7000 => {
            let small_font = Iq7000FontMap::from_rom(rom, 0x00F_1B45);
            let large_font = Iq7000LargeFontMap::from_rom(rom, 0x00F_2145);
            let matcher = LcdCharMatcher::from_iq7000_font_maps(&small_font, &large_font);
            if let Some(controller) = lcd.as_any_mut().downcast_mut::<Iq7000LcdController>() {
                controller.set_char_matcher(matcher);
            }
        }
    }
}

impl DeviceModel {
    pub const DEFAULT: Self = Self::PcE500;

    pub fn spec(self) -> DeviceSpec {
        match self {
            Self::Iq7000 => DeviceSpec {
                label: "iq-7000",
                rom_basename: "iq-7000.bin",
                lcd_kind: LcdKind::Iq7000Vram,
                rom_window_start: iq7000::ROM_WINDOW_START as u32,
                rom_window_len: iq7000::ROM_WINDOW_LEN,
                font_base_addr: Some(0x00F_1B45),
                text_decoder: Some(DeviceTextDecoderKind::Iq7000),
                timer: DeviceTimerProfile {
                    cpu_hz: pce500::DEFAULT_CPU_HZ,
                    mti_period: pce500::DEFAULT_MTI_PERIOD,
                    sti_period: pce500::DEFAULT_STI_PERIOD,
                    provenance: TimerProfileProvenance::PcE500CompatibilityFallback,
                },
                internal_ram_mirror: false,
                keyboard: DeviceKeyboardProfile {
                    columns_active_high: true,
                    fifo_mirroring: false,
                    keyi_on_any_press: true,
                    raw_kil: true,
                    press_threshold: 6,
                },
                sio_stub: false,
                default_memory_card: DeviceMemoryCardProfile::Absent,
            },
            Self::PcE500 => DeviceSpec {
                label: "pc-e500",
                rom_basename: "pc-e500-en.bin",
                lcd_kind: LcdKind::Hd61202,
                rom_window_start: pce500::ROM_WINDOW_START as u32,
                rom_window_len: pce500::ROM_WINDOW_LEN,
                font_base_addr: Some(pce500::ROM_ENGLISH_FONT_BASE_ADDR),
                text_decoder: Some(DeviceTextDecoderKind::Pce500),
                timer: DeviceTimerProfile {
                    cpu_hz: pce500::DEFAULT_CPU_HZ,
                    mti_period: pce500::DEFAULT_MTI_PERIOD,
                    sti_period: pce500::DEFAULT_STI_PERIOD,
                    provenance: TimerProfileProvenance::PcE500BoardEstimate,
                },
                internal_ram_mirror: true,
                keyboard: DeviceKeyboardProfile {
                    columns_active_high: true,
                    fifo_mirroring: true,
                    keyi_on_any_press: false,
                    raw_kil: false,
                    press_threshold: 1,
                },
                sio_stub: true,
                default_memory_card: DeviceMemoryCardProfile::BlankWritable64KiB,
            },
            Self::PcE500Jp => DeviceSpec {
                label: "pc-e500-jp",
                rom_basename: "pc-e500-jp.bin",
                lcd_kind: LcdKind::Hd61202,
                rom_window_start: pce500::ROM_WINDOW_START as u32,
                rom_window_len: pce500::ROM_WINDOW_LEN,
                font_base_addr: Some(pce500::ROM_JP_FONT_ATLAS_BASE_ADDR),
                text_decoder: Some(DeviceTextDecoderKind::Pce500),
                timer: DeviceTimerProfile {
                    cpu_hz: pce500::DEFAULT_CPU_HZ,
                    mti_period: pce500::DEFAULT_MTI_PERIOD,
                    sti_period: pce500::DEFAULT_STI_PERIOD,
                    provenance: TimerProfileProvenance::PcE500BoardEstimate,
                },
                internal_ram_mirror: true,
                keyboard: DeviceKeyboardProfile {
                    columns_active_high: true,
                    fifo_mirroring: true,
                    keyi_on_any_press: false,
                    raw_kil: false,
                    press_threshold: 1,
                },
                sio_stub: true,
                default_memory_card: DeviceMemoryCardProfile::BlankWritable64KiB,
            },
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.as_str() {
            "iq-7000" | "iq7000" | "iq_7000" => Some(Self::Iq7000),
            "pc-e500" | "pce500" | "pc_e500" => Some(Self::PcE500),
            "pc-e500-jp" | "pce500-jp" | "pc_e500_jp" | "pce500jp" => Some(Self::PcE500Jp),
            _ => None,
        }
    }

    pub fn is_pce500_family(self) -> bool {
        matches!(self, Self::PcE500 | Self::PcE500Jp)
    }

    pub fn label(self) -> &'static str {
        self.spec().label
    }

    pub fn rom_basename(self) -> &'static str {
        self.spec().rom_basename
    }

    pub fn lcd_kind(self) -> LcdKind {
        self.spec().lcd_kind
    }

    pub fn rom_window_start(self) -> u32 {
        self.spec().rom_window_start
    }

    pub fn rom_window_len(self) -> usize {
        self.spec().rom_window_len
    }

    pub fn font_base_addr(self) -> Option<u32> {
        self.spec().font_base_addr
    }

    pub fn text_decoder(self, rom: &[u8]) -> Option<DeviceTextDecoder> {
        self.spec().text_decoder.and_then(|kind| kind.build(rom))
    }

    pub fn timer_profile(self) -> DeviceTimerProfile {
        self.spec().timer
    }

    pub fn default_memory_card_profile(self) -> DeviceMemoryCardProfile {
        self.spec().default_memory_card
    }

    pub fn configure_keyboard(self, keyboard: &mut KeyboardMatrix) {
        self.spec().keyboard.apply(keyboard);
    }

    pub fn configure_runtime(&self, rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
        rt.set_device_model(*self)?;
        *rt.timer = self.timer_profile().new_context(true);
        rt.lcd = Some(create_lcd(self.lcd_kind()));
        if let Some(lcd) = rt.lcd.as_deref_mut() {
            configure_lcd_char_tracing(lcd, *self, rom);
        }
        if let Some(kb) = rt.keyboard.as_mut() {
            self.configure_keyboard(kb);
        }
        rt.sio = None;
        let result = match self {
            Self::Iq7000 => iq7000::load_iq7000_rom_image(rt, rom),
            Self::PcE500 => pce500::load_pce500_rom_window(rt, rom),
            Self::PcE500Jp => pce500::load_pce500_system_image(rt, rom),
        };
        result?;
        if self.spec().sio_stub {
            rt.enable_sio_stub();
        }
        self.default_memory_card_profile().apply(&mut rt.memory)
    }
}

#[derive(Debug, Clone)]
pub enum DeviceTextDecoder {
    Pce500(Pce500FontMap),
    Iq7000 {
        small_font: Iq7000FontMap,
        large_font: Iq7000LargeFontMap,
    },
}

impl DeviceTextDecoder {
    pub fn decode_display_text(&self, lcd: &dyn LcdHal) -> Vec<String> {
        match self {
            Self::Pce500(font) => decode_display_text(lcd, font),
            Self::Iq7000 {
                small_font,
                large_font,
            } => decode_iq7000_display_text_auto(lcd, small_font, large_font),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryCardMode;

    #[test]
    fn timer_profiles_preserve_model_specific_provenance() {
        let pc = DeviceModel::PcE500.timer_profile();
        let iq = DeviceModel::Iq7000.timer_profile();

        assert_eq!(pc.provenance, TimerProfileProvenance::PcE500BoardEstimate);
        assert!(!pc.is_provisional());
        assert_eq!(
            iq.provenance,
            TimerProfileProvenance::PcE500CompatibilityFallback
        );
        assert!(iq.is_provisional());
        assert_eq!(iq.mti_period, pc.mti_period);
        assert_eq!(iq.sti_period, pc.sti_period);
    }

    #[test]
    fn configure_runtime_installs_selected_models_timer_profile() {
        for model in [
            DeviceModel::PcE500,
            DeviceModel::PcE500Jp,
            DeviceModel::Iq7000,
        ] {
            let mut runtime = CoreRuntime::new();
            model
                .configure_runtime(&mut runtime, &[])
                .expect("configure model runtime");
            let expected = model.timer_profile();
            assert!(runtime.timer.enabled);
            assert_eq!(runtime.timer.mti_period, expected.mti_period);
            assert_eq!(runtime.timer.sti_period, expected.sti_period);
        }
    }

    #[test]
    fn complete_profiles_do_not_leak_pc_devices_into_iq7000() {
        let mut runtime =
            CoreRuntime::for_model(DeviceModel::Iq7000, &[]).expect("construct IQ-7000 runtime");
        let iq_keyboard = runtime
            .keyboard
            .as_ref()
            .expect("IQ-7000 keyboard")
            .snapshot_state();
        assert!(!iq_keyboard.emit_events);
        assert!(iq_keyboard.keyi_on_any_press);
        assert!(iq_keyboard.raw_kil);
        assert!(runtime.sio.is_none());
        assert_eq!(
            runtime
                .memory
                .memory_card_snapshot()
                .expect("IQ-7000 card snapshot")
                .expect("configured IQ-7000 slot")
                .mode,
            MemoryCardMode::Absent
        );

        DeviceModel::PcE500
            .configure_runtime(&mut runtime, &[])
            .expect("switch to PC-E500");
        let pc_keyboard = runtime
            .keyboard
            .as_ref()
            .expect("PC-E500 keyboard")
            .snapshot_state();
        assert!(pc_keyboard.emit_events);
        assert!(!pc_keyboard.keyi_on_any_press);
        assert!(!pc_keyboard.raw_kil);
        assert!(runtime.sio.is_some());
        assert_eq!(
            runtime
                .memory
                .memory_card_snapshot()
                .expect("PC-E500 card snapshot")
                .expect("configured PC-E500 slot")
                .mode,
            MemoryCardMode::Present
        );

        DeviceModel::Iq7000
            .configure_runtime(&mut runtime, &[])
            .expect("switch back to IQ-7000");
        assert!(runtime.sio.is_none());
    }
}
