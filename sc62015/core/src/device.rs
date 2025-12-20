// PY_SOURCE: pce500/emulator.py:PCE500Emulator
// PY_SOURCE: pce500/run_pce500.py

use crate::create_lcd;
use crate::lcd::{LcdHal, LcdKind};
use crate::lcd_text::{decode_display_text, Pce500FontMap};
use crate::pce500;
use crate::{CoreRuntime, Result};
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
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceTextDecoderKind {
    Pce500,
}

impl DeviceTextDecoderKind {
    fn build(self, rom: &[u8]) -> Option<DeviceTextDecoder> {
        match self {
            Self::Pce500 => pce500::pce500_font_map_from_rom(rom).map(DeviceTextDecoder::Pce500),
        }
    }
}

/// Supported device/ROM models. These primarily affect defaults (ROM selection, LCD controller,
/// font decoding) rather than the SC62015 CPU core.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceModel {
    #[cfg_attr(feature = "cli", value(name = "iq-7000"))]
    #[serde(rename = "iq-7000")]
    Iq7000,
    #[cfg_attr(feature = "cli", value(name = "pc-e500"))]
    #[serde(rename = "pc-e500")]
    PcE500,
}

impl DeviceModel {
    pub const DEFAULT: Self = Self::Iq7000;

    pub fn spec(self) -> DeviceSpec {
        match self {
            Self::Iq7000 => DeviceSpec {
                label: "iq-7000",
                rom_basename: "iq-7000.bin",
                lcd_kind: LcdKind::Unknown,
                // Placeholder: IQ-7000 mapping is not yet confirmed. Keep the PC-E500 window for now.
                rom_window_start: pce500::ROM_WINDOW_START as u32,
                rom_window_len: pce500::ROM_WINDOW_LEN,
                font_base_addr: None,
                text_decoder: None,
            },
            Self::PcE500 => DeviceSpec {
                label: "pc-e500",
                rom_basename: "pc-e500.bin",
                lcd_kind: LcdKind::Hd61202,
                rom_window_start: pce500::ROM_WINDOW_START as u32,
                rom_window_len: pce500::ROM_WINDOW_LEN,
                font_base_addr: Some(pce500::ROM_FONT_BASE_ADDR),
                text_decoder: Some(DeviceTextDecoderKind::Pce500),
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
            _ => None,
        }
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

    pub fn configure_runtime(&self, rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
        rt.set_device_model(*self);
        rt.lcd = Some(create_lcd(self.lcd_kind()));
        // Placeholder: share the existing PC-E500 window loader for now.
        pce500::load_pce500_rom_window(rt, rom)
    }
}

#[derive(Debug, Clone)]
pub enum DeviceTextDecoder {
    Pce500(Pce500FontMap),
}

impl DeviceTextDecoder {
    pub fn decode_display_text(&self, lcd: &dyn LcdHal) -> Vec<String> {
        match self {
            Self::Pce500(font) => decode_display_text(lcd, font),
        }
    }
}
